import asyncio
import os

import tiktoken
from openai import AsyncOpenAI
from tenacity import retry, wait_exponential, stop_after_attempt
from transformers import AutoTokenizer

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


class AdvancedLLM:
    def __init__(self, api_key: str, model: str = "qwen3.6-plus"):
        # 接入阿里云的 OpenAI 兼容接口
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.model = model
        self.history = [{"role": "system", "content": "你是一个很细心的助手"}]
        try:
            print("正在加载 Qwen Tokenizer，第一次运行可能需要几秒钟下载词表...")
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        except Exception as e:
            raise RuntimeError(f"分词器加载失败，请检查网络是否能连通HuggingFace，报错: {e}")

    """
    实现精确计算当前messages列表到底占用了多少Token。 
    """

    def _count_tokens(self, messages: list) -> int:
        """
        利用 Qwen 官方的 Chat Template 精确计算当前 messages 列表占用了多少 Token
        """
        try:
            # apply_chat_template 会自动把 messages 列表按照 Qwen 的格式拼成字符串并分词
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,  # 加上让模型开始生成的特殊 token
                return_tensors=None,  # 我们只需要普通的 Python list 长度
            )
            return len(input_ids)
        except Exception as e:
            # 生产环境遇到算Token报错绝不能让主流程崩溃，要有兜底机制
            print(f"[Warning] Token计算异常: {e}，将返回近似值。")
            return len(str(messages)) // 2  # 极其粗略的兜底

    """
    如果 history 的总 token 超过了 max_tokens，删除最早的对话，保留 system prompt。
    """

    def _trim_history(self, max_tokens: int = 3000):
        tokens = self._count_tokens(self.history)
        if tokens <= max_tokens:
            return

        while len(self.history) > 3 and self._count_tokens(self.history) > max_tokens:
            # user,assistant连着踢
            self.history.pop(1)
            self.history.pop(1)

        if self._count_tokens(self.history) > max_tokens:
            print("\n[警告] 当前单轮输入过长，触发强制失忆机制！")
            # 直接清空历史，只保留雷打不动的 System Prompt，丢弃前面所有的上下文
            self.history = [self.history[0]]

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    async def async_chat_stream(self, user_message: str):
        """
        1. 将 user_message 加入 history
        2. 触发 _trim_history 确保不超载
        3. 调用 client.chat.completions.create (必须带 stream=True)
        4. 异步 yield 解析出的文本块
        5. 在流输出结束后，把完整的 assistant 回答加入 history
        """
        current_message = {"role": "user", "content": user_message}
        self.history.append(current_message)

        self._trim_history()

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=self.history,
            stream=True,
            temperature=0.7
        )
        full_responst = ""
        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                full_responst += content
                yield content

        self.history.append({"role": "assistant", "content": full_responst})


# 测试入口
async def main():
    llm = AdvancedLLM(api_key=os.getenv("DASHSCOPE_API_KEY"))
    # 写一个 while True 循环，在终端里接收你的输入，并流式打印大模型的输出。
    while True:
        user_input = input("\n请开始输入：").strip()
        async for chunk in llm.async_chat_stream(user_input):
            print(chunk, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
