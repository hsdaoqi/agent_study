import asyncio
import tiktoken
from openai import AsyncOpenAI
from tenacity import retry, wait_exponential, stop_after_attempt


class AdvancedLLM:
    def __init__(self, api_key: str, model: str = "qwen3.6-plus"):
        self.client = AsyncOpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.model = model
        self.history = [{"role": "system", "content": "你是我的高冷御姐导师，要求极其严格..."}]
        self.tokenizer = tiktoken.encoding_for_model(self.model)

    def _count_tokens(self, messages: list) -> int:
        tokens_per_message = 3
        tokens_per_name = 1
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.item():
                num_tokens += len(self.tokenizer.encode(value))
        return num_tokens

    def _trim_history(self, max_tokens: int = 3000):
        """
        TODO: 如果 history 的总 token 超过了 max_tokens，删除最早的对话，保留 system prompt。
        """
        pass

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    async def async_chat_stream(self, user_message: str):
        """
        TODO:
        1. 将 user_message 加入 history
        2. 触发 _trim_history 确保不超载
        3. 调用 client.chat.completions.create (必须带 stream=True)
        4. 异步 yield 解析出的文本块
        5. 在流输出结束后，把完整的 assistant 回答加入 history
        """
        pass


# 测试入口
async def main():
    llm = AdvancedLLM(api_key="your-api-key")
    # TODO: 写一个 while True 循环，在终端里接收你的输入，并流式打印大模型的输出。


if __name__ == "__main__":
    asyncio.run(main())
