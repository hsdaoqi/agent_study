import json
import asyncio
import os

from openai import AsyncOpenAI
from loguru import logger
import traceback

from router import SemanticRouter
from tools import ToolRegistry


class SuperAgent:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = AsyncOpenAI(api_key=self.api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.router = SemanticRouter()
        self.tools_registry = ToolRegistry(api_key=self.api_key)
        self.model = "qwen3.6-plus"  # 复杂 Agent 推荐用 4 或者 Claude 3.5 Sonnet，小模型 hold 不住多工具

    async def fast_path_reply(self, intent: str, user_input: str) -> str:
        """快车道：简单的闲聊直接用小模型极速回复，不进工具循环，省时省钱！"""
        logger.info(f"⚡ [Agent|FastPath] 命中单意图快车道，直接生成回复...")
        system_prompt = "你是一个高冷、严厉但内心极其护短的AI导师。用御姐的口吻回答问题。"
        if intent == "chitchat":
            system_prompt += "用户在和你闲聊，你可以稍微傲娇一点回应。"

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
        )
        return response.choices[0].message.content

    async def react_loop(self, user_input: str, max_steps: int = 5) -> str:
        """
        🔥 工业级 ReAct (Tool Calling) 核心循环 🔥
        包含了：状态跟踪、死循环保护、异常回传自愈机制。
        """
        logger.info(f"🧠 [Agent|ReAct] 进入复杂任务处理模式，最大执行步数限制: {max_steps}")

        messages = [
            {"role": "system",
             "content": "你是一个顶级数据分析师兼AI专家导师。你可以使用多种工具组合来回答复杂问题。请确保回答排版精美（Markdown），并且信息准确。"},
            {"role": "user", "content": user_input}
        ]

        step_count = 0
        while step_count < max_steps:
            step_count += 1
            logger.info(f"🔄 [Agent|Loop] 第 {step_count}/{max_steps} 次思考迭代...")

            # 发起带有 Tools 的模型调用
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools_registry.tool_schemas,
                tool_choice="auto"  # 让模型自己决定调不调用工具
            )

            response_message = response.choices[0].message
            messages.append(response_message)  # 必须把模型的回复完整塞进历史记录

            # 判断 1：模型是否想调用工具？
            if response_message.tool_calls:
                for tool_call in response_message.tool_calls:
                    tool_name = tool_call.function.name
                    logger.info(f"🤖 [Agent|Thought] 大模型决定调用工具: {tool_name}")

                    try:
                        arguments = json.loads(tool_call.function.arguments)
                        logger.debug(f"🔧 [Agent|Action] 传递参数: {arguments}")

                        # 执行工具
                        tool_result = await self.tools_registry.execute_tool(tool_name, arguments)
                        logger.info(f"✅ [Agent|Observation] {tool_name} 执行完毕。")

                    except json.JSONDecodeError:
                        # 🚨 异常自愈机制 1：模型输出的 JSON 烂掉了
                        error_msg = f"系统报错：你传入的 JSON 格式错误。请修正后重试！"
                        logger.warning(f"⚠️ [Agent|Self-Correction] 参数解析失败，要求模型反思。")
                        tool_result = error_msg
                    except Exception as e:
                        # 🚨 异常自愈机制 2：其他未知错误，把 traceback 喂给模型
                        error_msg = f"系统执行崩溃，堆栈如下：{traceback.format_exc()}。请分析原因并采取备用方案。"
                        logger.error(f"❌ [Agent|Self-Correction] 工具崩溃，已回传堆栈。")
                        tool_result = error_msg

                    # 把工具的执行结果（Observation）作为特殊角色 "tool" 塞给大模型
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": tool_result
                    })
                # 工具执行完毕，进行下一次 while 循环，让大模型阅读结果并继续思考
                continue

            else:
                # 判断 2：大模型没有输出 tool_calls，说明它得出了最终答案！
                final_answer = response_message.content
                logger.info(f"🎉 [Agent|Finish] 任务完成，得出最终结论。")
                return final_answer

        # 熔断机制：如果跳出了 while 循环，说明超出了 max_steps
        logger.error(f"🛑 [Agent|KillSwitch] 检测到循环超过 {max_steps} 次，触发强制熔断！")
        return "导师系统警告：你的问题太复杂，导致计算资源超载，系统已强制中断处理。"

    async def run(self, user_input: str) -> str:
        """外部调用的统一入口：路由判定 + 任务分发"""
        print(f"\n👩‍🏫 用户提问: {user_input}\n{'-' * 50}")

        # 1. 语义路由拦截
        intent = self.router.route(user_input)

        # 2. 分支处理
        if intent in ["chitchat", "weather"]:
            # 注意：真实的系统里，如果是单纯查天气，可以直接拼个参数走快车道，这里简化处理
            if intent == "chitchat":
                return await self.fast_path_reply(intent, user_input)

        # 3. 复杂任务或意图不明，全部扔进重型 ReAct 循环
        return await self.react_loop(user_input)


# ================= 导师的魔鬼验收测试 =================
async def main():
    # 记得换成你自己的 API KEY

    API_KEY = os.getenv("DASHSCOPE_API_KEY")
    agent = SuperAgent(api_key=API_KEY)

    # 测试 1：纯闲聊，测试语义路由的拦截和低延迟
    res1 = await agent.run("哈喽导师，夸夸我今天有多努力！")
    print(f"\n最终回复:\n{res1}\n{'=' * 50}\n")

    # 测试 2：复杂串联任务（今天布置的终极考验）
    test_query = "嗨！帮我查一下华东师范大学今天的天气。另外，针对《Attention is All You Need》这篇论文（私有知识库），帮我总结一下它的核心架构，并去网上搜一下目前这篇论文的最新引用量是多少？最后把所有信息整合成一份简报给我。"
    res2 = await agent.run(test_query)
    print(f"\n最终简报:\n{res2}\n{'=' * 50}\n")


if __name__ == "__main__":
    asyncio.run(main())
