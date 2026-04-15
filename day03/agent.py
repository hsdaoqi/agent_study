import inspect
import os
import json
import asyncio
from typing import Callable

import pydantic
from openai import AsyncOpenAI
from agent_study.day03.tool_registry import registry
from dotenv import load_dotenv
import tools

load_dotenv()


class ReActAgent:
    def __init__(self, model: str = "qwen3.6-plus"):
        # 如果你没钱，把 base_url 换成国内的模型如 DeepSeek 或 硅基流动
        self.client = AsyncOpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 没钱就用这个
        )
        self.model = model
        self.system_prompt = (
            "你是华师大计算机系的顶级AI导师，高冷、严厉但极度专业。"
            "你需要根据用户要求，合理地使用工具。必须通过多步思考来解决问题。"
            "如果你需要通过工具调用的工程出现错误没得到信息，直接说没得到相关信息就好"
        )
        self.history = [{"role": "system", "content": self.system_prompt}]
        self.max_loops = 10  # 绝对防御：防止模型犯蠢陷入无限死循环刷爆信用卡！

    async def run_tool(self, func: Callable, argument_str: str, func_name: str, tool_call_id: str):
        try:
            model = registry.models.get(func_name)
            if not model:
                raise Exception(f"工具 {func_name} 未在注册表中找到模型")
            # 2. 【核心步骤】使用 Pydantic 进行强校验和类型转换
            # model_validate_json 会把字符串转成 Pydantic 对象，并自动处理类型转换
            try:
                validated_args = model.model_validate_json(argument_str)
                clean_args = validated_args.model_dump()
            except pydantic.ValidationError as e:
                # 如果校验失败，直接把报错信息扔回给大模型，让它自省（Self-Correction）
                error_msg = f"参数校验失败: {str(e)}"
                print(f"⚠️ [校验触发自愈] {error_msg}")
                return {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": func_name,
                    "content": error_msg
                }
                # 3. 执行真正的函数逻辑
            if inspect.iscoroutinefunction(func):
                result = await func(**clean_args)
            else:
                result = func(**clean_args)
            print(f"✅ [Result] {result}")

        except Exception as e:
            result = f"Error: {str(e)}"
            print(f"❌ [Error] {result}")

        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": func_name,
            "content": str(result)
        }

    async def chat(self, user_input: str):
        self.history.append({"role": "user", "content": user_input})

        loop_count = 0
        print("\n" + "=" * 50)
        print(f"👩‍🏫 导师接收到任务: {user_input}")

        # ---------------- 核心 ReAct 循环开始 ----------------
        while loop_count < self.max_loops:
            loop_count += 1
            print(f"\n[循环轮次 {loop_count}] 大脑思考中...")

            # 1. 带着工具向大模型发起请求
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=self.history,
                tools=registry.tools_schema,  # 挂载我们生成的 Schema
                tool_choice="auto",
                temperature=0.3
            )

            ai_msg = response.choices[0].message
            print(json.dumps(ai_msg.model_dump(), indent=2, ensure_ascii=False))
            # 将 AI 的回复（包含可能存在的 tool_calls）必须原封不动加入历史，这是 OpenAI 的硬性规范
            self.history.append(ai_msg)

            # 2. 判断是否有 Tool Calls（动作行动）
            if ai_msg.tool_calls:
                tasks = []
                for tool_call in ai_msg.tool_calls:
                    func_name = tool_call.function.name
                    arguments_str = tool_call.function.arguments
                    func = registry.tools.get(func_name)

                    tasks.append(self.run_tool(func, arguments_str, func_name, tool_call.id))

                results = await asyncio.gather(*tasks)
                for res in results:
                    self.history.append(res)
                continue
            else:
                # 如果没有 tool_calls，说明大模型认为已经得到最终答案了 (finish_reason == stop)
                print("\n🎉[思考结束] 输出最终回答:")
                return ai_msg.content

        # 如果超出了最大循环次数
        return "我已经思考了10次还是搞不定，你的问题太蠢了，或者我的逻辑陷入了死锁。"
