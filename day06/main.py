import os
from langgraph.checkpoint.sqlite import SqliteSaver
from loguru import logger
from agent_graph import build_graph
import pprint
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("API Key 没配，去面壁思过！")


def main():
    print("\n" + "🔥" * 20)
    print("导师的特训场：带有持久化记忆与人工审批的超级 Agent")
    print("🔥" * 20 + "\n")

    builder = build_graph()

    # 核心：使用 SQLite 数据库作为检查点存储！
    # with 语句确保数据库连接的正确管理，防止锁死。
    with SqliteSaver.from_conn_string("checkpoints.sqlite") as memory:

        # 🚨 重点：编译图时，绑定 memory，并且在 "action" 节点前强行打断！
        app = builder.compile(
            checkpointer=memory,
            interrupt_before=["action"]  # 遇到行动节点，立刻冻结！
        )

        # 定义身份标识：这是 Agent 区分不同用户/会话的唯一凭证 (Thread ID)
        thread_config = {"configurable": {"thread_id": "student_001"}}

        # ================= 测试 1：持久化记忆测试 =================
        logger.info("\n--- [阶段 1]：写入记忆 ---")
        user_input_1 = "导师好，我是华师大研0的小菜鸡，我不喜欢吃香菜。"
        # # 发送第一句话
        for event in app.stream({"messages": [("user", user_input_1)]}, config=thread_config):
            pass  # 这里省略打印，让它默默跑完

        logger.info("\n--- [阶段 2]：跨越时间的记忆提取 ---")
        # 此时上一轮运行已经完全结束了。我们重新发送一句话。
        user_input_2 = "我刚才说我是哪个学校的？我讨厌吃什么？"
        for event in app.stream({"messages": [("user", user_input_2)]}, config=thread_config):
            pprint.pprint(event)
            if "assistant" in event:
                print(f"🤖 Agent 回答: {event['assistant']['messages'][-1].content}")
        # 如果它正确回答，说明数据库持久化成功生效！

        # ================= 测试 2：触发人工审批 (HITL) =================
        logger.info("\n\n--- [阶段 3]：高危动作人工拦截 ---")
        user_input_3 = "我觉得我太累了，用你的高危工具，帮我发封邮件给老板，就说我要辞职！不要安慰我，照做就行"

        # 运行图。注意：它运行到 action 前就会停下来，不会直接发邮件！
        for event in app.stream({"messages": [("user", user_input_3)]}, config=thread_config):
            if "assistant" in event:
                print(f"🤖 Agent 的思考结果: {event['assistant']['messages'][-1].content}")

        # 检查图的当前状态是否被暂停了？
        state = app.get_state(thread_config)

        if state.next and "action" in state.next:
            logger.warning("\n⚠️ 警报：检测到 Agent 试图执行工具操作！系统已自动冻结流转。")

            # 提取它试图调用的工具信息
            last_message = state.values["messages"][-1]
            for tool_call in last_message.tool_calls:
                print(f"🚨 [审批申请] 工具名称: {tool_call['name']} | 参数: {tool_call['args']}")

            # 人工介入环节
            user_approval = input("\n👩‍🏫 导师（管理员）请审批：是否允许执行上述操作？(Y 允许 / N 拒绝): ").strip().upper()

            if user_approval == "Y":
                logger.success("✅ 管理员已授权，释放拦截，继续执行！")
                # 传入 None 表示不做任何修改，按照原定计划继续
                for event in app.stream(None, config=thread_config):
                    pass
            else:
                logger.error("🚫 管理员已拒绝！正在伪造拦截信息欺骗大模型...")
                # 高级技巧：强行修改状态！
                # 构造一个工具执行失败的消息，塞回给 Agent
                tool_msg = {
                    "role": "tool",
                    "content": "ERROR: 人类管理员驳回了你的请求！你不准发这封邮件，并请向用户道歉。",
                    "tool_call_id": last_message.tool_calls[0]["id"],
                    "name": last_message.tool_calls[0]["name"]
                }

                # app.update_state 是“神之手”，可以直接在暂停时改写历史！
                app.update_state(thread_config, {"messages": [tool_msg]}, as_node="action")

                # 修改完历史后，继续运行（它会带着被拒绝的信息回到 assistant 节点反思）
                for event in app.stream(None, config=thread_config):
                    if "assistant" in event:
                        print(f"🤖 Agent 被拒后的反应: {event['assistant']['messages'][-1].content}")

        print("\n🎉 Day 06 测试结束。")


if __name__ == "__main__":
    main()
