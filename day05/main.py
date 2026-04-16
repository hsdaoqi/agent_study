from langgraph.graph import StateGraph, START, END
from loguru import logger
from state import AgentState
from nodes import writer_node, critic_node
import os

# 检查你的 API 钥匙配好没，别惹我发火
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("你这个笨蛋，API Key 忘配了！")


# ==========================================
# 1. 定义条件边（路由逻辑）：决定走向
# ==========================================
def should_continue(state: AgentState) -> str:
    """
    判断是继续重写，还是结束流程。
    """
    score = state.get("score", 0)
    revisions = state.get("revision_count", 0)

    if score >= 8:
        logger.success(f"🎉 [Router] 得分 {score} >= 8，主编十分满意，文章通过！")
        return END
    elif revisions >= 3:
        logger.error(f"🛑 [Router] 已经修改了 {revisions} 次还是个垃圾，触发强制熔断！不再修改。")
        return END
    else:
        logger.info(f"🔁 [Router] 得分 {score} 太低，文章被打回，要求重新撰写！")
        return "writer"  # 返回节点名称，让图流转回 writer 节点


# ==========================================
# 2. 编排图结构 (构建状态机)
# ==========================================
# 初始化一个图构建器，告诉它我们要用 AgentState 作为全局状态
builder = StateGraph(AgentState)

# 把你的两个 Agent 作为节点添加进去
builder.add_node("writer", writer_node)
builder.add_node("critic", critic_node)

# 定义图的起点：一进来先去哪？当然是先去写草稿！
builder.add_edge(START, "writer")

# 写完草稿去哪？必须去接受审查！
builder.add_edge("writer", "critic")

# 审查完去哪？这就不一定了！必须用条件边 (Conditional Edges)
builder.add_conditional_edges(
    "critic",  # 从 critic 节点出发
    should_continue,  # 执行这个判断函数
    {
        "writer": "writer",  # 如果函数返回 "writer"，就走向 writer 节点
        END: END  # 如果函数返回 END，就走向 END 结束图的运行
    }
)

# 编译图，变成一个可执行的应用！
multi_agent_app = builder.compile()

# ==========================================
# 3. 运行测试
# ==========================================
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("👩‍🏫 导师的考验：开始执行 Actor-Critic 多智能体流")
    print("=" * 50 + "\n")

    # 初始状态输入
    initial_state = {
        "topic": "为什么大厂都在用 LangGraph 而放弃了简单的循环代码？",
        "draft": "",
        "critique": "",
        "score": 0,
        "revision_count": 0
    }

    # 运行图（使用 stream 可以实时看到流转过程，这是大厂做可视化监控的基础）
    for event in multi_agent_app.stream(initial_state):
        for node_name, state_update in event.items():
            print(f"\n--- 经过了节点: [{node_name}] ---")
            # 打印部分状态让你可以观察
            if "draft" in state_update:
                print(f"[最新草稿片段]: {state_update['draft']}...")
            final_state = state_update

    print("终于跑完了，最后的结果是：", final_state.get("draft"))
