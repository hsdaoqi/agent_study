import os

from langchain_openai import ChatOpenAI
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from loguru import logger

from agent_study.day06.tools_and_state import AgentState, tools

llm = ChatOpenAI(
    model="qwen3.6-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=1.3
)
llm_with_tools = llm.bind_tools(tools)


def assistant_node(state: AgentState):
    """大脑节点：负责思考和决定调用什么工具"""
    logger.info("🧠[Agent] 正在思考下一步行动...")
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# 工具节点我们直接用官方预置的 ToolNode，它会自动解析 LLM 的工具调用并执行
action_node = ToolNode(tools)


# 3. 路由判断：是否需要调工具？
def should_continue(state: AgentState):
    """判断大模型是否输出了 tool_calls"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        logger.info(f"🧭 [Router] 大模型决定调用工具，导向 action_node...")
        return "action"
    logger.success("🎉 [Router] 大模型没有调用工具，任务结束，导向 END。")
    return END


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node('assistant', assistant_node)
    builder.add_node('action', action_node)

    builder.add_edge(START, 'assistant')
    builder.add_conditional_edges(
        "assistant",
        should_continue,
        {
            "action": "action",
            END: END
        }
    )
    builder.add_edge("action", "assistant")
    return builder
