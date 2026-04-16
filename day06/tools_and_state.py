from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from loguru import logger
from langchain_core.tools import tool


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# 2. 定义一个极其危险的工具
@tool
def send_email_to_boss(content: str) -> str:
    """
    【高危工具】：向老板发送重要邮件。
    参数：
        content: 邮件的具体内容。
    """
    # 真实生产环境这里会调用 SMTP 或内部邮件网关
    logger.error(f"🚨 [DANGER] 正在执行真实物理操作：发送邮件给老板！")
    logger.warning(f"✉️ 邮件内容: {content}")
    return f"邮件已成功发送给老板。内容为：{content}"


@tool
def normal_search(query: str) -> str:
    """普通的搜索工具，无危险性"""
    logger.info(f"🔍 [Tool] 正在搜索: {query}")
    return f"搜索结果：关于 {query} 的资料显示，今年公司业绩压力很大。"


# 将工具打包
tools = [send_email_to_boss, normal_search]
