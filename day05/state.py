from typing import TypedDict
from pydantic import BaseModel, Field


# 1. 定义 Critic (审查员) 的强制输出格式
# 大厂绝对不允许审查员输出一段废话，必须是结构化的 JSON！
class CritiqueOutput(BaseModel):
    score: int = Field(description="给当前草稿打分，0到10分。")
    feedback: str = Field(description="详细的修改建议。如果满分，这里可以说'完美'。")
    pass_validation: bool = Field(description="如果分数 >= 8，则为 True，否则为 False。")


# 2. 定义全局共享状态 (State)
# 这就是整个图的血液，每个节点都可以读取它，并返回一个字典来更新它。
class AgentState(TypedDict):
    topic: str  # 用户输入的研究主题
    draft: str  # Writer 生成的当前草稿
    critique: str  # Critic 给出的反馈意见
    score: int  # 当前草稿的得分
    revision_count: int  # 记录修改了多少次，防止无限死循环！
