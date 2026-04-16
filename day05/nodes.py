import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from state import AgentState, CritiqueOutput

# 初始化模型（撰稿人可以用相对便宜的，但主编/裁判必须用最好的模型！）
actor_llm = ChatOpenAI(
    model="qwen3.6-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=1.3
)
critic_llm = ChatOpenAI(
    model="qwen3.6-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.1
)  # 裁判必须严谨，temperature 调低！


def writer_node(state: AgentState) -> dict:
    """Actor Agent: 负责根据反馈不断重写草稿"""
    logger.info(f"✍️ [Writer] 正在执行撰写... (当前修改次数: {state.get('revision_count', 0)})")

    topic = state["topic"]
    feedback = state.get("critique", "")

    if not feedback:
        # 第一次写，没有反馈
        prompt = f"你是一个资深的行业研究员。请写一份关于【{topic}】的研报。"
    else:
        # 被打回重写了！
        prompt = f"你需要修改关于【{topic}】的研报。上次的草稿被主编痛批了。\n主编的反馈意见是：{feedback}\n当前草稿内容：\n{state['draft']}\n\n请严格根据反馈重写一份更好的版本！"

    response = actor_llm.invoke(prompt)

    # 返回的内容会自动更新（覆盖）到 State 中对应的字段
    return {
        "draft": response.content,
        "revision_count": state.get("revision_count", 0) + 1  # 计数器 + 1
    }


def critic_node(state: AgentState) -> dict:
    """Critic Agent: 负责审阅草稿，并给出结构化打分"""
    logger.info("🧐 [Critic] 正在审查草稿...")

    # 强制让 LLM 输出我们刚才在 state.py 里定义的 Pydantic BaseModel 格式！
    # 这是构建稳定的 Agent 系统的神技！
    evaluator = critic_llm.with_structured_output(CritiqueOutput, method="json_mode")

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "你是一个极其严厉的大厂技术总监。你必须严格执行审查任务。"
            "你的打分必须极度苛刻。请分析草稿的逻辑和数据。,分数范围从0-10"
            "输出要求：禁止输出任何开场白、解释或列表格式。"  # 强力约束
            "输出必须是一个纯粹的、合法的 JSON 对象，符合以下字段定义：score, feedback, pass_validation。"
        )),
        ("user", "研报主题：{topic}\n\n待审查的草稿：\n{draft}")
    ])

    chain = prompt | evaluator
    result: CritiqueOutput = chain.invoke({"topic": state["topic"], "draft": state["draft"]})

    logger.warning(f"⚖️ [Critic] 审查完毕 | 得分: {result.score}/10 | 反馈: {result.feedback}...")

    return {
        "critique": result.feedback,
        "score": result.score
    }
