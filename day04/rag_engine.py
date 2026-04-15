import asyncio
import os
from typing import List, Dict
from openai import AsyncOpenAI
from sentence_transformers import CrossEncoder
from loguru import logger


# 假设你 Day 2 的文档入库逻辑已经写好，这里用一个 Mock 的数据库模拟粗排（召回）过程
# 真实生产中你应该连接 ChromaDB 或 Milvus
class MockVectorDB:
    def search(self, query: str, top_k: int = 10) -> List[str]:
        # 假装这里从向量数据库捞出了 10 个 Chunk
        return [
            "Transformer的核心是Self-Attention机制，公式为 QK^T / sqrt(d_k)...",
            "Attention is All You Need是2017年Google发表的论文。",
            "上海明天的天气是暴雨转阴。",  # 这是个干扰项
            "Scaled Dot-Product Attention计算中，除了QK，还要乘以V。",
            "作者包括Ashish Vaswani等人，最初用于机器翻译任务。",
            "ResNet通过残差连接解决了深层网络梯度消失的问题。"  # 干扰项
        ]


class AdvancedRAGEngine:
    def __init__(self, api_key: str):
        self.llm_client = AsyncOpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        # 交叉重排模型：与普通双塔模型不同，CrossEncoder将(query, doc)拼在一起输入模型，准确度极高！
        logger.info("🚀 [RAG] 正在加载交叉重排模型 (CrossEncoder)... 这玩意吃显存，稍等。")
        self.reranker = CrossEncoder('BAAI/bge-reranker-base')
        self.db = MockVectorDB()

    async def rewrite_query(self, original_query: str) -> str:
        """
        Step 1: 查询重写 (Query Rewrite)
        解决指代不明、语义不完整的问题。
        """
        prompt = f"""你是一个专业的学术搜索引擎助手。请将用户的口语化、可能包含代词的问题，
        重写为清晰、独立、高度结构化的学术搜索关键词（不需要任何回答，只输出重写后的句子）。
        原问题：{original_query}
        重写后：
        """

        response = await self.llm_client.chat.completions.create(
            model="qwen3.6-plus",  # 重写用小模型即可，大厂通常用自建小模型省钱
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        rewritten = response.choices[0].message.content.strip()
        logger.info(f"✍️ [RAG|Rewrite] 原始Query: '{original_query}' -> 重写后: '{rewritten}'")
        return rewritten

    def rerank_documents(self, query: str, docs: List[str], top_k: int = 3) -> List[Dict]:
        """
        Step 3: 重排 (Rerank)
        将粗召回的文档，通过 CrossEncoder 进行精准打分。
        """
        if not docs:
            return []

        # 构造 (query, doc) 句对
        sentence_pairs = [[query, doc] for doc in docs]

        # 批量预测打分
        scores = self.reranker.predict(sentence_pairs)

        # 将文档和分数打包并排序
        doc_score_pairs = list(zip(docs, scores))
        # 按照分数降序排列
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        # 截取 Top-K
        top_docs = [{"content": doc, "relevance_score": float(score)} for doc, score in doc_score_pairs[:top_k]]

        logger.debug(f"⚖️ [RAG|Rerank] 重排完成。Top 1 分数: {top_docs[0]['relevance_score']:.4f}")
        for idx, doc in enumerate(top_docs):
            logger.debug(f"  -> Rank {idx + 1}: {doc['content'][:30]}...")

        return top_docs

    async def search(self, query: str) -> str:
        """暴露给外部的 RAG Pipeline"""
        logger.info(f"🔍 [RAG] 启动高级检索 Pipeline: {query}")

        # 1. 意图重写
        better_query = await self.rewrite_query(query)
        # better_query = query
        # 2. 向量粗排召回 (Recall Top 10)
        retrieved_docs = self.db.search(better_query, top_k=10)
        logger.info(f"📚 [RAG|Retrieve] 粗召回完成，获取 {len(retrieved_docs)} 篇文档块。")

        # 3. 交叉重排精搜 (Rerank Top 3)
        final_docs = self.rerank_documents(better_query, retrieved_docs, top_k=3)

        # 4. 组装最终上下文
        context = "\n".join([f"[引用片段 {i + 1}]: {doc['content']}" for i, doc in enumerate(final_docs)])
        return context


async def test():
    rag = AdvancedRAGEngine(os.getenv("DASHSCOPE_API_KEY"))
    result = await rag.search("那个什么transformer啥的相关信息有啥")
    print(result)


if __name__ == "__main__":
    asyncio.run(test())
