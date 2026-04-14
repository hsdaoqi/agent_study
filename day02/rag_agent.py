# day02/rag_agent.py
import asyncio
import os

import jieba
from rank_bm25 import BM25Okapi
from openai import AsyncOpenAI
from document_processor import DocumentProcessor


class RAGAgent:
    def __init__(self, api_key: str):
        # 继承下午的数据库连接
        self.doc_processor = DocumentProcessor()
        self.collection = self.doc_processor.collection

        # 初始化大模型客户端
        self.client = AsyncOpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.model = "qwen3.6-plus"  # 推荐模型

        # 初始化 BM25 环境 (通常用 Elasticsearch，这里用本地模拟)
        self._init_bm25()

    def _init_bm25(self):
        """将 ChromaDB 中的数据提出来建立 BM25 索引"""
        print("[System] 正在构建本地 BM25 索引...")
        all_docs = self.collection.get()
        self.bm25_docs = all_docs['documents']
        self.bm25_metadatas = all_docs['metadatas']

        if not self.bm25_docs:
            print("[Warning] 知识库为空，请先运行 ingest.py 注入数据！")
            self.bm25 = None
            return

        # 使用 jieba 对中文文档进行分词
        tokenized_corpus = [list(jieba.cut(doc)) for doc in self.bm25_docs]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def hybrid_retrieve(self, query: str, top_k: int = 3) -> list:
        # 1. 向量检索：获取原始内容和对应的 Metadata
        vector_res = self.collection.query(query_texts=[query], n_results=top_k)
        v_docs = vector_res['documents'][0]
        v_metas = vector_res['metadatas'][0]

        # 2. BM25 检索：获取原始内容和对应的 Metadata
        # (假设你已经按我之前的逻辑存了 self.bm25_docs 和 self.bm25_metadatas)
        tokenized_query = list(jieba.cut(query))
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top_n_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
        b_docs = [self.bm25_docs[i] for i in bm25_top_n_idx]
        b_metas = [self.bm25_metadatas[i] for i in bm25_top_n_idx]

        # 3. RRF 计算
        rrf_scores = {}
        doc_to_meta = {}  # 用来记录内容对应的元数据，方便找回

        # 处理向量结果
        for rank, (doc, meta) in enumerate(zip(v_docs, v_metas)):
            rrf_scores[doc] = rrf_scores.get(doc, 0) + 1.0 / (60 + rank + 1)
            doc_to_meta[doc] = meta

        # 处理 BM25 结果
        for rank, (doc, meta) in enumerate(zip(b_docs, b_metas)):
            rrf_scores[doc] = rrf_scores.get(doc, 0) + 1.0 / (60 + rank + 1)
            # 如果重复了，meta 是一样的，直接覆盖也没关系
            doc_to_meta[doc] = meta

        # 4. 排序输出
        final_sorted = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # 组装最终给 LLM 的上下文
        results = []
        for doc, score in final_sorted:
            results.append({
                "content": doc,
                "meta": doc_to_meta[doc],
                "rrf_score": score
            })
        return results

    async def ask(self, query: str):
        """端到端问答与流式输出"""
        # 步骤 1：检索上下文
        retrieved_chunks = self.hybrid_retrieve(query, top_k=3)

        if not retrieved_chunks:
            print("你的知识库是空的，我没法回答。去查查 PDF 导进去没有。")
            return

        # 步骤 2：格式化检索到的上下文，要求带上引用溯源
        context_str = ""
        for i, chunk in enumerate(retrieved_chunks):
            context_str += f"【资料 {i + 1}】(来源: {chunk['meta']['source']}, 第{chunk['meta']['page']}页):\n{chunk['content']}\n\n"

        # 步骤 3：构建严苛的 RAG Prompt
        system_prompt = """你是一个极其严谨的高级学术导师 Agent。
你的任务是根据用户提供的【参考资料】来回答问题。
要求：
1. 你的回答必须、且只能基于提供的参考资料。
2. 如果资料中没有提到答案，你必须直接回答“根据当前资料无法回答”，绝对不允许动用你自己的预训练知识胡编乱造！
3. 回答时，必须在对应的句子后标注引用来源，例如：“...（见资料1，第3页）”。"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"参考资料如下：\n{context_str}\n\n用户问题：{query}"}
        ]

        print("\n导师思考中 [正在检索知识库...] ")
        print("-" * 50)

        # 步骤 4：调用大模型并流式输出 (结合Day 1的知识)
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                temperature=0.1  # 任务必须把温度调低，防止发散
            )

            async for chunk in response:
                if chunk.choices[0].delta.content:
                    # 模拟打字机效果
                    print(chunk.choices[0].delta.content, end="", flush=True)
            print("\n" + "-" * 50)

        except Exception as e:
            print(f"\n[Error] API 调用崩溃了，检查你的网络和 Token。错误详情: {e}")


# 测试入口
async def main():
    # TODO: 填入你的 API KEY
    agent = RAGAgent(api_key=os.getenv("DASHSCOPE_API_KEY"))

    print("\n高冷导师已上线。输入 'quit' 退出。")
    while True:
        user_input = input("\n你想问这篇论文什么问题？> ")
        if user_input.lower() in ['quit', 'exit']:
            break
        await agent.ask(user_input)


if __name__ == "__main__":
    asyncio.run(main())
