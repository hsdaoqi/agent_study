import os
import re
from typing import List, Dict
import dashscope
from chromadb.utils import embedding_functions
from pypdf import PdfReader
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings


# 首先要自己先实现这个类
class DashScopeEmbeddingFunction(EmbeddingFunction):
    def __init__(self, api_key: str, model_name: str = "text-embedding-v2"):
        print(f"[Debug] API Key is: {os.getenv('DASHSCOPE_API_KEY')}", flush=True)
        self.api_key = api_key
        self.model_name = model_name
        dashscope.api_key = self.api_key

    def __call__(self, input: Documents) -> Embeddings:
        # 提醒：大厂接口通常都有频率限制，如果你一次性传1000个文档，可能会报错
        # 这里直接调用阿里的SDK
        resp = dashscope.TextEmbedding.call(
            model=self.model_name,
            input=input
        )
        if resp.status_code == 200:
            # 提取向量结果，转换成 float 列表
            return [embeddings['embedding'] for embeddings in resp.output['embeddings']]
        else:
            raise Exception(f"DashScope Error: {resp.message}")


class DocumentProcessor:
    def __init__(self, persist_directory: str = './chroma_db', collection_name: str = "paper_db"):
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        # 创建向量化函数，有点慢
        self.embedding_fn = DashScopeEmbeddingFunction(os.getenv("DASHSCOPE_API_KEY"))

        # 获取或创建collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        """解析PDF，返回包含页码的文本块集合"""
        print(f"[System] 正在冷酷地撕碎并解析PDF: {pdf_path}...")
        reader = PdfReader(pdf_path)
        pages_data = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                clean_text = re.sub(r"\s+", " ", text).strip()
                pages_data.append({'text': clean_text, "page_num": i + 1})

        return pages_data

    def recursive_split(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        手搓的递归字符切分器雏形
        为什么要有 overlap？为了防止一句话被从中间硬生生切断，导致语义丢失！
        """
        chunks = []
        start = 0
        text_length = len(text)

        # 如果文本还没 overlap 长，直接返回整段，别瞎折腾
        if text_length <= chunk_size:
            return [text]

        while start < text_length:
            end = start + chunk_size
            if end >= text_length:
                chunks.append(text[start:])
                break

            # 寻找切分点
            split_point = end
            for punct in ['。', '！', '？', '\n']:
                last_punct = text.rfind(punct, start, end)
                if last_punct != -1:
                    # 找到了标点，切分点定在标点之后
                    split_point = last_punct + 1
                    break

            # 如果找到的切分点太靠前，导致 start 无法推进，则强制使用 end
            if split_point <= start + overlap:
                split_point = end

            chunks.append(text[start:split_point])

            # 确保下一次开始的位置一定比当前 start 大
            start = split_point - overlap

            # 终极保底：如果发生了意外导致 start 停滞，强制推进
            if start < 0:
                start = 0  # 防止负数索引

        return chunks

    def process_and_store(self, pdf_path: str, doc_name: str):
        """端到端：解析 -> 切块 -> 向量化入库"""
        pages_data = self.extract_text_from_pdf(pdf_path)

        all_chunks, all_metadatas, all_ids = [], [], []
        chunk_id_counter = 0

        for page in pages_data:
            chunks = self.recursive_split(page["text"])
            for chunk in chunks:
                all_chunks.append(chunk)
                all_metadatas.append({"source": doc_name, "page": page["page_num"]})
                all_ids.append(f"{doc_name}_chunk_{chunk_id_counter}")
                chunk_id_counter += 1

        total_chunks = len(all_chunks)
        print(f"[System] 共切分出 {total_chunks} 个 Chunk，开始分批注入...")

        # --- 核心改进：分批处理 (每批 20 条) ---
        batch_size = 20
        for i in range(0, total_chunks, batch_size):
            end_idx = min(i + batch_size, total_chunks)

            # 截取当前批次
            batch_chunks = all_chunks[i:end_idx]
            batch_metas = all_metadatas[i:end_idx]
            batch_ids = all_ids[i:end_idx]

            print(f"[Batch] 正在处理第 {i} 到 {end_idx} 条数据...")

            try:
                # 批量插入当前批次
                self.collection.add(
                    documents=batch_chunks,
                    metadatas=batch_metas,
                    ids=batch_ids
                )
            except Exception as e:
                print(f"[Error] 第 {i} 批数据注入失败: {e}")
                # 失败了可以考虑记录日志或重试
                continue

        print("[System] 知识库构建成功，逻辑闭环。")


# 测试入口
if __name__ == "__main__":
    processor = DocumentProcessor()
    # TODO: 去下载一篇你研究方向的PDF，放到同级目录下
    processor.process_and_store("fineLLM.pdf", "fine_Paper")
