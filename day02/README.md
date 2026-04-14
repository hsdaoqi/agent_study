
---

# Day 02:  RAG 架构引擎实现 (Hybrid Search & RRF)

## 📌 项目概述
本项目为 30 天从入门到精通 AI Agent 特训营的第二天成果。核心目标是打破大模型的“幻觉”限制，通过构建本地私有知识库，实现基于 **RAG (Retrieval-Augmented Generation)** 的精准学术论文分析助手。

相比于初级的 RAG，本项目实现了**混合检索 (Hybrid Search)** 方案，结合了语义向量检索与传统词频检索，并使用 **RRF (Reciprocal Rank Fusion)** 算法进行结果对齐。

## 🚀 核心特性
- **递归文档切片 (Recursive Character Splitting)**：自主实现带 Overlap 的切分逻辑，确保文本块语义不因物理切割而断裂。
- **国产大厂 Embedding 集成**：通过自定义 `EmbeddingFunction` 接入 **阿里百炼 (DashScope) text-embedding-v2**，深度优化中文语义表示。
- **本地持久化向量库**：利用 **ChromaDB** 实现向量数据的本地索引与持久化存储，支持 Cosine Similarity 度量。
- **混合检索逻辑 (Hybrid Retrieval)**：
    - **Vector Search**: 捕捉深层语义。
    - **BM25 (Best Matching 25)**: 捕捉关键词、术语及编号，解决语义检索在长尾词上的乏力问题。
- **RRF 排序对齐**：使用 Reciprocal Rank Fusion 算法（平滑系数 $k=60$），将不同量级的分数归一化为排名权重，确保 Top-K 结果的绝对相关性。
- **引用溯源系统**：生成回答时强制要求关联 `Metadata`，标注信息来源页码，消除 AI 幻觉。

## 🛠️ 技术栈
- **Language**: Python 3.12+
- **LLM/Embedding**: Alibaba DashScope (Qwen/BGE)
- **Vector DB**: ChromaDB
- **Search Alg**: BM25, RRF
- **PDF Logic**: PyPDF

## 📂 项目结构
```text
day02/
├── document_processor.py   # 知识库构建管道 (PDF解析、分批入库)
├── rag_agent.py            # RAG 核心逻辑 (混合检索、RRF排序、流式回答)
├── .gitignore              # 严苛的工程过滤配置
└── README.md               # 本文档
```

## ⚙️ 快速开始

### 1. 环境准备
```bash
pip install chromadb dashscope pypdf rank_bm25 jieba openai
```

### 2. 配置 API Key
```bash
# Windows
set DASHSCOPE_API_KEY=your_ali_key
# Linux/Mac
export DASHSCOPE_API_KEY=your_ali_key
```

### 3. 构建知识库
将 PDF 文件放入目录，修改 `document_processor.py` 中的路径并运行：
```bash
python document_processor.py
```

### 4. 启动 Agent 对话
```bash
python rag_agent.py
```

## 🧠 导师审计 (Self-Check)
- [x] 是否实现了 `RecursiveCharacterSplitter` 防止语义断裂？
- [x] `collection.add` 是否使用了 `batch_size` 避开 API 并发限制？
- [x] 检索过程中是否对 Vector 和 BM25 的结果进行了 RRF 融合？
- [x] LLM 的 Prompt 是否强制约束“仅根据参考资料回答”？

---

**Author:** [道柒] | **Mentor:** Gemini\
**Status:** Day 02 Completed. 

---
