
---

### 第一课：ChromaDB 的三种生存形态（极其重要）

ChromaDB 有三种运行模式，你必须根据场景选择：

1.  **内存模式（Ephemeral）**：纯测试用，程序一关，数据全没。**（大厂严禁使用）**
2.  **本地持久化模式（Persistent）**：存在本地文件夹里，我下午给你的代码用的就是这个。适合单机部署和个人项目。
3.  **客户端/服务器模式（Client/Server）**：真正的工业级用法！数据库跑在独立的 Docker 容器里，Python 代码通过 HTTP/gRPC 去连接它。

---

### 第二课：Python 核心操作全解析

#### 1. 连接数据库
```python
import chromadb

# 形态一：小白内存模式（别用）
# client = chromadb.Client()

# 形态二：持久化模式（你现在用的）
# 它会在当前目录下生成一个 db_data 文件夹，存 sqlite 格式的元数据和向量数据
client = chromadb.PersistentClient(path="./db_data")

# 形态三：服务器模式（进阶，假设你以后用 Docker 部署了 Chroma 镜像）
# client = chromadb.HttpClient(host='localhost', port=8000)
```

#### 2. Embedding 引擎：不要用默认的垃圾！
ChromaDB 默认用的是 `all-MiniLM-L6-v2` 这个英文模型。但是不能处理中文数据, 必须显式指定 Embedding 函数！

```python
from chromadb.utils import embedding_functions

# 方式A：使用 OpenAI / 硅基流动的 API (适合云端运算，不吃本地显卡)
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="your_api_key",
    model_name="text-embedding-3-small"
)

# 方式B：使用本地开源模型 (我下午给你写的，适合断网/保密环境)
# 内部调用了 sentence-transformers，你需要提前 pip install
huggingface_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-small-zh-v1.5"
)

# 方式C:使用其他的模型
import dashscope
from chromadb import EmbeddingFunction, Documents, Embeddings

# 首先要自己先实现这个类
class DashScopeEmbeddingFunction(EmbeddingFunction):
    def __init__(self, api_key: str, model_name: str = "text-embedding-v2"):
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

# 1. 实例化你刚才写的类 (以阿里为例)
my_ali_ef = DashScopeEmbeddingFunction(api_key="你的阿里API_KEY")
# 2. 连接数据库
client = chromadb.PersistentClient(path="./china_tech_db")

# 3. 创建 Collection 时把这个函数传进去
collection = client.get_or_create_collection(
    name="my_cn_collection",
    embedding_function=my_ali_ef # 重点：注入你自定义的函数！
)
```

#### 3. 核心容器：Collection（集合）
在关系型数据库（MySQL）里叫 Table（表），在向量数据库里叫 Collection。

```python
# 创建集合
# hnsw:space 是距离度量。默认是 l2（欧氏距离）。
# 严厉警告：做文本相似度，必须强制用 cosine（余弦相似度）！
collection = client.get_or_create_collection(
    name="my_academic_papers",
    embedding_function=huggingface_ef  # 或者my_ali_ef,
    metadata = {"hnsw:space": "cosine"}  # 细节：必须指定度量算法！
)

# 查看库里有多少数据
print(collection.count())
```

#### 4. CRUD（增删改查） —— 数据注入
向集合里塞数据。注意，`ids` 是必须且唯一的，重复的 `id` 会报错！

```python
# 添加数据 (Add)
collection.add(
    documents=[
        "大语言模型具备极强的上下文学习能力。",
        "B站大学的计算机学科在NLP领域有深厚积累。",
        "李雷今天早上吃了一个煎饼果子。"
    ],
    metadatas=[
        {"source": "AI_Paper", "author": "Andrew Ng"},
        {"source": "University_Info", "author": "Admin"},
        {"source": "Daily_Life", "author": "LiLei"}
    ],
    ids=["doc1", "doc2", "doc3"] 
    # 注意：这里我们没有传 embeddings=[]。因为你在上一步配置了 embedding_function，
    # Chroma 会自动在底层把 documents 变成向量！非常优雅。
)
```

#### 5. 向量检索 + 元数据过滤（Metadata Filtering）
如果你只是单纯做相似度搜索，那叫玩具。大厂真正的 RAG 一定会带上**元数据过滤**。

比如用户问：“吴恩达的论文里，提到了大模型什么能力？”
如果你不加过滤，可能会把“李雷吃煎饼”这种无关信息，或者其他作者的论文搜出来。

```python
results = collection.query(
    query_texts=["大模型的能力是什么？"], 
    n_results=2, # 取 Top-2 相似的
    # Where 过滤！在进行向量 KNN 搜索前，先过滤掉作者不是吴恩达的数据。
    where={"author": "Andrew Ng"}, 
    # where_document={"$contains": "模型"} # 甚至能针对文本做正则过滤
)

print(results)
# 返回的是一个巨型字典，包含 'ids', 'distances', 'metadatas', 'documents'
```

返回值解析：
```python
{
  'ids': [['doc1']], 
  'distances': [[0.15]], # 距离越小越相似（因为你选了 cosine）
  'metadatas': [[{'author': 'Andrew Ng', 'source': 'AI_Paper'}]], 
  'documents': [['大语言模型具备极强的上下文学习能力。']]
}
#返回多个答案

{
  'ids': [['doc_A', 'doc_B']], 
  'distances': [[0.12, 0.25]], # 按相似度排序，0.12 的最相似
  'metadatas': [
      [
          {'author': 'Andrew Ng', 'source': 'Paper_1'}, 
          {'author': 'Yann LeCun', 'source': 'Paper_2'}
      ]
  ], 
  'documents': [
      ['大语言模型具备极强的上下文学习能力。', '卷积神经网络在视觉领域占据统治地位。']
  ]
}
```
你一定会问：“为什么要搞两层列表 [[...]]，直接一层不香吗？”
原因： 这是为了支持批量查询（Multi-querying）。
外层列表：对应你的问题数量。如果你同时输入 query_texts=["问题1", "问题2"]，外层列表就会有两个元素。
内层列表：对应每个问题的 Top-K 结果。
#### 6. 更新与删除
别存进去了就不管了，知识库是要迭代的。

```python
# 更新某条数据 (比如文档内容有错别字，或者 metadata 要改)
collection.update(
    ids=["doc3"],
    documents=["李雷今天早上吃了两个煎饼果子！"], # 更新文本
    metadatas=[{"source": "Daily_Life", "author": "LiLei", "mood": "hungry"}]
)

# 删除特定的 chunk
collection.delete(
    ids=["doc3"]
)

# 或者直接把整个集合删了 (慎用)
# client.delete_collection(name="my_academic_papers")
```

---
