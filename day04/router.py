import os
import warnings
# 1. 关掉 HuggingFace 的 Windows 软链接警告
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
# 2. 关掉 TensorFlow/Keras 的烦人日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# 3. 忽略 Python 的 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Literal, List, Dict
from loguru import logger




class SemanticRouter:
    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", threshold: float = 0.60):
        """
        语义路由器
        :param model_name: 采用轻量级的中日英混合模型，速度极快
        :param threshold: 相似度阈值，低于此阈值说明不属于任何预设池
        """

        logger.info(f"🚀 [Router] 正在加载本地路由模型: {model_name}...")
        self.encoder = SentenceTransformer(model_name)
        self.threshold = threshold

        # 意图池定义 (在生产环境中，这部分应该存放在 Redis 或配置中心)
        self.intent_pools: Dict[str, List[str]] = {
            "chitchat": ["你好", "你是谁", "讲个笑话", "早安", "在吗", "夸夸我"],
            "weather": ["今天天气怎么样", "下雨了吗", "气温", "出门需要带伞吗", "加衣服吗", "冷不冷", "热不热"],
            "knowledge_rag": ["总结一下论文", "什么是Attention", "根据知识库回答", "帮我查一下内部资料"]
        }

        # 预计算向量缓存 (空间换时间)
        self._intent_embeddings = {}
        self._build_index()

    def _build_index(self):
        """将所有意图池中的句子预先向量化，避免每次请求时重复计算"""
        logger.info("📦 [Router] 正在预编译意图向量索引池...")
        for intent, phrases in self.intent_pools.items():
            embeddings = self.encoder.encode(phrases, normalize_embeddings=True)
            self._intent_embeddings[intent] = embeddings
        logger.info("✅ [Router] 意图池构建完成.")

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度 (大厂面试常考手写基础，给我记牢了)"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def route(self, query: str) -> Literal["chitchat", "weather", "knowledge_rag", "complex_agent"]:
        """
        核心路由决策方法
        """
        query_emb = self.encoder.encode([query], normalize_embeddings=True)[0]

        best_intent = "complex_agent"  # 默认兜底：如果都匹配不上，说明是个复杂任务，交给 ReAct Agent
        highest_score = 0.0

        for intent, embeddings in self._intent_embeddings.items():
            # 计算 query 与该意图池中所有句子的相似度
            similarities = [self._cosine_similarity(query_emb, emb) for emb in embeddings]
            max_sim = max(similarities)

            if max_sim > highest_score:
                highest_score = max_sim
                best_intent = intent

        logger.debug(f"🧭 [Router] Query: '{query}' | 最佳匹配意图: {best_intent} | 最高相似度: {highest_score:.4f}")

        # 阈值拦截机制
        if highest_score < self.threshold:
            logger.warning(f"⚠️ [Router] 相似度低于阈值 {self.threshold}，降级为默认复杂任务处理。")
            return "complex_agent"

        return best_intent


# 临时测试代码，看完删掉
if __name__ == "__main__":
    router = SemanticRouter()
    print(router.route("帮我总结一下Transformer论文的核心创新点"))  # 应输出 knowledge_rag
    print(router.route("上海明天冷不冷"))  # 应输出 weather
    print(router.route("去搜一下最近的AI新闻，然后结合知识库写一篇研报"))  # 应输出 complex_agent (因为没有高度匹配的单条意图)
