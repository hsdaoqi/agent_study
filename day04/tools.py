import json
import asyncio
from typing import Dict, Any
from loguru import logger
from rag_engine import AdvancedRAGEngine  # 引入你刚才写的模块


class ToolRegistry:
    def __init__(self, api_key: str):
        # 初始化私有知识库武器
        self.rag_engine = AdvancedRAGEngine(api_key=api_key)

        # 工业级 Agent 必须向 LLM 声明严格的 Tool Schema
        self.tool_schemas = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "获取指定城市的实时天气情况，适用于任何询问天气的场景。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "城市名称，例如：上海, 北京"}
                        },
                        "required": ["city"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "进行互联网搜索，获取最新的外部信息、新闻或实时数据。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "搜索关键词"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_internal_knowledge",
                    "description": "检索内部私有知识库，专用于查询学术论文（如Transformer/Attention）或内部规定。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "需要查询的具体学术问题或知识点"}
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

    # ---------- 工具具体实现逻辑 ----------

    async def get_weather(self, city: str) -> str:
        """模拟天气API调用"""
        logger.info(f"🌤️ [Tool|Weather] 正在查询 {city} 的天气...")
        await asyncio.sleep(0.5)  # 模拟网络请求延迟
        # 生产环境中这里应该接真实的API
        if "上海" in city or "华东师范大学" in city:
            return f"{city}今天大雨，气温 18-22 度，建议带伞并穿外套。"
        return f"{city}今天晴朗，气温 25 度。"

    async def web_search(self, query: str) -> str:
        """模拟外部搜索引擎"""
        logger.info(f"🌐 [Tool|WebSearch] 正在全网搜索: {query}...")
        await asyncio.sleep(1)
        return f"网搜结果：目前《Attention is All You Need》的最新引用量已超过 100,000 次，是 NLP 领域最具影响力的论文之一。"

    async def search_internal_knowledge(self, query: str) -> str:
        """打通刚才写的高阶 RAG 引擎"""
        logger.info(f"📚 [Tool|InternalRAG] 正在深入知识库剖析: {query}...")
        # 直接调用模块二中封装的强力 RAG
        context = await self.rag_engine.search(query)
        return f"内部知识库检索结果：\n{context}"

    # ---------- 动态分发路由 ----------

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """工具调用的统一入口（反射执行）"""
        try:
            # Python 的反射机制，根据字符串获取对应的方法并执行
            func = getattr(self, tool_name)
            result = await func(**arguments)
            return str(result)
        except Exception as e:
            # 🚨 导师警告：绝不允许工具崩溃导致整个进程挂掉！必须捕获并返回给LLM进行自愈！
            error_msg = f"工具执行内部报错: {str(e)}"
            logger.error(f"❌ [Tool|Error] {tool_name} 执行失败: {error_msg}")
            return error_msg
