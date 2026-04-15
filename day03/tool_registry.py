import inspect
from typing import Callable, Any, Dict, List
import json

from pydantic import create_model


class ToolRegistry:
    """工具注册表：负责管理所有挂载的工具函数及自动生成 Schema"""

    def __init__(self):
        print("\n[INIT] >>> ToolRegistry 唯一实例已初始化 <<<")
        self.tools: Dict[str, Callable] = {}
        self.tools_schema: List[Dict] = []
        self.models: Dict[str, Any] = {}

    def register(self, func: Callable):
        """核心装饰器： 利用Pydantic 自动生成 Schema"""
        name = func.__name__
        doc = func.__doc__ or "未提供工具说明"

        # 1. 提取函数的参数签名 (反射机制)
        sig = inspect.signature(func)
        fields = {}
        for param_name, param in sig.parameters.items():
            annotation = param.annotation if param.annotation != inspect.Parameter.empty else Any
            default = ... if param.default == inspect.Parameter.empty else param.default
            fields[param_name] = (annotation, default)

        PydanticModel = create_model(f"{name}_input", **fields)
        self.models[name] = PydanticModel
        self.tools[name] = func
        self.tools_schema.append({
            "type": "function",
            "function": {
                "name": name,
                "description": doc.strip(),
                "parameters": PydanticModel.model_json_schema()
            }
        })
        return func


# 全局单例注册表
registry = ToolRegistry()
tool = registry.register
