

---

### 1. 基础模型定义：`BaseModel`
在大模型开发里，我们不写原始字典（Dict），因为字典没有约束。我们写 `BaseModel`。

```python
from pydantic import BaseModel

class UserSearchQuery(BaseModel):
    name: str
    age: int
    tags: list[str] = []  # 设置默认值
```
*   **为什么用它？** 当你从大模型那里拿到一串 JSON 时，直接 `UserSearchQuery.model_validate_json(json_str)`，如果模型生成的 `age` 是个字符串 `"25"`，Pydantic 会自动帮你转换成 `int`。如果转换失败，直接抛错，**这叫第一道防线**。

---

### 2. 核心中的核心：`Field`（模型描述）
在大模型调用工具（Function Calling）时，大模型必须知道每个参数是干什么的。**`Field` 里的 `description` 就是写给大模型看的说明书。**

```python
from pydantic import BaseModel, Field

class GetWeather(BaseModel):
    city: str = Field(
        description="城市名称，例如 '上海' 或 'Beijing'", 
        examples=["上海"]
    )
    unit: str = Field(
        default="celsius", 
        pattern="^(celsius|fahrenheit)$", # 正则校验，只准是大模型选这两个之一
        description="温度单位"
    )
```
*   **导师点评：** 别偷懒不写 `description`。如果你不写，大模型在调用工具时就像个盲人摸象。大厂的 Prompt 工程里，有 30% 的优化是在调优这些 Field 的描述语。

---

### 3. 大厂级必杀技：`model_json_schema()`
OpenAI 的工具调用需要你提交一份复杂的 JSON Schema。Pydantic 可以一键生成。

```python
print(GetWeather.model_json_schema())
```
它会输出符合 JSON Schema 标准的字典，包含类型、描述、默认值。**你在写 `@tool` 装饰器时，就是靠这个方法来自动生成大模型看得懂的工具定义。**

---

### 4. 嵌套模型（Nested Models）
当你的工具非常复杂时，参数可能也是一个对象。

```python
class SearchScope(BaseModel):
    source: str = Field(description="搜索来源：zhihu, github, arxiv")
    limit: int = 5

class AdvancedSearch(BaseModel):
    query: str
    scope: SearchScope # 嵌套了
```
Pydantic 会自动处理递归的校验和 Schema 生成。

---

### 5. 进阶：如何用 Pydantic 辅助装饰器？（Day 3 核心提示）

你要写的 `@tool` 装饰器，核心逻辑应该是这样的：
1.  利用 `inspect.signature(func)` 拿到函数的参数名和类型。
2.  动态地为这个函数创建一个 Pydantic 模型。
3.  调用 `.model_json_schema()` 提取信息。

**这里我给你一个“大厂级封装”的思路片段（别直接抄，理解它）：**

```python
import inspect
from pydantic import create_model

def tool(func):
    # 1. 提取函数签名
    sig = inspect.signature(func)
    
    # 2. 构造 Pydantic 模型所需的字段定义
    fields = {}
    for name, param in sig.parameters.items():
        # 这里可以提取参数的类型注解和默认值
        fields[name] = (param.annotation, param.default)
    
    # 3. 动态创建一个 Pydantic 模型来代表这个函数的参数
    # 这就是所谓的“动态建模”
    dynamic_model = create_model(f"{func.__name__}_Schema", **fields)
    
    # 4. 把生成的 Schema 挂在函数对象上，方便后面读取
    func.args_schema = dynamic_model.model_json_schema()
    return func

@tool
def get_weather(city: str, unit: str = "celsius"):
    """获取指定城市的实时天气"""
    pass

# 现在 get_weather.args_schema 里就包含了 OpenAI 需要的一切！
```

---


