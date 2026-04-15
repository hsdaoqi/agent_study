import httpx
import os
from agent_study.day03.tool_registry import tool


@tool
async def get_weather(city: str) -> str:
    """
    当用户询问天气时调用此工具。
    参数 city 必须是城市名称，例如 "上海", "北京"。
    """
    # 这里为了方便你测试，不要求去申请复杂的 API Key，直接调用免费的 wttr.in 接口
    try:
        url = f"https://wttr.in/{city}?format=j1"
        response = httpx.get(url, timeout=5.0)
        response.raise_for_status()
        data = response.json()
        temp = data['current_condition'][0]['temp_C']
        desc = data['current_condition'][0]['weatherDesc'][0]['value']
        return f"{city}当前天气：{desc}，气温：{temp}℃。"
    except Exception as e:
        return f"获取天气失败，请重试或告知用户网络异常。错误信息: {str(e)}"


@tool
def web_search(query: str) -> str:
    """
    当需要搜索实时新闻、互联网信息、或回答不知道的问题时调用。
    参数 query 是你想搜索的关键词。
    """
    # 这里我简单用鸭鸭搜索的开源库逻辑模拟，生产环境用 SerpAPI
    # 为了保证你的代码能立刻跑通，我写死一个 mock，但在真实场景必须接入真实爬虫
    # 模拟真实检索结果返回
    if "纯音乐" in query:
        return "推荐纯音乐：《雨的印记》(Kiss the Rain), 《River Flows in You》。非常适合阴雨天气聆听。"
    return "搜索结果未找到相关确切信息，请尝试其他关键词。"


@tool
def read_local_file(file_path: str) -> str:
    """
    用于读取本地文件的内容。
    参数 file_path 是文件的绝对或相对路径。
    """
    if not os.path.exists(file_path):
        # 注意：这里抛出异常是为了后面测试 Agent 的【异常自愈】能力！
        raise FileNotFoundError(f"文件不存在: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()[:1000]  # 截断前1000字符防止Token爆炸
