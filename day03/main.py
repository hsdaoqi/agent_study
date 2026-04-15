import asyncio
from agent import ReActAgent


async def main():
    agent = ReActAgent()
    prompt = "帮我查下上海天气，然后搜一下适合雨天听的纯音乐，最后读一下 /non_exist.txt"
    answer = await agent.chat(prompt)
    print(f"\n{answer}")


if __name__ == "__main__":
    asyncio.run(main())
