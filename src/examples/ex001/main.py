import asyncio
import os

import rich as r
from langchain.chat_models import init_chat_model


async def main() -> None:
    model = os.getenv("CHAT_MODEL")
    llm = init_chat_model(model)

    # llm = init_chat_model("openai:gpt-4.1-nano")
    response = await llm.ainvoke("Olá, tudo bem?")
    r.print(response)


if __name__ == "__main__":
    asyncio.run(main())
