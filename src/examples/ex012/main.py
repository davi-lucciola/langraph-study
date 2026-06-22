import asyncio

from checkpointer import build_checkpointer_psql
from constants import DATABASE_URL
from context import ChatContext, UserType
from graph import ChatState, build_graph
from langchain.messages import AIMessage, HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from rich import print
from rich.markdown import Markdown
from utils import lifespan


async def run_graph(checkpointer: BaseCheckpointSaver) -> None:
    graph = build_graph(checkpointer=checkpointer)

    context = ChatContext(user_type=UserType.PLUS)
    config = RunnableConfig(configurable={"thread_id": 1})

    while True:
        print("[bold cyan]YOU: \n-> ", end="")
        user_input = await asyncio.to_thread(input)
        print(Markdown("-------"))

        if user_input.lower() in {"q", "quit"}:
            print("[bold green]Bye 👋")
            break

        human_message = HumanMessage(user_input)
        result = await graph.ainvoke(
            ChatState(messages=[human_message]), config=config, context=context
        )

        ai_message: AIMessage = result["messages"][-1]

        print("[bold cyan]AI: \n")
        print(Markdown(ai_message.text))
        print(Markdown("-------"))

    print(await graph.aget_state(config=config))


async def main() -> None:
    async with (
        lifespan(),
        build_checkpointer_psql(DATABASE_URL) as checkpointer,
    ):
        await run_graph(checkpointer)


if __name__ == "__main__":
    asyncio.run(main())
