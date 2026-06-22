import asyncio

from checkpointer import build_checkpointer
from context import ChatContext, UserType
from graph import ChatState, build_graph
from langchain.messages import AIMessage, HumanMessage
from langchain_core.runnables.config import RunnableConfig
from rich import print
from rich.markdown import Markdown
from utils import lifespan


def run_graph() -> None:
    checkpointer = build_checkpointer()
    graph = build_graph(checkpointer=checkpointer)

    context = ChatContext(user_type=UserType.PLUS)
    config = RunnableConfig(configurable={"thread_id": 1})

    while True:
        print("[bold cyan]YOU: \n-> ", end="")
        user_input = input()
        print(Markdown("-------"))

        if user_input.lower() in {"q", "quit"}:
            print("[bold green]Bye 👋")
            break

        human_message = HumanMessage(user_input)
        result = graph.invoke(
            ChatState(messages=[human_message]), config=config, context=context
        )

        ai_message: AIMessage = result["messages"][-1]

        print("[bold cyan]AI: \n")
        print(Markdown(ai_message.text))
        print(Markdown("-------"))


async def main():
    async with lifespan():
        run_graph()


if __name__ == "__main__":
    asyncio.run(main())
