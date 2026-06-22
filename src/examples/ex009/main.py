from graph import ChatState, build_graph
from langchain.messages import AIMessage, HumanMessage
from langchain_core.runnables.config import RunnableConfig
from rich import print
from rich.markdown import Markdown

from examples.ex009.context import ChatContext, UserType


def main() -> None:
    context = ChatContext(user_type=UserType.PLUS)
    config = RunnableConfig(configurable={"thread_id": 1})
    graph = build_graph()

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
        print(ai_message)
        print(Markdown("-------"))

    print(graph.get_state(config=config))


if __name__ == "__main__":
    main()
