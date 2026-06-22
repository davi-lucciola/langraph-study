from typing import Literal

from graph import ChatState, build_graph
from langchain.messages import AIMessage, HumanMessage
from langchain_core.runnables.config import RunnableConfig
from rich import print
from rich.markdown import Markdown

type UserType = Literal["plus", "enterprise"]


def main() -> None:
    user_type: UserType = "enterprise"
    config = RunnableConfig(
        run_name="meu_grafo",
        tags=["enterprise"],
        configurable={"thread_id": 1, "user_type": user_type},
        # callbacks=[FunctionCallbackHandler(function=builtins.print)],
    )
    graph = build_graph()

    while True:
        print("[bold cyan]YOU: \n-> ", end="")
        user_input = input()
        print(Markdown("-------"))

        if user_input.lower() in {"q", "quit"}:
            print("[bold green]Bye 👋")
            break

        human_message = HumanMessage(user_input)
        result = graph.invoke(ChatState(messages=[human_message]), config=config)

        ai_message: AIMessage = result["messages"][-1]

        print("[bold cyan]AI: \n")
        print(Markdown(ai_message.text))
        print(ai_message)
        print(Markdown("-------"))

    print(graph.get_state(config=config))


if __name__ == "__main__":
    main()
