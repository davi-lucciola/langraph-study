from typing import TYPE_CHECKING

from graph import ChatState, build_graph
from langchain.messages import HumanMessage
from langchain_core.runnables.config import RunnableConfig
from rich import print

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage


def main() -> None:
    thread_config = RunnableConfig(configurable={"thread_id": 1})
    graph = build_graph()

    user_input = input("HUMAN: ")

    human_message = HumanMessage(user_input)
    current_messages: list[BaseMessage] = [human_message]
    result = graph.invoke(ChatState(messages=current_messages), config=thread_config)

    print(result)
    # ai_message: AIMessage = result["messages"][-1]
    # print("\nAI:", Markdown(str(ai_message.content)))


if __name__ == "__main__":
    main()
