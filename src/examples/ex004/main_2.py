import os
from collections.abc import Sequence
from typing import Annotated, TypedDict

from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages import BaseMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph, add_messages
from rich import print
from rich.markdown import Markdown


# 1. Definir o Schema do Estado do Grafo
class ChatState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# 2. Definir os Nós
def call_llm(state: ChatState) -> ChatState:
    model = os.getenv("CHAT_MODEL")
    llm = init_chat_model(model)

    result = llm.invoke(state["messages"])
    return ChatState(messages=[result])


# 3. Definir o Builder
builder = StateGraph(
    ChatState, context_schema=None, input_schema=ChatState, output_schema=ChatState
)

builder.add_node("call_llm", call_llm)

builder.add_edge(START, "call_llm")
builder.add_edge("call_llm", END)


checkpoint = InMemorySaver()
thread_config = RunnableConfig(configurable={"thread_id": 1})
graph = builder.compile(checkpointer=checkpoint)

if __name__ == "__main__":
    os.system("clear")  # noqa: S605, S607

    system_message = SystemMessage("""
        Você é um assistente onde seu objetivo é ajudar as pessoas
        a aprender programação. Suas respostas devem ser sempre em markdown.
    """)

    human_message = HumanMessage("Olá, tudo bom?")
    result = graph.invoke({"messages": [human_message]}, config=thread_config)

    ai_message: AIMessage = result["messages"][-1]

    print("AI: \n", Markdown(str(ai_message.content)), "\n")

    while True:
        user_input = input("YOU: \n=> ")

        if user_input.lower() in {"q", "quit"}:
            print("Bye 👋")
            break

        human_message = HumanMessage(user_input)
        result = graph.invoke({"messages": [human_message]}, config=thread_config)

        ai_message: AIMessage = result["messages"][-1]

        print("AI: \n", Markdown(str(ai_message.content)))
        print(Markdown("-------"))
