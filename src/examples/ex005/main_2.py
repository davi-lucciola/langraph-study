import os
from dataclasses import dataclass
from typing import Annotated, Literal

from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool
from langchain_core.messages import BaseMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph, add_messages
from rich import print
from rich.markdown import Markdown


@tool
def multiply(a: float, b: float) -> float:
    """Multiply a * b and returns the result

    Args:
        a: float mutiplicant
        b: float multiplier

    Returns:
        the resulting float of the equation a * b
    """
    return a * b


tools = [multiply]
tools_by_name = {tool.name: tool for tool in tools}

model = os.getenv("CHAT_MODEL")
llm = init_chat_model(model).bind_tools(tools)

checkpoint = InMemorySaver()
thread_config = RunnableConfig(configurable={"thread_id": 1})


# 1. Definir Estado dos Grafo
@dataclass
class ChatState:
    messages: Annotated[list[BaseMessage], add_messages]

    def get_last_message(self) -> BaseMessage:
        return self.messages[-1]


# 2. Definir os Nós
def call_llm(state: ChatState) -> ChatState:
    ai_message = llm.invoke(state.messages)
    return ChatState(messages=[ai_message])


def tool_node(state: ChatState) -> ChatState:
    ai_message = state.get_last_message()

    if not isinstance(ai_message, AIMessage):
        return ChatState(messages=[])

    call = ai_message.tool_calls[-1]
    name, args, _id = (call["name"], call["args"], call["id"])

    try:
        content = tools_by_name[name].invoke(args)
        status = "success"
    except (KeyError, IndexError, TypeError) as err:
        content = f"Please, fix your mistakes: {err}"
        status = "error"

    tool_message = ToolMessage(content, tool_call_id=_id, status=status)
    return ChatState(messages=[tool_message])


def should_call_tool(state: ChatState) -> Literal["tool_node", "__end__"]:
    last_message = state.get_last_message()

    if isinstance(last_message, AIMessage) and getattr(
        last_message, "tool_calls", None
    ):
        return "tool_node"

    return "__end__"


# 3. Definir Graph Builder
def build_graph():  # noqa: ANN201
    builder = StateGraph(ChatState)

    builder.add_node("call_llm", call_llm)
    builder.add_node("tool_node", tool_node)

    builder.add_edge(START, "call_llm")
    builder.add_conditional_edges("call_llm", should_call_tool, ["tool_node", END])
    builder.add_edge("tool_node", "call_llm")

    return builder.compile(checkpointer=checkpoint)


if __name__ == "__main__":
    graph = build_graph()
    os.system("clear")  # noqa: S605, S607

    system_message = SystemMessage("""
        You are a helpful assistent. Your awnsers always have to be in Markdown.
    """)

    human_message = HumanMessage("Olá, tudo bom?")
    result = graph.invoke(ChatState(messages=[human_message]), config=thread_config)

    ai_message: AIMessage = result["messages"][-1]

    print("AI: \n", Markdown(str(ai_message.content)), "\n")

    while True:
        user_input = input("YOU: \n=> ")

        if user_input.lower() in {"q", "quit"}:
            print("Bye 👋")
            break

        human_message = HumanMessage(user_input)
        result = graph.invoke(ChatState(messages=[human_message]), config=thread_config)

        ai_message: AIMessage = result["messages"][-1]

        print("AI: \n", Markdown(str(ai_message.content)))
        print(Markdown("-------"))
