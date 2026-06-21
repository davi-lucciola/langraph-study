from typing import Literal

from langchain.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from state import ChatState
from tools import TOOLS, TOOLS_BY_NAME
from utils import load_llm


def call_llm(state: ChatState, config: RunnableConfig) -> ChatState:
    user_type = config.get("configurable", {}).get("user_type")
    temperature = 1 if user_type == "plus" else 0

    llm = (
        load_llm()
        .bind_tools(TOOLS)
        .with_config(config={"configurable": {"temperature": temperature}})
    )

    ai_message = llm.invoke(state.messages)
    return ChatState(messages=[ai_message])


def tool_node(state: ChatState) -> ChatState:
    ai_message = state.get_last_message()

    if not isinstance(ai_message, AIMessage):
        return ChatState(messages=[])

    call = ai_message.tool_calls[-1]
    name, args, _id = (call["name"], call["args"], call["id"])

    try:
        content = TOOLS_BY_NAME[name].invoke(args)
        status = "success"
    except (KeyError, IndexError, TypeError) as err:
        content = f"Please, fix your mistakes: {err}"
        status = "error"

    tool_message = ToolMessage(content, tool_call_id=_id, status=status)
    return ChatState(messages=[tool_message])


type NodeRouter = Literal["tool_node", "__end__"]


def router(state: ChatState) -> NodeRouter:
    last_message = state.get_last_message()

    is_ai_message = isinstance(last_message, AIMessage)
    has_tool_calls = getattr(last_message, "tool_calls", None)

    if is_ai_message and has_tool_calls:
        return "tool_node"

    return "__end__"


def build_graph() -> CompiledStateGraph[ChatState, None, ChatState, ChatState]:
    builder = StateGraph(ChatState)

    builder.add_node("call_llm", call_llm)
    builder.add_node("tool_node", tool_node)

    builder.add_edge(START, "call_llm")
    builder.add_conditional_edges("call_llm", router, ["tool_node", END])
    builder.add_edge("tool_node", "call_llm")

    return builder.compile(checkpointer=InMemorySaver())
