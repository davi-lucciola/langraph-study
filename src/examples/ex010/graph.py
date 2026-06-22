from context import ChatContext
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import tools_condition
from nodes import call_llm, tool_node
from state import ChatState


def build_graph(
    checkpointer: BaseCheckpointSaver,
) -> CompiledStateGraph[ChatState, ChatContext, ChatState, ChatState]:
    builder = StateGraph(
        state_schema=ChatState,
        context_schema=ChatContext,
        input_schema=ChatState,
        output_schema=ChatState,
    )

    builder.add_node("call_llm", call_llm)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "call_llm")
    builder.add_conditional_edges("call_llm", tools_condition, ["tools", END])
    builder.add_edge("tools", "call_llm")

    return builder.compile(checkpointer=checkpointer)
