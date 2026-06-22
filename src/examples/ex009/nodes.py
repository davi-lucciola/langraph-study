from context import ChatContext
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from state import ChatState
from tools import TOOLS
from utils import load_llm

tool_node = ToolNode(tools=TOOLS)


def call_llm(state: ChatState, runtime: Runtime[ChatContext]) -> ChatState:
    user_type = runtime.context.user_type
    temperature = 1 if user_type == "plus" else 0

    llm = (
        load_llm()
        .bind_tools(TOOLS)
        .with_config(config={"configurable": {"temperature": temperature}})
    )

    ai_message = llm.invoke(state.messages)
    return ChatState(messages=[ai_message])
