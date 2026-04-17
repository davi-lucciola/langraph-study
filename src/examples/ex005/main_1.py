import os

from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain.tools import BaseTool, tool
from langchain_core.messages import BaseMessage
from rich import print


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


model = os.getenv("CHAT_MODEL")
llm = init_chat_model(model)

messages: list[BaseMessage] = []

system_message = SystemMessage(
    "You are a helpful assistent,  You have access to tools. When the user asks for "
    "something, first look if you have a tool that solves that problem."
)

human_message = HumanMessage("Olá, sou Davi, quanto é 3.2 vezes 5?")
messages.extend((system_message, human_message))

tools: list[BaseTool] = [multiply]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)

llm_response = llm_with_tools.invoke(messages)
messages.append(llm_response)

if isinstance(llm_response, AIMessage) and getattr(llm_response, "tool_calls", None):
    call = llm_response.tool_calls[-1]
    name, args, _id = call["name"], call["args"], call["id"]

    try:
        content = tools_by_name[name].invoke(args)
        status = "success"
    except (KeyError, IndexError, TypeError) as err:
        content = f"Please, fix your mistakes: {err}"
        status = "error"

    tool_message = ToolMessage(content, tool_call_id=_id, status=status)
    messages.append(tool_message)

    ai_message = llm_with_tools.invoke(messages)
    messages.append(ai_message)


print(messages)
