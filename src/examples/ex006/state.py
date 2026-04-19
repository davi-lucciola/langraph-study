from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


@dataclass
class ChatState:
    messages: Annotated[Sequence[BaseMessage], add_messages]

    def get_last_message(self) -> BaseMessage:
        return self.messages[-1]
