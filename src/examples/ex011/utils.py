import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import cast

from langchain.chat_models import BaseChatModel, init_chat_model


def load_llm() -> BaseChatModel:
    model = os.getenv("CHAT_MODEL")

    if not model:
        error = 'Chat model is not defined in "CHAT_MODEL" environment variable.'
        raise ValueError(error)

    llm_configurable = init_chat_model(model, configurable_fields=("temperature",))
    llm = cast("BaseChatModel", llm_configurable)

    assert hasattr(llm, "bind_tools")  # noqa: S101
    assert hasattr(llm, "invoke")  # noqa: S101
    assert hasattr(llm, "with_config")  # noqa: S101

    return llm


@asynccontextmanager
async def lifespan() -> AsyncGenerator[None]:
    yield
