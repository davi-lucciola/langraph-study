import os

from langchain.chat_models import BaseChatModel, init_chat_model


def load_llm() -> BaseChatModel:
    model = os.getenv("CHAT_MODEL")

    if not model:
        error = 'Chat model is not defined in "CHAT_MODEL" environment variable.'
        raise ValueError(error)

    return init_chat_model(model)
