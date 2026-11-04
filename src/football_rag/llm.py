"""Chat model shared by the graphs and the judge."""

from langchain_openai import ChatOpenAI

from football_rag import config


def get_llm():
    """Return the LM Studio chat model at temperature 0."""

    llm = ChatOpenAI(
        model=config.CHAT_MODEL,
        base_url=config.LMSTUDIO_URL,
        api_key=config.LMSTUDIO_API_KEY,
        temperature=0
    )

    return llm