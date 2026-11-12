"""Chat model shared by the graphs and the judge."""

from langchain_openai import ChatOpenAI

from football_rag import config

LLM_TIMEOUT = 50


def get_llm():
    """Return the LM Studio chat model at temperature 0, with a request timeout."""

    llm = ChatOpenAI(
        model=config.CHAT_MODEL,
        base_url=config.LMSTUDIO_URL,
        api_key=config.LMSTUDIO_API_KEY,
        temperature=0,
        timeout=LLM_TIMEOUT,
    )

    return llm
