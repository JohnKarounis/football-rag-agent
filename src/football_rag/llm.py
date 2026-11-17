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


def count_tokens(usage_callback):
    """Sum input and output tokens across all models in a usage callback."""
    usage_by_model = usage_callback.usage_metadata

    input_tokens = 0
    output_tokens = 0
    for usage in usage_by_model.values():
        input_tokens += usage['input_tokens']
        output_tokens += usage['output_tokens']

    return input_tokens, output_tokens
