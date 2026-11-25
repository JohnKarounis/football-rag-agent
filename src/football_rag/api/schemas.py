"""Request and response models for the API."""

from pydantic import BaseModel, Field

MAX_QUESTION_CHARS = 500


class AskRequest(BaseModel):
    """Body of a POST /v1/ask request."""
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS, description="The question to ask the model.")
    conversation_id: str | None = Field(default=None, description="Id of the conversation to continue; leave it out to start a new one.")


class AskResponse(BaseModel):
    """Answer returned by /v1/ask, with its latency and token usage."""
    answer: str = Field(description="The answer to return.")
    latency_ms: float = Field(description="Time taken to answer, in milliseconds.")
    input_tokens: int = Field(description="Input tokens across all LLM calls for this request.")
    output_tokens: int = Field(description="Output tokens across all LLM calls for this request.")
    conversation_id: str = Field(description="Id of this conversation; send it with the next question to continue.")
