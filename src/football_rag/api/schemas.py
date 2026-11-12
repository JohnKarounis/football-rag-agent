"""Request and response models for the API."""

from pydantic import BaseModel, Field

MAX_QUESTION_CHARS = 500


class AskRequest(BaseModel):
    """Body of a POST /v1/ask request."""
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS, description="The question to ask the model.")


class AskResponse(BaseModel):
    """Answer returned by /v1/ask, with how long it took."""
    answer: str = Field(description="The answer to return.")
    latency_ms: float = Field(description="Time taken to answer, in milliseconds.")
