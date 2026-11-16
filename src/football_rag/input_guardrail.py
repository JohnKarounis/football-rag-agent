"""Input guardrail: decides whether a question may reach the agent."""

from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate

from football_rag.llm import get_llm
from football_rag.prompts import INPUT_GUARDRAIL_SYSTEM

class GuardrailDecision(BaseModel):
    """Guardrail decision on whether a question may reach the agent."""
    allowed: bool = Field(description='True if the question may go to the agent; otherwise false.')
    reason: str = Field(description='One short sentence explaining the decision, e.g. "not about football" or "tries to change the instructions".')