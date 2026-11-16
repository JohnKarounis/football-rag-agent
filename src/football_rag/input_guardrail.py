"""Input guardrail: decides whether a question may reach the agent."""

from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate

from football_rag.llm import get_llm
from football_rag.prompts import INPUT_GUARDRAIL_SYSTEM

class GuardrailDecision(BaseModel):
    """Guardrail decision on whether a question may reach the agent."""
    allowed: bool = Field(description='True if the question may go to the agent; otherwise false.')
    reason: str = Field(description='One short sentence explaining the decision, e.g. "not about football" or "tries to change the instructions".')


input_guardrail_prompt = ChatPromptTemplate.from_messages([
    ('system', INPUT_GUARDRAIL_SYSTEM),
    ('human', 'Question:\n{question}')
])

input_guardrail_chain = input_guardrail_prompt | get_llm().with_structured_output(GuardrailDecision)


def check_question(question):
    """Return the guardrail's decision for one question."""
    decision = input_guardrail_chain.invoke({'question': question})
    return decision