"""Output guardrail: checks whether the agent's answer is supported by the tool results."""

from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate

from football_rag.llm import get_llm
from football_rag.prompts import OUTPUT_GUARDRAIL_SYSTEM

class GroundedDecision(BaseModel):
    """Output guardrail decision on whether an answer is supported by the context."""
    grounded: bool = Field(description="True if every fact in the answer is supported by the context; otherwise false.")
    reason: str = Field(description="One short sentence explaining the reason, e.g., 'the score is not in the context'")

output_guardrail_prompt = ChatPromptTemplate.from_messages([
    ('system', OUTPUT_GUARDRAIL_SYSTEM),
    ('human', "Question:\n{question}\n\nContext:\n{context}\n\nAnswer:\n{answer}")
])

output_guardrail_chain = output_guardrail_prompt | get_llm().with_structured_output(GroundedDecision)


def check_answer(question, context, answer):
    """Return the output guardrail's decision on whether the answer is supported by the context."""
    decision = output_guardrail_chain.invoke({'question': question, 'context': context, 'answer': answer})
    return decision
