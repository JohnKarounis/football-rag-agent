"""Unit tests for the agent graph's routing."""
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import END

from football_rag.agent_graph import route_after_guardrail, REFUSAL 

def test_allowed_question_goes_to_model():
    """A question the guardrail allowed goes on to the agent."""
    state = {'messages': [HumanMessage(content='Who won Euro 2024')]}
    next_node = route_after_guardrail(state)
    assert next_node == 'model'


def test_refusal_goes_to_end():
    """A refusal added by the guardrail ends the graph."""
    state = {'messages': [HumanMessage(), AIMessage(content=REFUSAL)]}
    next_node = route_after_guardrail(state)
    assert next_node == END