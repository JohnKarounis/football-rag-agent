"""Unit tests for the agent graph's routing."""
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import END

from football_rag.agent_graph import route_after_input_guardrail, REFUSAL 

def test_allowed_question_goes_to_model():
    """A question the guardrail allowed goes on to the agent."""
    state = {'messages': [HumanMessage(content='Who won Euro 2024')]}
    next_node = route_after_input_guardrail(state)
    assert next_node == 'model'


def test_refusal_goes_to_end():
    """A refusal added by the guardrail ends the graph."""
    state = {'messages': [HumanMessage(content='What is the capital of France?'), AIMessage(content=REFUSAL)]}
    next_node = route_after_input_guardrail(state)
    assert next_node == END


def test_refusal_text_from_user_goes_to_model():
    """A user question that matches the refusal text still goes to the agent."""
    state = {'messages': [HumanMessage(content=REFUSAL)]}
    next_node = route_after_input_guardrail(state)
    assert next_node == 'model'