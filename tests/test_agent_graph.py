"""Unit tests for the agent graph's routing and helpers."""
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import END

from football_rag import agent_graph
from football_rag.agent_graph import route_after_input_guardrail, collect_tool_context, count_retries, REFUSAL, NOT_FOUND, RETRY_FEEDBACK

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


def test_collect_tool_context_joins_results():
    """Only tool results are kept from the messages, joined in order."""
    messages = [
        HumanMessage('Who won Euro 2024?'),
        ToolMessage(content='chunk A', tool_call_id='1'),
        ToolMessage(content='chunk B', tool_call_id='2'),
        AIMessage(content='Spain won.')
    ]
    context = collect_tool_context(messages)
    assert context == 'chunk A\n\nchunk B'


class FakeGraph:
    """Stands in for the compiled graph; always hits the step limit."""
    def invoke(self, inputs, config):
        raise GraphRecursionError('step limit')
    

def test_step_limit_returns_not_found(monkeypatch):
    """Hitting the step limit returns NOT_FOUND."""
    monkeypatch.setattr(agent_graph, 'graph', FakeGraph())
    answer = agent_graph.run_graph('Who scored the most goals ever?')
    assert answer == NOT_FOUND


def test_count_retries_counts_only_feedback():
    """Only feedback messages from the output guardrail are counted, not the user's question."""
    messages = [
        HumanMessage(content='Who won Euro 2024?'),
        AIMessage(content='Spain won, beating Germany.'),
        HumanMessage(content=f'{RETRY_FEEDBACK} The final was against England.')
    ]
    retries = count_retries(messages)
    assert retries == 1
