"""Tests for the API endpoints."""

from fastapi.testclient import TestClient

from football_rag.api import main
from football_rag.api.schemas import MAX_QUESTION_CHARS

client = TestClient(main.app)
FAKE_ANSWER = "Spain won Euro 2024" 
FAKE_INPUT_TOKENS = 100
FAKE_OUTPUT_TOKENS = 20

def fake_agent(question, thread_id=None):
    """Stand-in for run_graph_with_usage: ignores the question and returns a fixed answer and token counts."""
    return FAKE_ANSWER, FAKE_INPUT_TOKENS, FAKE_OUTPUT_TOKENS


def test_health():
    """/health returns 200 with status ok."""
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}


def test_ask_returns_answer(monkeypatch):
    """/v1/ask returns the answer, latency and token usage."""
    monkeypatch.setattr(main, 'run_graph_with_usage', fake_agent)
    response = client.post('/v1/ask', json={'question': 'Who won Euro 2024?'})
    body = response.json()
    assert response.status_code == 200
    assert body['answer'] == FAKE_ANSWER
    assert 'latency_ms' in body
    assert 'conversation_id' in body
    assert body['input_tokens'] == FAKE_INPUT_TOKENS
    assert body['output_tokens'] == FAKE_OUTPUT_TOKENS


def test_ask_rejects_empty_question():
    """An empty question is rejected with 422."""
    response = client.post('/v1/ask', json={'question': ''})
    assert response.status_code == 422


def test_ask_rejects_long_question():
    """A long question is rejected with 422."""
    long_question = 'a' * (MAX_QUESTION_CHARS + 1)
    response = client.post('/v1/ask', json={'question': long_question})
    assert response.status_code == 422


def test_ask_keeps_conversation_id(monkeypatch):
    """A conversation id sent by the client is returned unchanged."""
    monkeypatch.setattr(main, 'run_graph_with_usage', fake_agent)
    response = client.post('/v1/ask', json={'question': 'Who won Euro 2024?', 'conversation_id': 'abc'})
    body = response.json()
    assert response.status_code == 200
    assert body['conversation_id'] == 'abc'