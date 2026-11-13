"""Tests for the API endpoints."""

from fastapi.testclient import TestClient

from football_rag.api import main
from football_rag.api.schemas import MAX_QUESTION_CHARS

client = TestClient(main.app)
FAKE_ANSWER = "Spain won Euro 2024"

def fake_agent(question):
    """Stand-in for run_graph: ignores the question and returns FAKE_ANSWER."""
    return FAKE_ANSWER


def test_health():
    """/health returns 200 with status ok."""
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}


def test_ask_returns_answer(monkeypatch):
    """/v1/ask returns the answer and latency."""
    monkeypatch.setattr(main, 'run_graph', fake_agent)
    response = client.post('/v1/ask', json={'question': 'Who won the Euro 2024?'})
    body = response.json()
    assert response.status_code == 200
    assert body['answer'] == FAKE_ANSWER
    assert 'latency_ms' in body 


def test_ask_rejects_empty_question():
    """An empty question is rejected with 422."""
    response = client.post('/v1/ask', json={'question': ''})
    assert response.status_code == 422


def test_ask_rejects_long_question():
    """A long question is rejected with 422."""
    long_question = 'a' * (MAX_QUESTION_CHARS + 1)
    response = client.post('/v1/ask', json={'question': long_question})
    assert response.status_code == 422
