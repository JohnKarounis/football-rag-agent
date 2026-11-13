"""Tests for the API endpoints."""

from fastapi.testclient import TestClient

from football_rag.api import main

client = TestClient(main.app)
FAKE_ANSWER = "Spain won Euro 2024"

def fake_agent(question):
    """Stand-in for run_graph: ignores the question and returns FAKE_ANSWER."""
    return FAKE_ANSWER
