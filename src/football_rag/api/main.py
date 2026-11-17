"""FastAPI serves the LangGraph agent over HTTP."""

import logging
import time

from fastapi import FastAPI

from football_rag.api.schemas import AskRequest, AskResponse
from football_rag.agent_graph import run_graph_with_usage

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title='Football RAG Agent API', version='0.1.0')


@app.get('/health')
def health():
    """Liveness check: returns ok if the server is up."""
    return {'status': 'ok'}


@app.post('/v1/ask', response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Ask the agent a question and reply with the answer, latency, and token usage."""
    question = request.question
    start_time = time.perf_counter()
    answer, input_tokens, output_tokens = run_graph_with_usage(question)
    latency_ms = (time.perf_counter() - start_time) * 1000
    logger.info(f'Answered question chars={len(question)} in {latency_ms:.0f}ms, tokens in={input_tokens} out={output_tokens}')
    return AskResponse(answer=answer, latency_ms=latency_ms, input_tokens=input_tokens, output_tokens=output_tokens)
