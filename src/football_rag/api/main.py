"""FastAPI serves the LangGraph agent over HTTP."""

import logging
import time

from fastapi import FastAPI

from football_rag.api.schemas import AskRequest, AskResponse
from football_rag.agent_graph import run_graph

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title='Football RAG Agent API', version='0.1.0')

@app.get('/health')
def health():
    """Liveness check: returns ok if the server is up."""
    return {'status': 'ok'}

