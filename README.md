# Football RAG Agent

Football Q&A built in measured stages: RAG from scratch with numpy, then LangChain, then a tool-calling agent (hand-written loop and LangGraph), then Corrective RAG. Every step is evaluated with LLM-as-judge on tuning and held-out sets. Runs fully local with LM Studio, Qwen and nomic embeddings.

**Status:** Work in progress. The RAG chain, agents, CRAG, evaluation, retrieval experiments, tests, the FastAPI service, the input guardrail and per-request latency/token tracking are done; next are an output guardrail, tracing, Docker and CI.

**Stack:** LM Studio | Qwen | nomic-embed | LangChain | Chroma | LangGraph

## Results

LLM-as-judge (Qwen), judged against reference answers. Latency and tokens are averages per question on the tuning set, measured locally (so latency depends on the GPU); tokens are summed over every LLM call a question makes.

| System | Tuning (48) | Held-out (11) | Avg latency | Avg input tokens | Avg output tokens |
|---|---|---|---|---|---|
| LangChain RAG chain | 94% | 91% | 1.0 s | 1,379 | 26 |
| Hand-written agent loop | 100% | 91% | 3.1 s | n/a | n/a |
| LangGraph agent | 100% | 91% | 3.5 s | 3,256 | 161 |
| CRAG | 98–100% | 91% | 7.1 s | 2,495 | 136 |

The hand-written loop calls the OpenAI client directly, so LangChain's usage callback can't count its tokens. CRAG scored 100% on tuning in earlier runs and 98% (one question) in the latest; nothing in CRAG changed between them, so it is run-to-run variance.

The fixed RAG chain retrieves once and has no Wikipedia fallback or scope rule, so it misses tuning questions the agents get right. The held-out set has no Wikipedia-only or out-of-scope questions yet, which is why the chain ties there.

The agents and CRAG tie on accuracy, but CRAG is about twice as slow, so the LangGraph agent is the default design. CRAG actually uses *fewer* tokens than the agent: its cost is the number of calls, not their size. It makes one small grader call per retrieved chunk (10 per question), and each call is a sequential round trip to the model. The agent makes fewer, larger calls, with all the chunks in one prompt plus the guardrail call. The RAG chain is the cheapest at a single call, but it has no fallback when retrieval misses.

The one held-out miss is the same for all four: the corpus text names the wrong Morocco goalscorer against Belgium, so it is a data error, not an agent error.

The LangGraph agent scores above with the input guardrail in front of it.

### Input guardrail

Scored on its own set of 20 labelled questions (`data/eval/guardrail.json`): plain and tricky football questions that must pass, plus off-topic questions, near misses (Super Bowl, Premier League Darts) and prompt injections that must be blocked.

| Accuracy | False blocks | Missed blocks |
|---|---|---|
| 100% | 0 | 0 |

The first prompt said "football" and let the Super Bowl and darts through, so it now says "association football (soccer)" and names other sports as off-topic. The full eval then caught a false block that the guardrail set missed: a tuning question about Qatar revoking a news outlet's visas during the 2022 World Cup, which the guardrail read as politics. The prompt now allows news linked to a tournament, but not general facts about a country ("What is the capital of Qatar?" is blocked). The 20 questions were also used to tune the prompt, so 100% is a tuning score; there is no held-out guardrail set yet.

### Retrieval experiments

Measured on the tuning set in [notebooks/01_retrieval_experiments.ipynb](notebooks/01_retrieval_experiments.ipynb) (hit@3/5/10, MRR):

- **Structure-based chunking** beats fixed-size chunking at every size; the pipeline uses 100-word chunks with 20 overlap.
- **LLM query rewriting:** no gain (MRR 0.80 → 0.78), and an extra LLM call per query.
- **Cross-encoder reranking** (bge-reranker-base, ms-marco-MiniLM): mixed. One improves MRR, the other hit@10, and neither is worth a model plus latency on every query when the agents already score 100% on tuning.
- **Hybrid BM25 + cosine with RRF:** worse at every weighting.
- **`nomic-embed-text-v1.5`** beats `Qwen3-Embedding-0.6B` on every metric (MRR 0.79 vs 0.71).

## How it works

**Ingestion** (`ingest.py`): 5 Wikipedia articles → Markdown headings → section split → 100-word chunks with 20-word overlap → `title > section` breadcrumb on each chunk → latest year mentioned stored as metadata → Chroma (cosine).

**Tools** (`tools.py`), shared by all agents:
- `search_football_docs(query, prefer_recent=False)`: top 10 chunks; `prefer_recent` re-sorts the top 20 by year
- `search_wikipedia(query)`: top 3 Wikipedia article intros, the fallback for anything not in the corpus

**RAG chain** (`rag_chain.py`): LCEL `retriever | prompt | llm`. Always retrieves once, then answers only from those chunks.

**Input guardrail** (`input_guardrail.py`): one structured-output LLM call (`allowed` + `reason`) that blocks off-topic questions and prompt injections before they reach the agent. In the LangGraph agent it is the first node: `START → guardrail → model | END`. A blocked question ends the graph with `I can only help with football.`; an allowed one goes to the agent unchanged. It adds about 0.5 s per question. The agent's own prompt keeps its scope rule as a second layer.

**Agents:**
- `agent_loop.py`: hand-written tool-calling loop on the OpenAI client
- `agent_graph.py`: the same agent as a LangGraph `model ↔ tools` graph, behind the input guardrail
- `crag_graph.py`: Corrective RAG: retrieve → grade each chunk → generate, or rewrite the query → Wikipedia → generate, with up to 2 web attempts

**Evaluation** (`evaluate.py`): scores the guardrail's decisions against its labels (accuracy, false blocks, missed blocks), then runs the chain and each agent on the eval sets and scores the answers with an LLM judge. Unanswerable questions expect a refusal. Each agent call is timed and wrapped in LangChain's usage callback, so every system also gets average latency and input/output tokens per question.

**API** (`api/`): FastAPI service in front of the LangGraph agent.
- `POST /v1/ask`: takes `{"question": "..."}` (1–500 characters, validated by Pydantic; anything else gets a 422) and returns `{"answer": "...", "latency_ms": ..., "input_tokens": ..., "output_tokens": ...}`, with the tokens summed over every LLM call in the request (guardrail included)
- `GET /health`: liveness check
- Interactive docs at `/docs`. Each request is logged with the question length, latency and token counts; the LLM client has a 50-second timeout.

## Project layout

```
data/
  corpus/       5 Wikipedia articles
  eval/         hand_written (18), tuning (30), held_out (11), guardrail (20)
notebooks/
  01_retrieval_experiments.ipynb  numpy RAG baseline and retrieval experiments
scripts/
  fetch_corpus.py  re-download the Wikipedia corpus
tests/          unit tests for ingestion, the eval sets, the API and the guardrail router
src/football_rag/
  api/
    main.py     FastAPI app: /v1/ask and /health
    schemas.py  request and response models
  config.py     paths, models, chunking and retrieval settings
  embeddings.py nomic embeddings with document/query prefixes
  ingest.py     build and load the Chroma index
  tools.py      retrieval and Wikipedia tools
  prompts.py    all system prompts
  input_guardrail.py  scope and injection check
  llm.py        shared chat model
  rag_chain.py  LangChain RAG chain (LCEL)
  agent_loop.py hand-written agent
  agent_graph.py LangGraph agent
  crag_graph.py Corrective RAG graph
  evaluate.py   LLM-as-judge evaluation
```

## Setup

1. Install [LM Studio](https://lmstudio.ai), load `qwen/qwen3.8-27b` and `text-embedding-nomic-embed-text-v1.5`, and start the server on `http://localhost:1234`.
2. Install the package (`dev` adds pytest):
   ```
   pip install -e ".[dev]"
   ```
3. Build the index:
   ```
   python -m football_rag.ingest
   ```
4. Ask a question:
   ```
   python -c "from football_rag.agent_graph import run_graph; print(run_graph('Who won Euro 2024?'))"
   ```
5. Serve the API, then open http://127.0.0.1:8000/docs:
   ```
   python -m uvicorn football_rag.api.main:app --reload
   ```
6. Run the evaluation:
   ```
   python -m football_rag.evaluate
   ```
7. Run the tests (no LM Studio needed; the API tests replace the agent with a fake):
   ```
   python -m pytest
   ```

The corpus is committed, so step 3 works offline. `python scripts/fetch_corpus.py` re-downloads it from Wikipedia, but it overwrites the committed articles and the current text may no longer match the eval phrases.

## Known limitations

- Tables are dropped at ingestion, so e.g. the Premier League champions per season are missing.
- The year metadata is a heuristic: a 2022 chunk that mentions 2026 is tagged 2026.
- The model sometimes leaks its own knowledge into search queries.
- The held-out set has no Wikipedia-only, unanswerable or out-of-scope questions yet.
- The guardrail is only wired into the LangGraph agent (the one the API serves), and it has no held-out set yet.

## License

MIT
