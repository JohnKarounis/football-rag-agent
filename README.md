# Football RAG Agent

[![tests](https://github.com/JohnKarounis/football-rag-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/JohnKarounis/football-rag-agent/actions/workflows/tests.yml)

Football Q&A built in measured stages: RAG from scratch with numpy, then LangChain, then a tool-calling agent (hand-written loop and LangGraph), then Corrective RAG. Every step is evaluated with LLM-as-judge on tuning and held-out sets. Runs fully local with LM Studio, Qwen and nomic embeddings.

**Status:** Work in progress. The RAG chain, agents, CRAG, evaluation, retrieval experiments, tests, the FastAPI service, input and output guardrails, per-request latency/token tracking and CI are done; next are Docker and tracing (Langfuse self-hosted, LangSmith optional).

**Stack:** LM Studio | Qwen | nomic-embed | LangChain | Chroma | LangGraph

## Results

LLM-as-judge (Qwen), judged against reference answers. Latency and tokens are averages per question on the tuning set, measured locally (so latency depends on the GPU); tokens are summed over every LLM call a question makes.

| System | Tuning (48) | Held-out (11) | Avg latency | Avg input tokens | Avg output tokens |
|---|---|---|---|---|---|
| LangChain RAG chain | 94% | 91% | 1.0 s | 1,379 | 26 |
| Hand-written agent loop | 100% | 91% | 3.1 s | n/a | n/a |
| LangGraph agent (both guardrails) | 100% | 91% | 6.4 s | 4,926 | 231 |
| CRAG | 98–100% | 91% | 7.1 s | 2,513 | 137 |

The hand-written loop calls the OpenAI client directly, so LangChain's usage callback can't count its tokens. CRAG scored 98% (one question) in one run and 100% in every other, with no code change in between, so that is run-to-run variance.

The fixed RAG chain retrieves once and has no Wikipedia fallback or scope rule, so it misses tuning questions the agents get right. The held-out set has no Wikipedia-only or out-of-scope questions yet, which is why the chain ties there.

The agents and CRAG tie on accuracy. CRAG uses *fewer* tokens than the agent, but its cost is the number of calls, not their size: it makes one small grader call per retrieved chunk (10 per question), and each call is a sequential round trip to the model. The agent makes fewer, larger calls, with all the chunks in one prompt. Without guardrails the agent averaged 3.5 s and 3,256 input tokens, half of CRAG's latency; the two guardrails bring it to 6.4 s, close to CRAG. It stays the default design because the guardrails buy safety, not just accuracy. The RAG chain is the cheapest at a single call, but it has no fallback when retrieval misses.

The one held-out miss is the same for all four: the corpus text names the wrong Morocco goalscorer against Belgium, so it is a data error, not an agent error.

The LangGraph agent scores above run with both guardrails.

### Input guardrail

Scored on its own set of 20 labelled questions (`data/eval/input_guardrail.json`): plain and tricky football questions that must pass, plus off-topic questions, near misses (Super Bowl, Premier League Darts) and prompt injections that must be blocked.

| Accuracy | False blocks | Missed blocks |
|---|---|---|
| 100% | 0 | 0 |

The first prompt said "football" and let the Super Bowl and darts through, so it now says "association football (soccer)" and names other sports as off-topic. The full eval then caught a false block that the guardrail set missed: a tuning question about Qatar revoking a news outlet's visas during the 2022 World Cup, which the guardrail read as politics. The prompt now allows news linked to a tournament, but not general facts about a country ("What is the capital of Qatar?" is blocked). The 20 questions were also used to tune the prompt, so 100% is a tuning score; there is no held-out guardrail set yet.

### Output guardrail

Checks that every fact in the agent's answer is supported by the tool results; an unsupported answer is replaced with `I couldn't find that.` Scored on 16 hand-made cases (`data/eval/output_guardrail.json`): a short context, an answer and a label. The 8 grounded answers paraphrase, reformat or leave facts out; the 8 ungrounded ones have a wrong score, scorer, year or date, add a fact the context doesn't contain (including true facts, like Brazil's five titles when the context doesn't mention them), or come with an injection hidden in the context.

| Accuracy | False rejects | Missed hallucinations |
|---|---|---|
| 100% | 0 | 0 |

On the agent eval it changed nothing: no correct answer was rejected (tuning stays 100%), and the agent didn't hallucinate on these sets, so there was nothing to catch. It is insurance, and it costs about 3 s and 1,700 input tokens per answered question (6.4 s vs 3.5 s, 4,926 vs 3,256 tokens), because it re-reads all the tool results. Refusals and questions the input guardrail blocks skip it.

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

**Input guardrail** (`input_guardrail.py`): one structured-output LLM call (`allowed` + `reason`) that blocks off-topic questions and prompt injections before they reach the agent. In the LangGraph agent it is the first node: `START → input_guardrail → model | END`. A blocked question ends the graph with `I can only help with football.`; an allowed one goes to the agent unchanged. It adds about 0.5 s per question. The agent's own prompt keeps its scope rule as a second layer.

**Output guardrail** (`output_guardrail.py`): one structured-output LLM call (`grounded` + `reason`) that checks the final answer against the tool results collected from the conversation. When the model stops calling tools, the graph goes `model → output_guardrail → END` instead of ending. Ungrounded answers are replaced with `I couldn't find that.`; grounded answers and refusals pass through unchanged. An answer given without any tool call has no context, so it counts as ungrounded, which matches the agent's rule to answer only from tool results.

**Agents:**
- `agent_loop.py`: hand-written tool-calling loop on the OpenAI client
- `agent_graph.py`: the same agent as a LangGraph `model ↔ tools` graph, between the input and output guardrails
- `crag_graph.py`: Corrective RAG: retrieve → grade each chunk → generate, or rewrite the query → Wikipedia → generate, with up to 2 web attempts

**Evaluation** (`evaluate.py`): scores each guardrail against its labelled set (input: accuracy, false blocks, missed blocks; output: accuracy, false rejects, missed hallucinations), then runs the chain and each agent on the eval sets and scores the answers with an LLM judge. Unanswerable questions expect a refusal. Each agent call is timed and wrapped in LangChain's usage callback, so every system also gets average latency and input/output tokens per question.

**API** (`api/`): FastAPI service in front of the LangGraph agent.
- `POST /v1/ask`: takes `{"question": "..."}` (1–500 characters, validated by Pydantic; anything else gets a 422) and returns `{"answer": "...", "latency_ms": ..., "input_tokens": ..., "output_tokens": ...}`, with the tokens summed over every LLM call in the request (guardrail included)
- `GET /health`: liveness check
- Interactive docs at `/docs`. Each request is logged with the question length, latency and token counts; the LLM client has a 50-second timeout.

## Project layout

```
data/
  corpus/       5 Wikipedia articles
  eval/         hand_written (18), tuning (30), held_out (11), input_guardrail (20), output_guardrail (16)
notebooks/
  01_retrieval_experiments.ipynb  numpy RAG baseline and retrieval experiments
scripts/
  fetch_corpus.py  re-download the Wikipedia corpus
tests/          unit tests for ingestion, the eval sets, the API, the guardrail router and tool-context collection
.github/workflows/tests.yml  CI: runs the tests on every push and pull request
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
  output_guardrail.py groundedness check
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
   GitHub Actions runs the same tests on every push and pull request (Ubuntu, Python 3.11). They need neither LM Studio nor a built index: the API tests use a fake agent, and nothing calls the model or searches the store at import time.

The corpus is committed, so step 3 works offline. `python scripts/fetch_corpus.py` re-downloads it from Wikipedia, but it overwrites the committed articles and the current text may no longer match the eval phrases.

## Known limitations

- Tables are dropped at ingestion, so e.g. the Premier League champions per season are missing.
- The year metadata is a heuristic: a 2022 chunk that mentions 2026 is tagged 2026.
- The model sometimes leaks its own knowledge into search queries.
- The held-out set has no Wikipedia-only, unanswerable or out-of-scope questions yet.
- The guardrails are only wired into the LangGraph agent (the one the API serves), and neither has a held-out set yet.
- The output guardrail only skips its check for the exact refusal texts; when the model refuses in its own words, the check still runs (a wasted call, not a wrong answer).

## License

MIT
