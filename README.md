# Football RAG Agent

Football Q&A built in measured stages: RAG from scratch with numpy, then LangChain, then a tool-calling agent (hand-written loop and LangGraph), then Corrective RAG. Every step is evaluated with LLM-as-judge on tuning and held-out sets. Runs fully local with LM Studio, Qwen and nomic embeddings.

**Status:** Work in progress. The agents, CRAG, evaluation and retrieval experiments are done; the production phase (FastAPI, guardrail, tracing, Docker) is next.

**Stack:** LM Studio | Qwen | nomic-embed | LangChain | Chroma | LangGraph

## Results

LLM-as-judge (Qwen), judged against reference answers.

| Agent | Tuning (48) | Held-out (11) |
|---|---|---|
| Hand-written agent loop | 100% | 91% |
| LangGraph agent | 100% | 91% |
| CRAG | 100% | 91% |

The agents and CRAG tie on accuracy, but CRAG makes about twice as many LLM calls (one grader call per retrieved chunk), so the LangGraph agent is the default design.

The one held-out miss is the same for all three: the corpus text names the wrong Morocco goalscorer against Belgium, so it is a data error, not an agent error.

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

**Agents:**
- `agent_loop.py`: hand-written tool-calling loop on the OpenAI client
- `agent_graph.py`: the same agent as a LangGraph `model ↔ tools` graph
- `crag_graph.py`: Corrective RAG: retrieve → grade each chunk → generate, or rewrite the query → Wikipedia → generate, with up to 2 web attempts

**Evaluation** (`evaluate.py`): runs each agent on the eval sets and scores the answers with an LLM judge. Unanswerable questions expect a refusal.

## Project layout

```
data/
  corpus/       5 Wikipedia articles
  eval/         hand_written (18), tuning (30), held_out (11)
notebooks/
  01_retrieval_experiments.ipynb  numpy RAG baseline and retrieval experiments
src/football_rag/
  config.py     paths, models, chunking and retrieval settings
  embeddings.py nomic embeddings with document/query prefixes
  ingest.py     build and load the Chroma index
  tools.py      retrieval and Wikipedia tools
  prompts.py    all system prompts
  llm.py        shared chat model
  agent_loop.py hand-written agent
  agent_graph.py LangGraph agent
  crag_graph.py Corrective RAG graph
  evaluate.py   LLM-as-judge evaluation
```

## Setup

1. Install [LM Studio](https://lmstudio.ai), load `qwen/qwen3.8-27b` and `text-embedding-nomic-embed-text-v1.5`, and start the server on `http://localhost:1234`.
2. Install the package:
   ```
   pip install -e .
   ```
3. Build the index:
   ```
   python -m football_rag.ingest
   ```
4. Ask a question:
   ```
   python -c "from football_rag.agent_graph import run_graph; print(run_graph('Who won Euro 2024?'))"
   ```
5. Run the evaluation:
   ```
   python -m football_rag.evaluate
   ```

## Known limitations

- Tables are dropped at ingestion, so e.g. the Premier League champions per season are missing.
- The year metadata is a heuristic: a 2022 chunk that mentions 2026 is tagged 2026.
- The model sometimes leaks its own knowledge into search queries.
- The held-out set has no Wikipedia-only, unanswerable or out-of-scope questions yet.

## License

MIT
