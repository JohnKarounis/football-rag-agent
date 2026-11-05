# CLAUDE.md — Football RAG / Agent Project

## What this project is
A portfolio-quality AI engineering project: question answering over football knowledge, built in stages and measured at every step.
1. **RAG pipeline** (fixed workflow): raw Python/numpy first, then LangChain (LCEL).
2. **Agent** (model controls retrieval via tools): hand-written loop, then LangGraph.
3. **Corrective RAG (CRAG)** (graph-controlled workflow with chunk grading + web fallback), compared against the agent.

The original work lives in `C:\Users\User\Documents\Drills\AI_Engineering.ipynb` and is being migrated into this repo stage by stage. Next: `scripts/fetch_corpus.py`, tests, then **Phase 5 — production** (FastAPI, guardrail, cost/latency, observability, Docker).

Everything runs locally and free. No paid APIs.

## Stack
- **LLM + embeddings:** LM Studio, OpenAI-compatible server at `http://localhost:1234/v1`
  - Chat: Qwen (`qwen/qwen3.8-27b`), `temperature=0` everywhere
  - Embeddings: `text-embedding-nomic-embed-text-v1.5`. Requires prefixes `search_document: ` (indexing) and `search_query: ` (queries); handled by `NomicEmbeddings` in `embeddings.py`
- **Frameworks:** LangChain (LCEL, `ChatOpenAI`, `@tool`), LangGraph, Chroma (persisted in `chroma_db/`, cosine, not committed)
- **Environment:** Windows, conda env `ml` (Python 3.11), VS Code. John runs commands in cmd (Anaconda Prompt), so give cmd syntax, not bash.
- **Package:** src layout, `pyproject.toml` with pinned deps, installed with `pip install -e .`. Extras: `experiments` (rank-bm25, sentence-transformers), `dev` (pytest, ruff, ipykernel).

## Repo layout
```
data/corpus/      5 Wikipedia articles (.txt)
data/eval/        hand_written.json (18), tuning.json (30), held_out.json (11)
notebooks/        01_retrieval_experiments.ipynb (numpy RAG baseline + retrieval experiments; imports from football_rag)
src/football_rag/
  config.py       paths (ROOT, CORPUS_DIR, EVAL_DIR, CHROMA_DIR), models, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
  embeddings.py   NomicEmbeddings, get_embeddings()
  ingest.py       load_docs, split_docs, add_breadcrumbs, build_store, load_store; __main__ rebuilds the index
  tools.py        search_football_docs, search_wikipedia, agent_tools
  prompts.py      SYSTEM, GRADER_SYSTEM, REWRITE_SYSTEM, GENERATE_SYSTEM, JUDGE_SYSTEM
  llm.py          get_llm()
  agent_loop.py   run_agent (hand-written loop, OpenAI client)
  agent_graph.py  build_graph(llm, tools, system_prompt), run_graph
  crag_graph.py   CRAGState, GradeChunk, chains, nodes, build_crag_graph, run_crag
  evaluate.py     load_eval_set, evaluate_agent; __main__ scores all three agents
```
Run with `python -m football_rag.ingest` and `python -m football_rag.evaluate`.

## Data / corpus
- 5 Wikipedia articles fetched via the MediaWiki API (`prop=extracts`, `explaintext=1`): 2022 FIFA World Cup, UEFA Euro 2024, FIFA World Cup, Premier League, Offside (association football).
- Ingestion: `==` headings converted to Markdown → `MarkdownHeaderTextSplitter` → `RecursiveCharacterTextSplitter` (100 words, 20 overlap, word length function) → `title > section` breadcrumb on each chunk → `max_year` metadata (latest year in the chunk) → Chroma. 565 chunks.
- Retriever returns 10 chunks.
- All Wikipedia HTTP calls: `User-Agent` `rag-practice/1.0 (github.com/JohnKarounis)`, `timeout=10`, `raise_for_status()`.

## Key components
- **Tools** (`@tool(parse_docstring=True)`, shared by all agents; the docstrings are part of the tuned system — do not reword):
  - `search_football_docs(query, prefer_recent=False)`: top 10 by relevance; `prefer_recent=True` takes top 20 and re-sorts by `max_year`
  - `search_wikipedia(query)`: `generator=search`, top 3 article intros (1,500 chars each), sorted by search rank; returns an error string instead of raising
- **Prompts:** all live in `prompts.py`, copied verbatim from the notebook. Do not reword without re-running the evals.
- **`run_agent`:** hand-written tool-calling loop on the raw OpenAI client, `MAX_STEPS = 5`.
- **Agent graph:** `StateGraph(MessagesState)`: `model` ↔ `tools` (`ToolNode`) via `tools_condition`; `RECURSION_LIMIT = 12`, catches `GraphRecursionError`.
- **CRAG graph:** `retrieve → grade → generate | rewrite → web_search → generate → END | rewrite`. One grader call per chunk (structured `GradeChunk`). Retry when the answer contains `couldn't find that` and `attempts < MAX_WEB_ATTEMPTS` (2).
- **Evaluation:** `judge_chain` returns CORRECT/INCORRECT; `answer: None` means expected `Refusal`. `evaluate_agent(agent_func, eval_set)` returns judge rate + per-question results.

## Eval data (rules matter)
- `hand_written.json`: 18 items (`question`, `answer`, `phrase`): direct, reworded, 3 Wikipedia-only, 2 unanswerable (`answer: null`), 1 out-of-scope.
- `tuning.json` (30) and `held_out.json` (11): LLM-generated, manually reviewed, answers filled in. Committed as final files; no build script.
- Tuning set = `hand_written + tuning` (48). **Held-out is never used for tuning** — run it only to confirm a finished change.
- The raw `eval_generated.json` (questions without answers) stays in Drills and is not committed.

## Results (judge rate)
| System | Tuning (48) | Held-out (11) |
|---|---|---|
| Agent — hand-written loop | 1.00 | 0.91 |
| Agent — LangGraph | 1.00 | 0.91 |
| CRAG + retry | 1.00 | 0.91 |

Notebook timings: LangGraph 2m41s / 34s, CRAG 5m01s / 1m11s (tuning / held-out). Earlier, on the original 15 questions: raw RAG 0.87, LangChain chain 0.87, agent 0.93, agent + `prefer_recent` 1.00.

**Conclusion:** agent and CRAG tie on accuracy; CRAG costs ~2× latency/LLM calls. The agent is the default design; CRAG is kept as a measured alternative.

### Retrieval experiments
From `notebooks/01_retrieval_experiments.ipynb` (tuning set, hit@3/5/10 + MRR): structure-based chunking beats fixed at every size (100/20: MRR 0.80 vs 0.73); LLM query rewriting no gain (MRR 0.80 → 0.78); reranking mixed (MiniLM best MRR 0.85, bge best hit@10 1.0), not adopted; hybrid BM25 + RRF worse at every weighting; nomic beats Qwen3-Embedding-0.6B (MRR 0.79 vs 0.71). The notebook's rewrites showed no knowledge leakage, so don't claim it.

## Known issues
- **Morocco vs Belgium (held-out miss for all three):** the corpus names Saïss instead of Sabiri. Data error, not an agent error.
- **Tables are dropped** at ingestion (`explaintext`). Real fix: parse HTML tables to Markdown.
- `max_year` is a heuristic (2022 chunks mentioning 2026 get 2026).
- The model sometimes leaks its own knowledge into search queries.
- The judge can be lenient/wrong — always read failures before trusting a score.
- Held-out has no Wikipedia-only, unanswerable or out-of-scope questions yet.
- `evaluate.py` prints only at the end and saves nothing; a crash loses the whole run.

## Roadmap
1. Done: `notebooks/01_retrieval_experiments.ipynb` (numpy RAG class + 5 experiments, results in Markdown cells and a summary table).
2. `scripts/fetch_corpus.py`: re-download the corpus (User-Agent + timeout).
3. Tests.
4. **Phase 5:** FastAPI (`/ask`, streaming); input guardrail node (`START → guardrail → model | END`, scope + injection checks); per-request timing and token counts, caching; LangSmith tracing; Docker.

**Later / optional:** `grader_chain.batch(...)` to parallelise CRAG grading; small fine-tuned grader or guardrail classifier; table-aware ingestion; conversation memory; supervisor + SQL stats agent.

## Git workflow
- One feature branch per stage (`feat/...`), small tested commits, push, PR with **What / Test** sections, merge commit, delete the branch, then `git switch main`, `git pull`, `git branch -d`.
- Stage files explicitly when there are untracked files that don't belong in the commit.
- Every file ends with a trailing newline.

## How to work with John
- **John makes every change himself** (code, files, commits, pushes, PRs). Give him the code and commands; don't edit the repo unless he says so. Reading code, files, git and GitHub never needs permission.
- Short, direct answers. Elaborate only when asked.
- Plain, readable code: one step per line, named intermediates, no calls crammed into loop headers, no clever one-liners. Don't over-engineer; stay close to the notebook style.
- Explain how a concept works before implementing it, and lay out the whole plan (files, commits) up front.
- Before each commit, check the file: run it or test it live, look for typos and a missing trailing newline.
- Flag bugs clearly; own your own mistakes briefly and move on.
- Every change is judged by the evals: tuning set while iterating, held-out once at the end.
