# CLAUDE.md — Football RAG / Agent Project

## What this project is
A portfolio-quality AI engineering project: question answering over football knowledge, built in stages and measured at every step.
1. **RAG pipeline** (fixed workflow): raw Python/numpy first, then LangChain (LCEL).
2. **Agent** (model controls retrieval via tools): hand-written loop, then LangGraph.
3. **Corrective RAG (CRAG)** (graph-controlled workflow with chunk grading + web fallback), compared against the agent.

The original work lives in `C:\Users\User\Documents\Drills\AI_Engineering.ipynb` and has been fully migrated into this repo. Phase 5 — production — is in progress: FastAPI, the input guardrail and latency/token tracking are done; next are the output guardrail, LangSmith tracing, Docker and CI.

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
data/eval/        hand_written.json (18), tuning.json (30), held_out.json (11), guardrail.json (20: question, allowed)
notebooks/        01_retrieval_experiments.ipynb (numpy RAG baseline + retrieval experiments; imports from football_rag)
scripts/          fetch_corpus.py (re-downloads the 5 articles; overwrites data/corpus — don't run casually)
tests/            test_ingest.py (headings, years, breadcrumbs, 565 chunks), test_eval_data.py (sizes, keys, phrases in corpus, no held-out overlap, guardrail set format), test_api.py (health, ask with fake run_graph_with_usage via monkeypatch incl. token fields, 422 on empty/too-long), test_agent_graph.py (route_after_guardrail: allowed → model, refusal → END, refusal text typed by user → model); 17 tests, no LM Studio needed (test_api imports agent_graph, which loads the local chroma_db)
src/football_rag/
  api/main.py     FastAPI app: POST /v1/ask (run_graph_with_usage, latency_ms, one log line per request with question length, latency and tokens in/out), GET /health; run with `python -m uvicorn football_rag.api.main:app --reload`
  api/schemas.py  AskRequest (question, 1..MAX_QUESTION_CHARS=500), AskResponse (answer, latency_ms, input_tokens, output_tokens)
  config.py       paths (ROOT, CORPUS_DIR, EVAL_DIR, CHROMA_DIR), models, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
  embeddings.py   NomicEmbeddings, get_embeddings()
  ingest.py       load_docs, split_docs, add_breadcrumbs, build_store, load_store; __main__ rebuilds the index
  tools.py        search_football_docs, search_wikipedia, agent_tools
  prompts.py      SYSTEM, GRADER_SYSTEM, REWRITE_SYSTEM, GENERATE_SYSTEM, JUDGE_SYSTEM, RAG_SYSTEM, INPUT_GUARDRAIL_SYSTEM
  input_guardrail.py  GuardrailDecision (allowed, reason), input_guardrail_chain (structured output), check_question
  llm.py          get_llm() (temperature 0, LLM_TIMEOUT = 50 s), count_tokens(usage_callback) (sums input/output tokens over usage_callback.usage_metadata, which is keyed by model name)
  rag_chain.py    rag_chain, run_chain (retrieve once → answer; no tools, no scope rule)
  agent_loop.py   run_agent (hand-written loop, OpenAI client)
  agent_graph.py  REFUSAL, guardrail_node, route_after_guardrail, build_graph(llm, tools, system_prompt), run_graph, run_graph_with_usage (run_graph inside get_usage_metadata_callback → answer, input_tokens, output_tokens; used by the API)
  crag_graph.py   CRAGState, GradeChunk, chains, nodes, build_crag_graph, run_crag
  evaluate.py     load_eval_set, evaluate_agent (judge rate + avg latency/input/output tokens; timer and usage callback wrap only the agent call, not the judge), evaluate_guardrail; __main__ scores the guardrail, then the chain and all three agents
```
Run with `python -m football_rag.ingest`, `python -m football_rag.evaluate`, `python -m uvicorn football_rag.api.main:app --reload` and `python -m pytest` (Device Guard blocks `.exe` launchers like `pytest.exe` and `uvicorn.exe` on John's machine, so always use `python -m`). Run `python -m pytest` before every code commit. Teaching mode for new topics (e.g. the API): explain the concept and give a skeleton/step list, let John write the code, then review — don't hand over full code unless asked.

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
- **Agent graph:** `StateGraph(MessagesState)`: `START → guardrail → (route_after_guardrail, path map ['model', END]) → model | END`, then `model` ↔ `tools` (`ToolNode`) via `tools_condition`; `RECURSION_LIMIT = 12`, catches `GraphRecursionError`. `guardrail_node` appends `AIMessage(REFUSAL)` when blocked, else returns `{}`; the router checks `isinstance(last, AIMessage)`, not the text. `SYSTEM` keeps its scope rule as a second layer.
- **Input guardrail:** `INPUT_GUARDRAIL_SYSTEM` scopes to association football (soccer), allows news linked to a tournament (politics, human rights, media), blocks general country facts, other sports (NFL, rugby, darts) and injection attempts; unsure → allow. ~0.5 s per call. Only in the LangGraph agent (the API's agent).
- **CRAG graph:** `retrieve → grade → generate | rewrite → web_search → generate → END | rewrite`. One grader call per chunk (structured `GradeChunk`). Retry when the answer contains `couldn't find that` and `attempts < MAX_WEB_ATTEMPTS` (2).
- **Evaluation:** `judge_chain` returns CORRECT/INCORRECT; `answer: None` means expected `Refusal`. `evaluate_agent(agent_func, eval_set)` returns judge rate + per-question results. `evaluate_guardrail(eval_set)` calls `check_question` directly (no judge) and returns accuracy, false blocks (allowed but blocked), missed blocks (should block but allowed) + results with the reason.

## Eval data (rules matter)
- `hand_written.json`: 18 items (`question`, `answer`, `phrase`): direct, reworded, 3 Wikipedia-only, 2 unanswerable (`answer: null`), 1 out-of-scope.
- `tuning.json` (30) and `held_out.json` (11): LLM-generated, manually reviewed, answers filled in. Committed as final files; no build script.
- `guardrail.json`: 20 hand-labelled items: 5 plain football, 5 tricky football (governance, ownership, Qatar workers, VAR, TV money), 5 off-topic incl. near misses (Super Bowl, Premier League Darts), 5 injections (one wrapped in a football question). Also used to tune the guardrail prompt; no held-out guardrail set yet.
- Tuning set = `hand_written + tuning` (48). **Held-out is never used for tuning** — run it only to confirm a finished change.
- The raw `eval_generated.json` (questions without answers) stays in Drills and is not committed.

## Results (judge rate)
| System | Tuning (48) | Held-out (11) | Avg latency (tuning) | Avg input / output tokens (tuning) |
|---|---|---|---|---|
| RAG chain (LCEL) | 0.94 | 0.91 | 1.01 s | 1,379 / 26 |
| Agent — hand-written loop | 1.00 | 0.91 | 3.05 s | n/a (raw OpenAI client, callback can't see it) |
| Agent — LangGraph (+ guardrail) | 1.00 | 0.91 | 3.48 s | 3,256 / 161 |
| CRAG + retry | 0.98 (1.00 in earlier runs) | 0.91 | 7.07 s | 2,495 / 136 |

Held-out averages: chain 1.02 s, 1,384/30; loop 2.73 s; LangGraph 3.49 s, 2,934/172; CRAG 7.30 s, 2,554/143. CRAG's 0.98 is one tuning question in the latest run with no CRAG code change: run-to-run variance (temperature 0 is not fully deterministic in LM Studio, and the Wikipedia tool hits the live API). CRAG uses fewer tokens than the agent but ~2× the latency: 10 small sequential grader calls per question, so the cost is call count, not tokens. Live API spot check: a docs question ~2.6k input tokens, Wikipedia path ~5.4k, a guardrail-blocked question ~200.

Notebook timings: LangGraph 2m41s / 34s, CRAG 5m01s / 1m11s (tuning / held-out). Earlier, on the original 15 questions: raw RAG 0.87, LangChain chain 0.87, agent 0.93, agent + `prefer_recent` 1.00.

LangGraph agent scores are with the guardrail in front. Guardrail set: accuracy 1.00, 0 false blocks, 0 missed blocks. History: "football" let Super Bowl and darts through → "association football (soccer)" + other sports named; the full eval then showed a false block (Iran International visas question, tuning 0.98) → allow tournament news; that let "capital of Qatar" through → "General facts about a country are not about soccer".

**Conclusion:** agent and CRAG tie on accuracy; CRAG costs ~2× latency (more, smaller sequential calls) while using fewer tokens. The agent is the default design; CRAG is kept as a measured alternative.

### Retrieval experiments
From `notebooks/01_retrieval_experiments.ipynb` (tuning set, hit@3/5/10 + MRR): structure-based chunking beats fixed at every size (100/20: MRR 0.80 vs 0.73); LLM query rewriting no gain (MRR 0.80 → 0.78); reranking mixed (MiniLM best MRR 0.85, bge best hit@10 1.0), not adopted; hybrid BM25 + RRF worse at every weighting; nomic beats Qwen3-Embedding-0.6B (MRR 0.79 vs 0.71). The notebook's rewrites showed no knowledge leakage, so don't claim it.

## Known issues
- **Morocco vs Belgium (held-out miss for all four):** the corpus names Saïss instead of Sabiri. Data error, not an agent error.
- **Tables are dropped** at ingestion (`explaintext`). Real fix: parse HTML tables to Markdown.
- `max_year` is a heuristic (2022 chunks mentioning 2026 get 2026).
- The model sometimes leaks its own knowledge into search queries.
- The judge can be lenient/wrong — always read failures before trusting a score.
- Held-out has no Wikipedia-only, unanswerable or out-of-scope questions yet.
- `evaluate.py` prints only at the end and saves nothing; a crash loses the whole run.
- The guardrail only checks topic and injection on the input; no output guardrail yet. Not wired into `run_agent` or `run_crag`.
- Token counts come from LangChain's usage callback, so `run_agent` (raw OpenAI client) reports 0; it would need `response.usage` read in the loop.

## Roadmap
1. Done: `notebooks/01_retrieval_experiments.ipynb` (numpy RAG class + 5 experiments, results in Markdown cells and a summary table).
2. Done: `scripts/fetch_corpus.py` (User-Agent + timeout + `raise_for_status`).
3. Done: tests (pytest). Later: tests for CRAG routing functions (needs crag_graph import without loading the store/LLM).
4. Done: `feat/api` — FastAPI `/v1/ask` + `/health`, Pydantic schemas, LLM timeout, request logging, API tests. Decided: no rate limiting or background jobs (single-user local app, LM Studio is the bottleneck); API key auth only if deployed. Later option: strip whitespace-only questions; streaming.
5. Done: `feat/guardrail` — input guardrail node, guardrail eval set + `evaluate_guardrail`, router tests. Later: held-out guardrail set; output guardrails.
6. Done: `feat/observability` — `run_graph_with_usage` + `count_tokens`, token fields in `/v1/ask` and its log line, latency/tokens per question in `evaluate_agent`.
7. **Phase 5 (rest):** output guardrail (groundedness check: answer supported by tool results; measure its cost with the new metrics); LangSmith tracing; caching; Docker; GitHub Actions CI running `python -m pytest` on every push/PR.

**Later / optional:** `grader_chain.batch(...)` to parallelise CRAG grading; small fine-tuned grader or guardrail classifier; table-aware ingestion; conversation memory; supervisor + SQL stats agent.

## Git workflow
- One feature branch per stage (`feat/...`), small tested commits, push, PR with **What / Test** sections, merge commit, delete the branch, then `git switch main`, `git pull`, `git branch -d`.
- Stage files explicitly when there are untracked files that don't belong in the commit.

## How to work with John
- **John makes every change himself** (code, files, commits, pushes, PRs). Give him the code and commands; don't edit the repo unless he says so. **Exception:** Claude writes and updates `README.md` and `CLAUDE.md` directly; John reviews and commits them. Reading code, files, git and GitHub never needs permission.
- Short, direct answers. Elaborate only when asked.
- Plain, readable code: one step per line, named intermediates, no calls crammed into loop headers, no clever one-liners. Don't over-engineer; stay close to the notebook style.
- Explain how a concept works before implementing it, and lay out the whole plan (files, commits) up front.
- Before each commit, check the file: run it or test it live, look for bugs and typos. Don't flag whitespace (blank lines, trailing newlines).
- Flag bugs clearly; own your own mistakes briefly and move on.
- Every change is judged by the evals: tuning set while iterating, held-out once at the end.
