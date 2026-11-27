# Football RAG Agent

[![tests](https://github.com/JohnKarounis/football-rag-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/JohnKarounis/football-rag-agent/actions/workflows/tests.yml)

Football Q&A built in measured stages: RAG from scratch with numpy, then LangChain, then a tool-calling agent (hand-written loop and LangGraph), then Corrective RAG. Every step is evaluated with LLM-as-judge on tuning and held-out sets. Runs fully local with LM Studio, Qwen and nomic embeddings.

**Status:** Work in progress. The RAG chain, agents, CRAG, evaluation, retrieval experiments, tests, the FastAPI service, input and output guardrails, per-request latency/token tracking, CI, Docker, tracing (self-hosted Langfuse), a Gradio chat UI, a self-correcting retry after the output guardrail and conversation memory are done.

**Stack:** LM Studio | Qwen | nomic-embed | LangChain | Chroma | LangGraph | FastAPI | Docker | Langfuse | Gradio

## Results

LLM-as-judge (Qwen), judged against reference answers. Latency and tokens are averages per question on the tuning set, measured locally (so latency depends on the GPU); tokens are summed over every LLM call a question makes.

| System | Tuning (48) | Held-out (11) | Avg latency | Avg input tokens | Avg output tokens |
|---|---|---|---|---|---|
| LangChain RAG chain | 94% | 91% | 1.1 s | 1,386 | 26 |
| Hand-written agent loop | 98–100% | 91% | 3.0 s | n/a | n/a |
| LangGraph agent (both guardrails, memory) | 98–100% | 91% | 6.2 s | 4,818 | 226 |
| CRAG | 98–100% | 91% | 7.7 s | 2,515 | 135 |

The hand-written loop calls the OpenAI client directly, so LangChain's usage callback can't count its tokens. The 98% scores are one question in one run, a different question each time, with 100% in the other runs and no code change in between: run-to-run variance (temperature 0 is not fully deterministic in LM Studio, and the Wikipedia tool hits the live API). Latency also drifts by a few tenths of a second between runs.

The fixed RAG chain retrieves once and has no Wikipedia fallback or scope rule, so it misses tuning questions the agents get right. The held-out set has no Wikipedia-only or out-of-scope questions yet, which is why the chain ties there.

The agents and CRAG tie on accuracy. CRAG uses *fewer* tokens than the agent, but its cost is the number of calls, not their size: it makes one small grader call per retrieved chunk (10 per question), and each call is a sequential round trip to the model. The agent makes fewer, larger calls, with all the chunks in one prompt. Without guardrails the agent averaged 3.5 s and 3,256 input tokens, half of CRAG's latency; the two guardrails bring it to about 6 s, close to CRAG. It stays the default design because the guardrails buy safety, not just accuracy. The RAG chain is the cheapest at a single call, but it has no fallback when retrieval misses.

The one held-out miss is the same for all four: the corpus text names the wrong Morocco goalscorer against Belgium, so it is a data error, not an agent error.

The LangGraph agent scores above run with both guardrails.

### Input guardrail

Scored on its own set of 20 labelled questions (`data/eval/input_guardrail.json`): plain and tricky football questions that must pass, plus off-topic questions, near misses (Super Bowl, Premier League Darts) and prompt injections that must be blocked.

| Accuracy | False blocks | Missed blocks |
|---|---|---|
| 100% | 0 | 0 |

The first prompt said "football" and let the Super Bowl and darts through, so it now says "association football (soccer)" and names other sports as off-topic. The full eval then caught a false block that the guardrail set missed: a tuning question about Qatar revoking a news outlet's visas during the 2022 World Cup, which the guardrail read as politics. The prompt now allows news linked to a tournament, but not general facts about a country ("What is the capital of Qatar?" is blocked). The 20 questions were also used to tune the prompt, so 100% is a tuning score; there is no held-out guardrail set yet.

### Output guardrail

Checks that every fact in the agent's answer is supported by the tool results; an unsupported answer is replaced with `I couldn't find that.` Scored on 18 hand-made cases (`data/eval/output_guardrail.json`): a short context, an answer and a label. The 9 grounded answers paraphrase, reformat, leave facts out or combine facts from two chunks; the 9 ungrounded ones have a wrong score, scorer, year or date, add a fact the context doesn't contain (including true facts, like Brazil's five titles when the context doesn't mention them), come with an injection hidden in the context, or state an offside rule in reverse.

| Accuracy | False rejects | Missed hallucinations |
|---|---|---|
| 94% (17/18) | 0 | 1 |

The two offside cases come from a real trace (see Tracing below). The guardrail accepts the answer that combines two chunks, but it misses the reversed rule every time: the context says offside is judged when the ball is played, "not when the player receives the ball", and the answer says a player who is onside by the time they receive it is not penalised. The guardrail's reason fills the gap with its own inference ("implying no penalty if they become onside before receiving it"). Three prompt changes (allowing synthesis, naming reversed conditions, forbidding inference) didn't move it, so the prompt stays as it was and the case stays in the set as a known miss rather than being tuned away on an 18-item set. It catches wrong names, scores, dates and invented facts; subtly reversed rules need a stronger checker model or a sentence-by-sentence check.

On the agent eval it changed nothing: no correct answer was rejected (tuning stays 100%), and the agent didn't hallucinate on these sets, so there was nothing to catch. It is insurance, and it costs about 3 s and 1,700 input tokens per answered question (6.4 s vs 3.5 s, 4,926 vs 3,256 tokens), because it re-reads all the tool results. Refusals and questions the input guardrail blocks skip it.

### Conversation memory

Scored on 8 two-turn conversations (`data/eval/follow_up.json`): a question, then a follow-up that only makes sense with it ("Who won the 2022 World Cup?" → "Who was voted best player of that tournament?"). Both answers have a reference and are judged. The control run sends each follow-up to a new conversation, so the agent sees it without the first question.

| | First question | Follow-up |
|---|---|---|
| With memory | 100% | 100% |
| Without memory (control) | 100% | 25% |

The control shows the follow-ups really depend on the first turn: without it, the agent guesses (asked "Who did they beat in the final?" cold, it picks the 2022 World Cup). The two it still gets right are the ones where the follow-up names enough on its own.

The first run scored 7/8 with memory and found a real gap: "What was the largest stadium at Euro 2024?" → Olympiastadion in Berlin, then "In which city is it?" → `I can only help with football.` The agent had the whole conversation, but the input guardrail only saw the latest message, which on its own reads like a geography question. The guardrail now also gets the user's previous question for follow-ups. A first attempt also added a sentence to the guardrail prompt and sent `Previous question: None` with every first question; that fixed the follow-up, but flipped the tuning question about Qatar revoking a news outlet's visas back to a false block (3 of 3 tries in each variant, against 3 of 3 allowed before). So the prompt is unchanged, and the previous question is added only when there is one: first questions see exactly the input they did before, and all 8 follow-ups pass.

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

**Input guardrail** (`input_guardrail.py`): one structured-output LLM call (`allowed` + `reason`) that blocks off-topic questions and prompt injections before they reach the agent. In the LangGraph agent it is the first node: `START → input_guardrail → model | END`. A blocked question ends the graph with `I can only help with football.`; an allowed one goes to the agent unchanged. For a follow-up in a conversation it also gets the user's previous question, so "In which city is it?" can be read in context. It adds about 0.5 s per question. The agent's own prompt keeps its scope rule as a second layer.

**Output guardrail** (`output_guardrail.py`): one structured-output LLM call (`grounded` + `reason`) that checks the final answer against the tool results collected from the conversation. When the model stops calling tools, the graph goes `model → output_guardrail` instead of ending. Grounded answers and refusals pass through to `END` unchanged. An ungrounded answer gets **one retry**: the guardrail adds a feedback message with its reason ("Your answer is not supported by the search results. … Rewrite it using only facts from the search results, or reply exactly: 'I couldn't find that.'"), the graph routes back to the model, and the rewrite is checked again. Only if the rewrite also fails is the answer replaced with `I couldn't find that.` The retry count comes from the feedback messages already in the conversation, so the graph state stays a plain message list. With memory, the guardrail checks the current turn: the question is the latest user message that isn't feedback, and only that turn's feedback messages count as retries. An answer given without any tool call has no context, so it counts as ungrounded, which matches the agent's rule to answer only from tool results.

**Agents:**
- `agent_loop.py`: hand-written tool-calling loop on the OpenAI client
- `agent_graph.py`: the same agent as a LangGraph `model ↔ tools` graph, between the input and output guardrails, with conversation memory
- `crag_graph.py`: Corrective RAG: retrieve → grade each chunk → generate, or rewrite the query → Wikipedia → generate, with up to 2 web attempts

**Conversation memory** (`agent_graph.py`): the graph is compiled with LangGraph's `MemorySaver` checkpointer, and every run gets a `thread_id`. After each run the checkpointer stores the thread's messages; the next question in the same thread is appended to them, so the model sees the whole conversation and can resolve "they" or "that tournament". A call without a thread id gets a new one, so the eval sets still run one independent question at a time.

**Evaluation** (`evaluate.py`): scores the follow-up conversations with and without memory, then each guardrail against its labelled set (input: accuracy, false blocks, missed blocks; output: accuracy, false rejects, missed hallucinations), then runs the chain and each agent on the eval sets and scores the answers with an LLM judge. Unanswerable questions expect a refusal. Each agent call is timed and wrapped in LangChain's usage callback, so every system also gets average latency and input/output tokens per question.

**API** (`api/`): FastAPI service in front of the LangGraph agent.
- `POST /v1/ask`: takes `{"question": "...", "conversation_id": "..."}` (question 1–500 characters, validated by Pydantic; anything else gets a 422) and returns `{"answer": "...", "latency_ms": ..., "input_tokens": ..., "output_tokens": ..., "conversation_id": "..."}`, with the tokens summed over every LLM call in the request (guardrails included). `conversation_id` is optional: leave it out and the API starts a new conversation and returns its id; send it back with the next question to continue the conversation
- `GET /health`: liveness check
- Interactive docs at `/docs`. Each request is logged with the question length, latency and token counts; the LLM client has a 50-second timeout.

**Docker:** the API runs in a container (`Dockerfile`: `python:3.11-slim`, editable install, uvicorn on `0.0.0.0:8000`). LM Studio stays on the host: `config.py` reads `LMSTUDIO_URL` from the environment (default `http://localhost:1234/v1`), and `docker-compose.yml` sets it to `http://host.docker.internal:1234/v1`. The Chroma index isn't baked into the image; Compose mounts the local `chroma_db/` into the container.

**Tracing** (`tracing.py`): optional, with a self-hosted [Langfuse](https://langfuse.com). `get_callbacks()` returns Langfuse's LangChain callback handler when `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set, and an empty list otherwise, so tests, CI and anyone without Langfuse run unchanged. `run_graph` passes the callbacks to the graph, and every request becomes one trace: each node, LLM call (prompt, output, tokens, latency), tool call and retrieval as a tree. The keys live in a git-ignored `.env` (template: `.env.example`); uvicorn loads it with `--env-file`, Compose with `env_file`. Langfuse runs from its own Docker Compose stack outside this repo, at `http://localhost:3000` (the API container reaches it at `host.docker.internal:3000`).

Tracing already paid off: a question about the all-time top scorer hit the 12-step limit, and the trace showed why: one docs search, then four Wikipedia searches (one with a player name the model added from its own knowledge), none of which returned the answer, since the corpus has no scorer table and the Wikipedia tool returns only article intros. The API used to return the technical `Stopped: reached the step limit` in that case; it now returns `I couldn't find that.`, like any other unanswerable question. It also shows where the tokens go: the 10 retrieved chunks (~1,300 tokens) are read twice per answer, once by the agent and once by the output guardrail.

It also caught the output guardrail's first real block, and a weakness: "What is the offside rule?" got a long answer that stated one rule backwards (a player who was offside when the ball was played "becomes onside before receiving it" and is not penalised). The guardrail replaced it with `I couldn't find that.`, but its reason was a different, correct sentence that combined facts from two chunks, and it didn't mention the false one. The right outcome for the wrong reason. Both cases are now in the guardrail set (see Output guardrail above).

**Chat UI** (`gradio_ui.py`): a Gradio `ChatInterface` that renders the agent's Markdown answers. It is a separate client of the API, not part of the agent: each message is sent to `POST /v1/ask` with `requests` (URL from `API_URL`, default `http://127.0.0.1:8000`), so the flow is browser → Gradio → API → agent, the same as with any other front end. Gradio is an optional `ui` extra, so the API image and CI don't install it. The UI keeps the API's `conversation_id` in a `gr.State` for each browser session and sends it with every message, so follow-ups work; clearing the chat starts a new conversation. If the API can't be reached or returns an error (for example when `run` opens the UI before the API has finished starting), the chat shows "Couldn't get an answer from the API. Is it running? Try again in a few seconds." instead of an error badge, and keeps the conversation.

## Project layout

```
data/
  corpus/       5 Wikipedia articles
  eval/         hand_written (18), tuning (30), held_out (11), input_guardrail (20), output_guardrail (18), follow_up (8 conversations)
notebooks/
  01_retrieval_experiments.ipynb  numpy RAG baseline and retrieval experiments
scripts/
  fetch_corpus.py  re-download the Wikipedia corpus
tests/          unit tests for ingestion, the eval sets, the API, both guardrail routers, tool-context collection, the retry count, the current-question lookup, the step limit and the API's conversation id
.github/workflows/tests.yml  CI: runs the tests on every push and pull request
Dockerfile      API image
docker-compose.yml  runs the API container: port 8000, LMSTUDIO_URL, Langfuse URL, .env, chroma_db volume
.env.example    template for the Langfuse keys (copy to .env)
run.bat         starts the API and the chat UI in two windows (Windows)
src/football_rag/
  api/
    main.py     FastAPI app: /v1/ask and /health
    schemas.py  request and response models
  config.py     paths, models, chunking and retrieval settings (LMSTUDIO_URL overridable by env var)
  embeddings.py nomic embeddings with document/query prefixes
  ingest.py     build and load the Chroma index
  tools.py      retrieval and Wikipedia tools
  prompts.py    all system prompts
  input_guardrail.py  scope and injection check
  output_guardrail.py groundedness check
  llm.py        shared chat model
  tracing.py    optional Langfuse callback handler
  gradio_ui.py  Gradio chat UI that calls the API
  rag_chain.py  LangChain RAG chain (LCEL)
  agent_loop.py hand-written agent
  agent_graph.py LangGraph agent
  crag_graph.py Corrective RAG graph
  evaluate.py   LLM-as-judge evaluation
```

## Setup

1. Install [LM Studio](https://lmstudio.ai), load `qwen/qwen3.8-27b` and `text-embedding-nomic-embed-text-v1.5`, and start the server on `http://localhost:1234`.
2. Install the package (`dev` adds pytest, `ui` adds Gradio):
   ```
   pip install -e ".[dev,ui]"
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
   Or run it in Docker (needs [Docker Desktop](https://www.docker.com/products/docker-desktop/) and the index from step 3; LM Studio stays on the host):
   ```
   docker compose up --build
   ```
   Then open http://localhost:8000/docs; stop it with Ctrl+C.
6. Optional, tracing: start Langfuse with its own [Docker Compose setup](https://langfuse.com/self-hosting/docker-compose) (outside this repo), open http://localhost:3000, create a project and an API key pair, then copy `.env.example` to `.env` and fill in the keys. Start the API with the keys loaded:
   ```
   python -m uvicorn football_rag.api.main:app --reload --env-file .env
   ```
   `docker compose up` picks up `.env` by itself. Every question then appears as a trace in Langfuse.
7. Chat in the browser: with the API running (step 5), start the UI in a second terminal; it opens http://127.0.0.1:7860 by itself:
   ```
   python -m football_rag.gradio_ui
   ```
   Or start the API and the UI together on Windows, from a terminal with the environment active (each opens in its own window):
   ```
   run
   ```
8. Run the evaluation:
   ```
   python -m football_rag.evaluate
   ```
9. Run the tests (no LM Studio needed; the API tests replace the agent with a fake):
   ```
   python -m pytest
   ```
   GitHub Actions runs the same tests on every push and pull request (Ubuntu, Python 3.11). They need neither LM Studio nor a built index: the API tests use a fake agent, and nothing calls the model or searches the store at import time.

The corpus is committed, so step 3 works offline. `python scripts/fetch_corpus.py` re-downloads it from Wikipedia, but it overwrites the committed articles and the current text may no longer match the eval phrases.

## Known limitations

- Tables are dropped at ingestion, so e.g. the Premier League champions per season are missing.
- The year metadata is a heuristic: a 2022 chunk that mentions 2026 is tagged 2026.
- The model sometimes leaks its own knowledge into search queries, and into answers: asked for the top World Cup scorer, it answered Klose with 16 goals (true before 2026) although the retrieved text says Mbappé with 22, and the output guardrail blocked it. `SYSTEM` now says the tool results win when they disagree with what the model knows ("records and results change"). The agent then answers Mbappé with 22, and the evals are unchanged (loop and LangGraph still 100% tuning, 91% held-out), but it still adds Klose's old total of 16 as a side detail, which the guardrail rejects. With the retry, the trace shows the full loop: rejected for "the specific number 16", sent back with that reason, rewritten without it ("Mbappé… 22 goals… surpassed Messi and Klose"), accepted. The retry cost about 3.3 s (one model call, one guardrail check) on that question. On the full eval it changed nothing on tuning (100%, 5.9 s, apparently never triggered) and kept held-out at 91% (7.0 s vs 6.4 s, about one retry's worth of extra tokens).
- The held-out set has no Wikipedia-only, unanswerable or out-of-scope questions yet.
- The guardrails are only wired into the LangGraph agent (the one the API serves), and neither has a held-out set yet.
- The output guardrail only skips its check for the exact refusal texts; when the model refuses in its own words, the check still runs (a wasted call, not a wrong answer).
- No give-up rule: in rare cases, when no search returns the answer, the agent keeps searching until the 12-step limit (about 14 s and 17k input tokens) before returning `I couldn't find that.` Measured on all 59 eval questions, none hit the limit: the unanswerable ones gave up on their own in about 10–11 s, and the slowest answered ones were Wikipedia lookups. So no rule was added; it would save little and could cut off answers that need a later search.
- Coverage is much narrower than the scope the input guardrail allows: the corpus covers the World Cup, the 2022 World Cup, Euro 2024, the Premier League and offside; everything else depends on Wikipedia intros.
- Conversation memory lives in the API process (`MemorySaver`): it is lost on restart, grows without limit, and wouldn't be shared between several API workers. A database checkpointer (SQLite or Postgres) would fix all three.
- The whole conversation is re-sent to the model every turn, so input tokens grow with the length of the chat; there is no trimming or summarising.
- The input guardrail sees only the previous question, not the whole conversation, and the follow-up set is small (8 two-turn conversations, no longer chats or topic switches).
- Oyarzabal's winner in the Euro 2024 final isn't in the corpus (it is in a dropped table), so "Who scored their winning goal in the final?" gets `I couldn't find that.`
- The output guardrail misses subtly reversed rules: it infers its way to "supported" even when the context contradicts the answer (1 known miss in its set; prompt changes didn't fix it).

## License

MIT
