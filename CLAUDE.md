# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

MediGuard is a Chinese medical health consultation AI system with a **multi-agent safety pipeline**. It accepts user symptom descriptions, performs triage + diagnosis via LLM + RAG, and enforces legal compliance through a two-layer safety review with automatic retry. The system is governed by Chinese medical regulations (互联网诊疗监管细则, 医师法, 药品管理法) and must never provide prescriptions, definitive diagnoses, or discourage users from seeing a doctor.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Build the RAG knowledge base (first-time setup, runs ~10min)
python -m rag.ingest

# Run CLI test (single query, hardcoded in main.py)
python main.py

# Start FastAPI server (REST API on port 8000)
python -m api.server

# Start Gradio Web UI (chat interface on port 7860)
python -m api.web_ui

# Run safety evaluation harness
python -m evaluation.safety_harness --verbose
python -m evaluation.safety_harness --category 剂量           # Single category
python -m evaluation.safety_harness --limit 5                  # First 5 cases only

# Run a single RAG query for testing
python -m rag.retriever
```

## Architecture: Agent Pipeline (LangGraph)

The core is a **LangGraph StateGraph** in `agent/graph.py` with this flow:

```
triage → diagnosis → safety ──[safe]──→ output → END
                        │
                        └──[unsafe, retries < MAX_RETRIES]──→ diagnosis (retry)
                        └──[unsafe, retries >= MAX_RETRIES]──→ output (fallback message)
```

- **State** (`agent/state.py`): `AgentState` TypedDict — fields flow through all nodes including `user_input`, `department`, `urgency`, `diagnosis_result`, `rag_context`, `is_safe`, `safety_reason`, `retry_count`, `chat_history`, `memory_context`, `trace_id`.
- **Node wrapper** (`_wrap_node` in `graph.py`): Every node is wrapped with logging + `Timer` + trace_id propagation. Do not call node functions directly; the graph handles all context passing.
- **Retry mechanism**: When safety rejects a response, the graph routes back to `diagnosis_node` with `safety_reason` injected into the prompt as feedback. This loops up to `MAX_RETRIES` (default 3). On retry, RAG results are reused from the previous attempt.

## Two-Layer Safety System

Safety enforcement is the most critical subsystem (`agent/safety.py`):

1. **Layer 1 — Keyword block** (`SAFETY_KEYWORDS` in `config.py`): If the AI output contains any trigger phrase (e.g., "建议服用", "每天2", "mg每日"), it's immediately rejected without LLM call. Fast and deterministic.
2. **Layer 2 — LLM safety review** (`SAFETY_PROMPT` in `safety.py`): A separate model judges the output against a detailed legal framework. This catches soft/semantic violations that keywords miss. Uses `qwen-plus` with temperature 0.1.

The safety prompt defines **two violation tiers**:
- **Class 1 (一票否决)**: Prescription acts, diagnostic substitution, treatment plans, discouraging hospital visits, first-visit overreach. Any single hit → unsafe.
- **Class 2 (观察项)**: Edge dosage hints, info authenticity issues, over-promising. 2+ cumulative hits or high prominence → unsafe.

When safety rejection happens, `safety_reason` is fed back to the diagnosis node as `safety_feedback` in the prompt template, asking the model to generate a safer alternative.

## RAG: Hybrid Search + Rerank

`rag/retriever.py` implements a three-stage retrieval pipeline:
1. **Vector search** (ChromaDB, DashScope `text-embedding-v3` embeddings) → top-20
2. **BM25 keyword search** (jieba tokenization + `rank_bm25`) → top-20
3. **RRF fusion** (Reciprocal Rank Fusion, k=60) → merge rankings
4. **Rerank** (DashScope `qwen3-rerank`) → final top-K (default 3)

BM25 index is cached both on disk (`bm25_cache.pkl`) and in memory (`_bm25_index_cache` global). The cache auto-rebuilds when ChromaDB document count changes.

Knowledge base (`rag/ingest.py`) has three data sources:
- `shibing624/medical` (auto-downloaded from HuggingFace mirror)
- `Chinese-medical-dialogue-data` (local, 79万 real doctor-patient dialogues in CSV/GB18030)
- Built-in safety knowledge (15 hardcoded Q&A pairs)

Documents are embedded with `DashScopeEmbeddings` (`rag/embeddings.py`) — a LangChain `Embeddings` adapter wrapping DashScope's `TextEmbedding` API with batch processing (10 texts per call).

## Memory: Short-Term + Long-Term

`agent/memory.py` implements dual memory:
- **Short-term**: JSON file (`agent/memory/recent_sessions.json`), last 10 sessions with full conversation details.
- **Long-term**: ChromaDB vector store (`rag/chroma_db_memory`). Each session is summarized by LLM (using triage model for cost efficiency), then the summary is embedded and stored for semantic retrieval.

Memory is recalled in `diagnosis_node` via `recall_memory()` (long-term, top-2) and `get_recent_sessions()` (short-term, last 3). The prompt explicitly warns: memory is for background reference only, must not be treated as current user statements.

## Multi-Model Strategy

Different pipeline stages use different models (configurable via env vars):
| Stage | Default Model | Purpose |
|-------|--------------|---------|
| Triage | `qwen-turbo` | Fast, cheap classification |
| Diagnosis | `qwen-plus` | Main generation quality |
| Safety | `qwen-plus` | High-quality judgment |
| Embedding | `text-embedding-v3` | Vector embeddings |
| Rerank | `qwen3-rerank` | Result reranking |
| Memory summary | `qwen-turbo` | Cost-efficient summarization |

All LLM calls go through DashScope's OpenAI-compatible API (`https://dashscope.aliyuncs.com/compatible-mode/v1`) using `langchain-openai`'s `ChatOpenAI`.

## Logging

`utils/logger.py` provides structured logging with request-level tracing:
- `get_logger(__name__)` returns a child logger under `mediguard`
- `set_trace_id(tid)` sets a `ContextVar` that flows through all log messages for one request
- `Timer` context manager logs elapsed time at exit
- `log_llm_call()` records model, elapsed time, and prompt/response lengths
- Logs go to both console (INFO) and `logs/app.log` (DEBUG)

## Web UIs

Two entry points exist — they share the same graph but are independent:
- **FastAPI** (`api/server.py`): `POST /query` with `QueryRequest { user_input, chat_history }`. Returns structured `QueryResponse` with all state fields. Lazy-loads graph on first request.
- **Gradio** (`api/web_ui.py`): Chatbot interface at `http://localhost:7860`. Builds a fresh graph per message (stateless). Chat history is passed as alternating `[user, assistant, user, ...]` list.

## Evaluation

`evaluation/safety_harness.py` is a dedicated safety evaluation harness. Test cases in `evaluation/test_cases.json` are labeled with `should_block` (true/false) and `category`. The harness classifies outcomes:
- **PASS**: Actual matches expected
- **FALSE_NEGATIVE** (leak): Should have blocked but didn't — further split into `prompt_prevented` (diagnosis model self-censored), `safety_caught` (retry fixed it), and `true_leaks` (both layers missed)
- **FALSE_POSITIVE** (false alarm): Should have passed but got blocked

Results are saved to `evaluation/results/` with per-run JSON + cumulative `eval_history.jsonl`.

## Key Configuration

All in `config.py`, loaded from `.env`:
- `DASHSCOPE_API_KEY` — required, Alibaba Cloud API key
- `LLM_MODEL` — default model (usually `qwen-plus`)
- `TRIAGE_MODEL`, `DIAGNOSIS_MODEL`, `SAFETY_MODEL` — per-stage model overrides
- `MAX_RETRIES` — safety retry limit (default 3)
- `SAFETY_KEYWORDS` — Layer-1 keyword blocklist (modify with care: false positives here cause user-facing rejections)
- `RAG_PERSIST_DIR` — ChromaDB storage path (default `./rag/chroma_db`)

## Important Constraints

- **Never hardcode API keys** in source files. Keys are in `.env` (gitignored), with `.env.example` as a template.
- **Safety keywords in `config.py`** are tuned for Chinese medical compliance. Changes here directly affect the false-positive/false-negative balance. Run `evaluation/safety_harness.py` after any modification to `SAFETY_KEYWORDS` or safety/diagnosis prompts.
- **The diagnosis prompt** (`DIAGNOSIS_PROMPT` in `diagnosis.py`) is ~100 lines of legal compliance instructions. It explicitly encodes Chinese medical law constraints into the model's behavior. Changes to this prompt must be verified against the evaluation harness.
- **BM25 cache invalidation**: If you modify `rag/ingest.py` and rebuild the vector store, delete `rag/chroma_db/bm25_cache.pkl` to force rebuild.
- **Windows-specific**: RAG CSV data uses GB18030 encoding. BM25 uses `pickle` for caching — avoid cross-OS cache reuse.
