# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Identity

MediGuard is a Chinese medical health consultation AI system with a **multi-agent safety pipeline**. It accepts user symptom descriptions, performs triage + diagnosis via LLM + RAG, and enforces legal compliance through a two-layer safety review with automatic retry. Built with LangGraph + DashScope Qwen models + ChromaDB.

The system is governed by Chinese medical regulations: 互联网诊疗监管细则 (Internet Medical Supervision Rules), 医师法 (Physicians Law), 药品管理法 (Drug Administration Law), and 医疗纠纷预防和处理条例 (Medical Dispute Prevention & Resolution Regulations). It must never provide prescriptions, definitive diagnoses, or discourage users from seeing a doctor.

## Quick Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Build the RAG knowledge base (first-time setup)
python -m rag.ingest          # shibing624/medical 10,000条 + Chinese-medical-dialogue-data 50,000条

# Run CLI test (single query hardcoded in main.py)
python main.py

# Start FastAPI server (REST API on port 8000)
python -m api.server

# Start Gradio Web UI (chat interface on port 7860)
python -m api.web_ui

# Run safety evaluation harness
python -m evaluation.safety_harness --verbose
python -m evaluation.safety_harness --category 剂量           # Single category
python -m evaluation.safety_harness --limit 5                  # First 5 cases only
python -m evaluation.safety_harness --review                   # With interactive human review
```

## Architecture: Agent Pipeline (LangGraph StateGraph)

```
triage ──→ diagnosis ──→ safety ──[safe]──→ output ──→ END
                            │
                            └──[unsafe, retries < MAX_RETRIES]──→ diagnosis (retry)
                            └──[unsafe, retries >= MAX_RETRIES]──→ output (fallback)
```

Defined in `agent/graph.py:107-125` via `build_graph()` which returns a compiled StateGraph.

### AgentState (`agent/state.py`)

A `TypedDict` with 16 fields:
- `user_input` — raw user query
- `department` — routed specialty (e.g., 神经内科, 全科)
- `urgency` — high/medium/low
- `should_see_doctor` — bool
- `diagnosis_result` — LLM-generated response text
- `rag_context` — retrieved RAG knowledge merged into prompt
- `is_safe` — bool, final safety verdict
- `safety_reason` — human-readable violation reason (used as feedback for retry)
- `safety_history` — `List[str]` accumulating ALL safety rejection reasons across retries (used by evaluation harness to deconstruct which defense layer caught the violation)
- `retry_count` — integer, incremented by safety node on rejection
- `final_response` — what the user actually sees
- `chat_history` — `List[str]` supplied by API/Gradio callers, alternating [user, assistant, ...]
- `memory_context` — 长期记忆检索结果（ChromaDB 语义召回）
- `trace_id` — 12-char hex UUID, set once per request, propagated through all nodes

### Node Wrapper (`agent/graph.py:28-63`)

Every node function is wrapped via `_wrap_node(name, fn)`:
- Generates trace_id if missing (`uuid.uuid4().hex[:12]`)
- Wraps execution in `Timer` context manager for elapsed-time logging
- Injects `trace_id` into return dict to ensure propagation through LangGraph state chain
- Logs node-specific key info (department/urgency for triage, response_len/rag_hit for diagnosis, safety verdict for safety)
- Do NOT call node functions directly; the graph handles all context passing

### Retry Mechanism (`agent/graph.py:66-72`)

`should_retry()` is the conditional edge function:
- Routes back to `diagnosis` when `is_safe == False` AND `retry_count < MAX_RETRIES` (default 3)
- Routes to `finish` (output node) otherwise
- Safety reason is injected into diagnosis prompt as `{safety_feedback}` on retry
- RAG results are reused from the first attempt on retries (`diagnosis.py:103-106`)

### Output Node (`agent/graph.py:75-105`)

`format_output()`:
- If max retries exhausted → returns hardcoded fallback: "⚠️ 抱歉，多次尝试后仍无法给出安全的建议，请及时前往医院就诊。"
- If single rejection → generic rejection message
- If safe → returns `diagnosis_result` directly
- On every completion, calls `save_summary()` from `memory/` to extract a one-sentence summary and append to ChromaDB. Failures are logged as warnings (never crash the pipeline).

## Two-Layer Safety System

Safety is the most critical subsystem. Defined in `agent/safety.py`.

### Layer 1 — Keyword Block (`SAFETY_KEYWORDS` in `config.py:24-39`)

35 trigger phrases organized in 4 groups:
- **Prescription phrases**: 开处方, 推荐药, 推荐吃药, 推荐吃, 制定治疗方案, 治疗方案如下
- **Diagnosis replacement**: 确诊为, 诊断为您, 诊断为你, 您患有, 你患有, 您得了
- **Dosage phrases**: 建议服用, 可以服用, 推荐服用, 可以考虑服用, 每次服用, 每日服用, 一次服用, mg每日, mg每天, mg一次, mg每次, 一天3次, 一天2次, 一日3次, 一日2次
- **Discourage medical visits**: 不用就医, 不需要就医, 不必去医院, 无需就医, 不用听医生, 自己能好, 自己能调整, 自己能恢复

Only checks the AI output (`diagnosis_result`), NOT user input. If any keyword matches → immediate `is_safe=False` with reason "规则拦截：包含越界关键词「{keyword}」". Keyword interception is appended to `safety_history` and `retry_count` is incremented.

**IMPORTANT**: Changes to `SAFETY_KEYWORDS` directly affect false-positive/false-negative balance. Always run `python -m evaluation.safety_harness` after modifying these or any safety/diagnosis prompts.

### Layer 2 — LLM Safety Review (`SAFETY_PROMPT` in `safety.py:18-147`)

A separate `qwen-plus` model with temperature=0.1 judges the output against a detailed legal framework. The prompt defines:

- **Class 1 violations (一票否决 — one-vote veto)**: Prescription acts (1A), diagnostic substitution (1B), treatment plans (1C), discouraging hospital visits (1D), first-visit overreach (1E). Any single hit → immediate unsafe.
- **Class 2 risks (观察项 — observation items)**: Edge dosage hints (2A), information authenticity issues (2B), over-promising (2C). 2+ cumulative or high prominence → unsafe.
- **Boundary case examples** (Cases 1-8) with explicit safe/unsafe rulings to guide the LLM.
- Output format: strict JSON `{"is_safe": true/false, "reason": "..."}`.

On LLM call failure → conservative fallback: `is_safe=False`, reason="安全审查服务异常，为保障安全已拦截".

## Multi-Model Strategy

All LLM calls go through DashScope's OpenAI-compatible API (`https://dashscope.aliyuncs.com/compatible-mode/v1`) using `langchain-openai`'s `ChatOpenAI`.

| Stage | Model | Temperature | Rationale |
|-------|-------|------------|-----------|
| Triage | `qwen-turbo` | 0.3 | Fast, cheap classification |
| Diagnosis | `qwen-plus` | 0.3 | Main generation quality |
| Safety | `qwen-plus` | 0.1 | High-quality judgment, low randomness |
| Embedding | `text-embedding-v3` | — | DashScope native |
| Rerank | `qwen3-rerank` | — | DashScope native |
| Memory summary | `qwen-turbo` | 0.1 | Cost-efficient summarization |

Configurable via env vars: `TRIAGE_MODEL`, `DIAGNOSIS_MODEL`, `SAFETY_MODEL`, `EMBEDDING_MODEL`. All default to sensible values in `config.py`.

## RAG: Three-Stage Retrieval Pipeline

Defined in `rag/retriever.py`.

### Stage 1 — Vector Search
- ChromaDB via `langchain_chroma.Chroma`
- `DashScopeEmbeddings` wrapper (`rag/embeddings.py`) calling `dashscope.TextEmbedding`
- Batch embedding: 10 texts per API call
- Returns top-20 with scores

### Stage 2 — BM25 Keyword Search
- jieba tokenization + `rank_bm25.BM25Okapi`
- Corpus loaded from ChromaDB (paginated, 500 docs/batch to avoid SQLite variable limit)
- Results cached to both:
  - **Pickle file**: `rag/chroma_db/bm25_cache.pkl`, auto-rebuilds when ChromaDB doc count changes
  - **Memory**: `_bm25_index_cache` global variable, survives within process lifetime
- Returns top-20 with scores

### Stage 3 — RRF Fusion + Rerank
- Reciprocal Rank Fusion (k=60) merges vector + BM25 rankings
- DashScope `TextReRank` with `qwen3-rerank` model re-ranks to final top-K (default 3)
- On rerank failure → graceful fallback to top-K by RRF score

### Knowledge Base — 双数据源

采用**通用医普 + 真实医患对话**的双数据源架构：

| 数据源 | 规模 | 格式 | 定位 |
|---|---|---|---|
| **shibing624/medical** | 10,000 条 | Q&A 对（instruction + output） | 通用医学知识科普覆盖 |
| **Chinese-medical-dialogue-data** | 50,000 条 | 医患对话（按科室分类） | 真实医患对话场景，多科室覆盖 |

**shibing624/medical** (`rag/ingest.py:load_medical_qa_data`): HuggingFace 数据集，通过 hf-mirror 下载 JSON 格式的医疗 QA 训练数据。默认取 10,000 条，长度过滤 >=30 字符。

**Chinese-medical-dialogue-data** (`rag/ingest.py:load_chinese_medical_dialogue`): GitHub 开源数据集 (Toyhom/Chinese-medical-dialogue-data)，79 万条真实医患对话，覆盖男科、内科、妇产科、肿瘤科、儿科、外科 6 个科室。使用前需先 `git clone` 到本地。默认取 50,000 条。

向量库: ChromaDB `collection_name="medical_knowledge"`，两个来源共用。默认总规模 ~60,000 条。入库支持断点续传（中断后重跑自动跳过已入库文档）。

```bash
# 构建知识库（自动重建，清旧库）
python -m rag.ingest

# 如需使用对话数据集，先克隆到项目目录
git clone https://github.com/Toyhom/Chinese-medical-dialogue-data.git
```

### BM25 Cache & Prebuild
- 缓存文件: `rag/chroma_db/bm25_cache.pkl`，文档数变化时自动重建
- `rag/ingest.py` 入库后自动调用 `prebuild_bm25_index(force=True)` 预构建缓存
- 手动强制重建: 删除 `bm25_cache.pkl` 或调用 `from rag.retriever import prebuild_bm25_index; prebuild_bm25_index(force=True)`
- 构建过程有进度日志，约 60,000 条数据需 ~1-2 分钟

## Memory: 双层记忆 (`memory/` package)

### 短期: `chat_history` in AgentState
- 由 API/Gradio 调用方传入，全量注入 prompt 作为 `{conversation_history}`

### 长期: `memory/long_term.py`
- ChromaDB vector store at `memory/data/chroma_db/` (gitignored)，collection: `session_memory`
- 每轮对话结束时，`save_summary()` 用 qwen-turbo (temp=0.1) 提取一句摘要，追加到向量库
- 嵌入: DashScope text-embedding-v3
- `recall_memory(query, top_k=3)` 语义搜索召回相关历史摘要

### Entry Point: `memory/__init__.py`
- `save_summary(user_input, response)` — 提取摘要存入 ChromaDB
- `recall_memory(query, top_k)` — 语义检索历史记忆

## Web Interfaces

Two entry points share the same graph but are independent:

### FastAPI (`api/server.py`)
- `POST /query` with `QueryRequest { user_input, chat_history }`
- Returns `QueryResponse` with all state fields
- Lazy-loads graph on first request (`get_graph()` singleton)
- CORS enabled, all origins allowed

### Gradio (`api/web_ui.py`)
- Chatbot interface at `http://localhost:7860`
- Builds a fresh graph per message (stateless)
- Chat history passed as alternating `[user, assistant, user, ...]` list
- 8 example queries pre-loaded
- Displays department, urgency badge, safety status in response

## Configuration (`config.py`)

All settings loaded from `.env` via `python-dotenv` with `override=True`:
- `DASHSCOPE_API_KEY` — **required**, Alibaba Cloud API key
- `LLM_BASE_URL` — defaults to DashScope compatible-mode endpoint
- `LLM_MODEL` — default `qwen-plus`
- `TRIAGE_MODEL` — defaults to `qwen-turbo`
- `DIAGNOSIS_MODEL` — defaults to `qwen-plus`
- `SAFETY_MODEL` — defaults to `qwen-plus`
- `EMBEDDING_MODEL` — defaults to `text-embedding-v3`
- `RAG_PERSIST_DIR` — defaults to `./rag/chroma_db`
- `RAG_TOP_K` — 3
- `MAX_RETRIES` — 3
- `TEMPERATURE` — 0.3

## Logging (`utils/logger.py`)

Structured logging with request-level tracing:
- `get_logger(__name__)` returns a child logger under `mediguard`
- `set_trace_id(tid)` sets a `ContextVar` propagated through all log messages for one request
- `Timer` context manager logs elapsed time at exit (INFO level when used in node wrappers)
- `log_llm_call()` records model name, elapsed time, prompt length, and response length
- Logs go to: console (INFO level) + `logs/app.log` (DEBUG level)
- Custom `TraceFormatter` injects `trace_id` into every log record

## Evaluation (`evaluation/safety_harness.py`)

Dedicated safety evaluation framework:
- Test cases from `evaluation/test_cases_v2.json` (20 labeled cases, 7 categories)
- Runs each case through the full agent pipeline
- `categorize()` function classifies outcomes into 5 categories:
  - "安全通过" — passed safety without any retry
  - "关键词拦截" — caught by Layer 1 keyword block
  - "LLM审查拦截(重试1/2)" — caught by Layer 2 LLM review, repaired by retry
  - "LLM审查拦截(兜底)" — retries exhausted, returned fallback message
- `compute_metrics()` aggregates by label and by category
- `print_report()` outputs detailed breakdown with visual bars
- Results saved to `evaluation/results/eval_{timestamp}.json` + `evaluation/results/history.jsonl`
- `--review` flag enables interactive human-vs-system consistency analysis with precision/recall/F1

## File-by-File Reference

```
MediGuard/
├── main.py                         # CLI entry point: builds graph, runs one hardcoded query
├── config.py                       # ALL global settings, SAFETY_KEYWORDS, model configs
├── requirements.txt                # Dependencies (langgraph, langchain, chromadb, dashscope, gradio, fastapi, etc.)
├── .env.example                    # Template for .env (DASHSCOPE_API_KEY required)
├── .gitignore                      # Ignores chroma_db, logs, eval artifacts, .env, datasets
│
├── agent/                          # Core agent pipeline
│   ├── state.py                    # AgentState TypedDict (16 fields)
│   ├── graph.py                    # LangGraph StateGraph: triage→diagnosis→safety→output, retry loop, node wrapper, format_output
│   ├── triage.py                   # Triage node: department routing, urgency assessment (qwen-turbo)
│   ├── diagnosis.py                # Diagnosis node: RAG retrieval, memory recall, compliance prompt (~100 lines of legal constraints)
│   └── safety.py                   # Two-layer safety: keyword scan + LLM legal review (qwen-plus, temp=0.1)
│
├── memory/                         # 双层记忆系统
│   ├── __init__.py                 # Public API: save_summary(), recall_memory()
│   └── long_term.py                # 长期记忆：ChromaDB向量库存储每轮对话摘要，语义检索召回
│
├── rag/                            # RAG retrieval engine
│   ├── embeddings.py               # DashScopeEmbeddings: LangChain adapter for TextEmbedding API, batch=10
│   ├── ingest.py                   # Knowledge base builder: shibing624/medical (10K) + Chinese-medical-dialogue (50K)
│   └── retriever.py                # Hybrid search: Vector(top-20) + BM25(jieba+rank_bm25, top-20) → RRF(k=60) → Rerank(qwen3-rerank, top-3)
│
├── api/                            # Web interfaces (both share same graph)
│   ├── server.py                   # FastAPI: POST /query, GET /health, lazy graph loading, CORS enabled
│   └── web_ui.py                   # Gradio: chatbot interface, 8 example queries, per-message graph rebuild
│
├── utils/                          # Shared utilities
│   ├── __init__.py                 # Empty
│   └── logger.py                   # Structured logging: TraceFormatter, Timer, log_llm_call(), trace_id via ContextVar
│
├── evaluation/                     # Safety evaluation
│   ├── safety_harness.py           # Full evaluation framework: categorize→metrics→report→save→human review
│   └── test_cases_v2.json          # 20 test cases, 7 categories (剂量, 处方, 诊断, 疗程, 就医劝阻, 用药途径, 联合用药)
│
└── logs/                           # Application logs (gitignored)
    └── app.log                     # DEBUG-level file log
```

## Important Constraints

- **Never hardcode API keys** in source files. Keys go in `.env` (gitignored), template is `.env.example`.
- **Safety keywords in `config.py`** are tuned for Chinese medical compliance. Run `python -m evaluation.safety_harness` after any modification to `SAFETY_KEYWORDS` or safety/diagnosis prompts.
- **The diagnosis prompt** in `diagnosis.py:19-96` explicitly encodes Chinese medical law constraints into model behavior. It defines 5 "red lines" (处方, 诊断替代, 首诊, 就医劝阻, 信息真实性) and provides alternative compliant phrasings. Changes must be verified against the evaluation harness.
- **BM25 cache**: delete `rag/chroma_db/bm25_cache.pkl` after modifying `rag/ingest.py` and rebuilding the vector store.
- **Windows-specific**: BM25 uses `pickle` for caching — avoid cross-OS cache reuse.
- **The memory/ package** (`memory/__init__.py`, `long_term.py`) uses per-round summarization: each completed conversation round is extracted to a one-sentence summary and stored in ChromaDB for semantic recall.
- **ChromaDB path normalization**: `rag/retriever.py:23` converts `RAG_PERSIST_DIR` to absolute path to prevent `chromadb.PersistentClient` and `langchain_chroma.Chroma` from connecting to different databases when cwd differs.
- **Gradio rebuilds graph per message** (`web_ui.py:15`): intentional stateless design. FastAPI lazy-loads a singleton graph (`server.py:24-29`).
