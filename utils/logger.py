"""结构化日志 + 请求追踪 + 耗时统计"""

import logging
import time
import json
import os
from contextvars import ContextVar

# ── trace_id：用 ContextVar 实现请求级透传，不污染函数签名 ──
_trace_id: ContextVar[str] = ContextVar("trace_id", default="")

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# ── 根 logger 配置 ──
_logger = logging.getLogger("mediguard")
_logger.setLevel(logging.DEBUG)


class TraceFormatter(logging.Formatter):
    """自定义 Formatter：在格式化前注入 trace_id，比 Filter 更可靠"""
    def format(self, record):
        if not hasattr(record, 'trace') or not record.trace:
            record.trace = _trace_id.get() or "-"
        return super().format(record)


_formatter = TraceFormatter(
    "%(asctime)s [%(levelname)-5s] [%(trace)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# 控制台输出
_console = logging.StreamHandler()
_console.setLevel(logging.INFO)
_console.setFormatter(_formatter)
_logger.addHandler(_console)

# 文件输出（持久化）
_file = logging.FileHandler(os.path.join(LOG_DIR, "app.log"), encoding="utf-8")
_file.setLevel(logging.DEBUG)
_file.setFormatter(_formatter)
_logger.addHandler(_file)


def set_trace_id(trace_id: str):
    _trace_id.set(trace_id)


def get_logger(name: str) -> logging.Logger:
    """获取子 logger，用法: logger = get_logger(__name__)"""
    return _logger.getChild(name)


# ── 耗时统计工具 ──

class Timer:
    """上下文管理器，自动记录耗时。

    用法:
        with Timer("RAG检索", logger=logger, extra={"top_k": 3}):
            result = retrieve_context(query)
        # 退出时自动输出: RAG检索 完成, elapsed=0.32s
    """
    def __init__(self, label: str, logger: logging.Logger = None, level: int = logging.DEBUG, **extra):
        self.label = label
        self.logger = logger or _logger
        self.level = level
        self.extra = extra

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        extras = f", {json.dumps(self.extra, ensure_ascii=False)}" if self.extra else ""
        self.logger.log(self.level, f"{self.label} 完成, elapsed={elapsed:.2f}s{extras}")


def log_llm_call(model: str, elapsed: float, prompt_len: int = 0, response_len: int = 0):
    """记录 LLM 调用耗时，方便定位慢查询"""
    _logger.info(
        f"LLM调用 model={model}, elapsed={elapsed:.2f}s, "
        f"prompt_len={prompt_len}, response_len={response_len}"
    )
