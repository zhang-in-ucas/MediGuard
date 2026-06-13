import uuid
from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.triage import triage_node
from agent.diagnosis import diagnosis_node
import agent.safety as safety_mod
from config import MAX_RETRIES
from utils.logger import get_logger, set_trace_id, Timer

logger = get_logger(__name__)


def _ensure_trace_id(state: dict) -> str:
    """确保 state 中有 trace_id，没有则生成一个"""
    tid = state.get("trace_id", "")
    if not tid:
        tid = uuid.uuid4().hex[:12]
        state["trace_id"] = tid
        set_trace_id(tid)
        logger.info(f"请求开始 user_input={state.get('user_input','')[:50]}...")
    else:
        set_trace_id(tid)
    return tid


# ── 节点 wrapper：自动记录进入/离开 + 耗时，同时确保 trace_id 在状态链中传递 ──

def _wrap_node(name: str, fn):
    """给节点函数包一层日志 + 耗时统计"""

    def wrapped(state: dict) -> dict:
        tid = _ensure_trace_id(state)
        with Timer(f"[{name}]", logger=logger, level=20):  # INFO level
            result = fn(state)

        # 关键：把 trace_id 放进返回值，确保 LangGraph 状态链中不丢失
        result["trace_id"] = tid

        # 节点级关键信息日志
        if name == "triage":
            logger.info(
                f"分诊结果 department={result.get('department','?')}, "
                f"urgency={result.get('urgency','?')}, "
                f"should_see_doctor={result.get('should_see_doctor','?')}"
            )
        elif name == "diagnosis":
            retry = state.get("retry_count", 0)
            prefix = f"重试#{retry} " if retry > 0 else ""
            response_len = len(result.get("diagnosis_result", ""))
            rag_hit = bool(result.get("rag_context", "").strip())
            logger.info(f"{prefix}问诊完成 response_len={response_len}, rag_hit={rag_hit}")
        elif name == "safety":
            is_safe = result.get("is_safe", True)
            reason = result.get("safety_reason", "")
            retry = result.get("retry_count", 0)
            if is_safe:
                logger.info("安全审查通过")
            else:
                logger.warning(f"安全审查拦截 retry={retry} reason={reason[:80]}")

        return result

    return wrapped


def should_retry(state: dict) -> str:
    """安全审查不通过且未超重试次数，回退问诊；否则结束"""
    _ensure_trace_id(state)
    if not state.get("is_safe", True) and state.get("retry_count", 0) < MAX_RETRIES:
        logger.info(f"触发重试 retry_count={state['retry_count']}/{MAX_RETRIES}")
        return "diagnosis"
    return "finish"


def format_output(state: dict) -> dict:
    """格式化最终输出"""
    _ensure_trace_id(state)

    if not state.get("is_safe", True) and state.get("retry_count", 0) >= MAX_RETRIES:
        final = "⚠️ 抱歉，多次尝试后仍无法给出安全的建议，请及时前往医院就诊。"
        logger.warning("已达最大重试次数，返回兜底话术")
    elif not state.get("is_safe", True):
        final = "⚠️ 回答中包含不安全内容，已拦截，请咨询专业医生。"
    else:
        final = state.get("diagnosis_result", "")

    # 对话结束，保存记忆
    try:
        from memory import save_session_summary
        save_session_summary(
            user_input=state.get("user_input", ""),
            final_response=final,
            department=state.get("department", ""),
            urgency=state.get("urgency", ""),
        )
    except Exception as e:
        logger.warning(f"记忆保存失败 error={e}")

    logger.info(
        f"请求完成 final_safe={state.get('is_safe', True)}, "
        f"total_retries={state.get('retry_count', 0)}, "
        f"department={state.get('department','?')}"
    )
    return {"final_response": final}


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("triage", _wrap_node("triage", triage_node))
    graph.add_node("diagnosis", _wrap_node("diagnosis", diagnosis_node))
    graph.add_node("safety", _wrap_node("safety", safety_mod.safety_node))
    graph.add_node("output", format_output)

    graph.set_entry_point("triage")
    graph.add_edge("triage", "diagnosis")
    graph.add_edge("diagnosis", "safety")
    graph.add_conditional_edges(
        "safety",
        should_retry,
        {"diagnosis": "diagnosis", "finish": "output"},
    )
    graph.add_edge("output", END)

    return graph.compile()
