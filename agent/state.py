from typing import TypedDict, List

class AgentState(TypedDict):
    user_input: str
    department: str
    urgency: str
    should_see_doctor: bool
    diagnosis_result: str
    rag_context: str
    is_safe: bool
    safety_reason: str
    safety_history: List[str]  # 累积所有安全审查的拦截原因（用于评测拆解防线）
    retry_count: int
    final_response: str
    chat_history: List[str]
    memory_context: str  # 记忆检索结果（长期+短期记忆拼接）
    trace_id: str        # 请求唯一标识，用于日志串联