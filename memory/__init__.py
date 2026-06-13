# -*- coding: utf-8 -*-
"""记忆管理包：短期记忆(JSON) + 长期记忆(ChromaDB)

三层记忆架构：
- 短期 (Session): chat_history in AgentState，单次对话上下文
- 中期 (Recent):  JSON 文件存储最近 N 次完整对话记录
- 长期 (Semantic): ChromaDB 向量库存储对话摘要，语义检索召回
"""

from memory.short_term import get_recent_sessions
from memory.long_term import recall_memory

__all__ = ["save_session_summary", "get_recent_sessions", "recall_memory"]


def save_session_summary(user_input, final_response, department, urgency):
    """对话结束时提取摘要，存入中期 + 长期记忆"""
    from memory.long_term import _extract_summary, _save_to_vectorstore
    from memory.short_term import _save_short_term

    # 中期记忆：完整记录存 JSON
    _save_short_term(user_input, final_response, department, urgency)

    # 长期记忆：LLM 提取摘要，存向量库
    try:
        summary = _extract_summary(user_input, final_response)
        _save_to_vectorstore(summary)
    except Exception as e:
        import logging
        logging.getLogger("mediguard.memory").warning(f"长期记忆存储失败 error={e}")
