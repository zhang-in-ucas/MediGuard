# -*- coding: utf-8 -*-
"""记忆管理：双层记忆架构

- 短期: chat_history in AgentState，全量注入 prompt
- 长期: ChromaDB 向量库，每轮对话提取一句摘要存入，语义检索召回
"""

from memory.long_term import recall_memory, save_summary

__all__ = ["save_summary", "recall_memory"]
