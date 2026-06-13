# -*- coding: utf-8 -*-
"""中期记忆：JSON 文件存储最近 N 次完整对话记录"""
import json
import os
from utils.logger import get_logger

logger = get_logger(__name__)

MEMORY_DIR = "./memory/data"
SHORT_TERM_PATH = os.path.join(MEMORY_DIR, "recent_sessions.json")
MAX_SESSIONS = 10


def _save_short_term(user_input, final_response, department, urgency):
    """追加一条记录到短期记忆，保留最近 MAX_SESSIONS 条"""
    from datetime import datetime

    records = _load_short_term()
    records.append({
        "time": datetime.now().isoformat(),
        "user_input": user_input,
        "response": final_response,
        "department": department,
        "urgency": urgency,
    })
    records = records[-MAX_SESSIONS:]

    os.makedirs(MEMORY_DIR, exist_ok=True)
    with open(SHORT_TERM_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def get_recent_sessions(n=3):
    """获取最近 n 次对话记录"""
    records = _load_short_term()
    return records[-n:]


def _load_short_term():
    """从 JSON 文件加载短期记忆"""
    os.makedirs(MEMORY_DIR, exist_ok=True)
    if os.path.exists(SHORT_TERM_PATH):
        try:
            with open(SHORT_TERM_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []
