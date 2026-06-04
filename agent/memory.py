"""记忆管理：短期记忆(JSON) + 长期记忆(向量库)"""
import json
import os
from datetime import datetime
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rag.embeddings import DashScopeEmbeddings
from utils.logger import get_logger

logger = get_logger(__name__)

MEMORY_DIR = "./agent/memory"
SHORT_TERM_PATH = os.path.join(MEMORY_DIR, "recent_sessions.json")
LONG_TERM_DIR = "./rag/chroma_db_memory"


def save_session_summary(user_input, final_response, department, urgency):
    """对话结束时提取摘要，存入短期+长期记忆"""
    # 短期记忆：完整记录存JSON
    short_term = _load_short_term()
    short_term.append({
        "time": datetime.now().isoformat(),
        "user_input": user_input,
        "response": final_response,
        "department": department,
        "urgency": urgency,
    })
    # 只保留最近10次
    short_term = short_term[-10:]
    os.makedirs(MEMORY_DIR, exist_ok=True)
    with open(SHORT_TERM_PATH, "w", encoding="utf-8") as f:
        json.dump(short_term, f, ensure_ascii=False, indent=2)

    # 长期记忆：用LLM提取摘要，存向量库
    try:
        summary = _extract_summary(user_input, final_response)
        _save_to_vectorstore(summary)
    except Exception as e:
        logger.warning(f"长期记忆存储失败 error={e}")


def _extract_summary(user_input, response):
    """用LLM提取对话摘要"""
    from langchain_openai import ChatOpenAI
    from config import DASHSCOPE_API_KEY, LLM_BASE_URL, TRIAGE_MODEL
    llm = ChatOpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url=LLM_BASE_URL,
        model=TRIAGE_MODEL,  # 摘要提取用轻量模型
        temperature=0.1,
    )
    prompt = f"""请用一句话总结这次医疗咨询的关键信息（症状、科室、建议要点）：
用户问：{user_input}
AI答：{response}
摘要："""
    return llm.invoke(prompt).content


def _save_to_vectorstore(summary):
    """存入长期记忆向量库"""
    embeddings = DashScopeEmbeddings()
    vectorstore = Chroma(
        persist_directory=LONG_TERM_DIR,
        embedding_function=embeddings,
        collection_name="session_memory",
    )
    vectorstore.add_documents([Document(page_content=summary)])


def recall_memory(query, top_k=3):
    """检索相关历史记忆（长期记忆）"""
    embeddings = DashScopeEmbeddings()
    try:
        vectorstore = Chroma(
            persist_directory=LONG_TERM_DIR,
            embedding_function=embeddings,
            collection_name="session_memory",
        )
        docs = vectorstore.similarity_search(query, k=top_k)
        return [doc.page_content for doc in docs]
    except Exception:
        return []


def get_recent_sessions(n=3):
    """获取最近n次对话记录（短期记忆）"""
    short_term = _load_short_term()
    return short_term[-n:]


def _load_short_term():
    os.makedirs(MEMORY_DIR, exist_ok=True)
    if os.path.exists(SHORT_TERM_PATH):
        try:
            with open(SHORT_TERM_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []