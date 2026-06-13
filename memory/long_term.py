# -*- coding: utf-8 -*-
"""长期记忆：ChromaDB 向量库存储对话摘要，语义检索召回"""
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rag.embeddings import DashScopeEmbeddings
from utils.logger import get_logger

logger = get_logger(__name__)

LONG_TERM_DIR = "./memory/data/chroma_db"


def _extract_summary(user_input, response):
    """用 LLM 把对话压缩为一句摘要（用 triage 模型省成本）"""
    from langchain_openai import ChatOpenAI
    from config import DASHSCOPE_API_KEY, LLM_BASE_URL, TRIAGE_MODEL

    llm = ChatOpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url=LLM_BASE_URL,
        model=TRIAGE_MODEL,
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
    """检索相关历史记忆（语义相似度搜索）"""
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
