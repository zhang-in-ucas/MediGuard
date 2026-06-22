"""RAG检索器：Hybrid Search (向量+BM25) + Rerank重排序"""

import os
import sys
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jieba
import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from config import RAG_PERSIST_DIR, EMBEDDING_MODEL, RAG_TOP_K, DASHSCOPE_API_KEY
from rag.embeddings import DashScopeEmbeddings
from utils.logger import get_logger

# 转绝对路径，避免不同 cwd 连接到不同数据库
RAG_PERSIST_DIR = os.path.abspath(RAG_PERSIST_DIR)

logger = get_logger(__name__)

BM25_CACHE_PATH = os.path.join(RAG_PERSIST_DIR, "bm25_cache.pkl")

# BM25索引内存缓存（首次构建后常驻内存，避免每次检索重建）
_bm25_index_cache = None


def _get_embeddings():
    return DashScopeEmbeddings()


def _get_chroma_collection():
    """用PersistentClient获取collection，支持大数据量"""
    client = chromadb.PersistentClient(path=RAG_PERSIST_DIR)
    try:
        return client.get_collection(name="medical_knowledge")
    except Exception:
        raise RuntimeError(
            f"向量库不存在，请先运行: python -m rag.ingest\n"
            f"（路径: {RAG_PERSIST_DIR}）"
        )


def _vector_search(query: str, top_k: int = 20):
    """向量语义检索"""
    embeddings = _get_embeddings()
    vectorstore = Chroma(
        persist_directory=RAG_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name="medical_knowledge",
    )
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=top_k)
    return [(doc, score) for doc, score in docs_with_scores]


def _get_bm25_index():
    """获取BM25索引（首次构建后缓存到内存，后续直接用）"""
    global _bm25_index_cache
    if _bm25_index_cache is not None:
        return _bm25_index_cache

    # 构建索引
    all_docs = _load_bm25_corpus()
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        logger.warning("rank_bm25未安装，跳过BM25检索")
        return None

    tokenized_corpus = [list(jieba.cut(doc.page_content)) for doc in all_docs]
    bm25 = BM25Okapi(tokenized_corpus)

    _bm25_index_cache = {
        "bm25": bm25,
        "docs": all_docs,
    }
    logger.info(f"BM25索引已构建并缓存到内存（{len(all_docs)}条）")
    return _bm25_index_cache


def _bm25_search(query: str, top_k: int = 20):
    """BM25关键词检索（使用内存缓存的索引）"""
    cached = _get_bm25_index()
    if cached is None:
        return []

    bm25 = cached["bm25"]
    docs = cached["docs"]
    tokenized_query = list(jieba.cut(query))
    scores = bm25.get_scores(tokenized_query)
    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return scored_docs[:top_k]


def _rrf_merge(vector_results, bm25_results, k=60):
    """Reciprocal Rank Fusion：融合向量检索和BM25的排名"""
    doc_scores = {}
    doc_map = {}

    for rank, (doc, _) in enumerate(vector_results):
        doc_id = hash(doc.page_content)
        doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
        doc_map[doc_id] = doc

    for rank, (doc, _) in enumerate(bm25_results):
        doc_id = hash(doc.page_content)
        doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
        doc_map[doc_id] = doc

    sorted_ids = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_map[doc_id], score) for doc_id, score in sorted_ids]


def _rerank(query: str, documents: list, top_k: int = 3):
    """DashScope qwen3-rerank 重排序"""
    try:
        from dashscope import TextReRank
        doc_texts = [doc.page_content for doc in documents]
        resp = TextReRank.call(
            model="qwen3-rerank",
            query=query,
            documents=doc_texts,
            top_n=top_k,
        )
        if resp.status_code == 200:
            return [documents[item.index] for item in resp.output.results]
        else:
            logger.warning(f"Rerank API调用失败: {resp.message}，跳过重排序")
            return documents[:top_k]
    except Exception as e:
        logger.warning(f"Rerank失败，跳过: {e}")
        return documents[:top_k]


def _get_all_documents_batched():
    """分批加载全量文档，避免SQLite变量上限999"""
    collection = _get_chroma_collection()
    total = collection.count()
    logger.info(f"BM25 共{total}条文档，分批加载...")
    batch_size = 500
    all_texts = []
    offset = 0
    while offset < total:
        batch = collection.get(include=["documents"], limit=batch_size, offset=offset)
        all_texts.extend(batch["documents"])
        offset += batch_size
    return all_texts


def _load_bm25_corpus():
    """加载BM25语料（带pickle缓存，文档数变化时自动重建）"""
    collection = _get_chroma_collection()
    total = collection.count()

    # 缓存命中且数量一致 → 直接用
    if os.path.exists(BM25_CACHE_PATH):
        with open(BM25_CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
        if cache.get("count") == total:
            logger.info(f"BM25使用缓存（{total}条）")
            return [Document(page_content=t) for t in cache["texts"]]

    # 缓存不存在或过期
    all_texts = _get_all_documents_batched()

    # 写缓存
    os.makedirs(os.path.dirname(BM25_CACHE_PATH), exist_ok=True)
    with open(BM25_CACHE_PATH, "wb") as f:
        pickle.dump({"texts": all_texts, "count": total}, f)
    print(f"[BM25] 缓存已保存（{total}条）", flush=True)
    return [Document(page_content=t) for t in all_texts]


def retrieve_context(query: str, use_hybrid: bool = True, use_rerank: bool = True) -> str:
    """检索相关医学知识"""
    if not use_hybrid:
        docs = _vector_search(query, top_k=RAG_TOP_K)
        return "\n\n---\n\n".join([doc.page_content for doc, _ in docs])

    # Hybrid Search
    vector_results = _vector_search(query, top_k=20)
    bm25_results = _bm25_search(query, top_k=20)
    merged = _rrf_merge(vector_results, bm25_results)
    merged_docs = [doc for doc, _ in merged[:20]]

    if use_rerank:
        final_docs = _rerank(query, merged_docs, top_k=RAG_TOP_K)
    else:
        final_docs = merged_docs[:RAG_TOP_K]

    return "\n\n---\n\n".join([doc.page_content for doc in final_docs])


def prebuild_bm25_index(force: bool = False):
    """预构建BM25索引缓存（可提前调用避免首次检索延迟）"""
    if force:
        global _bm25_index_cache
        _bm25_index_cache = None
        if os.path.exists(BM25_CACHE_PATH):
            os.remove(BM25_CACHE_PATH)
            logger.info("已清除旧BM25缓存，将重新构建")
    _get_bm25_index()


if __name__ == "__main__":
    result = retrieve_context("头痛怎么办")
    print(result[:500])