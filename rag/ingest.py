"""RAG知识库构建：DashScope嵌入 + 多数据源"""
import os
import sys

os.environ["PYTHONUTF8"] = "1"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import tempfile
import httpx
import pandas as pd
from langchain_core.documents import Document
from langchain_chroma import Chroma
from config import RAG_PERSIST_DIR, EMBEDDING_MODEL, DASHSCOPE_API_KEY
from rag.embeddings import DashScopeEmbeddings


def load_medical_qa_data(max_items=5000):
    """直接下载shibing624/medical的JSON文件"""
    print("加载医疗QA数据集...", flush=True)
    cache_path = "./rag/medical_train_zh.json"

    if not os.path.exists(cache_path):
        url = "https://hf-mirror.com/datasets/shibing624/medical/resolve/main/finetune/train_zh_0.json"
        try:
            print(f"  下载 {url}（文件较大，请耐心等待）...", flush=True)
            resp = httpx.get(url, timeout=300, follow_redirects=True)
            resp.raise_for_status()
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(resp.text)
            print(f"  已缓存到 {cache_path}", flush=True)
        except Exception as e:
            print(f"shibing624/medical下载失败: {e}", flush=True)
            return []
    else:
        print(f"  使用本地缓存: {cache_path}", flush=True)

    documents = []
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                instruction = item.get("instruction", "").strip()
                inp = item.get("input", "").strip()
                output = item.get("output", "").strip()
                if not instruction or not output:
                    continue
                question = f"{instruction} {inp}".strip() if inp else instruction
                text = f"问：{question}\n答：{output}"
                if len(text) >= 30:
                    documents.append(Document(page_content=text))
                if len(documents) >= max_items:
                    break
        print(f"从shibing624/medical加载 {len(documents)} 条", flush=True)
        return documents
    except Exception as e:
        print(f"shibing624/medical加载失败: {e}", flush=True)
        return []


def load_chinese_medical_dialogue(data_dir=None, max_items=50000):
    """加载Chinese-medical-dialogue-data（79万条真实医患对话）

    使用前需先下载数据集到本地：
    git clone https://github.com/Toyhom/Chinese-medical-dialogue-data.git

    data_dir: 克隆下来的数据集根目录（含Data_数据文件夹）
    max_items: 最多加载多少条
    """
    if data_dir is None:
        data_dir = os.getenv("DIALOGUE_DATA_DIR", "./Chinese-medical-dialogue-data")

    data_folder = os.path.join(data_dir, "Data_数据")
    if not os.path.exists(data_folder):
        print(f"[对话数据] 目录不存在: {data_folder}", flush=True)
        print("请先下载数据集: git clone https://github.com/Toyhom/Chinese-medical-dialogue-data.git", flush=True)
        return []

    csv_files = {
        "Andriatria_男科": "男科5-13000.csv",
        "IM_内科": "内科5000-33000.csv",
        "OAGD_妇产科": "妇产科6-28000.csv",
        "Oncology_肿瘤科": "肿瘤科5-10000.csv",
        "Pediatric_儿科": "儿科5-14000.csv",
        "Surgical_外科": "外科5-14000.csv",
    }

    documents = []
    for folder_name, csv_name in csv_files.items():
        csv_path = os.path.join(data_folder, folder_name, csv_name)
        if not os.path.exists(csv_path):
            print(f"  [跳过] 文件不存在: {csv_path}", flush=True)
            continue

        try:
            print(f"  读取 {folder_name}...", flush=True)
            df = pd.read_csv(csv_path, encoding="gb18030", on_bad_lines="skip")
            for _, row in df.iterrows():
                department = str(row.get("department", "")).strip()
                question = str(row.get("ask", "")).strip()
                answer = str(row.get("answer", "")).strip()

                if not question or not answer:
                    continue
                if answer == "nan" or question == "nan":
                    continue

                dept_prefix = f"[{department}]" if department else ""
                text = f"问：{dept_prefix}{question}\n答：{answer}"

                if len(text) >= 30:
                    documents.append(Document(page_content=text))
                if len(documents) >= max_items:
                    break
            if len(documents) >= max_items:
                break
        except Exception as e:
            print(f"  [失败] {folder_name}: {e}", flush=True)
            continue

    print(f"从Chinese-medical-dialogue-data加载 {len(documents)} 条", flush=True)
    return documents



def build_vectorstore(documents, rebuild=False):
    """构建Chroma向量库（分批入库，支持断点续传）"""
    import chromadb
    embeddings = DashScopeEmbeddings()

    # 先建立 ChromaDB 客户端
    client = chromadb.PersistentClient(path=RAG_PERSIST_DIR)

    if rebuild:
        # 用 ChromaDB API 删除集合（避免 Windows 文件锁问题），
        # 不要用 shutil.rmtree 直接删文件——SQLite 可能被其他进程锁定
        try:
            client.delete_collection(name="medical_knowledge")
            print(f"已删除旧集合 medical_knowledge", flush=True)
        except Exception:
            print(f"集合 medical_knowledge 不存在，无需删除", flush=True)

    # 去重
    seen = set()
    unique_docs = []
    for doc in documents:
        content = doc.page_content[:100]
        if content not in seen:
            seen.add(content)
            unique_docs.append(doc)

    total = len(unique_docs)
    print(f"去重后 {total} 条文档，开始向量化...", flush=True)

    # 创建 langchain_chroma 向量库实例
    from langchain_chroma import Chroma
    vectorstore = Chroma(
        client=client,
        collection_name="medical_knowledge",
        embedding_function=embeddings,
    )

    # 检查已有向量库条数（断点续传）
    try:
        collection = client.get_collection(name="medical_knowledge")
        existing_count = collection.count()
    except Exception:
        existing_count = 0
    start_idx = existing_count
    print(f"已有 {existing_count} 条，从第 {start_idx} 条继续", flush=True)

    # 分批入库，每批500条
    BATCH_SIZE = 500
    for i in range(start_idx, total, BATCH_SIZE):
        batch = unique_docs[i:i + BATCH_SIZE]
        try:
            vectorstore.add_documents(batch)
            print(f"  [{i + len(batch)}/{total}] 已入库", flush=True)
        except Exception as e:
            print(f"  [{i}/{total}] 批次失败: {e}", flush=True)
            print(f"  等待10秒后重试...", flush=True)
            import time
            time.sleep(10)
            try:
                vectorstore.add_documents(batch)
                print(f"  [{i + len(batch)}/{total}] 重试成功", flush=True)
            except Exception as e2:
                print(f"  [{i}/{total}] 重试也失败: {e2}，跳过此批次", flush=True)
                continue

    collection = client.get_collection(name="medical_knowledge")
    count = collection.count()
    print(f"向量库已保存到 {RAG_PERSIST_DIR}，共 {count} 条", flush=True)
    return vectorstore


if __name__ == "__main__":
    all_docs = []

    # 数据源1：医疗QA（主体）
    all_docs.extend(load_medical_qa_data(max_items=10000))

    # 数据源2：Chinese-medical-dialogue-data（真实医患对话）
    all_docs.extend(load_chinese_medical_dialogue(max_items=50000))

    print(f"\n共加载 {len(all_docs)} 条文档", flush=True)

    # 重建向量库
    build_vectorstore(all_docs, rebuild=True)

    # 预构建BM25缓存（避免首次检索时的延迟）
    from rag.retriever import retrieve_context, prebuild_bm25_index
    prebuild_bm25_index(force=True)

    # 验证

    print("\n--- 验证检索 ---")
    for q in ["头痛怎么办", "发烧38.5度吃什么药", "高血压用药", "感冒了多喝水有用吗", "儿科发烧"]:
        result = retrieve_context(q, use_hybrid=False)
        print(f"Q: {q}")
        print(f"A: {result[:150]}...\n")
    print("RAG知识库构建完成")