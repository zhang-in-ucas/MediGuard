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


def load_safety_knowledge():
    """内置医疗安全知识（关键词检索的补充）"""
    safety_docs = [
        Document(
            page_content="问：发烧可以自己吃退烧药吗\n答：发热是身体的防御反应，体温38.5℃以下可先物理降温。如需用药，对乙酰氨基酚或布洛芬是常用退烧药，但必须遵医嘱或严格按说明书剂量使用，切勿自行增加剂量或联合使用多种退烧药。持续发热超过3天应及时就医。"),
        Document(
            page_content="问：头痛可以长期吃止痛药吗\n答：不建议长期自行服用止痛药。布洛芬等NSAIDs长期使用可能引起胃肠道损伤、肾功能损害。头痛持续应就医查明原因，而非依赖止痛药掩盖症状。"),
        Document(
            page_content="问：孩子发烧38.5度怎么办\n答：儿童发热38.5℃建议先物理降温，如温水擦浴、减少衣物。如需用药须遵医嘱，儿童退烧药剂量需按体重计算，不可使用成人剂量。3个月以下婴儿发热应立即就医。"),
        Document(
            page_content="问：高血压可以自己买降压药吃吗\n答：高血压用药需医生根据个体情况选择，不同类型降压药适用不同人群。自行用药可能导致血压控制不当或药物不良反应。请至心内科就诊，在医生指导下规范用药。"),
        Document(
            page_content="问：糖尿病感觉好转可以停药吗\n答：不可以自行停药。血糖正常可能是药物控制的结果，停药后血糖可能反弹。应在医生指导下调整用药方案，定期监测血糖。"),
        Document(
            page_content="问：感冒了需要吃抗生素吗\n答：普通感冒多由病毒引起，抗生素对病毒无效，不应自行服用。滥用抗生素可能导致耐药性和不良反应。如出现持续高热、脓性痰等细菌感染迹象，应就医检查后遵医嘱用药。"),
        Document(
            page_content="问：失眠可以长期吃安眠药吗\n答：安眠药不宜长期自行服用，可能产生依赖性和耐受性。失眠应先从改善睡眠习惯入手，如规律作息、减少咖啡因摄入等。持续失眠建议就医，在医生指导下治疗。"),
        Document(
            page_content="问：皮肤过敏自己买药膏涂可以吗\n答：皮肤过敏应先明确过敏原。外用药膏尤其是含激素类药物，需在医生指导下使用，自行长期使用可能引起皮肤萎缩等不良反应。反复过敏建议就诊皮肤科。"),
        Document(
            page_content="问：胃疼吃奥美拉唑可以吗\n答：奥美拉唑是处方药，需在医生诊断后使用。胃痛原因多样，自行用药可能掩盖病情。建议消化内科就诊，明确是溃疡、胃炎还是其他原因后遵医嘱治疗。"),
        Document(
            page_content="问：拉肚子吃诺氟沙星可以吗\n答：诺氟沙星是处方类抗生素，对病毒性腹泻无效。急性腹泻多为自限性，以补液防脱水为主。18岁以下人群禁用诺氟沙星。腹泻持续或伴发热应就医，不要自行服用抗生素。"),
        Document(
            page_content="问：焦虑症发作吃什么药\n答：焦虑症需精神心理科专业评估，药物治疗应在精神科医生指导下进行。抗焦虑药物有多种类型，需根据症状严重程度和个体情况选择，切勿自行购药服用。"),
        Document(
            page_content="问：月经不调自己买药调理可以吗\n答：月经不调原因复杂，包括内分泌失调、妇科炎症、器质性病变等，需要医生检查明确原因后针对性治疗。自行服药可能延误诊断，建议妇科就诊。"),
        Document(
            page_content="问：腰疼吃止痛药就行吗\n答：腰痛原因多样，包括肌肉劳损、椎间盘突出、肾脏疾病等。止痛药只能缓解症状不能治本，长期服用还有胃肠道和肾脏风险。腰痛持续或加重应就医检查。"),
        Document(
            page_content="问：咳嗽一周了还没好怎么办\n答：咳嗽超过一周未愈应就医，需排除肺炎、支气管炎等疾病。不要自行购买止咳药长期服用，特别是含可待因的止咳药。咳嗽是排痰保护机制，盲目止咳可能加重感染。"),
        Document(
            page_content="问：胸闷气短是怎么回事\n答：胸闷气短可能涉及心脏或呼吸系统问题，如冠心病、心律失常、哮喘等，属于需要重视的症状。建议及时就医，心内科或呼吸科检查，不要自行判断或用药。"),
    ]
    print(f"内置安全知识 {len(safety_docs)} 条", flush=True)
    return safety_docs


def build_vectorstore(documents, rebuild=False):
    """构建Chroma向量库（分批入库，支持断点续传）"""
    import shutil
    embeddings = DashScopeEmbeddings()

    if rebuild and os.path.exists(RAG_PERSIST_DIR):
        shutil.rmtree(RAG_PERSIST_DIR)
        print(f"已删除旧向量库 {RAG_PERSIST_DIR}", flush=True)

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

    # 检查已有向量库（断点续传）
    vectorstore = Chroma(
        persist_directory=RAG_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name="medical_knowledge",
    )
    existing_count = vectorstore._collection.count()
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

    count = vectorstore._collection.count()
    print(f"向量库已保存到 {RAG_PERSIST_DIR}，共 {count} 条", flush=True)
    return vectorstore


if __name__ == "__main__":
    all_docs = []

    # 数据源1：医疗QA（主体）
    all_docs.extend(load_medical_qa_data(max_items=10000))

    # 数据源2：Chinese-medical-dialogue-data（真实医患对话）
    all_docs.extend(load_chinese_medical_dialogue(max_items=50000))

    # 数据源3：内置安全知识
    all_docs.extend(load_safety_knowledge())

    print(f"\n共加载 {len(all_docs)} 条文档", flush=True)

    # 重建向量库
    build_vectorstore(all_docs, rebuild=True)

    # 验证
    from rag.retriever import retrieve_context

    print("\n--- 验证检索 ---")
    for q in ["头痛怎么办", "发烧38.5度吃什么药", "高血压用药", "感冒了多喝水有用吗", "儿科发烧"]:
        result = retrieve_context(q, use_hybrid=False)
        print(f"Q: {q}")
        print(f"A: {result[:150]}...\n")
    print("RAG知识库构建完成")