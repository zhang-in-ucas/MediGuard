import os
from dotenv import load_dotenv

load_dotenv(override=True)

# ==================== LLM 配置 ====================
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen-plus")

# ==================== 嵌入模型配置 ====================
EMBEDDING_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")

# ==================== RAG 配置 ====================
RAG_PERSIST_DIR = os.getenv("RAG_PERSIST_DIR", "./rag/chroma_db")
RAG_TOP_K = 3

# ==================== Agent 配置 ====================
MAX_RETRIES = 3
TEMPERATURE = 0.3

# ==================== 安全审查规则 ====================
SAFETY_KEYWORDS = [
    # 开处方类
    "开处方", "给我开", "帮我开药", "开个药",
    "吃什么药", "用什么药", "推荐药", "推荐个药",
    "推荐一些药", "制定治疗方案", "治疗方案",
    # 诊断替代类
    "确诊为", "诊断为您", "诊断为你", "您患有", "你患有",
    "诊断一下", "是不是得了", "得了什么病",
    # 处方剂量类
    "建议服用", "可以服用", "推荐服用", "可以考虑服用",
    "每次服用", "每日服用", "一次服用",
    #"每次2", "每次3", "每天2", "每天3",（通过harness修改删除，跑步每天20分钟被拦截了）
    "mg每日", "mg每天", "mg一次", "mg每次",
    "一天3次", "一天2次", "一日3次", "一日2次", "加剂量",
    # 劝阻就医类
    "不用去医院", "不用就医", "不需要就医", "不必去医院", "无需就医",
    "不用听医生", "自己能好", "自己能调整", "自己能恢复",
]

# ==================== Agent 多模型配置 ====================
TRIAGE_MODEL = os.getenv("TRIAGE_MODEL", "qwen-turbo")
DIAGNOSIS_MODEL = os.getenv("DIAGNOSIS_MODEL", "qwen-plus")
SAFETY_MODEL = os.getenv("SAFETY_MODEL", "qwen-plus")