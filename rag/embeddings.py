"""DashScope嵌入模型的LangChain适配器"""
import dashscope
from langchain_core.embeddings import Embeddings
from config import EMBEDDING_MODEL, DASHSCOPE_API_KEY


class DashScopeEmbeddings(Embeddings):
    """DashScope嵌入模型的LangChain适配器"""

    def __init__(self, model: str = EMBEDDING_MODEL):
        self.model = model
        dashscope.api_key = DASHSCOPE_API_KEY

    def embed_documents(self, texts: list) -> list:
        from dashscope import TextEmbedding
        embeddings = []
        for i in range(0, len(texts), 10):
            batch = texts[i:i + 10]
            resp = TextEmbedding.call(model=self.model, input=batch)
            if resp.status_code != 200:
                raise RuntimeError(f"Embedding API调用失败: {resp.message} (code={resp.status_code})")
            for item in resp.output["embeddings"]:
                embeddings.append(item["embedding"])
        return embeddings

    def embed_query(self, text: str) -> list:
        from dashscope import TextEmbedding
        resp = TextEmbedding.call(model=self.model, input=[text])
        if resp.status_code != 200:
            raise RuntimeError(f"Embedding API调用失败: {resp.message} (code={resp.status_code})")
        return resp.output["embeddings"][0]["embedding"]