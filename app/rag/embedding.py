# app/rag/embedding.py
import os
import numpy as np
from openai import OpenAI

EMBEDDING_BASE_URL = os.getenv("DASHSCOPE_BASE_URL")
EMBEDDING_MODEL = "text-embedding-v4"
EMBEDDING_DIM = 1024

_client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY", ""), base_url=EMBEDDING_BASE_URL)


def _normalize(vec: list[float]) -> list[float]:
    """L2 归一化，与数据端写入时的处理保持一致"""
    arr = np.array(vec)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr.tolist()


def embed_query(text: str) -> list[float]:
    """将查询文本转为 1024 维归一化向量"""
    resp = _client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return _normalize(resp.data[0].embedding)
