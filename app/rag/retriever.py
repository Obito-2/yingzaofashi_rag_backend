# app/rag/retriever.py
import os
import re
import jieba

from app.connect import execute_query
from app.rag.embedding import embed_query

# 加载古建筑领域自定义词表
_DICT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "custon_char.text")
if os.path.exists(_DICT_PATH):
    jieba.load_userdict(_DICT_PATH)

# 停用词 / 标点过滤
_STOP_RE = re.compile(r"^[\s\u3000\W]+$", re.UNICODE)

RRF_K = 60  # RRF 融合常数


def _tokenize(text: str) -> list[str]:
    """jieba 分词 + 过滤标点和空白 token"""
    return [w for w in jieba.cut(text) if not _STOP_RE.match(w)]


def _vector_search(query_vec: list[float], k: int = 3) -> list[dict]:
    """pgvector 余弦相似度检索"""
    sql = """
        SELECT c.id, c.content, c.metadata, c.content_type, c.toc_path,
               c.has_images, c.has_annotation, c.annotation, c.document_id,
               1 - (c.embedding <=> %s::vector) AS similarity
        FROM chunks c
        ORDER BY c.embedding <=> %s::vector
        LIMIT %s
    """
    vec_str = "[" + ",".join(str(v) for v in query_vec) + "]"
    rows = execute_query(sql, (vec_str, vec_str, k), fetch_all=True)
    return rows or []


def _keyword_search(tokens: list[str], k: int = 3) -> list[dict]:
    """tsvector 关键词检索（OR 语义提高召回）"""
    if not tokens:
        return []
    tsquery_str = " | ".join(f"'{t}'" for t in tokens)
    sql = """
        SELECT c.id, c.content, c.metadata, c.content_type, c.toc_path,
               c.has_images, c.has_annotation, c.annotation, c.document_id,
               ts_rank(c.ts_vector, to_tsquery('simple', %s)) AS rank
        FROM chunks c
        WHERE c.ts_vector @@ to_tsquery('simple', %s)
        ORDER BY rank DESC
        LIMIT %s
    """
    rows = execute_query(sql, (tsquery_str, tsquery_str, k), fetch_all=True)
    return rows or []


def _rrf_fuse(vec_results: list[dict], kw_results: list[dict], k_final: int, min_score: float) -> list[dict]:
    """RRF 融合两路排序结果，返回 Top-k_final"""
    scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    for rank, row in enumerate(vec_results, start=1):
        cid = row["id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
        chunk_map[cid] = row

    for rank, row in enumerate(kw_results, start=1):
        cid = row["id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
        chunk_map[cid] = row

    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)

    if not sorted_ids or scores[sorted_ids[0]] < min_score:
        return []

    results = []
    for cid in sorted_ids[:k_final]:
        row = chunk_map[cid]
        row["rrf_score"] = scores[cid]
        results.append(row)
    return results


def hybrid_search(
    query: str,
    k_vector: int = 3,
    k_keyword: int = 3,
    k_final: int = 3,
    min_score: float = 0.0,
) -> list[dict]:
    """
    混合检索：向量语义 + 关键词，RRF 融合排序。
    返回 list[dict]，每个 dict 包含 chunk 各字段及 rrf_score。
    """
    query_vec = embed_query(query)
    tokens = _tokenize(query)

    vec_results = _vector_search(query_vec, k=k_vector)
    kw_results = _keyword_search(tokens, k=k_keyword)

    return _rrf_fuse(vec_results, kw_results, k_final, min_score)
