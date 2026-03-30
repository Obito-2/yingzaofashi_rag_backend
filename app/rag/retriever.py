# app/rag/retriever.py
"""混合检索入口与数据库查询；查询解析与 RRF 见 retriever_helpers。"""
from langsmith import traceable

from app.connect import execute_query
from app.rag.embedding import embed_query
from app.rag.retriever_helpers import (
    ORIGINAL_TEXT_CONTENT_TYPE,
    apply_image_slot_limit,
    apply_original_text_boost,
    build_tsquery_loose,
    build_tsquery_strict,
    detect_query_intent,
    lane_weights,
    merge_text_vector_lanes,
    rrf_fuse,
    strip_fuse_debug_fields,
    tokenize_query_display,
)

MAX_TOTAL_ITEMS = 20


def _fetch_text_keyword_rows(tsquery_str: str, k: int) -> list[dict]:
    sql = """
        SELECT chunk_id AS id, main_text, book_id, content_type,
               closest_title, toc_path, search_text, other_metadata, chunk_size,
               ts_rank(ts_vector, to_tsquery('simple', %s)) AS score
        FROM text_chunks
        WHERE ts_vector @@ to_tsquery('simple', %s)
        ORDER BY score DESC
        LIMIT %s
    """
    rows = execute_query(sql, (tsquery_str, tsquery_str, k), fetch_all=True)
    for r in (rows or []):
        r["type"] = "text"
    return rows or []


def _fetch_image_keyword_rows(tsquery_str: str, k: int) -> list[dict]:
    sql = """
        SELECT image_id AS id, title, image_uri, local_path, alt_text, caption,
               book_id, closest_title, toc_path, search_text, format,
               ts_rank(ts_vector, to_tsquery('simple', %s)) AS score
        FROM image_chunks
        WHERE ts_vector @@ to_tsquery('simple', %s)
        ORDER BY score DESC
        LIMIT %s
    """
    rows = execute_query(sql, (tsquery_str, tsquery_str, k), fetch_all=True)
    for r in (rows or []):
        r["type"] = "image"
    return rows or []


# --------------- 四路检索 ---------------

def _text_vector_search(
    query_vec: list[float],
    k: int = 5,
    *,
    content_types: list[str] | None = None,
) -> list[dict]:
    vec_str = "[" + ",".join(str(v) for v in query_vec) + "]"
    if content_types:
        sql = """
            SELECT chunk_id AS id, main_text, book_id, content_type,
                   closest_title, toc_path, search_text, other_metadata, chunk_size,
                   1 - (embedding_values <=> %s::vector) AS score
            FROM text_chunks
            WHERE content_type = ANY(%s::text[])
            ORDER BY embedding_values <=> %s::vector
            LIMIT %s
        """
        rows = execute_query(
            sql, (vec_str, content_types, vec_str, k), fetch_all=True
        )
    else:
        sql = """
            SELECT chunk_id AS id, main_text, book_id, content_type,
                   closest_title, toc_path, search_text, other_metadata, chunk_size,
                   1 - (embedding_values <=> %s::vector) AS score
            FROM text_chunks
            ORDER BY embedding_values <=> %s::vector
            LIMIT %s
        """
        rows = execute_query(sql, (vec_str, vec_str, k), fetch_all=True)
    for r in (rows or []):
        r["type"] = "text"
    return rows or []


def _text_keyword_search(query: str, k: int = 5) -> list[dict]:
    tsq = build_tsquery_strict(query)
    if not tsq:
        return []
    rows = _fetch_text_keyword_rows(tsq, k)
    if not rows:
        loose = build_tsquery_loose(query)
        if loose and loose != tsq:
            rows = _fetch_text_keyword_rows(loose, k)
    return rows


def _image_vector_search(query_vec: list[float], k: int = 5) -> list[dict]:
    sql = """
        SELECT image_id AS id, title, image_uri, local_path, alt_text, caption,
               book_id, closest_title, toc_path, search_text, format,
               1 - (embedding_values <=> %s::vector) AS score
        FROM image_chunks
        ORDER BY embedding_values <=> %s::vector
        LIMIT %s
    """
    vec_str = "[" + ",".join(str(v) for v in query_vec) + "]"
    rows = execute_query(sql, (vec_str, vec_str, k), fetch_all=True)
    for r in (rows or []):
        r["type"] = "image"
    return rows or []


def _image_keyword_search(query: str, k: int = 5) -> list[dict]:
    tsq = build_tsquery_strict(query)
    if not tsq:
        return []
    rows = _fetch_image_keyword_rows(tsq, k)
    if not rows:
        loose = build_tsquery_loose(query)
        if loose and loose != tsq:
            rows = _fetch_image_keyword_rows(loose, k)
    return rows or []


def _has_col_data(table: str, column: str) -> bool:
    """快速判断指定表的指定列是否存在非 NULL 数据"""
    row = execute_query(
        f"SELECT 1 FROM {table} WHERE {column} IS NOT NULL LIMIT 1",
        fetch_one=True,
    )
    return bool(row)


# --------------- 关联数据补充 ---------------

def _fetch_relations(main_ids: list[str]) -> list[dict]:
    if not main_ids:
        return []
    sql = """
        SELECT relation_id, source_type, source_id, target_type, target_id, relation_type
        FROM relations
        WHERE source_id = ANY(%s::uuid[]) OR target_id = ANY(%s::uuid[])
    """
    rows = execute_query(sql, (main_ids, main_ids), fetch_all=True)
    return rows or []


def _fetch_text_chunks_by_ids(ids: list[str]) -> dict[str, dict]:
    if not ids:
        return {}
    sql = """
        SELECT chunk_id AS id, main_text, book_id, content_type,
               closest_title, toc_path, search_text, other_metadata, chunk_size
        FROM text_chunks
        WHERE chunk_id = ANY(%s::uuid[])
    """
    rows = execute_query(sql, (ids,), fetch_all=True)
    result = {}
    for r in (rows or []):
        r["type"] = "text"
        result[r["id"]] = r
    return result


def _fetch_image_chunks_by_ids(ids: list[str]) -> dict[str, dict]:
    if not ids:
        return {}
    sql = """
        SELECT image_id AS id, title, image_uri, local_path, alt_text, caption,
               book_id, closest_title, toc_path, search_text, format
        FROM image_chunks
        WHERE image_id = ANY(%s::uuid[])
    """
    rows = execute_query(sql, (ids,), fetch_all=True)
    result = {}
    for r in (rows or []):
        r["type"] = "image"
        result[r["id"]] = r
    return result


def _is_image_type(chunk_type: str) -> bool:
    return chunk_type == "image"


def _enrich_with_relations(main_results: list[dict]) -> dict:
    main_ids = [r["id"] for r in main_results]
    main_set = set(main_ids)

    all_relations = _fetch_relations(main_ids)

    related_text_ids: set[str] = set()
    related_image_ids: set[str] = set()

    for rel in all_relations:
        for id_field, type_field in [("source_id", "source_type"), ("target_id", "target_type")]:
            rid = rel[id_field]
            if rid not in main_set:
                if _is_image_type(rel[type_field]):
                    related_image_ids.add(rid)
                else:
                    related_text_ids.add(rid)

    related_text = _fetch_text_chunks_by_ids(list(related_text_ids))
    related_images = _fetch_image_chunks_by_ids(list(related_image_ids))

    items = [_build_item(r, is_main=True) for r in main_results]

    all_known_ids = set(main_ids)
    for rid, row in {**related_text, **related_images}.items():
        if len(items) >= MAX_TOTAL_ITEMS:
            break
        if rid not in all_known_ids:
            items.append(_build_item(row, is_main=False))
            all_known_ids.add(rid)

    final_ids = {item["id"] for item in items}
    filtered_relations = [
        {"source_id": rel["source_id"], "target_id": rel["target_id"], "relation_type": rel["relation_type"]}
        for rel in all_relations
        if rel["source_id"] in final_ids and rel["target_id"] in final_ids
    ]

    return {"items": items, "relations": filtered_relations}


def _build_item(row: dict, is_main: bool) -> dict:
    strip_fuse_debug_fields(row)
    item_type = row["type"]
    if item_type == "text":
        content = row.get("main_text", "")
        item_metadata = {
            "book_id": row.get("book_id"),
            "content_type": row.get("content_type"),
            "closest_title": row.get("closest_title"),
            "toc_path": row.get("toc_path"),
            "chunk_size": row.get("chunk_size"),
            "other_metadata": row.get("other_metadata"),
        }
    else:
        content = row.get("caption") or row.get("title") or ""
        item_metadata = {
            "book_id": row.get("book_id"),
            "title": row.get("title"),
            "image_uri": row.get("image_uri"),
            "local_path": row.get("local_path"),
            "alt_text": row.get("alt_text"),
            "closest_title": row.get("closest_title"),
            "toc_path": row.get("toc_path"),
            "format": row.get("format"),
        }

    return {
        "id": row["id"],
        "type": item_type,
        "content": content,
        "metadata": item_metadata,
        "score": row.get("rrf_score"),
        "is_main": is_main,
    }


# --------------- 对外入口 ---------------

def hybrid_search(
    query: str,
    k_vector: int = 5,
    k_keyword: int = 5,
    k_final: int = 10,
    with_relations: bool = False,
) -> dict:
    """
    混合检索入口。
    返回 {"items": [...], "relations": [...]}。
    with_relations=False 时 relations 为空列表。
    """
    intent = detect_query_intent(query)
    query_vec = embed_query(query)

    has_text_vec = _has_col_data("text_chunks", "embedding_values")
    has_text_kw = _has_col_data("text_chunks", "ts_vector")
    has_img_vec = _has_col_data("image_chunks", "embedding_values")
    has_img_kw = _has_col_data("image_chunks", "ts_vector")

    if has_text_vec:
        general_tv = _text_vector_search(query_vec, k=k_vector)
        if intent.wants_original_text and intent.wants_institution:
            orig_tv = _text_vector_search(
                query_vec,
                k=min(k_vector, 8),
                content_types=[ORIGINAL_TEXT_CONTENT_TYPE],
            )
            text_vec = merge_text_vector_lanes(orig_tv, general_tv, k_vector)
        else:
            text_vec = general_tv
    else:
        text_vec = []

    text_kw = _text_keyword_search(query, k=k_keyword) if has_text_kw else []
    img_vec = _image_vector_search(query_vec, k=k_vector) if has_img_vec else []
    img_kw = _image_keyword_search(query, k=k_keyword) if has_img_kw else []

    w_tv, w_tk, w_iv, w_ik = lane_weights(has_img_vec)

    main_results = rrf_fuse(
        [text_vec, text_kw, img_vec, img_kw],
        k_final * 2 if (intent.wants_original_text and intent.wants_institution) else k_final,
        weights=[w_tv, w_tk, w_iv, w_ik],
        lane_names=["text_vec", "text_kw", "img_vec", "img_kw"],
    )
    main_results = apply_original_text_boost(main_results, intent)

    max_images = k_final
    if intent.wants_original_text and intent.wants_institution:
        max_images = 1
    main_results = apply_image_slot_limit(main_results, k_final, max_images)

    if not main_results:
        return {"items": [], "relations": []}

    if with_relations:
        return _enrich_with_relations(main_results)

    items = [_build_item(r, is_main=True) for r in main_results]
    return {"items": items, "relations": []}


# --- 供脚本/调试：与历史导入名兼容 ---
_tokenize = tokenize_query_display
_rrf_fuse = rrf_fuse
_lane_weights = lane_weights
_merge_text_vector_lanes = merge_text_vector_lanes
_apply_original_text_boost = apply_original_text_boost
_apply_image_slot_limit = apply_image_slot_limit
