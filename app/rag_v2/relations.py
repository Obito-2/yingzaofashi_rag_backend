"""策略 B：主检索 top-k 之后按 relations 扩展关联块；可选 relation_type / book 范围过滤。"""
from __future__ import annotations

from app.connect import execute_query
from app.rag.retriever_helpers import strip_fuse_debug_fields
from app.rag_v2.schemas import IntentFilters

MAX_TOTAL_ITEMS = 20


def fetch_relations_for_ids(
    main_ids: list[str],
    relation_types: list[str] | None = None,
) -> list[dict]:
    if not main_ids:
        return []
    if relation_types:
        sql = """
            SELECT relation_id, source_type, source_id, target_type, target_id, relation_type
            FROM relations
            WHERE (source_id = ANY(%s::uuid[]) OR target_id = ANY(%s::uuid[]))
              AND relation_type = ANY(%s::text[])
        """
        rows = execute_query(
            sql, (main_ids, main_ids, relation_types), fetch_all=True
        )
    else:
        sql = """
            SELECT relation_id, source_type, source_id, target_type, target_id, relation_type
            FROM relations
            WHERE source_id = ANY(%s::uuid[]) OR target_id = ANY(%s::uuid[])
        """
        rows = execute_query(sql, (main_ids, main_ids), fetch_all=True)
    return rows or []


def fetch_text_chunks_by_ids(ids: list[str]) -> dict[str, dict]:
    if not ids:
        return {}
    sql = """
        SELECT chunk_id AS id, main_text, book_id, content_type,
               closest_title, toc_path, search_text, other_metadata, chunk_size
        FROM text_chunks
        WHERE chunk_id = ANY(%s::uuid[])
    """
    rows = execute_query(sql, (ids,), fetch_all=True)
    result: dict[str, dict] = {}
    for r in rows or []:
        r["type"] = "text"
        result[r["id"]] = r
    return result


def fetch_image_chunks_by_ids(ids: list[str]) -> dict[str, dict]:
    if not ids:
        return {}
    sql = """
        SELECT image_id AS id, title, image_uri, local_path, alt_text, caption,
               book_id, closest_title, toc_path, search_text, format
        FROM image_chunks
        WHERE image_id = ANY(%s::uuid[])
    """
    rows = execute_query(sql, (ids,), fetch_all=True)
    result: dict[str, dict] = {}
    for r in rows or []:
        r["type"] = "image"
        result[r["id"]] = r
    return result


def _is_image_type(chunk_type: str) -> bool:
    return chunk_type == "image"


def _allowed_book(book_id: str | None, book_allow: list[str] | None) -> bool:
    if not book_allow:
        return True
    if book_id is None:
        return False
    return book_id in book_allow


def enrich_main_with_relations(
    main_results: list[dict],
    filters: IntentFilters,
    *,
    source_retriever_default: str = "fusion",
    intent_type: str = "",
) -> dict:
    """
    main_results: 融合后的行（含 rrf_score 等），与 retriever 行结构一致。
    返回 {"items": [...], "relations": [...]}，items 含 is_main、与旧版一致字段。
    """
    main_ids = [r["id"] for r in main_results]
    main_set = set(main_ids)
    rel_types = filters.relation_types if filters.relation_types else None
    book_allow = filters.book_ids if filters.book_ids else None

    all_relations = fetch_relations_for_ids(main_ids, relation_types=rel_types)

    related_text_ids: set[str] = set()
    related_image_ids: set[str] = set()

    for rel in all_relations:
        for id_field, type_field in [
            ("source_id", "source_type"),
            ("target_id", "target_type"),
        ]:
            rid = rel[id_field]
            if rid not in main_set:
                if _is_image_type(rel[type_field]):
                    related_image_ids.add(rid)
                else:
                    related_text_ids.add(rid)

    related_text = fetch_text_chunks_by_ids(list(related_text_ids))
    related_images = fetch_image_chunks_by_ids(list(related_image_ids))

    # 按 book 过滤关联块
    if book_allow:
        related_text = {
            k: v
            for k, v in related_text.items()
            if _allowed_book(v.get("book_id"), book_allow)
        }
        related_images = {
            k: v
            for k, v in related_images.items()
            if _allowed_book(v.get("book_id"), book_allow)
        }

    items = [
        _build_item(
            r,
            is_main=True,
            source_retriever=r.get("_source_retriever") or source_retriever_default,
            intent_type=intent_type,
        )
        for r in main_results
    ]

    all_known_ids = set(main_ids)
    merged_related = {**related_text, **related_images}
    for rid, row in merged_related.items():
        if len(items) >= MAX_TOTAL_ITEMS:
            break
        if rid not in all_known_ids:
            row = dict(row)
            row["rrf_score"] = row.get("score")
            items.append(
                _build_item(
                    row,
                    is_main=False,
                    source_retriever="relation",
                    intent_type=intent_type,
                )
            )
            all_known_ids.add(rid)

    final_ids = {item["id"] for item in items}
    filtered_relations = [
        {
            "source_id": rel["source_id"],
            "target_id": rel["target_id"],
            "relation_type": rel["relation_type"],
        }
        for rel in all_relations
        if rel["source_id"] in final_ids and rel["target_id"] in final_ids
    ]

    return {"items": items, "relations": filtered_relations}


def _build_item(
    row: dict,
    *,
    is_main: bool,
    source_retriever: str,
    intent_type: str,
) -> dict:
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

    score = row.get("rrf_score")
    if score is None:
        score = row.get("score")

    return {
        "id": row["id"],
        "type": item_type,
        "content": content,
        "metadata": item_metadata,
        "score": score,
        "is_main": is_main,
        "source_retriever": source_retriever,
        "intent_type": intent_type,
    }


def items_only_from_rows(
    main_results: list[dict],
    *,
    source_retriever_default: str = "fusion",
    intent_type: str = "",
) -> dict:
    """无关联扩展时仅组装 items。"""
    items = [
        _build_item(
            r,
            is_main=True,
            source_retriever=r.get("_source_retriever") or source_retriever_default,
            intent_type=intent_type,
        )
        for r in main_results
    ]
    return {"items": items, "relations": []}
