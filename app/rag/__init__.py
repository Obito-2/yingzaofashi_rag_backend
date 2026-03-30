# app/rag/__init__.py
from app.connect import execute_query
from app.rag.retriever import hybrid_search


def _get_document_name(book_id: str) -> str:
    row = execute_query(
        "SELECT name FROM documents WHERE id = %s",
        (book_id,),
        fetch_one=True,
    )
    return row["name"] if row else "未知来源"


def _get_document_name_and_authors(book_id: str) -> tuple[str, list[str]]:
    row = execute_query(
        "SELECT name, authors FROM documents WHERE id = %s",
        (book_id,),
        fetch_one=True,
    )
    if not row:
        return "未知来源", []
    return row["name"], (row.get("authors") or [])


def _format_item(index: int, item: dict) -> str:
    """将单个 item 格式化为结构化文本块，供 LLM system prompt 使用"""
    parts: list[str] = [f"[{index}]"]
    meta = item.get("metadata") or {}

    book_id = meta.get("book_id")
    if book_id:
        doc_name, authors = _get_document_name_and_authors(book_id)
    else:
        doc_name, authors = "未知来源", []
    author_text = "、".join(authors) if authors else "未知作者"
    source_line = f"来源：{doc_name}"
    toc_path = meta.get("toc_path")
    if toc_path:
        source_line += "  " + " > ".join(toc_path)

    parts.append(source_line)

    if item["type"] == "text":
        content_type = meta.get("content_type") or "其他"
        content_type_label_map = {
            "original_text": "原文",
            "annotation": "注释",
            "modern_translation": "译文",
            "interpretation": "解读",
            "others_text": "其他",
        }
        item_type_label = content_type_label_map.get(content_type, "其他文本")
        parts.append(f"内容：{item['content']}")
        parts.append(f"类型：{item_type_label}")
        parts.append(f"作者：{author_text}")
    else:
        title = meta.get("title") or ""
        if title:
            parts.append(f"图名：{title}")
        if item["content"] and item["content"] != title:
            parts.append(f"图注：{item['content']}")
        image_uri = meta.get("image_uri")
        if image_uri:
            parts.append(f"地址：{image_uri}")

    return "\n".join(parts)


def _enrich_items_metadata(items: list[dict]) -> list[dict]:
    """为每个 item 的 metadata 补充 document name（批量查询减少 DB 往返）"""
    book_ids = {
        item.get("metadata", {}).get("book_id")
        for item in items
        if item.get("metadata", {}).get("book_id")
    }
    if not book_ids:
        return items

    book_names: dict[str, str] = {}
    for bid in book_ids:
        book_names[bid] = _get_document_name(bid)

    for item in items:
        bid = item.get("metadata", {}).get("book_id")
        if bid:
            item["metadata"]["book_name"] = book_names.get(bid, "未知来源")

    return items

def retrieve_context_structured(
    query: str,
    with_relations: bool = False,
    k_vector: int = 5,
    k_keyword: int = 5,
    k_final: int = 10,
) -> tuple[str, dict]:
    """
    RAG 检索入口：混合检索 → 双输出。
    返回 (prompt_text, search_result)：
      - prompt_text: 带编号的纯文本，注入 system prompt 供 LLM 引用
      - search_result: {"items": [...], "relations": [...]} 推送给前端
    无结果时返回 ("", {"items": [], "relations": []})。
    """
    result = hybrid_search(
        query,
        k_vector=k_vector,
        k_keyword=k_keyword,
        k_final=k_final,
        with_relations=with_relations,
    )
    items = result.get("items", [])

    if not items:
        return "", {"items": [], "relations": []}

    items = _enrich_items_metadata(items)
    result["items"] = items

    prompt_text = "\n\n".join(
        _format_item(i + 1, item) for i, item in enumerate(items)
    )
    return prompt_text, result


def retrieve_context(query: str) -> str:
    """向后兼容：返回纯文本格式的参考资料。"""
    prompt_text, _ = retrieve_context_structured(query)
    return prompt_text
