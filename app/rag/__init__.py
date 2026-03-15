# app/rag/__init__.py
from app.connect import execute_query
from app.rag.retriever import hybrid_search


def _get_document_name(document_id: str) -> str:
    """查询 chunk 关联的 document 名称"""
    row = execute_query(
        "SELECT name FROM documents WHERE id = %s",
        (document_id,),
        fetch_one=True,
    )
    return row["name"] if row else "未知来源"


def _get_chunk_images(chunk_id: str) -> list[dict]:
    """查询 chunk 关联的图片信息"""
    sql = """
        SELECT i.id, i.name, i.description, i.url
        FROM images i
        JOIN chunk_images ci ON ci.image_id = i.id
        WHERE ci.chunk_id = %s
    """
    rows = execute_query(sql, (chunk_id,), fetch_all=True)
    return rows or []


def _format_chunk(index: int, chunk: dict) -> str:
    """将单个 chunk 格式化为结构化文本块"""
    parts: list[str] = []

    # 编号
    parts.append(f"[{index}]")

    # 来源：(content_type) document_name  章节路径
    doc_name = _get_document_name(chunk["document_id"])
    content_type = chunk.get("content_type") or "其他"
    source_line = f"来源：({content_type}) {doc_name}"
    toc_path = chunk.get("toc_path")
    if toc_path:
        source_line += "  " + " > ".join(toc_path)
    parts.append(source_line)

    # 正文
    parts.append(f"正文内容：{chunk['content']}")

    # 关联图片
    if chunk.get("has_images"):
        images = _get_chunk_images(chunk["id"])
        if images:
            img_lines = []
            for img in images:
                desc = f"{img['name']}"
                if img.get("description"):
                    desc += f"（{img['description']}）"
                if img.get("url"):
                    desc += f" {img['url']}"
                img_lines.append(desc)
            parts.append("关联图片：" + "；".join(img_lines))

    # 关联注解
    if chunk.get("has_annotation") and chunk.get("annotation"):
        parts.append("关联注解：" + "；".join(chunk["annotation"]))

    return "\n".join(parts)


def retrieve_context(query: str) -> str:
    """
    RAG 检索入口：混合检索 → Context 组装。
    返回格式化的参考资料文本，无结果时返回空字符串。
    """
    chunks = hybrid_search(query)
    if not chunks:
        return ""

    blocks = [_format_chunk(i + 1, chunk) for i, chunk in enumerate(chunks)]
    return "\n\n".join(blocks)
