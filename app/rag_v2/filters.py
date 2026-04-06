"""校验后的 filters 仅作数据载体；SQL 条件在各 Retriever 内参数化拼接。"""
from __future__ import annotations

from app.rag_v2.schemas import IntentFilters


def filters_for_sql(f: IntentFilters) -> IntentFilters:
    """只读视图，供检索器读取 book_ids / content_types。"""
    return f
