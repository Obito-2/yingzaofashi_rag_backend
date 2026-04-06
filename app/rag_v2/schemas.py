"""意图载荷与检索路枚举；Pydantic v2 校验。"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from app.models import ContentTypeEnum, RelationTypeEnum


class IntentType(str, Enum):
    rare_char = "rare_char"
    term_explain = "term_explain"
    original_and_translation = "original_and_translation"
    image_by_text = "image_by_text"
    specific_book = "specific_book"
    complex = "complex"


class RetrieverId(str, Enum):
    text_toc_kw = "text_toc_kw"
    text_vec = "text_vec"
    text_kw = "text_kw"
    img_toc_kw = "img_toc_kw"
    img_content_kw = "img_content_kw"
    relation = "relation"


MAIN_RETRIEVER_IDS: tuple[RetrieverId, ...] = (
    RetrieverId.text_toc_kw,
    RetrieverId.text_vec,
    RetrieverId.text_kw,
    RetrieverId.img_toc_kw,
    RetrieverId.img_content_kw,
)

ALL_RETRIEVER_IDS: tuple[RetrieverId, ...] = MAIN_RETRIEVER_IDS + (RetrieverId.relation,)

_CONTENT_VALUES = frozenset(e.value for e in ContentTypeEnum)
_RELATION_VALUES = frozenset(e.value for e in RelationTypeEnum)


class IntentFilters(BaseModel):
    """空列表表示不限定。"""

    book_ids: list[str] = Field(default_factory=list)
    content_types: list[str] = Field(default_factory=list)
    relation_types: list[str] = Field(default_factory=list)

    @field_validator("content_types", mode="before")
    @classmethod
    def _strip_unknown_content(cls, v: Any) -> list[str]:
        if not v:
            return []
        out = [x for x in v if isinstance(x, str) and x in _CONTENT_VALUES]
        return out

    @field_validator("relation_types", mode="before")
    @classmethod
    def _strip_unknown_relation(cls, v: Any) -> list[str]:
        if not v:
            return []
        return [x for x in v if isinstance(x, str) and x in _RELATION_VALUES]


class SingleIntentBlock(BaseModel):
    type: str = "original_and_translation"
    filters: IntentFilters = Field(default_factory=IntentFilters)
    enabled_retrievers: list[str] = Field(default_factory=list)

    @field_validator("enabled_retrievers", mode="before")
    @classmethod
    def _normalize_retrievers(cls, v: Any) -> list[str]:
        if not v:
            return [r.value for r in MAIN_RETRIEVER_IDS]
        seen: set[str] = set()
        out: list[str] = []
        for x in v:
            if not isinstance(x, str):
                continue
            try:
                rid = RetrieverId(x)
            except ValueError:
                continue
            if rid.value not in seen:
                seen.add(rid.value)
                out.append(rid.value)
        return out if out else [r.value for r in MAIN_RETRIEVER_IDS]


class SubQueryItem(BaseModel):
    query: str
    type: str = "original_and_translation"
    filters: IntentFilters = Field(default_factory=IntentFilters)
    enabled_retrievers: list[str] = Field(default_factory=list)

    @field_validator("enabled_retrievers", mode="before")
    @classmethod
    def _normalize_sq_retrievers(cls, v: Any) -> list[str]:
        if not v:
            return [r.value for r in MAIN_RETRIEVER_IDS]
        seen: set[str] = set()
        out: list[str] = []
        for x in v:
            if not isinstance(x, str):
                continue
            try:
                rid = RetrieverId(x)
            except ValueError:
                continue
            if rid.value not in seen:
                seen.add(rid.value)
                out.append(rid.value)
        return out if out else [r.value for r in MAIN_RETRIEVER_IDS]


class IntentPayload(BaseModel):
    query: str = ""
    intents: list[SingleIntentBlock] = Field(default_factory=list)
    is_complex: bool = False
    sub_queries: list[SubQueryItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def _defaults(self) -> IntentPayload:
        if not self.intents:
            self.intents = [
                SingleIntentBlock(
                    type="original_and_translation",
                    filters=IntentFilters(),
                    enabled_retrievers=[r.value for r in MAIN_RETRIEVER_IDS],
                )
            ]
        return self


def load_valid_book_ids() -> frozenset[str]:
    """从 documents 表加载 id 白名单（每次调用查询；上层可缓存）。"""
    from app.connect import execute_query

    rows = execute_query("SELECT id FROM documents", fetch_all=True) or []
    return frozenset(r["id"] for r in rows if r.get("id"))


_valid_book_ids_cache: frozenset[str] | None = None


def get_valid_book_ids() -> frozenset[str]:
    global _valid_book_ids_cache
    if _valid_book_ids_cache is None:
        _valid_book_ids_cache = load_valid_book_ids()
    return _valid_book_ids_cache


def invalidate_book_id_cache() -> None:
    global _valid_book_ids_cache
    _valid_book_ids_cache = None


def sanitize_book_ids(book_ids: list[str] | None) -> list[str]:
    """剔除不存在于 documents 的 id；若全部非法则返回空列表（表示不限定）。"""
    if not book_ids:
        return []
    valid = get_valid_book_ids()
    out = [b for b in book_ids if isinstance(b, str) and b in valid]
    return out


def sanitize_filters(f: IntentFilters) -> IntentFilters:
    return IntentFilters(
        book_ids=sanitize_book_ids(f.book_ids),
        content_types=list(f.content_types),
        relation_types=list(f.relation_types),
    )


def parse_intent_result(
    raw: dict[str, Any] | None,
    *,
    refresh_book_cache: bool = False,
) -> IntentPayload:
    """
    将外部 dict 解析为 IntentPayload；非法 book_id 已剔除。
    raw 为 None 或解析失败时使用默认单意图、五路全开。
    """
    if refresh_book_cache:
        invalidate_book_id_cache()

    if not raw:
        return IntentPayload()

    try:
        payload = IntentPayload.model_validate(raw)
    except Exception:
        return IntentPayload()

    # 净化每个 intent / sub_query 的 book_ids
    new_intents: list[SingleIntentBlock] = []
    for it in payload.intents:
        sf = sanitize_filters(it.filters)
        new_intents.append(
            SingleIntentBlock(
                type=it.type,
                filters=sf,
                enabled_retrievers=it.enabled_retrievers,
            )
        )
    new_subs: list[SubQueryItem] = []
    for sq in payload.sub_queries:
        sf = sanitize_filters(sq.filters)
        new_subs.append(
            SubQueryItem(
                query=sq.query,
                type=sq.type,
                filters=sf,
                enabled_retrievers=sq.enabled_retrievers,
            )
        )
    return IntentPayload(
        query=payload.query or "",
        intents=new_intents or payload.intents,
        is_complex=payload.is_complex,
        sub_queries=new_subs,
    )
