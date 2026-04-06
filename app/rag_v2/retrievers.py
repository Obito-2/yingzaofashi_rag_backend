"""五路主检索 + RelationRetriever；行 dict 与旧 hybrid_search 对齐。"""
from __future__ import annotations

from abc import ABC, abstractmethod

from app.connect import execute_query
from app.rag.embedding import embed_query
from app.rag.retriever_helpers import build_tsquery_loose, build_tsquery_strict
from app.rag_v2.schemas import MAIN_RETRIEVER_IDS, IntentFilters, RetrieverId


def _text_row_defaults(r: dict) -> None:
    r["type"] = "text"


def _image_row_defaults(r: dict) -> None:
    r["type"] = "image"


def _build_text_filter_sql(
    book_ids: list[str], content_types: list[str]
) -> tuple[str, list]:
    """返回附加 WHERE 片段（以 AND 开头）及参数。"""
    parts: list[str] = []
    params: list = []
    if book_ids:
        parts.append("AND book_id = ANY(%s::text[])")
        params.append(book_ids)
    if content_types:
        parts.append("AND content_type = ANY(%s::text[])")
        params.append(content_types)
    return (" " + " ".join(parts) if parts else ""), params


def _build_image_filter_sql(book_ids: list[str]) -> tuple[str, list]:
    if not book_ids:
        return "", []
    return " AND book_id = ANY(%s::text[])", [book_ids]


class BaseRetriever(ABC):
    retriever_id: str

    @abstractmethod
    def retrieve(
        self,
        query: str,
        filters: IntentFilters,
        k: int,
        *,
        query_vec: list[float] | None = None,
    ) -> list[dict]:
        pass


class TextTitleKeywordRetriever(BaseRetriever):
    retriever_id = RetrieverId.text_toc_kw.value

    def retrieve(
        self,
        query: str,
        filters: IntentFilters,
        k: int,
        *,
        query_vec: list[float] | None = None,
    ) -> list[dict]:
        tsq = build_tsquery_strict(query)
        if not tsq:
            return []
        extra, extra_params = _build_text_filter_sql(
            filters.book_ids, filters.content_types
        )
        sql = f"""
            SELECT chunk_id AS id, main_text, book_id, content_type,
                   closest_title, toc_path, search_text, other_metadata, chunk_size,
                   ts_rank(toc_tsvector, to_tsquery('simple', %s)) AS score
            FROM text_chunks
            WHERE toc_tsvector IS NOT NULL
              AND toc_tsvector @@ to_tsquery('simple', %s)
              {extra}
            ORDER BY score DESC
            LIMIT %s
        """
        params = [tsq, tsq, *extra_params, k]
        rows = execute_query(sql, tuple(params), fetch_all=True) or []
        for r in rows:
            _text_row_defaults(r)
            r["_source_retriever"] = self.retriever_id
        if rows:
            return rows
        loose = build_tsquery_loose(query)
        if not loose or loose == tsq:
            return []
        params2 = [loose, loose, *extra_params, k]
        rows = execute_query(sql, tuple(params2), fetch_all=True) or []
        for r in rows:
            _text_row_defaults(r)
            r["_source_retriever"] = self.retriever_id
        return rows


class TextContentKeywordRetriever(BaseRetriever):
    retriever_id = RetrieverId.text_kw.value

    def retrieve(
        self,
        query: str,
        filters: IntentFilters,
        k: int,
        *,
        query_vec: list[float] | None = None,
    ) -> list[dict]:
        tsq = build_tsquery_strict(query)
        if not tsq:
            return []
        extra, extra_params = _build_text_filter_sql(
            filters.book_ids, filters.content_types
        )
        sql = f"""
            SELECT chunk_id AS id, main_text, book_id, content_type,
                   closest_title, toc_path, search_text, other_metadata, chunk_size,
                   ts_rank(ts_vector, to_tsquery('simple', %s)) AS score
            FROM text_chunks
            WHERE ts_vector IS NOT NULL
              AND ts_vector @@ to_tsquery('simple', %s)
              {extra}
            ORDER BY score DESC
            LIMIT %s
        """
        params = [tsq, tsq, *extra_params, k]
        rows = execute_query(sql, tuple(params), fetch_all=True) or []
        for r in rows:
            _text_row_defaults(r)
            r["_source_retriever"] = self.retriever_id
        if rows:
            return rows
        loose = build_tsquery_loose(query)
        if not loose or loose == tsq:
            return []
        params2 = [loose, loose, *extra_params, k]
        rows = execute_query(sql, tuple(params2), fetch_all=True) or []
        for r in rows:
            _text_row_defaults(r)
            r["_source_retriever"] = self.retriever_id
        return rows


class TextContentVectorRetriever(BaseRetriever):
    retriever_id = RetrieverId.text_vec.value

    def retrieve(
        self,
        query: str,
        filters: IntentFilters,
        k: int,
        *,
        query_vec: list[float] | None = None,
    ) -> list[dict]:
        if not query_vec:
            query_vec = embed_query(query)
        vec_str = "[" + ",".join(str(v) for v in query_vec) + "]"
        conds = ["embedding_values IS NOT NULL"]
        params: list = [vec_str]
        if filters.book_ids:
            conds.append("book_id = ANY(%s::text[])")
            params.append(filters.book_ids)
        if filters.content_types:
            conds.append("content_type = ANY(%s::text[])")
            params.append(filters.content_types)
        where_sql = " AND ".join(conds)
        sql = f"""
            SELECT chunk_id AS id, main_text, book_id, content_type,
                   closest_title, toc_path, search_text, other_metadata, chunk_size,
                   1 - (embedding_values <=> %s::vector) AS score
            FROM text_chunks
            WHERE {where_sql}
            ORDER BY embedding_values <=> %s::vector
            LIMIT %s
        """
        params.append(vec_str)
        params.append(k)
        rows = execute_query(sql, tuple(params), fetch_all=True) or []
        for r in rows:
            _text_row_defaults(r)
            r["_source_retriever"] = self.retriever_id
        return rows


class ImageTitleKeywordRetriever(BaseRetriever):
    retriever_id = RetrieverId.img_toc_kw.value

    def retrieve(
        self,
        query: str,
        filters: IntentFilters,
        k: int,
        *,
        query_vec: list[float] | None = None,
    ) -> list[dict]:
        tsq = build_tsquery_strict(query)
        if not tsq:
            return []
        extra, extra_params = _build_image_filter_sql(filters.book_ids)
        sql = f"""
            SELECT image_id AS id, title, image_uri, local_path, alt_text, caption,
                   book_id, closest_title, toc_path, search_text, format,
                   ts_rank(toc_tsvector, to_tsquery('simple', %s)) AS score
            FROM image_chunks
            WHERE toc_tsvector IS NOT NULL
              AND toc_tsvector @@ to_tsquery('simple', %s)
              {extra}
            ORDER BY score DESC
            LIMIT %s
        """
        params = [tsq, tsq, *extra_params, k]
        rows = execute_query(sql, tuple(params), fetch_all=True) or []
        for r in rows:
            _image_row_defaults(r)
            r["_source_retriever"] = self.retriever_id
        if rows:
            return rows
        loose = build_tsquery_loose(query)
        if not loose or loose == tsq:
            return []
        params2 = [loose, loose, *extra_params, k]
        rows = execute_query(sql, tuple(params2), fetch_all=True) or []
        for r in rows:
            _image_row_defaults(r)
            r["_source_retriever"] = self.retriever_id
        return rows


class ImageContentKeywordRetriever(BaseRetriever):
    retriever_id = RetrieverId.img_content_kw.value

    def retrieve(
        self,
        query: str,
        filters: IntentFilters,
        k: int,
        *,
        query_vec: list[float] | None = None,
    ) -> list[dict]:
        tsq = build_tsquery_strict(query)
        if not tsq:
            return []
        extra, extra_params = _build_image_filter_sql(filters.book_ids)
        sql = f"""
            SELECT image_id AS id, title, image_uri, local_path, alt_text, caption,
                   book_id, closest_title, toc_path, search_text, format,
                   ts_rank(ts_vector, to_tsquery('simple', %s)) AS score
            FROM image_chunks
            WHERE ts_vector IS NOT NULL
              AND ts_vector @@ to_tsquery('simple', %s)
              {extra}
            ORDER BY score DESC
            LIMIT %s
        """
        params = [tsq, tsq, *extra_params, k]
        rows = execute_query(sql, tuple(params), fetch_all=True) or []
        for r in rows:
            _image_row_defaults(r)
            r["_source_retriever"] = self.retriever_id
        if rows:
            return rows
        loose = build_tsquery_loose(query)
        if not loose or loose == tsq:
            return []
        params2 = [loose, loose, *extra_params, k]
        rows = execute_query(sql, tuple(params2), fetch_all=True) or []
        for r in rows:
            _image_row_defaults(r)
            r["_source_retriever"] = self.retriever_id
        return rows


class RelationRetriever(BaseRetriever):
    """占位：实际扩展在 relations.enrich 中按主结果 chunk id 拉取。"""

    retriever_id = RetrieverId.relation.value

    def retrieve(
        self,
        query: str,
        filters: IntentFilters,
        k: int,
        *,
        query_vec: list[float] | None = None,
    ) -> list[dict]:
        return []


RETRIEVER_REGISTRY: dict[str, BaseRetriever] = {
    RetrieverId.text_toc_kw.value: TextTitleKeywordRetriever(),
    RetrieverId.text_vec.value: TextContentVectorRetriever(),
    RetrieverId.text_kw.value: TextContentKeywordRetriever(),
    RetrieverId.img_toc_kw.value: ImageTitleKeywordRetriever(),
    RetrieverId.img_content_kw.value: ImageContentKeywordRetriever(),
    RetrieverId.relation.value: RelationRetriever(),
}


def get_main_retrievers(enabled_ids: list[str]) -> list[BaseRetriever]:
    """enabled 中若仅有 relation 或为空，则回退为五路主检索 id 顺序。"""
    mains = [rid for rid in enabled_ids if rid != RetrieverId.relation.value]
    if not mains:
        mains = [r.value for r in MAIN_RETRIEVER_IDS]
    out: list[BaseRetriever] = []
    for rid in mains:
        r = RETRIEVER_REGISTRY.get(rid)
        if r and r.retriever_id not in {x.retriever_id for x in out}:
            out.append(r)
    return out
