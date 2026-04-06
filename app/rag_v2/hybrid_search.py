"""hybrid_search_v2 入口：意图解析、五路并行、子查询 RRF、relation 后置。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from langsmith import traceable

from app.connect import execute_query
from app.rag.embedding import embed_query
from app.rag_v2.fusion import lane_weights_five, rrf_fuse_v2, rrf_merge_subquery_results
from app.rag_v2.relations import enrich_main_with_relations, items_only_from_rows
from app.rag_v2.retrievers import RETRIEVER_REGISTRY, get_main_retrievers
from app.rag_v2.schemas import (
    MAIN_RETRIEVER_IDS,
    IntentFilters,
    RetrieverId,
    SingleIntentBlock,
    SubQueryItem,
    parse_intent_result,
)
from app.rag_v2.intent_llm import recognize_intent_llm

LANE_ORDER = [r.value for r in MAIN_RETRIEVER_IDS]


def _has_col_data(table: str, column: str) -> bool:
    row = execute_query(
        f"SELECT 1 FROM {table} WHERE {column} IS NOT NULL LIMIT 1",
        fetch_one=True,
    )
    return bool(row)


def _retriever_allowed(
    retriever_id: str,
    has_text_vec: bool,
    has_text_ts: bool,
    has_text_toc: bool,
    has_img_ts: bool,
    has_img_toc: bool,
) -> bool:
    if retriever_id == RetrieverId.text_vec.value:
        return has_text_vec
    if retriever_id == RetrieverId.text_kw.value:
        return has_text_ts
    if retriever_id == RetrieverId.text_toc_kw.value:
        return has_text_toc
    if retriever_id == RetrieverId.img_content_kw.value:
        return has_img_ts
    if retriever_id == RetrieverId.img_toc_kw.value:
        return has_img_toc
    return False


def _col_flags() -> tuple[bool, bool, bool, bool, bool]:
    has_text_vec = _has_col_data("text_chunks", "embedding_values")
    has_text_ts = _has_col_data("text_chunks", "ts_vector")
    has_text_toc = _has_col_data("text_chunks", "toc_tsvector")
    has_img_ts = _has_col_data("image_chunks", "ts_vector")
    has_img_toc = _has_col_data("image_chunks", "toc_tsvector")
    return has_text_vec, has_text_ts, has_text_toc, has_img_ts, has_img_toc


@traceable(name="rag_v2_parallel_main", run_type="retriever")
def _run_parallel_main(
    query: str,
    intent: SingleIntentBlock,
    k_per: int,
    k_final: int,
    query_vec: list[float] | None,
) -> tuple[list[dict], dict[str, Any]]:
    """五路并行 + RRF；返回融合行列表与 debug 片段。"""
    has_text_vec, has_text_ts, has_text_toc, has_img_ts, has_img_toc = _col_flags()
    enabled = [
        r
        for r in intent.enabled_retrievers
        if r != RetrieverId.relation.value
        and _retriever_allowed(
            r,
            has_text_vec,
            has_text_ts,
            has_text_toc,
            has_img_ts,
            has_img_toc,
        )
    ]
    retrievers = get_main_retrievers(enabled)
    if not retrievers:
        return [], {"enabled_retrievers": enabled, "note": "no_retriever_after_col_check"}
    effective_ids = [r.retriever_id for r in retrievers]

    need_vec = any(r.retriever_id == RetrieverId.text_vec.value for r in retrievers)
    if need_vec and not query_vec:
        query_vec = embed_query(query)
    elif not need_vec:
        query_vec = None

    f = intent.filters
    results_by_id: dict[str, list[dict]] = {}

    def run_one(ret) -> tuple[str, list[dict]]:
        lst = ret.retrieve(query, f, k_per, query_vec=query_vec)
        return ret.retriever_id, lst

    with ThreadPoolExecutor(max_workers=max(4, len(retrievers))) as ex:
        futs = [ex.submit(run_one, r) for r in retrievers]
        for fut in as_completed(futs):
            rid, lst = fut.result()
            results_by_id[rid] = lst

    result_lists: list[list[dict]] = []
    lane_names_active: list[str] = []
    weights_active: list[float] = []
    w_all = lane_weights_five()
    for i, lid in enumerate(LANE_ORDER):
        if lid not in results_by_id:
            continue
        result_lists.append(results_by_id[lid])
        lane_names_active.append(lid)
        weights_active.append(w_all[i])

    if not result_lists:
        return [], {"enabled_retrievers": enabled}

    fused = rrf_fuse_v2(
        result_lists,
        k_final,
        weights=weights_active,
        lane_names=lane_names_active,
        primary_lane_tiebreak=(
            RetrieverId.text_vec.value
            if RetrieverId.text_vec.value in lane_names_active
            else lane_names_active[0]
        ),
    )
    debug = {
        "enabled_retrievers": enabled,
        "effective_retrievers": effective_ids,
        "lane_sizes": {ln: len(results_by_id.get(ln, [])) for ln in LANE_ORDER},
    }
    return fused, debug


@traceable(name="hybrid_search_v2", run_type="retriever")
def hybrid_search_v2(
    query: str,
    intent_result: dict[str, Any] | None = None,
    k_per_retriever: int = 5,
    k_final: int = 10,
    with_relations: bool = False,
    *,
    refresh_book_cache: bool = False,
) -> dict[str, Any]:
    """
    多路检索 v2。intent_result 为 None 时五路全开、无 filters。
    复合问题：sub_queries 非空时对每条递归检索再 RRF 合并。

    非复合路径下仅使用 ``intents[0]``；多意图时请让上游输出 is_complex=true 并用 sub_queries 拆分。
    """
    payload = parse_intent_result(intent_result, refresh_book_cache=refresh_book_cache)
    q = query or payload.query

    # 复合：仅使用 sub_queries
    if payload.is_complex and payload.sub_queries:
        sub_lists: list[list[dict]] = []
        sub_debug: list[dict] = []
        for sq in payload.sub_queries:
            inner = _hybrid_single_subquery(sq, k_per_retriever, k_final)
            sub_lists.append(inner["rows"])
            sub_debug.append(inner.get("debug", {}))
        merged_rows = rrf_merge_subquery_results(sub_lists, k_final)
        return _finalize_output(
            q,
            merged_rows,
            payload.sub_queries[0].filters,
            with_relations=with_relations,
            intent_type="complex",
            relation_explicit=_any_sub_enabled_relation(payload.sub_queries),
            extra_debug={
                "sub_queries_debug": sub_debug,
                "intent": payload.model_dump(),
            },
        )

    intent0 = payload.intents[0]
    rows, dbg = _run_parallel_main(
        q, intent0, k_per_retriever, k_final, query_vec=None
    )
    return _finalize_output(
        q,
        rows,
        intent0.filters,
        with_relations=with_relations,
        intent_type=intent0.type,
        relation_explicit=RetrieverId.relation.value
        in intent0.enabled_retrievers,
        extra_debug={"single_debug": dbg, "intent": payload.model_dump()},
    )


def _hybrid_single_subquery(
    sq: SubQueryItem,
    k_per: int,
    k_final: int,
) -> dict[str, Any]:
    intent = SingleIntentBlock(
        type=sq.type,
        filters=sq.filters,
        enabled_retrievers=sq.enabled_retrievers,
    )
    rows, dbg = _run_parallel_main(
        sq.query, intent, k_per, k_final, query_vec=None
    )
    return {"rows": rows, "debug": dbg}


def _any_sub_enabled_relation(subs: list[SubQueryItem]) -> bool:
    for s in subs:
        if RetrieverId.relation.value in s.enabled_retrievers:
            return True
    return False


def _finalize_output(
    query: str,
    main_rows: list[dict],
    filters: Any,
    *,
    with_relations: bool,
    intent_type: str,
    relation_explicit: bool,
    extra_debug: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(filters, IntentFilters):
        filters = IntentFilters()

    want_rel = with_relations or relation_explicit
    if not main_rows:
        out: dict[str, Any] = {
            "items": [],
            "relations": [],
            "debug_info": {"query": query, **extra_debug},
        }
        return out

    if want_rel:
        enriched = enrich_main_with_relations(
            main_rows,
            filters,
            intent_type=intent_type,
        )
    else:
        enriched = items_only_from_rows(main_rows, intent_type=intent_type)

    # 主路径也写入 source_retriever：融合行已有 _source_retriever
    enriched["debug_info"] = {
        "query": query,
        "intent_type": intent_type,
        **extra_debug,
    }
    return enriched


@traceable(name="hybrid_search_v2_with_llm", run_type="retriever")
def hybrid_search_v2_with_llm(
    query: str,
    *,
    use_llm: bool = True,
    k_per_retriever: int = 5,
    k_final: int = 10,
    with_relations: bool = False,
    refresh_book_cache: bool = False,
) -> dict[str, Any]:
    """
    先经 ``recognize_intent_llm`` 得到意图 JSON，再调用 :func:`hybrid_search_v2`。

    - ``use_llm=False``：等价于无意图默认（五路全开）。
    - LLM 失败或返回空对象时与 ``intent_result=None`` 行为一致。

    非复合路径仅使用 ``intents[0]``；复合问题需 LLM 输出 ``is_complex=true`` 并填充 ``sub_queries``。
    """
    if not use_llm:
        return hybrid_search_v2(
            query,
            intent_result=None,
            k_per_retriever=k_per_retriever,
            k_final=k_final,
            with_relations=with_relations,
            refresh_book_cache=refresh_book_cache,
        )
    raw = recognize_intent_llm(query)
    intent_in: dict[str, Any] | None = raw if raw else None
    return hybrid_search_v2(
        query,
        intent_result=intent_in,
        k_per_retriever=k_per_retriever,
        k_final=k_final,
        with_relations=with_relations,
        refresh_book_cache=refresh_book_cache,
    )
