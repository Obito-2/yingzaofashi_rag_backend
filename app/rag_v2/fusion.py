"""五路加权 RRF；可配置 tie-break 主 lane；子查询间 RRF 合并。"""
from __future__ import annotations

from app.rag.retriever_helpers import RRF_K, make_fusion_key


def lane_weights_five() -> tuple[float, float, float, float, float]:
    """text_toc_kw, text_vec, text_kw, img_toc_kw, img_content_kw"""
    return (0.9, 1.0, 0.75, 0.25, 0.35)


def rrf_fuse_v2(
    result_lists: list[list[dict]],
    k_final: int,
    *,
    weights: list[float] | None = None,
    lane_names: list[str] | None = None,
    primary_lane_tiebreak: str = "text_vec",
) -> list[dict]:
    """
    加权 RRF；tie-break 主 lane 可配置（旧版写死 text_vec）。
    """
    n = len(result_lists)
    if weights is None:
        weights = [1.0] * n
    if lane_names is None:
        lane_names = [f"lane_{i}" for i in range(n)]
    if len(weights) != n or len(lane_names) != n:
        raise ValueError("weights/lane_names 与 result_lists 长度须一致")

    acc_rrf: dict[str, float] = {}
    row_map: dict[str, dict] = {}
    lane_scores: dict[str, dict[str, float]] = {}
    lane_ranks: dict[str, dict[str, int]] = {}

    for lane_name, weight, result_list in zip(lane_names, weights, result_lists):
        for rank, row in enumerate(result_list, start=1):
            key = make_fusion_key(row)
            contrib = weight * (1.0 / (RRF_K + rank))
            acc_rrf[key] = acc_rrf.get(key, 0.0) + contrib
            if key not in row_map:
                row_map[key] = row
            if key not in lane_scores:
                lane_scores[key] = {}
                lane_ranks[key] = {}
            sc = row.get("score")
            if sc is not None:
                lane_scores[key][lane_name] = float(sc)
            lane_ranks[key][lane_name] = rank

    def sort_key(key: str) -> tuple:
        rrf = acc_rrf[key]
        scores = lane_scores.get(key, {})
        mx = max(scores.values()) if scores else 0.0
        tv_r = lane_ranks.get(key, {}).get(primary_lane_tiebreak, 999)
        tv_boost = 1 if tv_r == 1 else 0
        return (-rrf, -mx, -tv_boost, key)

    sorted_keys = sorted(acc_rrf.keys(), key=sort_key)

    results: list[dict] = []
    for key in sorted_keys[:k_final]:
        row = dict(row_map[key])
        row["rrf_score"] = acc_rrf[key]
        row["_lane_scores"] = dict(lane_scores.get(key, {}))
        row["_lane_ranks"] = dict(lane_ranks.get(key, {}))
        lr = lane_ranks.get(key, {})
        if lr:
            best_lane = min(lr.items(), key=lambda x: x[1])[0]
        else:
            best_lane = primary_lane_tiebreak
        row["_primary_lane"] = best_lane
        row["_source_retriever"] = best_lane
        results.append(row)
    return results


def rrf_merge_subquery_results(
    per_sub_lists: list[list[dict]],
    k_final: int,
    *,
    weights: list[float] | None = None,
) -> list[dict]:
    """每个子查询一条已排序列表，当作多 lane 做 RRF。"""
    n = len(per_sub_lists)
    if not n:
        return []
    if weights is None:
        weights = [1.0] * n
    names = [f"sub_{i}" for i in range(n)]
    return rrf_fuse_v2(
        per_sub_lists,
        k_final,
        weights=weights,
        lane_names=names,
        primary_lane_tiebreak="sub_0",
    )
