# app/rag/retriever_helpers.py
"""检索辅助：查询分词/tsquery 构造、意图检测、RRF 融合与后处理（与具体 SQL 无关）。"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import jieba
from langsmith import traceable

_DICT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "resources", "jieba_userdict.txt"
)
if os.path.exists(_DICT_PATH):
    jieba.load_userdict(_DICT_PATH)

_STOP_RE = re.compile(r"^[\s\u3000\W]+$", re.UNICODE)

RRF_K = 60

# 与 app/rag/__init__.py 中 content_type_label_map 键一致（DB 存英文枚举）
ORIGINAL_TEXT_CONTENT_TYPE = "original_text"

_SEARCH_STOPWORDS = frozenset(
    """
    的 了 和 与 或 及 等 在 是 有 为 以 与 其 这 那 一个 什么 哪些 如何 怎么 吗 呢 吧 啊
    有关 关于 请问 请 问 吗 是否 可以 能否 相关 一些 以及 还有 还是 或者 如果 就 也 都 而
    要 会 能 可 把 被 从 到 对 于 由 将 与 之 亦 即 又 及 或 但 并 且 若 则 所 着 给 让 向
    """.split()
)

_RE_BOOK = re.compile(r"《([^》]{1,50})》")
_RE_ZHIZHI = re.compile(r"[\u4e00-\u9fff]{2,15}之制")
_RE_JUAN = re.compile(r"第[一二三四五六七八九十百千0-9]+卷")


@dataclass
class QueryIntent:
    wants_original_text: bool
    wants_institution: bool


@traceable(name="jieba_query_tokenize", run_type="tool")
def tokenize_query_display(text: str) -> list[str]:
    """展示/调试用分词（不过滤停用词）。"""
    return [w for w in jieba.cut(text) if not _STOP_RE.match(w)]


def tokenize_for_search(text: str) -> list[str]:
    """检索用分词：jieba + 去停用词 + 去重保序。"""
    seen: set[str] = set()
    out: list[str] = []
    for w in jieba.cut(text):
        if _STOP_RE.match(w) or w in _SEARCH_STOPWORDS:
            continue
        if len(w.strip()) == 0:
            continue
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def tsquery_lex_escape(term: str) -> str:
    """PostgreSQL tsquery 词元转义。"""
    t = term.replace("'", "''").strip()
    if not t:
        return "''"
    return "'" + t + "'"


def extract_must_terms(query: str) -> list[str]:
    """从问句中提取应加强匹配的短语：《书名》、…之制。"""
    terms: list[str] = []
    for m in _RE_BOOK.finditer(query):
        t = m.group(1).strip()
        if len(t) >= 2:
            terms.append(t)
    for m in _RE_ZHIZHI.finditer(query):
        terms.append(m.group(0))
    dedup: list[str] = []
    seen: set[str] = set()
    for t in terms:
        if t not in seen:
            seen.add(t)
            dedup.append(t)
    return dedup


def _subterms_for_must(term: str) -> list[str]:
    parts = [w for w in jieba.cut(term) if not _STOP_RE.match(w) and w not in _SEARCH_STOPWORDS]
    if not parts:
        parts = [term]
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _must_group_sql(term: str) -> str | None:
    subs = _subterms_for_must(term)
    if not subs:
        return None
    if len(subs) == 1:
        return tsquery_lex_escape(subs[0])
    return "(" + " & ".join(tsquery_lex_escape(s) for s in subs) + ")"


def build_tsquery_strict(query: str) -> str | None:
    """必选短语 AND + 其余词 OR；无必选时等价于纯 OR。"""
    must_terms = extract_must_terms(query)
    tokens = tokenize_for_search(query)
    if not tokens and not must_terms:
        return None

    must_groups: list[str] = []
    for t in must_terms:
        g = _must_group_sql(t)
        if g:
            must_groups.append(g)

    opt_tokens = list(tokens)
    if must_groups:
        core = " & ".join(must_groups)
        if opt_tokens:
            opt_part = " | ".join(tsquery_lex_escape(t) for t in opt_tokens)
            return f"({core}) & ({opt_part})"
        return core
    if opt_tokens:
        return " | ".join(tsquery_lex_escape(t) for t in opt_tokens)
    return None


def build_tsquery_loose(query: str) -> str | None:
    """纯 OR，用于严格查询无结果时的回退。"""
    tokens = tokenize_for_search(query)
    if not tokens:
        return None
    return " | ".join(tsquery_lex_escape(t) for t in tokens)


def detect_query_intent(query: str) -> QueryIntent:
    wants_original = "原文" in query
    has_inst = bool(_RE_ZHIZHI.search(query) or _RE_JUAN.search(query))
    return QueryIntent(wants_original_text=wants_original, wants_institution=has_inst)


# ---------- RRF 与后处理 ----------


def make_fusion_key(row: dict) -> str:
    """用 type:id 作为去重键，避免跨表 ID 碰撞"""
    return f"{row['type']}:{row['id']}"


def lane_weights(has_image_vector: bool) -> tuple[float, float, float, float]:
    """text_vec, text_kw, img_vec, img_kw"""
    w_ik = 0.35 if has_image_vector else 0.2
    return (1.0, 0.75, 1.0, w_ik)


def merge_text_vector_lanes(
    primary: list[dict], secondary: list[dict], k: int
) -> list[dict]:
    """先去重保留 primary 顺序，再补 secondary，截断为 k。"""
    seen: set[str] = set()
    out: list[dict] = []
    for row in primary + secondary:
        key = row["id"]
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
        if len(out) >= k:
            break
    return out


def rrf_fuse(
    result_lists: list[list[dict]],
    k_final: int,
    *,
    weights: list[float] | None = None,
    lane_names: list[str] | None = None,
) -> list[dict]:
    """
    加权 RRF；写入 tie-break 用字段：rrf_score, _lane_scores, _lane_ranks。
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
        tv_r = lane_ranks.get(key, {}).get("text_vec", 999)
        tv_boost = 1 if tv_r == 1 else 0
        return (-rrf, -mx, -tv_boost, key)

    sorted_keys = sorted(acc_rrf.keys(), key=sort_key)

    results: list[dict] = []
    for key in sorted_keys[:k_final]:
        row = dict(row_map[key])
        row["rrf_score"] = acc_rrf[key]
        row["_lane_scores"] = dict(lane_scores.get(key, {}))
        row["_lane_ranks"] = dict(lane_ranks.get(key, {}))
        results.append(row)
    return results


def apply_original_text_boost(rows: list[dict], intent: QueryIntent) -> list[dict]:
    """「原文」意图下略抬 original_text。"""
    if not intent.wants_original_text:
        return rows
    boosted: list[tuple[float, dict]] = []
    for row in rows:
        s = float(row.get("rrf_score") or 0.0)
        if row.get("type") == "text" and row.get("content_type") == ORIGINAL_TEXT_CONTENT_TYPE:
            s *= 1.12
        boosted.append((s, row))
    boosted.sort(key=lambda x: -x[0])
    for s, row in boosted:
        row["rrf_score"] = s
    return [row for _, row in boosted]


def apply_image_slot_limit(
    rows: list[dict], k_final: int, max_images: int
) -> list[dict]:
    """在保持相对顺序下，限制图片条数，不足 k_final 时从剩余行继续取。"""
    if max_images >= k_final:
        return rows[:k_final]
    picked: list[dict] = []
    picked_ids: set[str] = set()
    n_img = 0

    def kid(r: dict) -> str:
        return f"{r.get('type')}:{r.get('id')}"

    for r in rows:
        if len(picked) >= k_final:
            break
        if kid(r) in picked_ids:
            continue
        if r.get("type") == "image" and n_img >= max_images:
            continue
        picked.append(r)
        picked_ids.add(kid(r))
        if r.get("type") == "image":
            n_img += 1

    if len(picked) < k_final:
        for r in rows:
            if len(picked) >= k_final:
                break
            if kid(r) in picked_ids:
                continue
            if r.get("type") == "image" and n_img >= max_images:
                continue
            picked.append(r)
            picked_ids.add(kid(r))
            if r.get("type") == "image":
                n_img += 1

    return picked[:k_final]


def strip_fuse_debug_fields(row: dict) -> None:
    row.pop("_lane_scores", None)
    row.pop("_lane_ranks", None)
