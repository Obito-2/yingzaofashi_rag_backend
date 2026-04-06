"""
检索器离线评测：读取 jsonl（query + evidence_chunk_id），调用检索入口，
计算 HR@3/5、MRR、NDCG@3/5，分层统计，检索延迟汇总。

- 默认 --backend v1：hybrid_search（四路，含 img_vec）。
- --backend v2：hybrid_search_v2；可加 --intent-llm 先走 LLM 意图再检索。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

_here = Path(__file__).resolve().parent
PROJECT_ROOT = _here.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from app.rag.retriever import hybrid_search
from app.rag_v2.hybrid_search import hybrid_search_v2, hybrid_search_v2_with_llm

K_VECTOR = 5
K_KEYWORD = 5
K_FINAL = 5
K_PER_RETRIEVER = 5


def _log2(x: float) -> float:
    return math.log(x, 2.0)


def _rank_of_gt(ids: list[str], gt: str) -> int | None:
    try:
        return ids.index(gt) + 1
    except ValueError:
        return None


def _ndcg_at_k(rank: int | None, k: int) -> float:
    """单相关文档、rel∈{0,1}；理想情况排在第 1 位，IDCG@K = 1/log2(2) = 1。"""
    if rank is None or rank > k:
        return 0.0
    idcg = 1.0 / _log2(2.0)  # 1.0
    dcg = 1.0 / _log2(float(rank + 1))
    return dcg / idcg


def _mrr(rank: int | None, max_rank: int) -> float:
    if rank is None or rank > max_rank:
        return 0.0
    return 1.0 / float(rank)


def _hr_at_k(rank: int | None, k: int) -> float:
    if rank is None:
        return 0.0
    return 1.0 if rank <= k else 0.0


def _percentile_linear(sorted_vals: list[float], p: float) -> float:
    """p ∈ [0, 100]，线性插值；与 numpy percentile 行为接近。"""
    if not sorted_vals:
        return 0.0
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    idx = (n - 1) * (p / 100.0)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (idx - lo) * (sorted_vals[hi] - sorted_vals[lo])


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _safe_run_name(name: str) -> str:
    """用于子目录名，避免路径穿越与非法字符。"""
    s = name.strip()
    if not s:
        return "run"
    s = s.replace("..", "_").replace("/", "_").replace("\\", "_")
    s = re.sub(r'[<>:"|?*]', "_", s)
    return (s[:120] or "run").strip("._") or "run"


def _aggregate_metrics(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "hr3": 0.0,
            "hr5": 0.0,
            "mrr": 0.0,
            "ndcg3": 0.0,
            "ndcg5": 0.0,
        }
    return {
        "n": n,
        "hr3": _mean([float(r["hr3"]) for r in rows]),
        "hr5": _mean([float(r["hr5"]) for r in rows]),
        "mrr": _mean([float(r["mrr"]) for r in rows]),
        "ndcg3": _mean([float(r["ndcg3"]) for r in rows]),
        "ndcg5": _mean([float(r["ndcg5"]) for r in rows]),
    }


def _latency_summary(latencies_s: list[float]) -> dict:
    if not latencies_s:
        return {
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "n": 0,
        }
    ms = [x * 1000.0 for x in latencies_s]
    ms_sorted = sorted(ms)
    return {
        "mean_ms": _mean(ms),
        "p50_ms": statistics.median(ms),
        "p95_ms": _percentile_linear(ms_sorted, 95.0),
        "min_ms": min(ms),
        "max_ms": max(ms),
        "n": len(ms),
    }


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_report_md(
    path: Path,
    summary: dict,
) -> None:
    o = summary["overall"]
    lat = summary["latency_ms"]
    ret = summary["retrieval"]
    backend = ret.get("backend", "v1")
    if backend == "v2":
        k_line = (
            f"- **backend** = v2，**intent_llm** = {ret.get('intent_llm', False)}，"
            f"`k_per_retriever` = {ret.get('k_per_retriever')}, `k_final` = {ret.get('k_final')}"
        )
        note = ret.get("note_compare")
        if note:
            k_line += f"\n- {note}"
    else:
        k_line = (
            f"- **backend** = v1，`k_vector` = {ret['k_vector']}, "
            f"`k_keyword` = {ret['k_keyword']}, `k_final` = {ret['k_final']}"
        )
    lines = [
        "# 检索器离线评测报告（自动生成）",
        "",
        "## 实验设置",
        "",
        f"- 样本数 **N** = {summary['n']}",
        k_line,
        "- `with_relations` = false",
        "",
        "## 准确性（总体）",
        "",
        "| 指标 | 值 |",
        "|------|-----|",
        f"| HR@3 | {o['hr3']:.4f} |",
        f"| HR@5 | {o['hr5']:.4f} |",
        f"| MRR | {o['mrr']:.4f} |",
        f"| NDCG@3 | {o['ndcg3']:.4f} |",
        f"| NDCG@5 | {o['ndcg5']:.4f} |",
        "",
        "## 分层：query_type",
        "",
        "| query_type | N | HR@3 | HR@5 | MRR | NDCG@3 | NDCG@5 |",
        "|------------|---|------|------|-----|--------|--------|",
    ]
    for qt in sorted(summary["by_query_type"].keys()):
        m = summary["by_query_type"][qt]
        lines.append(
            f"| {qt} | {m['n']} | {m['hr3']:.4f} | {m['hr5']:.4f} | {m['mrr']:.4f} | {m['ndcg3']:.4f} | {m['ndcg5']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 分层：content_type",
            "",
            "| content_type | N | HR@3 | HR@5 | MRR | NDCG@3 | NDCG@5 |",
            "|--------------|---|------|------|-----|--------|--------|",
        ]
    )
    for ct in sorted(summary["by_content_type"].keys()):
        m = summary["by_content_type"][ct]
        lines.append(
            f"| {ct} | {m['n']} | {m['hr3']:.4f} | {m['hr5']:.4f} | {m['mrr']:.4f} | {m['ndcg3']:.4f} | {m['ndcg5']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 检索延迟（毫秒）",
            "",
            "| mean | P50 | P95 | min | max | 次数 |",
            "|------|-----|-----|-----|-----|------|",
            f"| {lat['mean_ms']:.2f} | {lat['p50_ms']:.2f} | {lat['p95_ms']:.2f} | {lat['min_ms']:.2f} | {lat['max_ms']:.2f} | {lat['n']} |",
            "",
        ]
    )
    miss_n = summary.get("missed_top5_count", 0)
    examples = summary.get("missed_examples") or []
    lines.extend(
        [
            "## 未命中（GT 不在融合前 5 条）",
            "",
            f"- 条数：**{miss_n}** / {summary['n']}",
            "",
        ]
    )
    if examples:
        lines.append("| # | query（节选） | evidence_chunk_id |")
        lines.append("|---|---------------|-------------------|")
        for i, ex in enumerate(examples, 1):
            q = ex.get("query", "")
            if len(q) > 80:
                q = q[:80] + "…"
            gid = ex.get("evidence_chunk_id", "")
            lines.append(f"| {i} | {q} | `{gid}` |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _run_retrieval(
    query: str,
    *,
    backend: str,
    intent_llm: bool,
    k_vector: int,
    k_keyword: int,
    k_final: int,
    k_per_retriever: int,
) -> dict:
    if backend == "v1":
        return hybrid_search(
            query,
            k_vector=k_vector,
            k_keyword=k_keyword,
            k_final=k_final,
            with_relations=False,
        )
    if intent_llm:
        return hybrid_search_v2_with_llm(
            query,
            use_llm=True,
            k_per_retriever=k_per_retriever,
            k_final=k_final,
            with_relations=False,
        )
    return hybrid_search_v2(
        query,
        intent_result=None,
        k_per_retriever=k_per_retriever,
        k_final=k_final,
        with_relations=False,
    )


def run_eval(
    input_path: Path,
    detail_jsonl: Path,
    detail_csv: Path | None,
    summary_json: Path,
    report_md: Path | None,
    *,
    backend: str = "v1",
    intent_llm: bool = False,
    k_vector: int = K_VECTOR,
    k_keyword: int = K_KEYWORD,
    k_final: int = K_FINAL,
    k_per_retriever: int = K_PER_RETRIEVER,
) -> dict:
    records = _load_jsonl(input_path)
    detail_rows: list[dict] = []
    latencies: list[float] = []

    by_qt: dict[str, list[dict]] = defaultdict(list)
    by_ct: dict[str, list[dict]] = defaultdict(list)

    for i, rec in enumerate(records, 1):
        query = (rec.get("query") or "").strip()
        gt = str(rec.get("evidence_chunk_id") or "").strip()
        qtype = str(rec.get("query_type") or "").strip() or "(empty)"
        ctype = str(rec.get("content_type") or "").strip() or "(empty)"

        if not query or not gt:
            row = {
                "index": i,
                "query": query,
                "evidence_chunk_id": gt,
                "query_type": qtype,
                "content_type": ctype,
                "error": "missing_query_or_gt",
                "retrieved_ids": [],
                "rank": None,
                "hr3": 0.0,
                "hr5": 0.0,
                "mrr": 0.0,
                "ndcg3": 0.0,
                "ndcg5": 0.0,
                "latency_s": 0.0,
            }
            detail_rows.append(row)
            by_qt[qtype].append(row)
            by_ct[ctype].append(row)
            continue

        t0 = time.perf_counter()
        result = _run_retrieval(
            query,
            backend=backend,
            intent_llm=intent_llm,
            k_vector=k_vector,
            k_keyword=k_keyword,
            k_final=k_final,
            k_per_retriever=k_per_retriever,
        )
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

        items = result.get("items") or []
        ids = [str(it.get("id")) for it in items if it.get("id") is not None]
        rank = _rank_of_gt(ids, gt)

        row = {
            "index": i,
            "query": query,
            "evidence_chunk_id": gt,
            "query_type": qtype,
            "content_type": ctype,
            "retrieved_ids": ids,
            "rank": rank,
            "hr3": _hr_at_k(rank, 3),
            "hr5": _hr_at_k(rank, 5),
            "mrr": _mrr(rank, k_final),
            "ndcg3": _ndcg_at_k(rank, 3),
            "ndcg5": _ndcg_at_k(rank, 5),
            "latency_s": round(elapsed, 6),
        }
        detail_rows.append(row)
        by_qt[qtype].append(row)
        by_ct[ctype].append(row)

    overall = _aggregate_metrics(detail_rows)
    missed = [
        r
        for r in detail_rows
        if r.get("rank") is None and not r.get("error")
    ]
    missed_examples = [
        {"query": r["query"], "evidence_chunk_id": r["evidence_chunk_id"]}
        for r in missed[:8]
    ]
    if backend == "v2":
        retrieval_meta = {
            "backend": "v2",
            "intent_llm": intent_llm,
            "k_per_retriever": k_per_retriever,
            "k_final": k_final,
            "with_relations": False,
            "note_compare": "v2 不含 img_vec 路，与 v1 四路混合检索指标仅作参考对比",
        }
    else:
        retrieval_meta = {
            "backend": "v1",
            "k_vector": k_vector,
            "k_keyword": k_keyword,
            "k_final": k_final,
            "with_relations": False,
        }
    summary = {
        "n": len(detail_rows),
        "input": str(input_path.resolve()),
        "retrieval": retrieval_meta,
        "overall": {k: v for k, v in overall.items() if k != "n"},
        "overall_n": overall["n"],
        "missed_top5_count": len(missed),
        "missed_examples": missed_examples,
        "by_query_type": {k: _aggregate_metrics(v) for k, v in sorted(by_qt.items())},
        "by_content_type": {k: _aggregate_metrics(v) for k, v in sorted(by_ct.items())},
        "latency_ms": _latency_summary(latencies),
    }

    detail_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with detail_jsonl.open("w", encoding="utf-8") as f:
        for row in detail_rows:
            out = dict(row)
            # JSON 序列化：query 可能含换行
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    if detail_csv:
        detail_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "index",
            "query",
            "evidence_chunk_id",
            "query_type",
            "content_type",
            "rank",
            "hr3",
            "hr5",
            "mrr",
            "ndcg3",
            "ndcg5",
            "latency_s",
            "retrieved_ids",
        ]
        with detail_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for row in detail_rows:
                r = dict(row)
                r["retrieved_ids"] = json.dumps(r.get("retrieved_ids") or [], ensure_ascii=False)
                if "error" in row:
                    r.setdefault("rank", "")
                w.writerow(r)

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if report_md:
        _write_report_md(report_md, summary)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="检索器离线评测（HR@3/5、MRR、NDCG@3/5 + 分层 + 延迟）")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "out.jsonl",
        help="输入 jsonl（含 query、evidence_chunk_id）",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        metavar="NAME",
        help="批次名：未单独指定各输出路径时，写入 exper_data/<NAME>/，避免覆盖历史结果",
    )
    parser.add_argument(
        "--detail-jsonl",
        type=Path,
        default=None,
        help="逐条明细 JSONL 输出路径（默认：exper_data[/批次]/retriever_eval_detail.jsonl）",
    )
    parser.add_argument(
        "--detail-csv",
        type=Path,
        default=None,
        help="逐条明细 CSV 输出路径（默认：exper_data[/批次]/retriever_eval_detail.csv）",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="汇总 JSON（默认：exper_data[/批次]/retriever_eval_summary.json）",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=None,
        help="Markdown 报告（默认：exper_data[/批次]/retriever_eval_report.md）",
    )
    parser.add_argument("--no-report", action="store_true", help="不生成 Markdown 报告")
    parser.add_argument("--no-csv", action="store_true", help="不生成 CSV 明细")
    parser.add_argument(
        "--backend",
        choices=["v1", "v2"],
        default="v1",
        help="检索后端：v1=hybrid_search（四路含 img_vec）；v2=hybrid_search_v2（五路关键词+向量，无 img_vec）",
    )
    parser.add_argument(
        "--intent-llm",
        action="store_true",
        help="仅 --backend v2：先 LLM 意图识别再检索（耗时含意图 API）",
    )
    parser.add_argument("--k-final", type=int, default=K_FINAL, help="融合后取前 k 条（v1/v2 共用）")
    parser.add_argument(
        "--k-per-retriever",
        type=int,
        default=K_PER_RETRIEVER,
        help="v2 每路检索条数（--backend v2）",
    )
    parser.add_argument("--k-vector", type=int, default=K_VECTOR, help="v1 向量路每路条数")
    parser.add_argument("--k-keyword", type=int, default=K_KEYWORD, help="v1 关键词路每路条数")
    args = parser.parse_args()

    if args.intent_llm and args.backend != "v2":
        parser.error("--intent-llm 仅可与 --backend v2 同用")

    out_base = PROJECT_ROOT / "exper_reports"
    if args.run_name:
        out_base = out_base / _safe_run_name(args.run_name)

    detail_jsonl = args.detail_jsonl or (out_base / "retriever_eval_detail.jsonl")
    detail_csv = args.detail_csv or (out_base / "retriever_eval_detail.csv")
    summary_json = args.summary_json or (out_base / "retriever_eval_summary.json")
    report_md = args.report_md or (out_base / "retriever_eval_report.md")

    if args.run_name:
        print(f"输出目录（批次）: {out_base}", file=sys.stderr)

    csv_path = None if args.no_csv else detail_csv
    report = None if args.no_report else report_md

    summary = run_eval(
        input_path=args.input,
        detail_jsonl=detail_jsonl,
        detail_csv=csv_path,
        summary_json=summary_json,
        report_md=report,
        backend=args.backend,
        intent_llm=args.intent_llm,
        k_vector=args.k_vector,
        k_keyword=args.k_keyword,
        k_final=args.k_final,
        k_per_retriever=args.k_per_retriever,
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
