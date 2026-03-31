"""
RAG 检索器评测用伪查询数据集构建脚本。

环境变量（与 app/agent/nodes.py 一致）:
  SEU_API_KEY 或 DASHSCOPE_API_KEY — API Key
  SEU_BASE_URL 或 DASHSCOPE_BASE_URL — 兼容 OpenAI 的 Base URL
  CHAT_MODEL_NAME — 对话模型；也可用 EXPERIMENT_LLM_MODEL 覆盖本脚本专用模型
  DB_URL — PostgreSQL 连接串（覆盖 app/connect.py 默认值）

输出 JSONL 每行字段:
  query — 伪查询（≤max-query-chars，默认 50）
  query_type — 六类意图之一
  evidence_chunk_id — 金标 text chunk_id 或 image image_id
  evidence_kind — \"text\" 或 \"image\"
  source_book — documents.name
  content_type — 原文|译文|解读|注释|其他|image

用法: python scripts/build_retriever_experiment_dataset.py --out data/retriever_experiment.jsonl

并发与容错:
  --llm-concurrency 限制同时进行的 LLM 请求数，减轻限流与失败率。
  --llm-retries / --retry-backoff 控制单条失败后的重试与线性退避等待。
  仍失败的 chunk 会记录 [skip] 并跳过，不中断整体任务。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from app.connect import execute_query

# --- 六类意图（与计划一致）---
QUERY_TYPES: tuple[str, ...] = (
    "生僻字查询",
    "术语名词查询",
    "查找原文/文言文翻译",
    "文搜索图",
    "复合问题",
    "查找指定书籍/作者内容",
)

# text_chunks.content_type -> JSONL 展示名
TEXT_CONTENT_TYPE_LABEL: dict[str | None, str] = {
    "original_text": "原文",
    "modern_translation": "译文",
    "interpretation": "解读",
    "annotation": "注释",
    "others_text": "其他文本",
    None: "不明",
}

SYSTEM_PROMPT_TEXT = """你是《营造法式》与传统木构建筑知识库的测试数据构造助手。
根据下面给出的「单条文本块上下文」，生成 1～3 条模拟真实用户的检索问句（伪查询）。

硬性要求：
1. 每条伪查询必须能从该上下文中找到依据；不得编造上下文中未出现的书籍名、作者、术语、原文或事实。
2. 多条伪查询应针对正文的不同片段或不同维度（如字词、术语、原文句、与章节/书名相关的问题等），避免同义查询反复。
3. 每条 query 长度不超过 {max_q} 个字符（中文一字算一字符）。
4. query_type 必须从以下六类中择一（字符串完全一致）：
   {types_line}
5. 只输出一个 JSON 对象，不要 markdown 代码块，不要其它说明文字。格式：
{{"queries":[{{"query":"...","query_type":"..."}},...]}}
"""

SYSTEM_PROMPT_IMAGE = """你是《营造法式》与传统木构建筑知识库的测试数据构造助手。
下面给出的是「单条图像块」的检索用文本说明（search_text）。请根据该说明生成 1～3 条模拟用户会输入的检索问句（伪查询），可侧重「文搜索图」等能从说明中支撑的类型。

硬性要求：
1. 每条伪查询必须能从该说明中找到依据；不得编造说明中未出现的书籍、作者或图像内容。
2. 多条伪查询角度尽量多样化。
3. 每条 query 长度不超过 {max_q} 个字符。
4. query_type 必须从以下六类中择一（字符串完全一致）：
   {types_line}
5. 只输出一个 JSON 对象，不要 markdown 代码块。格式：
{{"queries":[{{"query":"...","query_type":"..."}},...]}}
"""


def _seu_api_key() -> str:
    return os.getenv("SEU_API_KEY") or os.getenv("DASHSCOPE_API_KEY", "")


def _seu_base_url() -> str:
    return os.getenv("SEU_BASE_URL") or os.getenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


def _experiment_llm() -> ChatOpenAI:
    model = (os.getenv("EXPERIMENT_LLM_MODEL") or "").strip() or os.getenv(
        "CHAT_MODEL_NAME", ""
    )
    return ChatOpenAI(
        model=model,
        api_key=SecretStr(_seu_api_key()),
        base_url=_seu_base_url(),
        temperature=0.3,
    )


def normalize_text_content_type(raw: str | None) -> str:
    if not raw:
        return "其他"
    return TEXT_CONTENT_TYPE_LABEL.get(raw, "其他")


def _sample_size(n: int, ratio: float, min_when_nonzero: int) -> int:
    if n <= 0:
        return 0
    k = max(min_when_nonzero, int(n * ratio))
    return min(k, n)


def _group_ids_by_book(
    rows: list[dict[str, Any]], id_key: str
) -> dict[str, list[str]]:
    by_book: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        bid = r.get("book_id")
        if not bid:
            continue
        by_book[str(bid)].append(str(r[id_key]))
    return dict(by_book)


def fetch_document_names() -> dict[str, str]:
    rows = execute_query(
        'SELECT id, name FROM "documents";', fetch_all=True
    )
    if not rows:
        return {}
    return {str(r["id"]): (r.get("name") or "") for r in rows}


def fetch_text_chunk_index() -> dict[str, list[str]]:
    rows = execute_query(
        'SELECT chunk_id, book_id FROM "text_chunks" WHERE book_id IS NOT NULL;',
        fetch_all=True,
    )
    return _group_ids_by_book(rows or [], "chunk_id")


def fetch_image_chunk_index() -> dict[str, list[str]]:
    rows = execute_query(
        'SELECT image_id, book_id FROM "image_chunks" WHERE book_id IS NOT NULL;',
        fetch_all=True,
    )
    return _group_ids_by_book(rows or [], "image_id")


def fetch_text_chunks_full(chunk_ids: list[str]) -> list[dict[str, Any]]:
    if not chunk_ids:
        return []
    ph = ",".join(["%s"] * len(chunk_ids))
    rows = execute_query(
        f"""
        SELECT t.chunk_id, t.book_id, t.main_text, t.content_type,
               t.closest_title, t.toc_path, d.name AS source_book
        FROM text_chunks t
        LEFT JOIN documents d ON t.book_id = d.id
        WHERE t.chunk_id IN ({ph});
        """,
        tuple(chunk_ids),
        fetch_all=True,
    )
    return list(rows or [])


def fetch_image_chunks_full(image_ids: list[str]) -> list[dict[str, Any]]:
    if not image_ids:
        return []
    ph = ",".join(["%s"] * len(image_ids))
    rows = execute_query(
        f"""
        SELECT i.image_id, i.book_id, i.search_text, i.title, i.caption,
               d.name AS source_book
        FROM image_chunks i
        LEFT JOIN documents d ON i.book_id = d.id
        WHERE i.image_id IN ({ph});
        """,
        tuple(image_ids),
        fetch_all=True,
    )
    return list(rows or [])


def build_text_context(row: dict[str, Any], max_chars: int) -> str:
    parts: list[str] = []
    sb = row.get("source_book") or ""
    if sb:
        parts.append(f"书名/来源：{sb}")
    ct = row.get("content_type")
    parts.append(f"内容类型（库内枚举）：{ct or '未知'}")
    title = row.get("closest_title")
    if title:
        parts.append(f"章节标题：{title}")
    toc = row.get("toc_path")
    if toc:
        if isinstance(toc, list):
            parts.append(f"目录路径：{' / '.join(str(x) for x in toc)}")
        else:
            parts.append(f"目录路径：{toc}")
    body = row.get("main_text") or ""
    if len(body) > max_chars:
        body = body[:max_chars] + "\n…（正文已截断）"
    parts.append(f"正文：\n{body}")
    return "\n".join(parts)


def build_image_context(row: dict[str, Any], max_chars: int) -> str:
    """PRD：图像块仅将 search_text 作为 LLM 上下文。"""
    st = (row.get("search_text") or "").strip()
    if len(st) > max_chars:
        st = st[:max_chars] + "…"
    return st


def _msg_content_str(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and isinstance(p.get("text"), str):
                parts.append(p["text"])
            else:
                parts.append(str(p))
        return "".join(parts)
    return str(content)


def strip_json_fence(raw: str) -> str:
    t = raw.strip()
    m = re.match(r"^```(?:json)?\s*([\s\S]*?)```\s*$", t, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    if t.startswith("```"):
        lines = t.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t


def parse_llm_queries(raw: str) -> list[dict[str, str]]:
    text = strip_json_fence(raw)
    data = json.loads(text)
    arr = data.get("queries")
    if not isinstance(arr, list):
        raise ValueError("missing queries array")
    out: list[dict[str, str]] = []
    for item in arr:
        if not isinstance(item, dict):
            continue
        q = item.get("query")
        qt = item.get("query_type")
        if not isinstance(q, str) or not isinstance(qt, str):
            continue
        out.append({"query": q.strip(), "query_type": qt.strip()})
    return out


def validate_and_trim_query(
    q: str, max_len: int, log: list[str]
) -> str | None:
    s = q.strip()
    if not s:
        return None
    if len(s) > max_len:
        log.append(f"trim query {len(s)}->{max_len} chars")
        return s[:max_len]
    return s


def run_llm_text(
    llm: ChatOpenAI,
    context: str,
    max_q: int,
    retries: int,
    retry_backoff: float = 0.0,
) -> list[dict[str, str]]:
    types_line = "、".join(QUERY_TYPES)
    sys_content = SYSTEM_PROMPT_TEXT.format(
        max_q=max_q, types_line=types_line
    )
    user_content = f"【文本块上下文】\n{context}"
    messages = [
        SystemMessage(content=sys_content),
        HumanMessage(content=user_content),
    ]
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = llm.invoke(messages)
            raw = _msg_content_str(resp.content)
            return parse_llm_queries(raw)
        except Exception as e:
            last_err = e
            if attempt < retries and retry_backoff > 0:
                time.sleep(retry_backoff * (attempt + 1))
            continue
    raise RuntimeError(f"LLM text failed after {retries + 1} attempts: {last_err}")


def run_llm_image(
    llm: ChatOpenAI,
    context: str,
    max_q: int,
    retries: int,
    retry_backoff: float = 0.0,
) -> list[dict[str, str]]:
    types_line = "、".join(QUERY_TYPES)
    sys_content = SYSTEM_PROMPT_IMAGE.format(
        max_q=max_q, types_line=types_line
    )
    messages = [
        SystemMessage(content=sys_content),
        HumanMessage(content=f"【图像块上下文】\n{context}"),
    ]
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = llm.invoke(messages)
            raw = _msg_content_str(resp.content)
            return parse_llm_queries(raw)
        except Exception as e:
            last_err = e
            if attempt < retries and retry_backoff > 0:
                time.sleep(retry_backoff * (attempt + 1))
            continue
    raise RuntimeError(f"LLM image failed after {retries + 1} attempts: {last_err}")


class ProgressBar:
    """简易终端进度条（stderr），线程安全。"""

    def __init__(self, total: int, desc: str, enabled: bool = True) -> None:
        self.total = max(0, total)
        self.desc = desc
        self.enabled = enabled and total > 0
        self._done = 0
        self._ok = 0
        self._skip = 0
        self._lock = threading.Lock()

    def update(self, ok: bool) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._done += 1
            if ok:
                self._ok += 1
            else:
                self._skip += 1
            n = self._done
            w = 36
            filled = int(w * n / self.total) if self.total else w
            bar = "#" * filled + "-" * (w - filled)
            pct = 100.0 * n / self.total if self.total else 100.0
            line = (
                f"\r{self.desc} |{bar}| {n}/{self.total} "
                f"({pct:.1f}%) 成功:{self._ok} 跳过:{self._skip}"
            )
            sys.stderr.write(line.ljust(100))
            sys.stderr.flush()

    def finish(self) -> None:
        if self.enabled:
            sys.stderr.write("\n")
            sys.stderr.flush()


def stratified_sample(
    by_book_text: dict[str, list[str]],
    by_book_image: dict[str, list[str]],
    book_ids: list[str],
    text_ratio: float,
    image_ratio: float,
    min_per_modality: int,
    rng: random.Random,
) -> tuple[list[str], list[str]]:
    text_ids: list[str] = []
    image_ids: list[str] = []
    for bid in book_ids:
        tlist = by_book_text.get(bid, [])
        ilist = by_book_image.get(bid, [])
        nt = _sample_size(len(tlist), text_ratio, min_per_modality)
        ni = _sample_size(len(ilist), image_ratio, min_per_modality)
        if tlist and nt > 0:
            text_ids.extend(rng.sample(tlist, nt))
        if ilist and ni > 0:
            image_ids.extend(rng.sample(ilist, ni))
    return text_ids, image_ids


def main() -> None:
    p = argparse.ArgumentParser(
        description="从数据库抽样 chunk，经 LLM 生成伪查询并写入 JSONL。详见文件头注释。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
环境变量: SEU_API_KEY/DASHSCOPE_API_KEY, SEU_BASE_URL, CHAT_MODEL_NAME,
         EXPERIMENT_LLM_MODEL(可选), DB_URL

JSONL 字段: query, query_type, evidence_chunk_id, evidence_kind,
            source_book, content_type

并发/容错: --llm-concurrency, --llm-retries, --retry-backoff, --no-progress
""",
    )
    p.add_argument(
        "--out",
        default="data/retriever_experiment.jsonl",
        help="输出 JSONL 路径（默认 data/retriever_experiment.jsonl）",
    )
    p.add_argument("--text-ratio", type=float, default=0.005, help="每书文本块抽样比例")
    p.add_argument("--image-ratio", type=float, default=0.005, help="每书图像块抽样比例")
    p.add_argument(
        "--min-per-modality",
        type=int,
        default=1,
        help="该书该模态块数>0 时至少抽取条数（与比例取 max 后 cap 到总数）",
    )
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument(
        "--max-text-chars",
        type=int,
        default=12000,
        help="拼进 prompt 的正文最大字符数",
    )
    p.add_argument(
        "--max-query-chars",
        type=int,
        default=50,
        help="伪查询最大长度（默认 50）",
    )
    p.add_argument(
        "--llm-retries", type=int, default=3, help="单条 chunk 调用失败时的重试次数（含首次共 retries+1 次）"
    )
    p.add_argument(
        "--retry-backoff",
        type=float,
        default=0.8,
        help="重试前等待秒数基数，第 n 次重试前等待 基数×n 秒；0 表示不等待",
    )
    p.add_argument(
        "--llm-concurrency",
        type=int,
        default=4,
        help="LLM 并发上限（线程数），过大易触发限流，建议 2～8",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="关闭 stderr 进度条",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="只统计抽样数量，不写文件、不调 LLM",
    )
    args = p.parse_args()

    rng = random.Random(args.seed)
    doc_names = fetch_document_names()
    book_ids = sorted(doc_names.keys())
    if not book_ids:
        print("documents 表无数据，退出。", file=sys.stderr)
        sys.exit(1)

    by_t = fetch_text_chunk_index()
    by_i = fetch_image_chunk_index()

    text_sampled, image_sampled = stratified_sample(
        by_t,
        by_i,
        book_ids,
        args.text_ratio,
        args.image_ratio,
        args.min_per_modality,
        rng,
    )

    print(
        f"书籍数={len(book_ids)}，抽样文本块={len(text_sampled)}，图像块={len(image_sampled)}",
        file=sys.stderr,
    )

    if args.dry_run:
        return

    if not _seu_api_key():
        print("未设置 SEU_API_KEY 或 DASHSCOPE_API_KEY，无法调用 LLM。", file=sys.stderr)
        sys.exit(1)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    llm = _experiment_llm()
    max_q = args.max_query_chars
    retries = args.llm_retries
    backoff = args.retry_backoff

    text_rows = fetch_text_chunks_full(text_sampled)
    image_rows = fetch_image_chunks_full(image_sampled)

    tasks: list[tuple[int, str, dict[str, Any]]] = []
    for i, row in enumerate(text_rows):
        tasks.append((i, "text", row))
    off = len(text_rows)
    for j, row in enumerate(image_rows):
        tasks.append((off + j, "image", row))

    total_tasks = len(tasks)
    if total_tasks == 0:
        print("无待处理 chunk，已写出空文件。", file=sys.stderr)
        open(args.out, "w", encoding="utf-8").close()
        return

    stderr_lock = threading.Lock()
    results: dict[int, list[dict[str, Any]]] = {}

    def process_one(
        task: tuple[int, str, dict[str, Any]],
    ) -> tuple[int, list[dict[str, Any]], str | None]:
        idx, kind, row = task
        log: list[str] = []

        try:
            return _process_one_body(idx, kind, row, log)
        except Exception as e:
            with stderr_lock:
                print(f"[skip] task {idx} ({kind}): {e}", file=sys.stderr)
            return idx, [], str(e)

    def _process_one_body(
        idx: int,
        kind: str,
        row: dict[str, Any],
        log: list[str],
    ) -> tuple[int, list[dict[str, Any]], str | None]:
        if kind == "text":
            ctx = build_text_context(row, args.max_text_chars)
            try:
                items = run_llm_text(llm, ctx, max_q, retries, backoff)
            except Exception as e:
                with stderr_lock:
                    print(
                        f"[skip] text chunk {row.get('chunk_id')}: {e}",
                        file=sys.stderr,
                    )
                return idx, [], str(e)
            cid = str(row["chunk_id"])
            sbook = row.get("source_book") or ""
            ct_label = normalize_text_content_type(row.get("content_type"))
            records: list[dict[str, Any]] = []
            for it in items:
                qt = it["query_type"]
                if qt not in QUERY_TYPES:
                    with stderr_lock:
                        print(
                            f"[skip] invalid query_type {qt!r} for text {cid}",
                            file=sys.stderr,
                        )
                    continue
                q = validate_and_trim_query(it["query"], max_q, log)
                if not q:
                    continue
                records.append(
                    {
                        "query": q,
                        "query_type": qt,
                        "evidence_chunk_id": cid,
                        "evidence_kind": "text",
                        "source_book": sbook,
                        "content_type": ct_label,
                    }
                )
            for line in log:
                with stderr_lock:
                    print(line, file=sys.stderr)
            return idx, records, None

        # image
        st = (row.get("search_text") or "").strip()
        if not st:
            with stderr_lock:
                print(
                    f"[skip] image {row.get('image_id')}: empty search_text",
                    file=sys.stderr,
                )
            return idx, [], "empty search_text"
        ctx = build_image_context(row, args.max_text_chars)
        try:
            items = run_llm_image(llm, ctx, max_q, retries, backoff)
        except Exception as e:
            with stderr_lock:
                print(
                    f"[skip] image {row.get('image_id')}: {e}",
                    file=sys.stderr,
                )
            return idx, [], str(e)
        iid = str(row["image_id"])
        sbook = row.get("source_book") or ""
        records = []
        for it in items:
            qt = it["query_type"]
            if qt not in QUERY_TYPES:
                with stderr_lock:
                    print(
                        f"[skip] invalid query_type {qt!r} for image {iid}",
                        file=sys.stderr,
                    )
                continue
            q = validate_and_trim_query(it["query"], max_q, log)
            if not q:
                continue
            records.append(
                {
                    "query": q,
                    "query_type": qt,
                    "evidence_chunk_id": iid,
                    "evidence_kind": "image",
                    "source_book": sbook,
                    "content_type": "image",
                }
            )
        for line in log:
            with stderr_lock:
                print(line, file=sys.stderr)
        return idx, records, None

    progress = ProgressBar(
        total_tasks,
        "LLM",
        enabled=not args.no_progress,
    )
    concurrency = max(1, args.llm_concurrency)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(process_one, t) for t in tasks]
        for fut in as_completed(futures):
            idx, recs, err = fut.result()
            results[idx] = recs
            progress.update(ok=(err is None))
    progress.finish()

    written = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(total_tasks):
            for rec in results.get(i, []):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1

    print(f"已写入 {written} 行 -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
