
import os
import sys
from pathlib import Path

# 项目根目录
_here = Path(__file__).resolve().parent
PROJECT_ROOT = _here.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from scripts.retriever_test import run_single_query

# ── 批量查询语句（按需修改）──
QUERIES = [
    "木+龙念什么字？",
    "七朱八白”是什么术语？",
    "《营造法式》有关‘造梁之制’的原文是什么？",
    "若地势偏邪，既以景表望筒取正，四方或有可疑处，则更以水池景表较",
    "栱是由哪些工具做成的？安装顺序是什么？"
]

# ── 与 `python scripts/test.py` 对应的参数 ──
PARAMS = dict(
    top_k=3,
    final_k=3,
    relations=True,
    steps=True,
    prompt_preview_chars=2000,
)


def batch_run(queries: list, **params) -> None:
    valid = [str(q).strip() for q in queries if q is not None and str(q).strip()]
    if not valid:
        print("QUERIES 为空或全部为空白，请编辑本文件配置。")
        return
    for i, q in enumerate(valid, 1):
        sep = "#" * 90
        print(f"\n{sep}\n  批量进度 [{i}/{len(valid)}]\n{sep}")
        run_single_query(q, **params)


if __name__ == "__main__":
    print("PROJECT_ROOT =", PROJECT_ROOT)
    batch_run(QUERIES, **PARAMS)
