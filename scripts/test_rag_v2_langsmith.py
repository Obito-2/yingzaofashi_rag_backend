#!/usr/bin/env python3
"""
本地测试 rag_v2 混合检索，并在 LangSmith 中查看 trace。

用法示例：
  export LANGCHAIN_TRACING_V2=true
  export LANGCHAIN_API_KEY=lsv2_...
  export LANGCHAIN_PROJECT=你的项目名   # 可选

  python scripts/test_rag_v2_langsmith.py "什么是铺作"
  python scripts/test_rag_v2_langsmith.py "铺作" --no-intent-llm
  python scripts/test_rag_v2_langsmith.py "梁思成关于斗栱的描述" --with-relations --k-final 5

LangSmith trace 链路：
  hybrid_search_v2_with_llm
    └─ recognize_intent_llm（LLM 意图识别）
    └─ hybrid_search_v2
         └─ rag_v2_parallel_main（五路并行检索）

其它常用环境变量（可在 .env 中配置）：
  INTENT_LLM_MODEL / CHAT_MODEL_NAME / DASHSCOPE_API_KEY 等
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _tracing_hint() -> None:
    v = os.getenv("LANGCHAIN_TRACING_V2", "").lower()
    if v not in ("true", "1", "yes", "on"):
        print(
            "提示: 未检测到 LANGCHAIN_TRACING_V2=true，LangSmith 通常不会有 trace。\n"
            "      请设置: export LANGCHAIN_TRACING_V2=true\n"
            "      以及:     export LANGCHAIN_API_KEY=<LangSmith API Key>",
            file=sys.stderr,
        )
    if not (os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")):
        print(
            "提示: 未设置 LANGCHAIN_API_KEY（或 LANGSMITH_API_KEY），追踪可能无法上报。",
            file=sys.stderr,
        )


def _print_result(result: dict, use_llm: bool) -> None:
    items = result.get("items") or []
    relations = result.get("relations") or []
    debug = result.get("debug_info") or {}

    print(f"\n--- 检索模式 ---")
    print(f"  意图 LLM：{'启用' if use_llm else '跳过（五路全开）'}")

    intent_type = debug.get("intent_type")
    if intent_type:
        print(f"  意图类型：{intent_type}")

    single_debug = debug.get("single_debug") or {}
    enabled = single_debug.get("enabled_retrievers") or single_debug.get("effective_retrievers")
    if enabled:
        print(f"  启用检索路：{enabled}")

    lane_sizes = single_debug.get("lane_sizes")
    if lane_sizes:
        print(f"  各路命中数：{lane_sizes}")

    intent_payload = debug.get("intent")
    if intent_payload:
        intents = intent_payload.get("intents") or []
        if intents:
            print(f"  意图详情：{intents[0]}")

    print(f"\n--- 检索结果 ---")
    print(f"  items 数量：{len(items)}")
    print(f"  relations 数量：{len(relations)}")

    if items:
        print("\n  前几条 items 摘要：")
        for i, item in enumerate(items[:5]):
            item_type = item.get("type", "?")
            score = item.get("score")
            score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
            meta = item.get("metadata") or {}
            book_id = meta.get("book_id", "")
            content_type = meta.get("content_type", "")
            source_ret = item.get("_source_retriever", "")
            content = item.get("content") or item.get("main_text") or ""
            preview = content[:100].replace("\n", " ")
            print(
                f"  [{i+1}] type={item_type} score={score_str} "
                f"book={book_id} ct={content_type} ret={source_ret}"
            )
            print(f"       {preview}…" if len(content) > 100 else f"       {preview}")

    if relations:
        print(f"\n  前几条 relations：")
        for rel in relations[:3]:
            print(
                f"    {rel.get('source_id')} --[{rel.get('relation_type')}]--> {rel.get('target_id')}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="测试 rag_v2 混合检索并在 LangSmith 产生追踪。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("query", help="用户查询文本")
    parser.add_argument(
        "--no-intent-llm",
        action="store_true",
        help="跳过 LLM 意图识别，直接五路全开（等同于 use_llm=False）",
    )
    parser.add_argument(
        "--with-relations",
        action="store_true",
        help="启用关系检索（enrich_main_with_relations）",
    )
    parser.add_argument(
        "--k-per",
        type=int,
        default=5,
        metavar="N",
        help="每路检索器召回数量（默认 5）",
    )
    parser.add_argument(
        "--k-final",
        type=int,
        default=10,
        metavar="N",
        help="RRF 融合后最终返回数量（默认 10）",
    )
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))

    _tracing_hint()

    from app.rag_v2 import hybrid_search_v2_with_llm

    query = args.query.strip()
    if not query:
        print("错误: query 为空", file=sys.stderr)
        sys.exit(2)

    use_llm = not args.no_intent_llm

    print(f"--- 查询 ---\n{query}\n")

    result = hybrid_search_v2_with_llm(
        query,
        use_llm=use_llm,
        k_per_retriever=args.k_per,
        k_final=args.k_final,
        with_relations=args.with_relations,
    )

    _print_result(result, use_llm)

    print(
        "\n完成。请在 LangSmith 中查看本次 trace"
        "（链路：hybrid_search_v2_with_llm → recognize_intent_llm → hybrid_search_v2 → rag_v2_parallel_main）。"
    )


if __name__ == "__main__":
    main()
