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
import json
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

    if relations:
        print(f"\n  前几条 relations：")
        for rel in relations[:3]:
            print(
                f"    {rel.get('source_id')} --[{rel.get('relation_type')}]--> {rel.get('target_id')}"
            )


def main() -> None:
    # Windows / PowerShell 下避免中文输出乱码
    try:
        # typing 上 TextIO 不一定有 reconfigure，这里用 getattr 规避静态检查报错
        getattr(sys.stdout, "reconfigure")(encoding="utf-8")
        getattr(sys.stderr, "reconfigure")(encoding="utf-8")
    except Exception:
        pass

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
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="直接以 JSON 输出检索结果（items/relations/debug_info）",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=0,
        metavar="N",
        help="最多打印多少条 items（0 表示全部，默认 0）",
    )
    parser.add_argument(
        "--content-len",
        type=int,
        default=300,
        metavar="N",
        help="每条 item 的 content 最多打印字符数（默认 300）",
    )
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))

    _tracing_hint()

    try:
        from app.rag_v2 import hybrid_search_v2_with_llm
    except ModuleNotFoundError as e:
        print(f"依赖缺失：{e}", file=sys.stderr)
        print(
            "请先安装依赖（例如在虚拟环境中）：\n"
            "  pip install -r requirements.txt\n"
            "或至少安装缺失包（例如）：\n"
            "  pip install sqlmodel",
            file=sys.stderr,
        )
        raise

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

    if args.print_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    _print_result(result, use_llm)

    items = result.get("items") or []
    max_items = args.max_items if args.max_items and args.max_items > 0 else len(items)
    content_len = max(0, int(args.content_len))

    if items:
        print("\n--- items 逐条输出 ---")
        for i, item in enumerate(items[:max_items]):
            item_type = item.get("type", "?")
            item_id = item.get("id", "")
            score = item.get("score")
            score_str = f"{score:.6f}" if isinstance(score, float) else str(score)
            meta = item.get("metadata") or {}
            book_id = meta.get("book_id", "")
            content_type = meta.get("content_type", "")
            is_main = item.get("is_main")
            source_ret = item.get("source_retriever", "")
            intent_type = item.get("intent_type", "")
            content = (item.get("content") or "").replace("\r\n", "\n")
            shown = content if content_len == 0 else content[:content_len]

            print(
                f"\n[{i+1}] type={item_type} score={score_str} is_main={is_main} "
                f"intent={intent_type} ret={source_ret} book={book_id} ct={content_type}"
            )
            if item_id:
                print(f"  id: {item_id}")
            if meta:
                print("  metadata:")
                for k, v in meta.items():
                    if v is None or v == "":
                        continue
                    print(f"    - {k}: {v}")
            if shown:
                print("  content:")
                print(shown)
                if content_len != 0 and len(content) > content_len:
                    print("  ...(content 已截断)")

    print(
        "\n完成。请在 LangSmith 中查看本次 trace"
        # "（链路：hybrid_search_v2_with_llm → recognize_intent_llm → hybrid_search_v2 → rag_v2_parallel_main）。"
    )


if __name__ == "__main__":
    main()
