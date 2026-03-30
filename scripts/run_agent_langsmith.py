#!/usr/bin/env python3
"""
本地一键跑通 Agent，便于在 LangSmith 查看 trace（run_agent_rag + 可选 stream_final_answer）。

用法示例：
  export LANGCHAIN_TRACING_V2=true
  export LANGCHAIN_API_KEY=lsv2_...
  export LANGCHAIN_PROJECT=你的项目名   # 可选

  python scripts/run_agent_langsmith.py "你好"
  python scripts/run_agent_langsmith.py "什么是铺作" --session-id test-sess-1
  python scripts/run_agent_langsmith.py "你好" --gate on
  python scripts/run_agent_langsmith.py "铺作" --no-stream   # 只跑图，不流式终答

多轮对话（与线上一致：检索仍只针对当前这句 query，history 只喂给终答 LLM）：
  python scripts/run_agent_langsmith.py "那斗栱呢" --history-json '[{"role":"user","content":"什么是铺作"},{"role":"assistant","content":"铺作是..."}]'
  python scripts/run_agent_langsmith.py "追问" --history-file /path/to/history.json

其它常用环境变量（与线上一致，可在 .env 中配置）：
  AGENT_GATE_MODE / AGENT_GATE_MODEL / CHAT_MODEL_NAME / DASHSCOPE_API_KEY 等
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="运行 Agent 查询并在 LangSmith 中产生追踪（需配置追踪环境变量）。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("query", help="用户查询文本")
    parser.add_argument(
        "--session-id",
        default=None,
        help="传入 LangGraph/LangSmith run 的 metadata.session_id（可选）",
    )
    parser.add_argument(
        "--gate",
        choices=("on", "off"),
        default=None,
        help="仅本次进程覆盖 AGENT_GATE_MODE：on=启用 Gate LLM；off=首轮必检索",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="只执行 run_agent_rag（检索图），不调用 stream_final_answer，少一次终答 LLM",
    )
    parser.add_argument(
        "--history-json",
        default=None,
        metavar="JSON",
        help='此前多轮消息 JSON 数组，形如 [{"role":"user","content":"..."},{"role":"assistant","content":"..."}]',
    )
    parser.add_argument(
        "--history-file",
        default=None,
        metavar="PATH",
        help="同上，从 UTF-8 文件读取 JSON（与 --history-json 二选一，同时指定时以文件为准）",
    )
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv(os.path.join(ROOT, ".env"))

    if args.gate is not None:
        os.environ["AGENT_GATE_MODE"] = args.gate

    _tracing_hint()

    from app.agent.graph import run_agent_rag
    from app.agent.nodes import merged_search_result, stream_final_answer

    query = args.query.strip()
    if not query:
        print("错误: query 为空", file=sys.stderr)
        sys.exit(2)

    history: list[dict] = []
    if args.history_file:
        try:
            with open(args.history_file, encoding="utf-8") as f:
                raw = json.load(f)
        except OSError as e:
            print(f"错误: 无法读取 --history-file: {e}", file=sys.stderr)
            sys.exit(2)
        except json.JSONDecodeError as e:
            print(f"错误: --history-file 不是合法 JSON: {e}", file=sys.stderr)
            sys.exit(2)
    elif args.history_json:
        try:
            raw = json.loads(args.history_json)
        except json.JSONDecodeError as e:
            print(f"错误: --history-json 不是合法 JSON: {e}", file=sys.stderr)
            sys.exit(2)
    else:
        raw = None

    if raw is not None:
        if not isinstance(raw, list):
            print("错误: history 必须是 JSON 数组", file=sys.stderr)
            sys.exit(2)
        for i, m in enumerate(raw):
            if not isinstance(m, dict):
                print(f"错误: history[{i}] 必须是对象", file=sys.stderr)
                sys.exit(2)
            role = m.get("role")
            if role not in ("user", "assistant"):
                print(
                    f"错误: history[{i}].role 须为 user 或 assistant",
                    file=sys.stderr,
                )
                sys.exit(2)
            if "content" not in m:
                print(f"错误: history[{i}] 缺少 content", file=sys.stderr)
                sys.exit(2)
        history = [dict(x) for x in raw]

    print("--- 查询 ---", query, sep="\n", end="\n\n")
    if history:
        print(f"--- 已载入 {len(history)} 条历史消息（仅用于终答）---\n")

    state = run_agent_rag(query, session_id=args.session_id)

    print("--- AgentState 摘要 ---")
    print(f"  skip_rag:        {state.get('skip_rag')}")
    print(f"  depth:           {state.get('depth')}")
    print(f"  is_sufficient:   {state.get('is_sufficient')}")
    print(f"  empty_streak:    {state.get('empty_retrieval_streak')}")
    cit = merged_search_result(state)
    print(f"  citations items: {len(cit.get('items') or [])}")

    if args.no_stream:
        print("\n已跳过终答流式（--no-stream）。到 LangSmith 查看 run_agent_rag 与子步骤。")
        return

    print("\n--- 终答（stream_final_answer）---\n")
    for delta in stream_final_answer(state, history):
        if delta:
            print(delta, end="", flush=True)
    print("\n")
    print("完成。请在 LangSmith 项目中查看本次 run（trace 名称含 run_agent_rag、stream_final_answer）。")


if __name__ == "__main__":
    main()
