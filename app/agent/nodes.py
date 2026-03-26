# app/agent/nodes.py
from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Iterator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langsmith import traceable
from pydantic import BaseModel, Field

from app.agent.prompts import (
    BOUNDARY_MAX_DEPTH,
    BOUNDARY_NO_HITS,
    DECIDE_SYSTEM,
    FINAL_SYSTEM_BASE,
    SUMMARIZE_SYSTEM,
)
from app.agent.state import CLUES_CHAR_THRESHOLD, MAX_RETRIEVE_DEPTH, AgentState
from app.rag import retrieve_context_structured


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name, "").lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return default


def _chat_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("CHAT_MODEL_NAME", ""),
        api_key=os.getenv("DASHSCOPE_API_KEY", ""),
        base_url=os.getenv(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        temperature=0,
    )


def _merge_relations(existing: list[dict], new_rows: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    out: list[dict] = []
    for rel in existing + new_rows:
        key = (rel.get("source_id"), rel.get("target_id"), rel.get("relation_type"))
        if key in seen:
            continue
        seen.add(key)
        out.append(rel)
    return out

# 检索节点
def retrieve_node(state: AgentState) -> dict:
    with_relations = _env_bool("RAG_WITH_RELATIONS", False)
    prompt_text, search_result = retrieve_context_structured(
        state["current_query"],
        with_relations=with_relations,
    )
    items = search_result.get("items") or []
    rels = search_result.get("relations") or []

    empty = len(items) == 0
    streak = state["empty_retrieval_streak"] + 1 if empty else 0

    citation_items = dict(state.get("citation_items") or {})
    for it in items:
        citation_items[str(it["id"])] = it

    citation_relations = _merge_relations(
        list(state.get("citation_relations") or []), rels
    )

    clues = list(state.get("clues") or [])
    if prompt_text:
        round_no = state.get("depth", 0) + 1
        clues.append(
            f"--- 第{round_no}轮检索 (查询: {state['current_query']}) ---\n{prompt_text}"
        )

    return {
        "depth": state.get("depth", 0) + 1,
        "clues": clues,
        "empty_retrieval_streak": streak,
        "citation_items": citation_items,
        "citation_relations": citation_relations,
    }


def summarize_node(state: AgentState) -> dict:
    clues = list(state.get("clues") or [])
    if not clues:
        return {}
    total_chars = sum(len(c) for c in clues)
    if total_chars <= CLUES_CHAR_THRESHOLD:
        return {}

    llm = _chat_llm()
    body = "\n\n".join(clues)
    resp = llm.invoke(
        [
            SystemMessage(content=SUMMARIZE_SYSTEM),
            HumanMessage(content=f"--- 待压缩线索 ---\n{body}"),
        ]
    )
    text = (resp.content or "").strip()
    if not text:
        return {}
    return {"clues": [f"--- 线索摘要（原约{total_chars}字已压缩）---\n{text}"]}


class DecisionOutput(BaseModel):
    sufficient: bool = Field(description="当前线索是否足以回答用户问题")
    thought: str = Field(description="简短推理")
    next_query: str | None = Field(
        default=None,
        description="若需继续检索，给出下一条检索查询；否则可留空",
    )


def decide_node(state: AgentState) -> dict:
    llm = _chat_llm().with_structured_output(DecisionOutput)
    clues = state.get("clues") or []
    clues_text = "\n\n".join(clues) if clues else "（尚无检索片段）"
    user_block = (
        f"用户问题：{state['input']}\n"
        f"本轮用于检索的查询：{state['current_query']}\n"
        f"当前检索轮次 depth：{state.get('depth', 0)} / 上限 {MAX_RETRIEVE_DEPTH}\n"
        f"连续无结果次数：{state.get('empty_retrieval_streak', 0)}\n\n"
        f"已累积知识线索：\n{clues_text}"
    )
    out: DecisionOutput = llm.invoke(
        [
            SystemMessage(content=DECIDE_SYSTEM),
            HumanMessage(content=user_block),
        ]
    )

    scratchpad = list(state.get("scratchpad") or [])
    scratchpad.append(out.thought)

    next_q = state["current_query"]
    if not out.sufficient and out.next_query and out.next_query.strip():
        next_q = out.next_query.strip()

    return {
        "is_sufficient": out.sufficient,
        "scratchpad": scratchpad,
        "current_query": next_q,
    }


def route_after_decide(state: AgentState) -> str:
    if state.get("is_sufficient"):
        return "done"
    if state.get("depth", 0) >= MAX_RETRIEVE_DEPTH:
        return "done"
    if state.get("empty_retrieval_streak", 0) >= 2:
        return "done"
    return "retrieve"


def _reduce_stream_chunks(chunks: Sequence[str]) -> dict:
    text = "".join(chunks)
    return {"answer_length": len(text)}


def merged_search_result(state: AgentState) -> dict:
    items = list((state.get("citation_items") or {}).values())
    items.sort(key=lambda x: (x.get("score") is None, -(x.get("score") or 0)))
    return {
        "items": items,
        "relations": list(state.get("citation_relations") or []),
    }


@traceable(
    name="stream_final_answer",
    run_type="chain",
    reduce_fn=_reduce_stream_chunks,
)
def stream_final_answer(
    state: AgentState,
    history: list[dict],
) -> Iterator[str]:
    llm = _chat_llm()
    clues = state.get("clues") or []
    clues_text = "\n\n".join(clues) if clues else ""

    system = FINAL_SYSTEM_BASE
    if clues_text:
        system += f"\n\n知识线索：\n{clues_text}"
    else:
        system += BOUNDARY_NO_HITS

    if state.get("depth", 0) >= MAX_RETRIEVE_DEPTH and not state.get(
        "is_sufficient", False
    ):
        system += BOUNDARY_MAX_DEPTH
    elif state.get("empty_retrieval_streak", 0) >= 2 and clues_text:
        system += BOUNDARY_NO_HITS

    system += "\n\n请在回答中引用参考资料时，在相关句子末尾标注来源编号，如[1]、[2]。不要在回答末尾生成引用列表。"

    messages: list = [SystemMessage(content=system)]
    for m in history:
        role, content = m.get("role"), m.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=state["input"]))

    for chunk in llm.stream(messages):
        if chunk.content:
            yield str(chunk.content)
