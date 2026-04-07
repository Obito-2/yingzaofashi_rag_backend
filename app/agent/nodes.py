# app/agent/nodes.py
from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from typing import Iterator

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langsmith import traceable
from pydantic import BaseModel, Field, SecretStr

from app.agent.prompts import (
    BOUNDARY_MAX_DEPTH,
    BOUNDARY_NO_HITS,
    DECIDE_SYSTEM,
    FINAL_SYSTEM_BASE,
    FINAL_SYSTEM_NO_RAG,
    GATE_SYSTEM,
    SUMMARIZE_SYSTEM,
)
from app.agent.state import CLUES_CHAR_THRESHOLD, MAX_RETRIEVE_DEPTH, AgentState
from app.rag import _enrich_items_metadata, _format_item
from app.rag_v2 import hybrid_search_v2_with_llm


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name, "").lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return default


def _message_content_str(msg: BaseMessage) -> str:
    c = msg.content
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts: list[str] = []
        for p in c:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and isinstance(p.get("text"), str):
                parts.append(p["text"])
            else:
                parts.append(str(p))
        return "".join(parts)
    return ""


def _seu_api_key() -> str:
    return os.getenv("SEU_API_KEY") or os.getenv("DASHSCOPE_API_KEY", "")


def _seu_base_url() -> str:
    return os.getenv("SEU_BASE_URL") or os.getenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


def _chat_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("CHAT_MODEL_NAME", ""),
        api_key=SecretStr(_seu_api_key()),
        base_url=_seu_base_url(),
        temperature=0,
    )


def _agent_gate_enabled() -> bool:
    v = os.getenv("AGENT_GATE_MODE", "off").lower().strip()
    return v in ("on", "llm", "1", "true", "yes")


def _gate_llm() -> ChatOpenAI:
    # 生产环境建议单独配置 AGENT_GATE_MODEL（轻量、非思考模型）以降低首轮延迟。
    model = (os.getenv("AGENT_GATE_MODEL") or "").strip() or os.getenv(
        "CHAT_MODEL_NAME", ""
    )
    temp_s = os.getenv("AGENT_GATE_TEMPERATURE", "0").strip()
    try:
        temperature = float(temp_s)
    except ValueError:
        temperature = 0.0
    return ChatOpenAI(
        model=model,
        api_key=SecretStr(_seu_api_key()),
        base_url=_seu_base_url(),
        temperature=temperature,
    )


class GateOutput(BaseModel):
    need_kb: bool = Field(description="是否需要检索《营造法式》知识库")
    need_clarify: bool = Field(default=False, description="用户问题是否模糊需要澄清")
    clarify_question: str = Field(default="", description="当 need_clarify=true 时向用户提出的澄清问题")
    thought: str = Field(description="简短推理")


def _parse_gate_output(raw: object) -> GateOutput:
    """兼容部分模型返回非标准 JSON（如 need_kb=false / thought: ... 键值行）。"""
    if isinstance(raw, GateOutput):
        return raw
    if isinstance(raw, dict):
        return GateOutput.model_validate(raw)
    if not isinstance(raw, str):
        return GateOutput.model_validate(raw)

    s = raw.strip()
    if s.startswith("```"):
        lines = s.split("\n")
        lines = lines[1:] if lines else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines).strip()

    try:
        return GateOutput.model_validate_json(s)
    except Exception:
        pass
    try:
        return GateOutput.model_validate(json.loads(s))
    except Exception:
        pass

    m = re.search(r"need_kb\s*[=：:]\s*(true|false)\b", s, re.I)
    need_kb = m.group(1).lower() == "true" if m else None
    tm = re.search(r"thought\s*[=：:]\s*(.*)", s, re.I | re.S)
    thought = (tm.group(1).strip() if tm else "") or ""
    if need_kb is not None:
        return GateOutput(need_kb=need_kb, thought=thought or s)

    return GateOutput.model_validate_json(s)


def gate_node(state: AgentState) -> dict:
    if not _agent_gate_enabled():
        return {"skip_rag": False, "clarification_question": ""}
    llm = _gate_llm().with_structured_output(GateOutput)
    raw = llm.invoke(
        [
            SystemMessage(content=GATE_SYSTEM),
            HumanMessage(content=f"用户输入：{state['input']}"),
        ]
    )
    out = _parse_gate_output(raw)
    scratchpad = list(state.get("scratchpad") or [])
    scratchpad.append(f"[gate] {out.thought}")
    clarification_question = out.clarify_question.strip() if out.need_clarify else ""
    return {
        "skip_rag": not out.need_kb,
        "clarification_question": clarification_question,
        "scratchpad": scratchpad,
    }


def route_after_gate(state: AgentState) -> str:
    if state.get("skip_rag") or state.get("clarification_question"):
        return "done"
    return "retrieve"


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

# 数据库检索节点（rag_v2：五路并行 + LLM 意图识别）
def retrieve_node(state: AgentState) -> dict:
    with_relations = _env_bool("RAG_WITH_RELATIONS", False)
    result = hybrid_search_v2_with_llm(
        state["current_query"],
        use_llm=True,
        k_per_retriever=5,
        k_final=10,
        with_relations=with_relations,
    )
    items = result.get("items") or []
    rels = result.get("relations") or []
    debug = result.get("debug_info") or {}

    empty = len(items) == 0
    streak = state["empty_retrieval_streak"] + 1 if empty else 0

    # 格式化 prompt_text（复用旧版格式化逻辑）
    if items:
        items = _enrich_items_metadata(items)
        prompt_text = "\n\n".join(
            _format_item(i + 1, item) for i, item in enumerate(items)
        )
    else:
        prompt_text = ""

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

    # intent_type 写入 scratchpad 便于追踪
    scratchpad = list(state.get("scratchpad") or [])
    intent_type = debug.get("intent_type", "")
    scratchpad.append(f"[retrieve] intent={intent_type or 'unknown'}, hits={len(items)}")

    return {
        "depth": state.get("depth", 0) + 1,
        "clues": clues,
        "empty_retrieval_streak": streak,
        "citation_items": citation_items,
        "citation_relations": citation_relations,
        "scratchpad": scratchpad,
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
    text = _message_content_str(resp).strip()
    if not text:
        return {}
    return {"clues": [f"--- 线索摘要（原约{total_chars}字已压缩）---\n{text}"]}


class DecisionOutput(BaseModel):
    thought: str = Field(description="先推理：用户核心问题是什么、线索已覆盖哪些、还缺少哪些关键内容")
    sufficient: bool = Field(description="基于 thought 的分析，当前线索是否足以回答用户问题")
    next_query: str | None = Field(
        default=None,
        description="若需继续检索，根据 thought 中的缺口改写查询；否则为 null",
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
    raw = llm.invoke(
        [
            SystemMessage(content=DECIDE_SYSTEM),
            HumanMessage(content=user_block),
        ]
    )
    out: DecisionOutput = (
        raw if isinstance(raw, DecisionOutput) else DecisionOutput.model_validate(raw)
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
    skip_rag = bool(state.get("skip_rag"))

    if skip_rag and not clues_text:
        system = FINAL_SYSTEM_NO_RAG
    else:
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
