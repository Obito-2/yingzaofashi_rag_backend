# app/agent/langgraph_app.py
"""供 LangGraph CLI（langgraph dev）与 Agent Chat UI 加载的编译图。

在自定义 AgentState 上增加 messages（add_messages），并由 init 节点从最后一条
Human 消息填充与 run_agent_rag 一致的字段，便于 Agent Server / Chat UI 使用标准
messages 输入。
"""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from app.agent.graph import initial_agent_state
from app.agent.nodes import (
    decide_node,
    gate_node,
    retrieve_node,
    route_after_decide,
    route_after_gate,
    stream_final_answer,
    summarize_node,
)
from app.agent.state import AgentState


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


def _last_human_text(messages: list[AnyMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return _message_content_str(m).strip()
        if getattr(m, "type", None) == "human":
            return _message_content_str(m).strip()
    return ""


class AgentStateWithMessages(TypedDict, total=False):
    """与 AgentState 一致字段全部可选，便于仅传入 messages；init 节点会补全。"""

    messages: Annotated[list[AnyMessage], add_messages]
    input: str
    current_query: str
    scratchpad: list[str]
    clues: list[str]
    depth: int
    is_sufficient: bool
    empty_retrieval_streak: int
    citation_items: dict[str, dict]
    citation_relations: list[dict]
    skip_rag: bool


def init_from_messages(state: AgentStateWithMessages) -> dict[str, Any]:
    """从 messages 取最后一条用户文本，对齐 run_agent_rag 的 initial_agent_state。"""
    msgs = state.get("messages") or []
    text = _last_human_text(msgs)
    if not text:
        text = (state.get("input") or state.get("current_query") or "").strip()
    base: AgentState = initial_agent_state(text)
    return dict(base)


def _prior_history_for_final(state: AgentStateWithMessages) -> list[dict]:
    """当前轮之前的多轮对话，供 stream_final_answer（与 FastAPI 一致）。"""
    msgs = state.get("messages") or []
    last_h = -1
    for i in range(len(msgs) - 1, -1, -1):
        m = msgs[i]
        if isinstance(m, HumanMessage) or getattr(m, "type", None) == "human":
            last_h = i
            break
    prior = msgs[:last_h] if last_h > 0 else []
    out: list[dict] = []
    for m in prior:
        if isinstance(m, HumanMessage) or getattr(m, "type", None) == "human":
            out.append({"role": "user", "content": _message_content_str(m)})
        elif isinstance(m, AIMessage) or getattr(m, "type", None) == "ai":
            out.append({"role": "assistant", "content": _message_content_str(m)})
    return out


def final_answer_node(state: AgentStateWithMessages) -> dict[str, Any]:
    """与 FastAPI 一致：在图结束前流式聚合终答，并写入 messages 供 Studio / Chat UI 展示。"""
    history = _prior_history_for_final(state)
    parts: list[str] = []
    for delta in stream_final_answer(state, history):
        if delta:
            parts.append(delta)
    text = "".join(parts)
    return {"messages": [AIMessage(content=text)]}


def build_server_graph():
    g = StateGraph(AgentStateWithMessages)
    g.add_node("init", init_from_messages)
    g.add_node("gate", gate_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("summarize", summarize_node)
    g.add_node("decide", decide_node)
    g.add_node("final_answer", final_answer_node)
    g.add_edge(START, "init")
    g.add_edge("init", "gate")
    g.add_conditional_edges(
        "gate",
        route_after_gate,
        {"retrieve": "retrieve", "done": "final_answer"},
    )
    g.add_edge("retrieve", "summarize")
    g.add_edge("summarize", "decide")
    g.add_conditional_edges(
        "decide",
        route_after_decide,
        {"retrieve": "retrieve", "done": "final_answer"},
    )
    g.add_edge("final_answer", END)
    return g.compile()


graph = build_server_graph()
