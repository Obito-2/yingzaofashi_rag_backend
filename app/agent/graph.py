# app/agent/graph.py
from typing import cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langsmith import traceable

from app.agent.nodes import (
    decide_node,
    gate_node,
    retrieve_node,
    route_after_decide,
    route_after_gate,
    summarize_node,
)
from app.agent.state import AgentState

_compiled = None

#定义图结构 
def build_graph():
    g = StateGraph(AgentState)
    g.add_node("gate", gate_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("summarize", summarize_node)
    g.add_node("decide", decide_node)
    g.add_edge(START, "gate")
    g.add_conditional_edges(
        "gate",
        route_after_gate,
        {"retrieve": "retrieve", "done": END},
    )
    g.add_edge("retrieve", "summarize")
    g.add_edge("summarize", "decide")
    g.add_conditional_edges(
        "decide",
        route_after_decide,
        {"retrieve": "retrieve", "done": END},
    )
    return g.compile()


def get_agent_graph():
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    return _compiled


def initial_agent_state(user_input: str) -> AgentState:
    return {
        "input": user_input,
        "current_query": user_input,
        "scratchpad": [],
        "clues": [],
        "depth": 0,
        "is_sufficient": False,
        "empty_retrieval_streak": 0,
        "citation_items": {},
        "citation_relations": [],
        "skip_rag": False,
    }

#执行入口
@traceable(name="run_agent_rag", run_type="chain")
def run_agent_rag(user_input: str, *, session_id: str | None = None) -> AgentState:
    config: RunnableConfig | dict = {}
    if session_id:
        config = {"metadata": {"session_id": session_id}}
    return cast(
        AgentState,
        get_agent_graph().invoke(initial_agent_state(user_input), config=config or None),
    )
