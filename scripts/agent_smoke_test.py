#!/usr/bin/env python3
"""Agent 冒烟：路由逻辑 + 图可编译（不调用 LLM / 数据库）。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.agent.graph import build_graph, initial_agent_state  # noqa: E402
from app.agent.nodes import route_after_decide  # noqa: E402


def test_route_after_decide():
    assert route_after_decide({"is_sufficient": True, "depth": 1}) == "done"
    assert (
        route_after_decide(
            {"is_sufficient": False, "depth": 3, "empty_retrieval_streak": 0}
        )
        == "done"
    )
    assert (
        route_after_decide(
            {"is_sufficient": False, "depth": 1, "empty_retrieval_streak": 2}
        )
        == "done"
    )
    assert (
        route_after_decide(
            {"is_sufficient": False, "depth": 1, "empty_retrieval_streak": 1}
        )
        == "retrieve"
    )
    print("route_after_decide: ok")


def test_graph_compile():
    g = build_graph()
    s = initial_agent_state("ping")
    assert s["depth"] == 0
    assert s["current_query"] == "ping"
    print("graph compile + initial state: ok", g)


if __name__ == "__main__":
    test_route_after_decide()
    test_graph_compile()
    print("agent_smoke_test: all passed")
