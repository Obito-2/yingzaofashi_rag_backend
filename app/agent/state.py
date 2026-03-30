# app/agent/state.py
from typing import TypedDict

#最大检索深度
MAX_RETRIEVE_DEPTH = int(__import__("os").getenv("AGENT_MAX_RETRIEVE_DEPTH", "3"))
#线索字符阈值，触发摘要的字符上限
CLUES_CHAR_THRESHOLD = int(__import__("os").getenv("AGENT_CLUES_CHAR_THRESHOLD", "20000"))


class AgentState(TypedDict):
    """LangGraph 状态：检索—摘要—决策循环。"""

    input: str
    current_query: str
    scratchpad: list[str]
    clues: list[str]
    depth: int
    is_sufficient: bool
    empty_retrieval_streak: int
    citation_items: dict[str, dict]
    citation_relations: list[dict]
    # True：Gate 判定无需检索（闲聊等），非检索失败
    skip_rag: bool
