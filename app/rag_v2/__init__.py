"""
rag_v2：多路检索（五路主检索并行 + relation 后置），不含图像向量路 img_vec。

与旧版 `app.rag.hybrid_search` 的差异：目录/正文关键词分列（toc_tsvector vs ts_vector），
支持 intent_result 过滤与复合子查询 RRF；离线评测见 `scripts/retriever_offline_eval.py --backend v2`。

环境变量（LLM 意图）：INTENT_LLM_MODEL、INTENT_LLM_TEMPERATURE、SEU_* 或 DASHSCOPE_*。
"""
from app.rag_v2.hybrid_search import hybrid_search_v2, hybrid_search_v2_with_llm
from app.rag_v2.intent_llm import recognize_intent_llm
from app.rag_v2.schemas import (
    IntentFilters,
    IntentPayload,
    IntentType,
    MAIN_RETRIEVER_IDS,
    RetrieverId,
    SingleIntentBlock,
    SubQueryItem,
    parse_intent_result,
)

__all__ = [
    "hybrid_search_v2",
    "hybrid_search_v2_with_llm",
    "recognize_intent_llm",
    "IntentFilters",
    "IntentPayload",
    "IntentType",
    "MAIN_RETRIEVER_IDS",
    "RetrieverId",
    "SingleIntentBlock",
    "SubQueryItem",
    "parse_intent_result",
]
