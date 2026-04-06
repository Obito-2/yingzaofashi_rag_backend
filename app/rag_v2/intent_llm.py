"""
LLM 意图识别：输出与 PRD 一致的 JSON，供 `parse_intent_result` 消费。

环境变量（与项目其它 OpenAI 兼容调用一致）：
- SEU_API_KEY 或 DASHSCOPE_API_KEY
- SEU_BASE_URL 或 DASHSCOPE_BASE_URL
- INTENT_LLM_MODEL（可选；否则用 CHAT_MODEL_NAME）
- INTENT_LLM_TEMPERATURE（默认 0）
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import OpenAI

_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.MULTILINE)


def _llm_client() -> OpenAI:
    api_key = os.getenv("SEU_API_KEY") or os.getenv("DASHSCOPE_API_KEY", "")
    base_url = os.getenv("SEU_BASE_URL") or os.getenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    return OpenAI(api_key=api_key, base_url=base_url)


def _intent_model_name() -> str:
    return (
        (os.getenv("INTENT_LLM_MODEL") or "").strip()
        or (os.getenv("CHAT_MODEL_NAME") or "").strip()
        or "qwen-turbo"
    )


def _intent_temperature() -> float:
    raw = (os.getenv("INTENT_LLM_TEMPERATURE") or "0").strip()
    try:
        return float(raw)
    except ValueError:
        return 0.0


SYSTEM_PROMPT = """你是古籍检索系统的意图解析器。根据用户问题输出**仅一个 JSON 对象**，不要其它文字。

JSON 字段（必须齐全）：
- "query": string，用户原问题
- "intents": 数组，每项含：
  - "type": 以下之一：rare_char | term_explain | original_and_translation | image_by_text | specific_book | complex
  - "filters": { "book_ids": string[], "content_types": string[], "relation_types": string[] }，不需要过滤时用 []
  - "enabled_retrievers": string[]，从下列 id 中选（可多选）：text_toc_kw, text_vec, text_kw, img_toc_kw, img_content_kw, relation
- "is_complex": boolean，是否为需拆成多子问题的复合问法
- "sub_queries": 当 is_complex 为 true 时必填；每项含 "query","type","filters","enabled_retrievers"，结构同单意图

documents.id（book_ids 只能使用这些语义 id，无法确定时 book_ids 用 []）：
- terms_brief：《法式》术语简要
- yzfs_liang：梁思成注释《营造法式》
- yzfs_interpretation_rev：《营造法式》解读（修订版）
- yzfs_wang：王贵祥译注《营造法式》
- rare_chars：《法式》生僻字库

content_types 仅 text 有效：original_text, annotation, modern_translation, interpretation, others_text。
relation_types：illustrates, annotates。

规则简述：
- 生僻字、部件：type 用 rare_char，book_ids 常含 rare_chars；检索路可 text_vec,text_kw,text_toc_kw。
- 术语释义：term_explain，可含 terms_brief。
- 原文/译文/出处：original_and_translation，content_types 可含 original_text、modern_translation。
- 找图、示意图：image_by_text，**不要**启用文本表相关路时只开 img_toc_kw, img_content_kw。
- 指定译本/章节：specific_book，book_ids 填对应译本 id。
- 多跳/对比/图文混合：is_complex true，用 sub_queries 拆分，每子项独立 filters 与 enabled_retrievers。

非复合问题时 is_complex 为 false，sub_queries 为 []。复合问题时 intents 可置一条占位，以 sub_queries 为准。"""


def _parse_json_content(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    m = _JSON_FENCE.search(text)
    if m:
        try:
            obj = json.loads(m.group(1).strip())
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            pass
    return None


def recognize_intent_llm(user_query: str) -> dict[str, Any]:
    """
    调用轻量 LLM 得到意图 JSON dict；失败或空查询返回 {}。
    返回值可直接作为 `hybrid_search_v2(..., intent_result=...)` 的输入（内部会经 parse_intent_result 校验）。
    """
    q = (user_query or "").strip()
    if not q:
        return {}

    client = _llm_client()
    model = _intent_model_name()
    temp = _intent_temperature()
    user_content = f"用户问题：\n{q}"

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": temp,
    }

    try:
        resp = client.chat.completions.create(
            **kwargs,
            response_format={"type": "json_object"},
        )
    except Exception:
        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception:
            return {}

    try:
        text = (resp.choices[0].message.content or "").strip()
    except Exception:
        return {}

    parsed = _parse_json_content(text)
    return parsed if parsed is not None else {}
