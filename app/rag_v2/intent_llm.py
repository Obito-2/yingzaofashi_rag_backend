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

from langsmith import traceable
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

检索器语义说明：
- text_toc_kw：按目录路径/卷章关键词检索，适合含"第X卷""XX作制度"等章节词的查询
- text_vec：文本向量语义检索，适合模糊语义匹配
- text_kw：文本正文关键词检索，适合精确词汇匹配
- img_toc_kw：图像目录关键词检索
- img_content_kw：图像内容/描述关键词检索
- relation：触发图文关联扩展（后置增强，非主检索路），找图时可加入

documents.id（book_ids 只能使用这些语义 id，无法确定时 book_ids 用 []）：
- terms_brief：《法式》术语简要【数据特征：content_type 全部为 interpretation】
- yzfs_liang：梁思成注释《营造法式》【含 original_text / annotation / modern_translation / others_text】
- yzfs_interpretation_rev：《营造法式》解读（修订版）【只有 annotation 和 interpretation，无 original_text 和 modern_translation】
- yzfs_wang：王贵祥译注《营造法式》【含 original_text / interpretation / modern_translation / others_text】
- rare_chars：《法式》生僻字库【数据特征：content_type 全部为 annotation】

content_types 仅 text 检索有效：original_text, annotation, modern_translation, interpretation, others_text。
relation_types：illustrates（图文配对）, annotates（注解关联）。

规则（严格遵守）：
- 生僻字、读音、部件拆字查询：type=rare_char，book_ids=["rare_chars"]，content_types=["annotation"]，enabled_retrievers=["text_vec","text_kw"]。
- 术语释义、定义查询：type=term_explain，book_ids=["terms_brief"]，content_types=["interpretation"]，enabled_retrievers=["text_vec","text_kw"]。
- 查原文（法式原文/条文）：type=original_and_translation，content_types=["original_text"]，book_ids=[]，enabled_retrievers=["text_toc_kw","text_vec","text_kw"]。
- 查今译/白话译文：type=original_and_translation，content_types=["modern_translation"]，book_ids 排除 yzfs_interpretation_rev，enabled_retrievers=["text_toc_kw","text_vec","text_kw"]。
- 查原文+译文/出处对照：type=original_and_translation，content_types=["original_text","modern_translation"]，book_ids 排除 yzfs_interpretation_rev，enabled_retrievers=["text_toc_kw","text_vec","text_kw"]。
- 找图、示意图、插图：type=image_by_text，enabled_retrievers=["img_toc_kw","img_content_kw","relation"]，不开任何文本路，book_ids 可限定来源书目。
- 指定译本/版本/作者的查询：type=specific_book，book_ids 填对应译本 id，content_types 按问题需求设置，enabled_retrievers=["text_toc_kw","text_vec","text_kw"]。
- 多跳/对比/图文混合：is_complex=true，用 sub_queries 拆分，每子项独立设置 filters 与 enabled_retrievers。

注意：yzfs_interpretation_rev 无 original_text 和 modern_translation，若用户要查这两种内容请勿将其加入 book_ids。
非复合问题时 is_complex 为 false，sub_queries 为 []。复合问题时 intents 填一条 type=complex 占位，以 sub_queries 为准。

示例：

用户问题：拆字为金巢的字怎么读，是什么意思？
{"query":"拆字为金巢的字怎么读，是什么意思？","intents":[{"type":"rare_char","filters":{"book_ids":["rare_chars"],"content_types":["annotation"],"relation_types":[]},"enabled_retrievers":["text_vec","text_kw"]}],"is_complex":false,"sub_queries":[]}

用户问题：琴面昂是什么意思？
{"query":"琴面昂是什么意思？","intents":[{"type":"term_explain","filters":{"book_ids":["terms_brief"],"content_types":["interpretation"],"relation_types":[]},"enabled_retrievers":["text_vec","text_kw"]}],"is_complex":false,"sub_queries":[]}

用户问题：垒脊瓦的定义是什么？
{"query":"垒脊瓦的定义是什么？","intents":[{"type":"term_explain","filters":{"book_ids":["terms_brief"],"content_types":["interpretation"],"relation_types":[]},"enabled_retrievers":["text_vec","text_kw"]}],"is_complex":false,"sub_queries":[]}

用户问题：营造法式第四卷大木作制度中斗栱的原文是什么？
{"query":"营造法式第四卷大木作制度中斗栱的原文是什么？","intents":[{"type":"original_and_translation","filters":{"book_ids":[],"content_types":["original_text"],"relation_types":[]},"enabled_retrievers":["text_toc_kw","text_vec","text_kw"]}],"is_complex":false,"sub_queries":[]}

用户问题：找一下乌头门的示意图
{"query":"找一下乌头门的示意图","intents":[{"type":"image_by_text","filters":{"book_ids":[],"content_types":[],"relation_types":["illustrates"]},"enabled_retrievers":["img_toc_kw","img_content_kw","relation"]}],"is_complex":false,"sub_queries":[]}

用户问题：梁思成对斗栱的解读是什么？
{"query":"梁思成对斗栱的解读是什么？","intents":[{"type":"specific_book","filters":{"book_ids":["yzfs_liang"],"content_types":["interpretation"],"relation_types":[]},"enabled_retrievers":["text_toc_kw","text_vec","text_kw"]}],"is_complex":false,"sub_queries":[]}

用户问题：给我找斗栱的原文，以及对应的插图
{"query":"给我找斗栱的原文，以及对应的插图","intents":[{"type":"complex","filters":{"book_ids":[],"content_types":[],"relation_types":[]},"enabled_retrievers":[]}],"is_complex":true,"sub_queries":[{"query":"斗栱原文","type":"original_and_translation","filters":{"book_ids":[],"content_types":["original_text"],"relation_types":[]},"enabled_retrievers":["text_toc_kw","text_vec","text_kw"]},{"query":"斗栱插图","type":"image_by_text","filters":{"book_ids":[],"content_types":[],"relation_types":["illustrates"]},"enabled_retrievers":["img_toc_kw","img_content_kw","relation"]}]}
"""


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


@traceable(name="recognize_intent_llm", run_type="llm")
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
