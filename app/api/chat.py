# app/api/chat.py
import os
import json
import time
import uuid
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from openai import OpenAI

from app.api.auth import get_current_user
from app.connect import execute_query
from app.agent import merged_search_result, run_agent_rag, stream_final_answer
from app.models import ChatRequest, RegenerateRequest

router = APIRouter()

# ----------------- LLM 配置（OpenAI 兼容接口：优先 SEU_*，兼容 DASHSCOPE_*） -----------------
OPENAI_API_KEY = os.getenv("SEU_API_KEY") or os.getenv("DASHSCOPE_API_KEY", "")
OPENAI_BASE_URL = os.getenv("SEU_BASE_URL") or os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
OPENAI_MODEL = os.getenv("TITLE_GENERATE_MODEL", "deepseek-v3.2")

MAX_HISTORY_ROUNDS = int(os.getenv("MAX_HISTORY_ROUNDS", "5"))
AGENT_TRACE_SSE = os.getenv("AGENT_TRACE_SSE", "").lower() in ("1", "true", "yes", "on")

llm = wrap_openai(OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL))


# ----------------- 历史消息加载 -----------------
def _load_history(session_id: str, k: int = MAX_HISTORY_ROUNDS) -> list[dict]:
    """加载会话历史消息，截断到最近 k 轮（1轮 = user + assistant），保证成对性。"""
    rows = execute_query(
        "SELECT role, content FROM messages WHERE session_id = %s ORDER BY created_at ASC",
        (session_id,),
        fetch_all=True,
    )
    if not rows:
        return []
    history = [{"role": r["role"], "content": r["content"]} for r in rows]
    if history[-1]["role"] == "user":
        history.pop()
    max_msgs = k * 2
    if len(history) > max_msgs:
        history = history[-max_msgs:]
    return history


# ----------------- SSE 工具 -----------------
def sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

# ----------------- 流式生成器 -----------------
@traceable(name="sse_chat_completions", run_type="chain")
def _stream_chat(session_id: str, user_id: str, query: str, is_new_session: bool):
    """核心 SSE 生成器，供 completions 和 regenerate 复用"""
    # 1. 在保存当前消息之前，先加载历史（避免当前 query 被重复纳入历史）
    history = _load_history(session_id)

    # 2. 保存 user message
    user_msg_id = str(uuid.uuid4())
    now = int(time.time() * 1000)
    execute_query(
        "INSERT INTO messages (id, session_id, role, content, created_at, updated_at) VALUES (%s, %s, 'user', %s, %s, %s)",
        (user_msg_id, session_id, query, now, now),
    )

    # 3. 创建 assistant message 占位
    assistant_msg_id = str(uuid.uuid4())
    execute_query(
        "INSERT INTO messages (id, session_id, role, content, feedback, created_at, updated_at) VALUES (%s, %s, 'assistant', '', 'none', %s, %s)",
        (assistant_msg_id, session_id, now, now),
    )

    # 4. 推送 meta
    yield sse_event("meta", {"session_id": session_id, "message_id": assistant_msg_id})

    # 5. LangGraph Agent：多轮检索—摘要—决策，再流式终答
    try:
        agent_state = run_agent_rag(query, session_id=session_id)
    except Exception as e:
        yield sse_event("error", {"code": 5003, "msg": str(e)})
        return

    yield sse_event("citations", merged_search_result(agent_state))
    if AGENT_TRACE_SSE and agent_state.get("scratchpad"):
        yield sse_event("agent_trace", {"scratchpad": agent_state["scratchpad"]})

    full_content = ""
    try:
        for delta in stream_final_answer(agent_state, history):
            if delta:
                full_content += delta
                yield sse_event("message", {"content": delta})
    except Exception as e:
        yield sse_event("error", {"code": 5003, "msg": str(e)})
        return

    # 7. 更新 assistant message 内容
    done_at = int(time.time() * 1000)
    execute_query(
        "UPDATE messages SET content = %s, updated_at = %s WHERE id = %s",
        (full_content, done_at, assistant_msg_id),
    )

    # 8. 新会话：生成标题并推送 title 事件
    if is_new_session:
        try:
            title_resp = llm.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "user", "content": f"请用不超过15个字概括以下问题的主题，只输出标题本身：\n{query}"},
                ],
                stream=False,
            )
            raw_title = title_resp.choices[0].message.content or ""
            title = raw_title.strip()
        except Exception:
            title = query[:30]

        execute_query(
            "UPDATE sessions SET title = %s, updated_at = %s WHERE id = %s",
            (title, done_at, session_id),
        )
        yield sse_event("title", {"title": title})

    # 9. 推送 done
    yield sse_event("done", {"status": "finished"})


@traceable(name="sse_chat_regenerate", run_type="chain")
def _stream_regenerate(session_id: str, user_id: str, last_user_content: str):
    history = _load_history(session_id)

    assistant_msg_id = str(uuid.uuid4())
    now = int(time.time() * 1000)
    execute_query(
        "INSERT INTO messages (id, session_id, role, content, feedback, created_at, updated_at) VALUES (%s, %s, 'assistant', '', 'none', %s, %s)",
        (assistant_msg_id, session_id, now, now),
    )

    yield sse_event("meta", {"session_id": session_id, "message_id": assistant_msg_id})

    try:
        agent_state = run_agent_rag(last_user_content, session_id=session_id)
    except Exception as e:
        yield sse_event("error", {"code": 5003, "msg": str(e)})
        return

    yield sse_event("citations", merged_search_result(agent_state))
    if AGENT_TRACE_SSE and agent_state.get("scratchpad"):
        yield sse_event("agent_trace", {"scratchpad": agent_state["scratchpad"]})

    full_content = ""
    try:
        for delta in stream_final_answer(agent_state, history):
            if delta:
                full_content += delta
                yield sse_event("message", {"content": delta})
    except Exception as e:
        yield sse_event("error", {"code": 5003, "msg": str(e)})
        return

    done_at = int(time.time() * 1000)
    execute_query(
        "UPDATE messages SET content = %s, updated_at = %s WHERE id = %s",
        (full_content, done_at, assistant_msg_id),
    )
    yield sse_event("done", {"status": "finished"})


# ----------------- 接口实现 -----------------

@router.post("/completions", summary="发送对话并获取流式回复")
def chat_completions(
    req: ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=422, detail="query 不能为空")

    user_id = current_user["id"]
    is_new_session = not req.session_id

    if is_new_session:
        session_id = str(uuid.uuid4())
        now = int(time.time() * 1000)
        execute_query(
            "INSERT INTO sessions (id, user_id, title, created_at, updated_at, is_deleted) VALUES (%s, %s, '', %s, %s, false)",
            (session_id, user_id, now, now),
        )
    else:
        session_id = req.session_id
        session = execute_query(
            "SELECT id FROM sessions WHERE id = %s AND user_id = %s AND is_deleted = false",
            (session_id, user_id),
            fetch_one=True,
        )
        if not session:
            raise HTTPException(status_code=404, detail="会话不存在")

    return StreamingResponse(
        _stream_chat(session_id, user_id, req.query.strip(), is_new_session),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/regenerate", summary="重新生成回复")
def chat_regenerate(
    req: RegenerateRequest,
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["id"]

    # 校验会话归属
    session = execute_query(
        "SELECT id FROM sessions WHERE id = %s AND user_id = %s AND is_deleted = false",
        (req.session_id, user_id),
        fetch_one=True,
    )
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    # 取最后一条 user 消息作为 query
    last_user_msg = execute_query(
        "SELECT id, content FROM messages WHERE session_id = %s AND role = 'user' ORDER BY created_at DESC LIMIT 1",
        (req.session_id,),
        fetch_one=True,
    )
    if not last_user_msg:
        raise HTTPException(status_code=404, detail="该会话中没有用户消息")

    # 删除最后一条 assistant 消息
    execute_query(
        """
        DELETE FROM messages WHERE id = (
            SELECT id FROM messages WHERE session_id = %s AND role = 'assistant'
            ORDER BY created_at DESC LIMIT 1
        )
        """,
        (req.session_id,),
    )

    return StreamingResponse(
        _stream_regenerate(req.session_id, user_id, last_user_msg["content"]),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
