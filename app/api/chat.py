# app/api/chat.py
import os
import json
import time
import uuid
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from openai import OpenAI

from app.api.auth import get_current_user
from app.connect import execute_query
from app.models import ChatRequest, RegenerateRequest

router = APIRouter()

# ----------------- LLM 配置 -----------------
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

llm = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

# ----------------- RAG 占位 -----------------
def retrieve_context(query: str) -> str:
    """向量检索占位，后续接入向量库后在此实现"""
    return ""

# ----------------- SSE 工具 -----------------
def sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

# ----------------- 流式生成器 -----------------
def _stream_chat(session_id: str, user_id: str, query: str, is_new_session: bool):
    """核心 SSE 生成器，供 completions 和 regenerate 复用"""
    # 1. 保存 user message
    user_msg_id = str(uuid.uuid4())
    now = int(time.time() * 1000)
    execute_query(
        "INSERT INTO messages (id, session_id, role, content, created_at, updated_at) VALUES (%s, %s, 'user', %s, %s, %s)",
        (user_msg_id, session_id, query, now, now),
    )

    # 2. 创建 assistant message 占位
    assistant_msg_id = str(uuid.uuid4())
    execute_query(
        "INSERT INTO messages (id, session_id, role, content, feedback, created_at, updated_at) VALUES (%s, %s, 'assistant', '', 'none', %s, %s)",
        (assistant_msg_id, session_id, now, now),
    )

    # 3. 推送 meta
    yield sse_event("meta", {"session_id": session_id, "message_id": assistant_msg_id})

    # 4. 构建 prompt（含 RAG 上下文）
    context = retrieve_context(query)
    system_prompt = "你是一个专业的建筑历史问答助手，专注于中国古代建筑知识。"
    if context:
        system_prompt += f"\n\n参考资料：\n{context}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    # 5. 流式调用 LLM
    full_content = ""
    try:
        stream = llm.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                full_content += delta
                yield sse_event("message", {"content": delta})
    except Exception as e:
        yield sse_event("error", {"code": 5003, "msg": str(e)})
        return

    # 6. 更新 assistant message 内容
    done_at = int(time.time() * 1000)
    execute_query(
        "UPDATE messages SET content = %s, updated_at = %s WHERE id = %s",
        (full_content, done_at, assistant_msg_id),
    )

    # 7. 新会话：生成标题并推送 title 事件
    if is_new_session:
        try:
            title_resp = llm.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "user", "content": f"请用不超过15个字概括以下问题的主题，只输出标题本身：\n{query}"},
                ],
                stream=False,
            )
            title = title_resp.choices[0].message.content.strip()
        except Exception:
            title = query[:30]

        execute_query(
            "UPDATE sessions SET title = %s, updated_at = %s WHERE id = %s",
            (title, done_at, session_id),
        )
        yield sse_event("title", {"title": title})

    # 8. 推送 done
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

    def regenerate_stream():
        # 不保存新的 user message，直接用原 query 重新生成
        assistant_msg_id = str(uuid.uuid4())
        now = int(time.time() * 1000)
        execute_query(
            "INSERT INTO messages (id, session_id, role, content, feedback, created_at, updated_at) VALUES (%s, %s, 'assistant', '', 'none', %s, %s)",
            (assistant_msg_id, req.session_id, now, now),
        )

        yield sse_event("meta", {"session_id": req.session_id, "message_id": assistant_msg_id})

        context = retrieve_context(last_user_msg["content"])
        system_prompt = "你是一个专业的建筑历史问答助手，专注于中国古代建筑知识。"
        if context:
            system_prompt += f"\n\n参考资料：\n{context}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": last_user_msg["content"]},
        ]

        full_content = ""
        try:
            stream = llm.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
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

    return StreamingResponse(
        regenerate_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
