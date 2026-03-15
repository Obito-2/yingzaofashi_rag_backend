# app/api/sessions.py
import time
from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.auth import get_current_user
from app.connect import execute_query

router = APIRouter()


@router.get("", summary="获取历史会话列表")
def get_sessions(
    page: int = Query(default=1, ge=1),
    size: int = Query(default=20, ge=1, le=100),
    current_user: dict = Depends(get_current_user),
):
    offset = (page - 1) * size
    rows = execute_query(
        """
        SELECT id, title, updated_at
        FROM sessions
        WHERE user_id = %s AND is_deleted = false
        ORDER BY updated_at DESC
        LIMIT %s OFFSET %s
        """,
        (current_user["id"], size, offset),
        fetch_all=True,
    )
    return {
        "code": 200,
        "msg": "success",
        "data": [dict(r) for r in (rows or [])],
    }


@router.get("/{session_id}/messages", summary="获取指定会话的消息记录")
def get_messages(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    # 校验会话归属
    session = execute_query(
        "SELECT id FROM sessions WHERE id = %s AND user_id = %s AND is_deleted = false",
        (session_id, current_user["id"]),
        fetch_one=True,
    )
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    rows = execute_query(
        "SELECT id, role, content, feedback, created_at FROM messages WHERE session_id = %s ORDER BY created_at ASC",
        (session_id,),
        fetch_all=True,
    )

    data = []
    for r in (rows or []):
        item = {
            "id": r["id"],
            "role": r["role"],
            "content": r["content"],
            "created_at": r["created_at"],
        }
        if r["role"] == "assistant":
            item["feedback"] = r["feedback"]
        data.append(item)

    return {"code": 200, "msg": "success", "data": data}


@router.delete("/{session_id}", summary="删除会话")
def delete_session(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    session = execute_query(
        "SELECT id FROM sessions WHERE id = %s AND user_id = %s AND is_deleted = false",
        (session_id, current_user["id"]),
        fetch_one=True,
    )
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")

    execute_query(
        "UPDATE sessions SET is_deleted = true, updated_at = %s WHERE id = %s",
        (int(time.time() * 1000), session_id),
    )
    return {"code": 200, "msg": "success"}
