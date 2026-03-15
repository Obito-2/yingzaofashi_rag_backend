# app/api/messages.py
import time
from fastapi import APIRouter, Depends, HTTPException

from app.api.auth import get_current_user
from app.connect import execute_query
from app.models import FeedbackRequest

router = APIRouter()

VALID_ACTIONS = {"like", "dislike", "none"}


@router.post("/{message_id}/feedback", summary="提交点赞/踩反馈")
def submit_feedback(
    message_id: str,
    req: FeedbackRequest,
    current_user: dict = Depends(get_current_user),
):
    if req.action not in VALID_ACTIONS:
        raise HTTPException(status_code=400, detail="action 值无效，仅支持 like / dislike / none")

    # 校验消息存在且归属当前用户（通过 session → user 关联）
    msg = execute_query(
        """
        SELECT m.id FROM messages m
        JOIN sessions s ON s.id = m.session_id
        WHERE m.id = %s AND s.user_id = %s AND m.role = 'assistant'
        """,
        (message_id, current_user["id"]),
        fetch_one=True,
    )
    if not msg:
        raise HTTPException(status_code=404, detail="消息不存在")

    execute_query(
        "UPDATE messages SET feedback = %s, remark = %s, updated_at = %s WHERE id = %s",
        (req.action, req.remark, int(time.time() * 1000), message_id),
    )
    return {"code": 200, "msg": "success"}
