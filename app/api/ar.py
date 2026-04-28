# app/api/ar.py
"""AR 设备专用接口 - 无认证、无会话、流式 SSE 响应"""

import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langsmith import traceable

from app.agent import run_agent_rag, stream_final_answer, merged_search_result

router = APIRouter(prefix="/ar", tags=["ar"])


class ARChatRequest(BaseModel):
    """AR 设备提问请求"""
    query: str  # 用户问题


def sse_event(event: str, data: dict) -> str:
    """格式化 SSE 事件"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@traceable(name="ar_chat", run_type="chain")
def _stream_ar_chat(query: str):
    """
    AR 设备专用流式生成器
    - 无认证、无会话、无历史
    - 单轮问答，流式推送结果
    """
    # 1. 执行 Agent（无会话 ID）
    try:
        agent_state = run_agent_rag(query)
    except Exception as e:
        yield sse_event("error", {"code": 5003, "msg": str(e)})
        return

    # 2. 推送引文信息
    yield sse_event("citations", merged_search_result(agent_state))

    # 3. 推送 Agent 思考过程（可选，若启用）
    if agent_state.get("scratchpad"):
        yield sse_event("agent_trace", {"scratchpad": agent_state["scratchpad"]})

    # 4. 推送澄清问题（若问题模糊）
    if agent_state.get("clarification_question"):
        yield sse_event(
            "clarification",
            {"question": agent_state["clarification_question"]}
        )
        return

    # 5. 流式推送最终答案（无历史）
    full_content = ""
    try:
        for delta in stream_final_answer(agent_state, []):
            if delta:
                full_content += delta
                yield sse_event("message", {"content": delta})
    except Exception as e:
        yield sse_event("error", {"code": 5003, "msg": str(e)})
        return

    # 6. 推送完成事件
    yield sse_event("done", {"status": "finished"})


@router.post("/chat", summary="AR 设备知识问答（无认证）")
def ar_chat(req: ARChatRequest):
    """
    AR 设备专用接口 - 流式 SSE 返回

    使用示例（Unity C#）：
    ```csharp
    var request = new UnityWebRequest("http://内网IP:8000/ar/chat", "POST");
    var json = JsonConvert.SerializeObject(new { query = "斗栱是什么？" });
    request.uploadHandler = new UploadHandlerRaw(Encoding.UTF8.GetBytes(json));
    request.SetRequestHeader("Content-Type", "application/json");

    // 处理 SSE 流
    var response = UnityWebRequest.Post("...", "application/json");
    ```

    **SSE 事件说明：**
    - `citations`: {merged search results} - 引文信息
    - `agent_trace`: {scratchpad: [...]} - Agent 思考过程
    - `clarification`: {question: str} - 需要澄清的问题
    - `message`: {content: str} - 流式答案（多次）
    - `done`: {status: "finished"} - 完成标记
    - `error`: {code, msg} - 错误信息
    """
    if not req.query or not req.query.strip():
        return StreamingResponse(
            (sse_event("error", {"code": 400, "msg": "query 不能为空"}) for _ in [1]),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return StreamingResponse(
        _stream_ar_chat(req.query.strip()),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
