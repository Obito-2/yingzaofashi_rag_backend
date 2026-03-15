**RAG 问答应用 API 接口文档**

# 0. 全局约定

- **Base URL**: `https://gpumcpsgnnjp.sealoshzh.site` 
- **数据传输格式**: `application/json` (除核心问答接口采用 `text/event-stream` 外)
- **鉴权方式**: 统一在 HTTP Header 中携带 `User-Id: <用户唯一标识>` (或根据项目后续需求改为 `Authorization: Bearer <token>`)。
- **时间戳**: 所有时间相关的字段均采用 **13位 Unix 时间戳 (毫秒)**。

### 0.1 标准失败返回值格式
非流式接口在发生业务错误或系统异常时，统一返回如下结构（HTTP 状态码视情况为 200 或 4xx/5xx）：
```json
{
  "code": 400, // 或 500, 404, 422(参数校验失败)
  "msg": "错误提示信息",
  "data": null
}
```
### 0.2 错误码对照表
|HTTP |状态码|	业务码 (code)|	说明|
| :--- | :--- | :--- | :--- |
|400	|4000	|请求参数错误（如缺少必填字段、格式错误）|
|401	|4001|	未认证或认证失败（如缺少 User-Id）|
|404	|4004	|资源不存在（如会话 ID 无效）|
|422	|4220	|参数校验失败（如 query 为空字符串）|
|500	|5000|	服务器内部错误|
|503	|5003	|服务不可用（如 RAG 检索超时、模型服务异常）|
|5001	|5001	|流式响应中的错误（仅出现在 SSE error 事件中）|

# 1. 会话与历史记录管理

## 1.1 获取历史会话列表
前端在进入首页或初始加载侧边栏时调用，展示左侧的历史会话记录。

接口路径: GET /sessions

Headers:
User-Id: string (必填)

Query 参数:

page (int, 可选): 页码，默认 1

size (int, 可选): 每页数量，默认 20
成功响应示例:
{
  "code": 200,
  "msg": "success",
  "data": [
    {
      "id": "session_123456",
      "title": "什么是《营造法式》中的「材分制」？",
      "updated_at": 1715432123456
    }
  ]
}

## 1.2 获取指定会话的消息记录
用户在侧边栏点击某条历史记录时调用，用于在主聊天区还原当时的对话上下文。

接口路径: GET /sessions/{session_id}/messages

Headers:

User-Id: string (必填)

Path 参数:

session_id (string, 必填): 会话的唯一 ID

成功响应示例:
{
  "code": 200,
  "msg": "success",
  "data": [
    {
      "id": "msg_001",
      "role": "user",
      "content": "分析宋代建筑与唐代建筑的结构差异",
      "created_at": 1715432120000
    },
    {
      "id": "msg_002",
      "role": "assistant",
      "content": "宋代建筑相比唐代，其结构差异主要体现在...",
      "feedback": "none", // 值域: like, dislike, none (用于回显评价状态)
      "created_at": 1715432125000
    }
  ]
}
## 1.3 删除会话
用户清理特定历史记录的需求。

接口路径: DELETE /sessions/{session_id}

Headers:

User-Id: string (必填)

Path 参数:

session_id (string, 必填): 会话的唯一 ID

成功响应示例:
{
  "code": 200,
  "msg": "success"
}

# 2. 核心问答接口 (流式输出 SSE)

⚠️ 前端对接须知：
由于微信小程序不支持原生的 EventSource，前端需使用 uni.request 配合 enableChunked: true，在 onChunkReceived 中手动将 ArrayBuffer 转为字符串，并按照 \n\n 切割解析 event 和 data。

## 2.1 发送对话并获取流式回复
当 session_id 为空时，后端会生成全新会话并触发“标题总结事件”。

接口路径: POST /chat/completions

Headers:

User-Id: string (必填)

Accept: text/event-stream

请求体 (JSON):

{
  "session_id": "", // (可选) 如果是全新提问，传空字符串或不传；如果是历史会话继续提问，传入对应的 session_id
  "query": "如何通过榫卯结构实现建筑的抗震？" // (必填) 用户输入的问题
}

成功响应流 (SSE):
后端以 event: <类型>\ndata: <JSON数据>\n\n 格式不断推送增量块。

| 发生顺序 | Event 类型 | 说明 | Data 示例 |
| :--- | :--- | :--- | :--- |
| 1 | meta | 元数据：返回本次对话实际的 session_id 和 message_id | `{"session_id": "session_123", "message_id": "msg_abc"}` |
| 2 | message | 文本增量：大模型生成的正文内容（高频触发） | `{"content": "榫"}` |
| 3 | title | 标题生成：仅首轮对话触发。前端收到后更新会话列表 | `{"title": "榫卯结构抗震原理"}` |
| 4 | done | 结束标识：代表本次流式输出彻底完成 | `{"status": "finished"}` |

## 2.2 重新生成回复
当用户对最后一次 AI 回复不满意时调用，针对最新 Query 重新走一遍 RAG 推流。

接口路径: POST /chat/regenerate

Headers:

User-Id: string (必填)

Accept: text/event-stream

请求体 (JSON):
{
  "session_id": "session_123456" // 必填，当前所在会话的 ID
}
成功响应流 (SSE):
与 2.1 接口完全一致，持续推送 meta -> message -> done 事件。

# 3. 用户交互与反馈
## 3.1 提交点赞/踩反馈
持续优化检索和生成质量的依据。

接口路径: POST /messages/{message_id}/feedback

Headers:

User-Id: string (必填)

Path 参数:

message_id (string, 必填): 2.1 或 2.2 接口中通过 meta 事件返回的消息 ID。

请求体 (JSON):
{
  "action": "dislike", // 必填，值域: "like" (点赞), "dislike" (踩), "none" (取消表态)
  "remark": "回答不准确，没有结合营造法式的原文" // 可选，主要用于踩的时候收集用户的备注原因
}
成功响应示例:
{
  "code": 200,
  "msg": "success"
}