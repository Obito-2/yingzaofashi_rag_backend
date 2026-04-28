# AR 设备集成指南

## 接口说明

**端点**：`POST /ar/chat`

无需认证，内网直接访问。

### 请求格式

```json
{
  "query": "斗栱是什么结构？"
}
```

### 响应格式（SSE 流式）

响应为 Server-Sent Events (SSE) 格式，包含以下事件类型：

#### 1. **citations** - 引文信息
```json
{
  "event": "citations",
  "data": {
    "citations": [
      {
        "id": "doc1",
        "content": "斗栱是传统建筑中的...",
        "source": "营造法式"
      }
    ]
  }
}
```

#### 2. **agent_trace** - Agent 思考过程（可选）
```json
{
  "event": "agent_trace",
  "data": {
    "scratchpad": [
      "第1轮检索：查询'斗栱定义'",
      "找到3条相关条文"
    ]
  }
}
```

#### 3. **clarification** - 澄清问题（若问题模糊）
```json
{
  "event": "clarification",
  "data": {
    "question": "你是想了解斗栱的结构，还是施工方法？"
  }
}
```

#### 4. **message** - 流式答案（多次）
```json
{
  "event": "message",
  "data": {
    "content": "斗栱是"
  }
}
```

#### 5. **done** - 完成标记
```json
{
  "event": "done",
  "data": {
    "status": "finished"
  }
}
```

#### 6. **error** - 错误信息
```json
{
  "event": "error",
  "data": {
    "code": 5003,
    "msg": "具体错误原因"
  }
}
```

---


---

## 部署配置

### 1. 服务器启动

```bash
cd yingzaofashi_rag_backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 2. 环境变量（可选）

```bash
# 启用 Agent 思考过程推送
AGENT_TRACE_SSE=true

# Agent 配置
AGENT_MAX_RETRIEVE_DEPTH=3
AGENT_GATE_MODE=on
```

### 3. 防火墙配置（内网）

确保 AR 设备所在网络可以访问服务器的 8000 端口：
- Windows：允许 Python/uvicorn 进程通过防火墙
- Linux/Mac：`ufw allow 8000`

### 4. 跨域配置

已在 `app/main.py` 中配置，允许内网所有来源。如需限制，可修改 `origins` 列表。

---

## 测试

### 使用 curl 测试

```bash
curl -X POST http://localhost:8000/ar/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"斗栱是什么？"}' \
  -N  # -N: 禁用缓冲以查看实时 SSE 流
```

### 使用 Python 测试

```python
import requests
import json

response = requests.post(
    "http://localhost:8000/ar/chat",
    json={"query": "斗栱是什么？"},
    stream=True
)

for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))
```

---

## 常见问题

### Q: 为什么没有收到 Agent 思考过程？
**A**: 需要在环境变量中设置 `AGENT_TRACE_SSE=true`

### Q: 多个 AR 设备同时提问会不会冲突？
**A**: 不会。每个请求都是独立的，无会话状态。

### Q: 内网 IP 访问时出现 CORS 错误？
**A**: 检查 `app/main.py` 中 `origins` 列表是否包含该 IP 段。

### Q: Unity 接收流式数据时出现乱码？
**A**: 确保使用 UTF-8 编码。检查 `UploadHandlerRaw` 和 `StreamReader` 的编码设置。

---

## API 文档（OpenAPI/Swagger）

启动服务后，访问：
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

可以在 Web 界面中测试 `/ar/chat` 接口（若浏览器支持 SSE）。
