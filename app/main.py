# 程序的入口 (FastAPI 或 Flask 实例)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import auth, sessions, chat, messages

app = FastAPI(title="YingZaoFaShi RAG API")


# 配置允许跨域的列表
origins = [
    "http://localhost:5173", # 你的前端开发地址
    "http://127.0.0.1:5173",
]

# 配置跨域，解决前端 Web 调试时的跨域问题
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 路由注册，将各个模块的路由注册到住应用
app.include_router(auth.router,     prefix="/api/v1/auth",tags=["用户认证"])
app.include_router(sessions.router, prefix="/api/v1/sessions",tags=["会话管理"])
app.include_router(chat.router,     prefix="/api/v1/chat",tags=["聊天服务"])
app.include_router(messages.router, prefix="/api/v1/messages",tags=["用户反馈"])


@app.get("/")
def root():
    return {"message": "Welcome to YingZaoFaShi API Server. Base path is /api/v1"}

if __name__ == "__main__":
    import uvicorn
    # 建议使用 uvicorn 启动
    uvicorn.run("app.main:app", host="0.0.0.0", port=8080, reload=True)