# app/api/login.py
import os
import time
import uuid
import jwt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext

from app.models import RegisterRequest, LoginRequest
from app.connect import execute_query

router = APIRouter()
security = HTTPBearer()

# ----------------- 配置区 -----------------
# 密码哈希上下文
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

# JWT 配置项
JWT_SECRET = os.getenv("JWT_SECRET", "your-super-secret-key-change-in-prod")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_DAYS = 7

# ----------------- 工具函数 -----------------
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(user_id: str, username: str) -> str:
    expire = time.time() + (JWT_EXPIRATION_DAYS * 24 * 60 * 60)
    payload = {
        "user_id": user_id,
        "username": username,
        "exp": expire
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

# ----------------- 认证依赖 (用于保护后续接口) -----------------
def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """校验 JWT Token，并返回当前登录用户的信息字典"""
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id: str = payload.get("user_id")
        if user_id is None:
            raise ValueError()
    except (jwt.ExpiredSignatureError, jwt.PyJWTError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="token无效或过期"
        )

    # 验证用户是否存在且未被删除
    query = "SELECT * FROM users WHERE id = %s AND is_deleted = false"
    user = execute_query(query, (user_id,), fetch_one=True)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户不存在或已被禁用")
    
    return user

# ----------------- 接口实现 -----------------

@router.post("/register", summary="用户注册")
def register(req: RegisterRequest):
    # 1. 检查用户名是否存在
    check_query = "SELECT id FROM users WHERE username = %s AND is_deleted = false"
    existing_user = execute_query(check_query, (req.username,), fetch_one=True)
    if existing_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="用户名已存在")
        
    # 转换为 bytes 计算长度
    password_bytes = req.password.encode('utf-8')
    if len(password_bytes) > 72:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="密码超出长度限制（最大支持72个英文字符或约24个汉字）"
        )

    # 2. 创建新用户数据
    user_id = str(uuid.uuid4())
    hashed_pwd = get_password_hash(req.password)
    now_ms = int(time.time() * 1000)

    insert_query = """
        INSERT INTO users (id, username, password_hash, nickname, created_at, updated_at, is_deleted)
        VALUES (%s, %s, %s, %s, %s, %s, false)
    """
    execute_query(insert_query, (user_id, req.username, hashed_pwd, req.nickname, now_ms, now_ms))

    # 3. 生成 Token
    token = create_access_token(user_id, req.username)

    # 4. 返回响应
    return {
        "code": 200,
        "message": "注册成功",
        "data": {
            "token": token,
            "user": {
                "id": user_id,
                "username": req.username,
                "nickname": req.nickname,
                "avatar": None
            }
        }
    }


@router.post("/login", summary="用户登录")
def login(req: LoginRequest):
    # 1. 查询用户
    query = "SELECT id, username, password_hash, nickname, avatar_url FROM users WHERE username = %s AND is_deleted = false"
    user = execute_query(query, (req.username,), fetch_one=True)

    # 2. 校验密码
    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户名或密码错误")

    # 3. 签发 Token
    token = create_access_token(user["id"], user["username"])

    return {
        "code": 200,
        "message": "登录成功",
        "data": {
            "token": token,
            "user": {
                "id": user["id"],
                "username": user["username"],
                "nickname": user["nickname"],
                "avatar": user["avatar_url"]
            }
        }
    }


@router.get("/info", summary="获取当前用户信息")
def get_user_info(current_user: dict = Depends(get_current_user)):
    # current_user 已经在 get_current_user 依赖中完成解析和数据库查询
    return {
        "code": 200,
        "message": "success",
        "data": {
            "id": current_user["id"],
            "username": current_user["username"],
            "nickname": current_user["nickname"],
            "avatar": current_user["avatar_url"],
            "created_at": current_user["created_at"]
        }
    }
