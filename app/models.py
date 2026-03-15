# app/models.py
import time
import uuid
from enum import Enum
from typing import Optional, List
from sqlmodel import SQLModel, Field
from sqlalchemy import Column, String, Float, LargeBinary
from sqlalchemy.dialects.postgresql import JSONB, ARRAY, TSVECTOR
from pydantic import validator, root_validator
import re

# ----------------- users 数据库模型 -----------------
class User(SQLModel, table=True):
    __tablename__ = "users"
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True, max_length=50)
    username: str = Field(unique=True, index=True, max_length=100)
    password_hash: str = Field(max_length=255)
    nickname: Optional[str] = Field(default=None, max_length=100)
    avatar_url: Optional[str] = Field(default=None)
    created_at: int = Field(default_factory=lambda: int(time.time() * 1000))
    updated_at: int = Field(default_factory=lambda: int(time.time() * 1000))
    is_deleted: bool = Field(default=False)

# ----------------- sessions 数据库模型 -----------------
class Session(SQLModel, table=True):
    __tablename__ = "sessions"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True, max_length=50)
    user_id: str = Field(foreign_key="users.id", max_length=50)
    title: Optional[str] = Field(default=None)
    created_at: int = Field(default_factory=lambda: int(time.time() * 1000))
    updated_at: int = Field(default_factory=lambda: int(time.time() * 1000))
    is_deleted: bool = Field(default=False)

# ----------------- messages 数据库模型 -----------------
class Message(SQLModel, table=True):
    __tablename__ = "messages"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True, max_length=50)
    session_id: str = Field(foreign_key="sessions.id", max_length=50)
    role: str = Field(max_length=10)
    content: str = Field()
    feedback: str = Field(default="none", max_length=10)
    remark: Optional[str] = Field(default=None)
    created_at: int = Field(default_factory=lambda: int(time.time() * 1000))
    updated_at: int = Field(default_factory=lambda: int(time.time() * 1000))

# ----------------- 注册API 请求体模型 -----------------
class RegisterRequest(SQLModel):
    username: str = Field(min_length=3, max_length=100)
    password: str = Field(min_length=6)
    nickname: Optional[str] = Field(default=None, max_length=100)
    
    @validator('username')
    def validate_username(cls, v):
        if len(v) > 20:
            raise ValueError('用户名长度不能超过20个字符')
        # 可以添加更多验证规则
        if not v.isalnum() and '_' not in v:
            raise ValueError('用户名只能包含字母、数字和下划线_')
        return v
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 6:
            raise ValueError('密码长度不能少于6个字符')
        return v
    
    @validator('nickname')
    def validate_nickname(cls, v):
        if v is not None and len(v) > 10:
            raise ValueError('昵称长度不能超过10个字符')
        return v

class LoginRequest(SQLModel):
    username: str
    password: str

# ----------------- 对话 API 请求体模型 -----------------
class ChatRequest(SQLModel):
    session_id: Optional[str] = Field(default="")
    query: str

class RegenerateRequest(SQLModel):
    session_id: str

class FeedbackRequest(SQLModel):
    action: str  # like / dislike / none
    remark: Optional[str] = Field(default=None)

# ----------------- API 响应体模型结构 -----------------
class SessionItem(SQLModel):
    """GET /sessions 列表项"""
    id: str
    title: Optional[str]
    updated_at: int

class MessageItem(SQLModel):
    """GET /sessions/{session_id}/messages 列表项"""
    id: str
    role: str
    content: str
    feedback: Optional[str] = None  # 仅 assistant 消息携带
    created_at: int


# ==================== RAG 知识库数据库模型 ====================

class ContentTypeEnum(str, Enum):
    original  = "原文"
    annotation = "注解"
    translation = "译文"
    other     = "其他"

# ----------------- documents 书籍知识库表 -----------------
class Document(SQLModel, table=True):
    __tablename__ = "documents"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    name: str = Field()
    authors: Optional[List[str]] = Field(default=None, sa_column=Column(ARRAY(String)))
    publish_info: Optional[str] = Field(default=None)          # 出版社、出版时间等版权信息
    # SQLAlchemy Declarative 中 metadata 是保留属性名，Python 侧需避开；
    # 数据库列名仍保持 metadata，避免影响现有表结构。
    doc_metadata: Optional[dict] = Field(default=None, sa_column=Column("metadata", JSONB))
    chunks_count: int = Field(default=0)
    vector_dimensions: int = Field(default=1024)
    created_at: int = Field(default_factory=lambda: int(time.time() * 1000))
    updated_at: int = Field(default_factory=lambda: int(time.time() * 1000))

# ----------------- chunks 文本块表 -----------------
class Chunk(SQLModel, table=True):
    __tablename__ = "chunks"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    document_id: str = Field(foreign_key="documents.id")
    content: str = Field()
    # 同上，Python 属性名避免使用保留字 metadata。
    chunk_metadata: Optional[dict] = Field(default=None, sa_column=Column("metadata", JSONB))
    # embedding 用 ARRAY(Float) 存储，避免强依赖 pgvector Python 包；
    # 数据端写入时通过原生 SQL 使用 ::vector 强制转型写入真实 vector 列
    embedding: Optional[List[float]] = Field(default=None, sa_column=Column(ARRAY(Float)))
    ts_vector: Optional[str] = Field(default=None, sa_column=Column(TSVECTOR))
    content_type: Optional[str] = Field(default=None)          # ContentTypeEnum 值
    toc_path: Optional[List[str]] = Field(default=None, sa_column=Column(ARRAY(String)))
    has_images: bool = Field(default=False)
    has_annotation: bool = Field(default=False)
    annotation: Optional[List[str]] = Field(default=None, sa_column=Column(ARRAY(String)))
    created_at: int = Field(default_factory=lambda: int(time.time() * 1000))
    updated_at: int = Field(default_factory=lambda: int(time.time() * 1000))

# ----------------- images 图片表 -----------------
class Image(SQLModel, table=True):
    __tablename__ = "images"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    document_id: Optional[str] = Field(default=None, foreign_key="documents.id")
    name: str = Field()
    description: Optional[str] = Field(default=None)
    url: Optional[str] = Field(default=None)
    binary_data: Optional[bytes] = Field(default=None, sa_column=Column(LargeBinary))
    created_at: int = Field(default_factory=lambda: int(time.time() * 1000))
    updated_at: int = Field(default_factory=lambda: int(time.time() * 1000))

# ----------------- chunk_images chunk与图片关联表 -----------------
class ChunkImage(SQLModel, table=True):
    __tablename__ = "chunk_images"

    chunk_id: str = Field(foreign_key="chunks.id", primary_key=True)
    image_id: str = Field(foreign_key="images.id", primary_key=True)
