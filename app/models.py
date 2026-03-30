# app/models.py
import time
import uuid
from enum import Enum
from typing import List, Optional
from sqlmodel import SQLModel, Field
from sqlalchemy import Column, String
from sqlalchemy.dialects.postgresql import JSONB, ARRAY, TSVECTOR
from pgvector.sqlalchemy import Vector
from pydantic import validator, root_validator
import re

# ----------------- users 数据库模型 -----------------
class User(SQLModel, table=True):
    __tablename__ = "users"  # pyright: ignore[reportAssignmentType]

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
    __tablename__ = "sessions"  # pyright: ignore[reportAssignmentType]

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True, max_length=50)
    user_id: str = Field(foreign_key="users.id", max_length=50)
    title: Optional[str] = Field(default=None)
    created_at: int = Field(default_factory=lambda: int(time.time() * 1000))
    updated_at: int = Field(default_factory=lambda: int(time.time() * 1000))
    is_deleted: bool = Field(default=False)

# ----------------- messages 数据库模型 -----------------
class Message(SQLModel, table=True):
    __tablename__ = "messages"  # pyright: ignore[reportAssignmentType]

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
    """文本 chunk 的内容类型"""
    original_text = "original_text"
    annotation = "annotation"
    modern_translation = "modern_translation"
    interpretation = "interpretation"
    others_text = "others_text"

class ChunkTypeEnum(str, Enum):
    """relations 表中 source_type / target_type 使用，文本类型与 ContentTypeEnum 一致，图像固定为 image"""
    original_text = "original_text"
    annotation = "annotation"
    modern_translation = "modern_translation"
    interpretation = "interpretation"
    others_text = "others_text"
    image = "image"

class RelationTypeEnum(str, Enum):
    illustrates = "illustrates"
    annotates = "annotates"

# ----------------- documents 书籍知识库表 -----------------
class Document(SQLModel, table=True):
    __tablename__ = "documents"  # pyright: ignore[reportAssignmentType]

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    name: str = Field()
    authors: Optional[List[str]] = Field(default=None, sa_column=Column(ARRAY(String)))
    other_metadata: Optional[dict] = Field(default=None, sa_column=Column(JSONB))
    content: Optional[str] = Field(default=None)
    created_at: int = Field(default_factory=lambda: int(time.time() * 1000))
    updated_at: int = Field(default_factory=lambda: int(time.time() * 1000))

# ----------------- text_chunks 文本块表 -----------------
class TextChunk(SQLModel, table=True):
    __tablename__ = "text_chunks"  # pyright: ignore[reportAssignmentType]

    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    content_type: Optional[str] = Field(default=None)
    chunk_size: Optional[int] = Field(default=None)
    main_text: str = Field()
    book_id: str = Field(foreign_key="documents.id")
    closest_title: Optional[str] = Field(default=None)
    toc_path: Optional[List[str]] = Field(default=None, sa_column=Column(ARRAY(String)))
    search_text: Optional[str] = Field(default=None)
    ts_vector: Optional[str] = Field(default=None, sa_column=Column(TSVECTOR))
    other_metadata: Optional[dict] = Field(default=None, sa_column=Column(JSONB))
    embedding_values: Optional[List[float]] = Field(default=None, sa_column=Column(Vector(1024)))
    created_at: int = Field(default_factory=lambda: int(time.time() * 1000))
    updated_at: int = Field(default_factory=lambda: int(time.time() * 1000))

# ----------------- image_chunks 图像块表 -----------------
class ImageChunk(SQLModel, table=True):
    __tablename__ = "image_chunks"  # pyright: ignore[reportAssignmentType]

    image_id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    title: Optional[str] = Field(default=None)
    image_uri: Optional[str] = Field(default=None)
    local_path: Optional[str] = Field(default=None)
    alt_text: Optional[str] = Field(default=None)
    caption: Optional[str] = Field(default=None)
    book_id: Optional[str] = Field(default=None, foreign_key="documents.id")
    closest_title: Optional[str] = Field(default=None)
    toc_path: Optional[List[str]] = Field(default=None, sa_column=Column(ARRAY(String)))
    search_text: Optional[str] = Field(default=None)
    ts_vector: Optional[str] = Field(default=None, sa_column=Column(TSVECTOR))
    embedding_values: Optional[List[float]] = Field(default=None, sa_column=Column(Vector(1024)))
    format: Optional[str] = Field(default=None)
    created_at: int = Field(default_factory=lambda: int(time.time() * 1000))
    updated_at: int = Field(default_factory=lambda: int(time.time() * 1000))

# ----------------- relations 关联关系表 -----------------
class Relation(SQLModel, table=True):
    __tablename__ = "relations"  # pyright: ignore[reportAssignmentType]

    relation_id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    source_type: str = Field()
    source_id: str = Field()
    target_type: str = Field()
    target_id: str = Field()
    relation_type: str = Field()
    created_at: int = Field(default_factory=lambda: int(time.time() * 1000))
