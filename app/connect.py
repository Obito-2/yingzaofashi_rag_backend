# app/connect.py
import os
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor  # 返回字典格式的结果

# 从环境变量读取数据库配置（Sealos 会自动注入）
DB_HOST = os.getenv("DB_HOST", "test-db-postgresql.ns-q5nnz4bx.svc")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "lchgjt88")

# 创建连接池（最小1个连接，最大10个连接）
connection_pool = psycopg2.pool.SimpleConnectionPool(
    1, 10,
    host=DB_HOST,
    port=DB_PORT,
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    cursor_factory=RealDictCursor  # 让查询结果以字典形式返回
)

def get_connection():
    """从连接池获取一个连接"""
    return connection_pool.getconn()
def release_connection(conn):
    """释放连接回连接池"""
    connection_pool.putconn(conn)

def execute_query(query, params=None, fetch_one=False, fetch_all=False):
    """
    执行 SQL 并自动管理连接和游标
    fetch_one: 返回单条记录（字典）
    fetch_all: 返回所有记录（字典列表）
    否则执行 INSERT/UPDATE/DELETE 并提交
    """
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(query, params)
        if fetch_one:
            result = cur.fetchone()
        elif fetch_all:
            result = cur.fetchall()
        else:
            conn.commit()
            result = None
        return result
    except Exception as e:
        conn.rollback()  # 出错时回滚
        raise e
    finally:
        cur.close()
        release_connection(conn)