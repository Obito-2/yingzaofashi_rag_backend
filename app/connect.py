# app/connect.py
import os
import sys
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor  # 返回字典格式的结果
from urllib.parse import urlparse
from dotenv import load_dotenv

# 让直接运行 `python app/connect.py ...` 也能读取 .env 中的 DB_URL
load_dotenv()

# 直接粘贴最新连接地址到这里即可，或通过环境变量 DB_URL 覆盖
_DEFAULT_URL = "postgresql://postgres:lchgjt88@dbconn.sealoshzh.site:47025/postgres"

_url = urlparse(os.getenv("DB_URL", _DEFAULT_URL))

DB_HOST     = _url.hostname
DB_PORT     = str(_url.port or 5432)
DB_NAME     = _url.path.lstrip("/") or "postgres"
DB_USER     = _url.username
DB_PASSWORD = _url.password

connection_pool: pool.SimpleConnectionPool | None = None


class DatabaseUnavailableError(RuntimeError):
    pass


def _ensure_pool() -> pool.SimpleConnectionPool:
    """
    按需初始化连接池：
    - 避免模块 import 阶段就强依赖数据库（会导致服务启动直接失败）
    - 真正需要执行 SQL 时再建立连接
    """
    global connection_pool
    if connection_pool is not None:
        return connection_pool
    try:
        connection_pool = pool.SimpleConnectionPool(
            1,
            10,
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            cursor_factory=RealDictCursor,  # 让查询结果以字典形式返回
        )
        return connection_pool
    except psycopg2.OperationalError as e:
        raise DatabaseUnavailableError(
            f"数据库连接失败：host={DB_HOST} port={DB_PORT} dbname={DB_NAME} user={DB_USER}；"
            "请检查 DB_URL/网络/端口/服务是否已启动。"
        ) from e

def get_connection():
    """从连接池获取一个连接"""
    return _ensure_pool().getconn()
def release_connection(conn):
    """释放连接回连接池"""
    p = _ensure_pool()
    p.putconn(conn)

def execute_query(query, params=None, fetch_one=False, fetch_all=False):
    """
    执行 SQL 并自动管理连接和游标
    fetch_one: 返回单条记录（字典）
    fetch_all: 返回所有记录（字典列表）
    否则执行 INSERT/UPDATE/DELETE 并提交
    """
    try:
        conn = get_connection()
    except DatabaseUnavailableError:
        raise
    except psycopg2.OperationalError as e:
        raise DatabaseUnavailableError("数据库连接失败，请检查连接配置与网络状态。") from e

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
    except psycopg2.OperationalError as e:
        # 运行时断连 / 连接池取到坏连接等
        raise DatabaseUnavailableError("数据库连接异常（可能已断开），请稍后重试。") from e
    except Exception as e:
        conn.rollback()  # 出错时回滚
        raise e
    finally:
        cur.close()
        release_connection(conn)


def parse_table_name(table_name: str):
    """
    解析表名，支持:
    - users -> public.users
    - public.users -> public.users
    """
    if "." in table_name:
        schema_name, pure_table_name = table_name.split(".", 1)
    else:
        schema_name, pure_table_name = "public", table_name
    return schema_name, pure_table_name


def fetch_table_schema(table_name: str):
    """查询指定表的结构信息。"""
    schema_name, pure_table_name = parse_table_name(table_name)
    sql = """
    SELECT
        ordinal_position,
        column_name,
        data_type,
        is_nullable,
        column_default
    FROM information_schema.columns
    WHERE table_schema = %s
      AND table_name = %s
    ORDER BY ordinal_position;
    """
    return execute_query(sql, params=(schema_name, pure_table_name), fetch_all=True)


def fetch_table_preview(table_name: str, limit: int = 10):
    """查询指定表前 N 条数据。"""
    schema_name, pure_table_name = parse_table_name(table_name)
    sql = f'SELECT * FROM "{schema_name}"."{pure_table_name}" LIMIT %s;'
    return execute_query(sql, params=(limit,), fetch_all=True)


def print_table_info(table_name: str):
    """打印表结构与前10条数据。"""
    schema = fetch_table_schema(table_name)
    if not schema:
        print(f"表不存在或无可读列: {table_name}")
        return

    preview_rows = fetch_table_preview(table_name, limit=10) or []

    print(f"===== 表结构: {table_name} =====")
    for col in schema:
        print(
            f"{col['ordinal_position']:>2}. "
            f"{col['column_name']} "
            f"({col['data_type']}) "
            f"NULLABLE={col['is_nullable']} "
            f"DEFAULT={col['column_default']}"
        )

    print(f"\n===== 前 {len(preview_rows)} 条数据: {table_name} =====")
    for i, row in enumerate(preview_rows, start=1):
        print(f"{i:>2}. {row}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python app/connect.py <table_name>")
        print("示例: python app/connect.py users")
        print("示例: python app/connect.py public.users")
        sys.exit(1)

    input_table_name = sys.argv[1]
    print_table_info(input_table_name)