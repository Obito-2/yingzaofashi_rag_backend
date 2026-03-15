"""查询 chunks 表前 N 条数据并可视化展示"""
import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.connect import execute_query
from tabulate import tabulate


def ts_to_str(ts_ms):
    """毫秒时间戳转可读字符串"""
    if not ts_ms:
        return "-"
    return datetime.fromtimestamp(ts_ms / 1000).strftime("%Y-%m-%d %H:%M:%S")


def truncate(text, max_len=60):
    if not text:
        return "-"
    text = str(text).replace("\n", " ")
    return text[:max_len] + "..." if len(text) > max_len else text


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    sql = """
        SELECT id, document_id, content, metadata, content_type,
               toc_path, has_images, has_annotation, annotation,
               created_at, updated_at
        FROM chunks
        ORDER BY created_at ASC
        LIMIT %s
    """
    rows = execute_query(sql, (limit,), fetch_all=True)

    if not rows:
        print("chunks 表中没有数据。")
        return

    print(f"\n{'='*100}")
    print(f"  chunks 表数据概览（前 {limit} 条，共查到 {len(rows)} 条）")
    print(f"{'='*100}\n")

    # --- 汇总表 ---
    table_data = []
    for i, row in enumerate(rows, 1):
        table_data.append([
            i,
            row["id"][:8] + "...",
            row["document_id"][:8] + "..." if row.get("document_id") else "-",
            truncate(row.get("content"), 40),
            row.get("content_type") or "-",
            "是" if row.get("has_images") else "否",
            "是" if row.get("has_annotation") else "否",
            ts_to_str(row.get("created_at")),
        ])

    headers = ["#", "ID", "文档ID", "内容(截断)", "类型", "含图", "含注", "创建时间"]
    print(tabulate(table_data, headers=headers, tablefmt="grid", stralign="left"))

    # --- 逐条详情 ---
    print(f"\n{'='*100}")
    print("  逐条详细信息")
    print(f"{'='*100}")

    for i, row in enumerate(rows, 1):
        print(f"\n{'─'*80}")
        print(f"  [{i}] chunk id: {row['id']}")
        print(f"{'─'*80}")
        print(f"  document_id  : {row.get('document_id', '-')}")
        print(f"  content_type : {row.get('content_type') or '-'}")
        print(f"  toc_path     : {row.get('toc_path') or '-'}")
        print(f"  has_images   : {'是' if row.get('has_images') else '否'}")
        print(f"  has_annotation: {'是' if row.get('has_annotation') else '否'}")
        print(f"  annotation   : {truncate(str(row.get('annotation')), 80) if row.get('annotation') else '-'}")
        print(f"  created_at   : {ts_to_str(row.get('created_at'))}")
        print(f"  updated_at   : {ts_to_str(row.get('updated_at'))}")

        metadata = row.get("metadata")
        if metadata:
            print(f"  metadata     :")
            if isinstance(metadata, dict):
                for k, v in metadata.items():
                    print(f"    {k}: {truncate(str(v), 70)}")
            else:
                print(f"    {truncate(str(metadata), 80)}")

        content = row.get("content", "")
        print(f"  content ({len(content)} 字符):")
        print(f"    {truncate(content, 200)}")

    # --- 统计信息 ---
    count_sql = "SELECT COUNT(*) AS total FROM chunks"
    total = execute_query(count_sql, fetch_one=True)

    type_sql = """
        SELECT content_type, COUNT(*) AS cnt
        FROM chunks
        GROUP BY content_type
        ORDER BY cnt DESC
    """
    type_stats = execute_query(type_sql, fetch_all=True)

    print(f"\n{'='*100}")
    print("  统计信息")
    print(f"{'='*100}")
    print(f"  chunks 表总记录数: {total['total']}")
    print(f"\n  按 content_type 分布:")
    for row in type_stats:
        ct = row["content_type"] or "(NULL)"
        print(f"    {ct}: {row['cnt']} 条")

    print()


if __name__ == "__main__":
    main()
