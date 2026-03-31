"""
将 documents.id 从中文 slug 刷为约定英文短 id，并同步 text_chunks / image_chunks.book_id。

用法:
  python scripts/migrate_document_ids_en_short.py
  python scripts/migrate_document_ids_en_short.py --dry-run
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.connect import get_connection, release_connection

# 中文 id -> 英文短 id（用户确认「短版」）
ZH_TO_EN_SHORT: dict[str, str] = {
    "法式术语简要": "terms_brief",
    "梁思成注释营造法式": "yzfs_liang",
    "营造法式解读_修订版": "yzfs_interpretation_rev",
    "王贵祥译注营造法式": "yzfs_wang",
    "法式生僻字库": "rare_chars",
}


def _find_fk_book_id_to_documents(cur, table: str) -> str | None:
    cur.execute(
        """
        SELECT tc.constraint_name
        FROM information_schema.table_constraints AS tc
        JOIN information_schema.key_column_usage AS kcu
          ON tc.constraint_schema = kcu.constraint_schema
         AND tc.constraint_name = kcu.constraint_name
         AND tc.table_name = kcu.table_name
        JOIN information_schema.constraint_column_usage AS ccu
          ON ccu.constraint_schema = tc.constraint_schema
         AND ccu.constraint_name = tc.constraint_name
        WHERE tc.table_schema = 'public'
          AND tc.table_name = %s
          AND tc.constraint_type = 'FOREIGN KEY'
          AND ccu.table_name = 'documents'
          AND kcu.column_name = 'book_id'
        LIMIT 1;
        """,
        (table,),
    )
    r = cur.fetchone()
    return r["constraint_name"] if r else None


def _constraint_exists(cur, name: str) -> bool:
    cur.execute("SELECT 1 FROM pg_constraint WHERE conname = %s;", (name,))
    return cur.fetchone() is not None


def run_migrate(dry_run: bool) -> None:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT id FROM documents;")
        existing = {str(r["id"]) for r in cur.fetchall()}

        to_apply: list[tuple[str, str]] = []
        for zh, en in ZH_TO_EN_SHORT.items():
            if zh not in existing:
                continue
            if zh == en:
                continue
            to_apply.append((zh, en))

        print("=== 将执行的映射 (old -> new) ===")
        if not to_apply:
            print("  (无：库中无待替换的中文 id，或已全部为英文)")
        for old, new in to_apply:
            print(f"  {old!r} -> {new!r}")

        if dry_run:
            print("\n[dry-run] 未写入数据库。")
            conn.rollback()
            return

        if not to_apply:
            return

        cur.execute("BEGIN;")

        fk_tc = _find_fk_book_id_to_documents(cur, "text_chunks")
        fk_ic = _find_fk_book_id_to_documents(cur, "image_chunks")
        if fk_tc:
            cur.execute(f'ALTER TABLE text_chunks DROP CONSTRAINT "{fk_tc}";')
        if fk_ic:
            cur.execute(f'ALTER TABLE image_chunks DROP CONSTRAINT "{fk_ic}";')

        for old_id, new_id in to_apply:
            cur.execute(
                "UPDATE text_chunks SET book_id = %s WHERE book_id = %s;",
                (new_id, old_id),
            )
            cur.execute(
                "UPDATE image_chunks SET book_id = %s WHERE book_id = %s;",
                (new_id, old_id),
            )
            cur.execute(
                "UPDATE documents SET id = %s WHERE id = %s;",
                (new_id, old_id),
            )

        if not _constraint_exists(cur, "text_chunks_book_id_fkey"):
            cur.execute(
                """
                ALTER TABLE text_chunks ADD CONSTRAINT text_chunks_book_id_fkey
                  FOREIGN KEY (book_id) REFERENCES documents(id);
                """
            )
        if not _constraint_exists(cur, "image_chunks_book_id_fkey"):
            cur.execute(
                """
                ALTER TABLE image_chunks ADD CONSTRAINT image_chunks_book_id_fkey
                  FOREIGN KEY (book_id) REFERENCES documents(id);
                """
            )

        conn.commit()
        print("\n迁移已提交。")
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        release_connection(conn)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    run_migrate(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
