"""
将 documents.id 改为由 name 生成的可读 slug，并同步 text_chunks / image_chunks.book_id；
为 text_chunks、image_chunks 增加 toc_tsvector（jieba 分词 + simple 词典）及 GIN 索引。

用法:
  python scripts/migrate_readable_book_ids_and_toc_tsvector.py           # 执行迁移
  python scripts/migrate_readable_book_ids_and_toc_tsvector.py --dry-run  # 仅打印映射与步骤
"""
from __future__ import annotations

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jieba

from app.connect import get_connection, release_connection

_ID_MAX = 200
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.I,
)


def _load_jieba_userdict() -> None:
    path = os.path.join(
        os.path.dirname(__file__), "..", "resources", "jieba_userdict.txt"
    )
    if os.path.exists(path):
        jieba.load_userdict(path)


def slug_from_name(name: str, fallback_id: str) -> str:
    raw = str(fallback_id).replace("-", "")
    fb = f"doc_{raw[:8]}"
    if not name or not str(name).strip():
        return fb
    s = str(name).strip()
    s = re.sub(r"[《》「」『』]", "", s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^\w\u4e00-\u9fff]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        return fb
    if len(s) > _ID_MAX:
        s = s[:_ID_MAX]
    return s


def build_slug_mapping(rows: list[dict]) -> dict[str, str]:
    """old_id -> new_id，保证 new_id 唯一。"""
    used: set[str] = set()
    mapping: dict[str, str] = {}
    for row in rows:
        old_id = str(row["id"])
        base = slug_from_name(row.get("name") or "", old_id)
        new_id = base
        n = 2
        while new_id in used:
            suf = f"_{n}"
            new_id = (base[: _ID_MAX - len(suf)] + suf) if len(base) + len(suf) > _ID_MAX else base + suf
            n += 1
        used.add(new_id)
        mapping[old_id] = new_id
    return mapping


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


def needs_book_id_migration(doc_rows: list[dict]) -> bool:
    return any(_UUID_RE.match(str(r["id"])) for r in doc_rows)


def _constraint_exists(cur, name: str) -> bool:
    cur.execute("SELECT 1 FROM pg_constraint WHERE conname = %s;", (name,))
    return cur.fetchone() is not None


def run_migrate(dry_run: bool) -> None:
    _load_jieba_userdict()
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute('SELECT id, name FROM documents ORDER BY id;')
        doc_rows = list(cur.fetchall())
        if not doc_rows:
            print("documents 为空，跳过 book_id 迁移。")
            mapping: dict[str, str] = {}
            do_book = False
        else:
            do_book = needs_book_id_migration(doc_rows)
            mapping = build_slug_mapping(doc_rows) if do_book else {}

        if do_book:
            print("=== book_id 映射 (old -> new) ===")
            for old, new in sorted(mapping.items(), key=lambda x: x[0]):
                if old != new:
                    print(f"  {old} -> {new}")
                else:
                    print(f"  {old} (不变)")
        else:
            print("=== 跳过 documents.id / book_id 迁移（当前 id 已非 UUID 或无需改写）===")

        cur.execute(
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'text_chunks' AND column_name = 'toc_tsvector';
            """
        )
        has_toc_text = cur.fetchone() is not None
        cur.execute(
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'image_chunks' AND column_name = 'toc_tsvector';
            """
        )
        has_toc_img = cur.fetchone() is not None
        print(f"\n=== toc_tsvector 列: text_chunks={has_toc_text}, image_chunks={has_toc_img} ===")

        if dry_run:
            print("\n[dry-run] 未写入数据库。")
            conn.rollback()
            return

        cur.execute("BEGIN;")

        if do_book and mapping:
            fk_tc = _find_fk_book_id_to_documents(cur, "text_chunks")
            fk_ic = _find_fk_book_id_to_documents(cur, "image_chunks")
            if fk_tc:
                cur.execute(f'ALTER TABLE text_chunks DROP CONSTRAINT "{fk_tc}";')
            if fk_ic:
                cur.execute(f'ALTER TABLE image_chunks DROP CONSTRAINT "{fk_ic}";')

            cur.execute(
                "ALTER TABLE documents ALTER COLUMN id TYPE TEXT USING id::text;"
            )
            cur.execute(
                "ALTER TABLE text_chunks ALTER COLUMN book_id TYPE TEXT USING book_id::text;"
            )
            cur.execute(
                "ALTER TABLE image_chunks ALTER COLUMN book_id TYPE TEXT USING book_id::text;"
            )

            for old_id, new_id in mapping.items():
                if old_id == new_id:
                    continue
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

        if not has_toc_text:
            cur.execute(
                "ALTER TABLE text_chunks ADD COLUMN toc_tsvector TSVECTOR;"
            )
        if not has_toc_img:
            cur.execute(
                "ALTER TABLE image_chunks ADD COLUMN toc_tsvector TSVECTOR;"
            )

        cur.execute(
            """
            SELECT chunk_id, toc_path FROM text_chunks
            WHERE toc_path IS NOT NULL AND cardinality(toc_path) > 0;
            """
        )
        for row in cur.fetchall():
            toc_path = row["toc_path"]
            chunk_id = row["chunk_id"]
            text = " ".join(toc_path)
            joined = " ".join(jieba.cut(text)).strip()
            if not joined:
                continue
            cur.execute(
                "UPDATE text_chunks SET toc_tsvector = to_tsvector('simple', %s) WHERE chunk_id = %s;",
                (joined, chunk_id),
            )

        cur.execute(
            """
            SELECT image_id, toc_path FROM image_chunks
            WHERE toc_path IS NOT NULL AND cardinality(toc_path) > 0;
            """
        )
        for row in cur.fetchall():
            toc_path = row["toc_path"]
            image_id = row["image_id"]
            text = " ".join(toc_path)
            joined = " ".join(jieba.cut(text)).strip()
            if not joined:
                continue
            cur.execute(
                "UPDATE image_chunks SET toc_tsvector = to_tsvector('simple', %s) WHERE image_id = %s;",
                (joined, image_id),
            )

        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_text_chunks_toc_tsvector ON text_chunks USING gin (toc_tsvector);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_image_chunks_toc_tsvector ON image_chunks USING gin (toc_tsvector);"
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
