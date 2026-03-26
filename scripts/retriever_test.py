"""RAG 混合检索测试脚本 —— 逐步展示四路检索 → RRF 融合 → 最终输出"""
import sys
import os
import time
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from app.rag.embedding import embed_query
from app.rag.retriever import (
    _tokenize,
    _has_col_data,
    _text_vector_search,
    _text_keyword_search,
    _image_vector_search,
    _image_keyword_search,
    _rrf_fuse,
    hybrid_search,
)
from app.rag import retrieve_context_structured


SEP = "=" * 90
THIN_SEP = "─" * 90


def truncate(text, max_len=120):
    if not text:
        return "-"
    text = str(text).replace("\n", " ")
    return text[:max_len] + "..." if len(text) > max_len else text


def fmt_score(score):
    if score is None:
        return "-"
    return f"{score:.6f}"


def print_rows(rows: list[dict], label: str):
    print(f"\n{THIN_SEP}")
    print(f"  {label}  （{len(rows)} 条）")
    print(THIN_SEP)
    if not rows:
        print("  (无结果)")
        return
    for i, r in enumerate(rows, 1):
        rtype = r.get("type", "?")
        rid = r.get("id", "?")
        score = r.get("rrf_score") or r.get("score")
        if rtype == "text":
            content = truncate(r.get("main_text") or r.get("content", ""))
        else:
            content = truncate(r.get("caption") or r.get("title") or r.get("content", ""))
        print(f"  {i:>2}. [{rtype:>5}] score={fmt_score(score)}  id={rid}")
        print(f"      {content}")


def print_items(items: list[dict], label: str):
    print(f"\n{THIN_SEP}")
    print(f"  {label}  （{len(items)} 条）")
    print(THIN_SEP)
    if not items:
        print("  (无结果)")
        return
    for i, item in enumerate(items, 1):
        itype = item.get("type", "?")
        iid = item.get("id", "?")
        score = item.get("score")
        is_main = item.get("is_main")
        tag = "主" if is_main else "关联"
        meta = item.get("metadata") or {}
        book_name = meta.get("book_name", "")
        content = truncate(item.get("content", ""), 100)
        print(f"  {i:>2}. [{itype:>5}] [{tag}] score={fmt_score(score)}  id={iid}")
        if book_name:
            print(f"      来源: {book_name}")
        toc_path = meta.get("toc_path")
        if toc_path:
            print(f"      目录: {' > '.join(toc_path) if isinstance(toc_path, list) else toc_path}")
        print(f"      内容: {content}")


def run_single_query(
    query: str,
    *,
    top_k: int = 3,
    final_k: int = 5,
    relations: bool = False,
    steps: bool = False,
    prompt_preview_chars: int = 2000,
) -> None:
    """执行一次与命令行脚本相同的混合检索流程，并打印到 stdout。"""
    print(f"\n{SEP}")
    print(f"  RAG 混合检索测试")
    print(f"{SEP}")
    print(f"  Query       : {query}")
    print(f"  每路 Top-K  : {top_k}")
    print(f"  融合 Top-N  : {final_k}")
    print(f"  补充关联    : {'是' if relations else '否'}")
    print(f"  展示中间步骤: {'是' if steps else '否'}")

    # ── Step 1: embedding + 分词 ──
    t0 = time.perf_counter()
    query_vec = embed_query(query)
    t_embed = time.perf_counter() - t0

    tokens = _tokenize(query)
    print(f"\n  Embedding   : {len(query_vec)} 维, 耗时 {t_embed:.3f}s")
    print(f"  分词结果    : {' / '.join(tokens)}")

    if steps:
        # ── Step 2: 数据列检查 ──
        has_text_vec = _has_col_data("text_chunks", "embedding_values")
        has_text_kw  = _has_col_data("text_chunks", "ts_vector")
        has_img_vec  = _has_col_data("image_chunks", "embedding_values")
        has_img_kw   = _has_col_data("image_chunks", "ts_vector")
        print(f"\n  数据列检查: 文本向量={has_text_vec} 文本关键词={has_text_kw} "
              f"图片向量={has_img_vec} 图片关键词={has_img_kw}")

        # ── Step 3: 四路检索（跳过无数据路）──
        t1 = time.perf_counter()
        text_vec = _text_vector_search(query_vec, k=top_k) if has_text_vec else []
        t2 = time.perf_counter()
        text_kw = _text_keyword_search(tokens, k=top_k) if has_text_kw else []
        t3 = time.perf_counter()
        img_vec = _image_vector_search(query_vec, k=top_k) if has_img_vec else []
        t4 = time.perf_counter()
        img_kw = _image_keyword_search(tokens, k=top_k) if has_img_kw else []
        t5 = time.perf_counter()

        label_tv = f"① 文本向量检索  ({t2-t1:.3f}s)" if has_text_vec else "① 文本向量检索  (跳过，无数据)"
        label_tk = f"② 文本关键词检索 ({t3-t2:.3f}s)" if has_text_kw else "② 文本关键词检索 (跳过，无数据)"
        label_iv = f"③ 图片向量检索  ({t4-t3:.3f}s)" if has_img_vec else "③ 图片向量检索  (跳过，无数据)"
        label_ik = f"④ 图片关键词检索 ({t5-t4:.3f}s)" if has_img_kw else "④ 图片关键词检索 (跳过，无数据)"
        print_rows(text_vec, label_tv)
        print_rows(text_kw, label_tk)
        print_rows(img_vec, label_iv)
        print_rows(img_kw, label_ik)

        # ── Step 4: RRF 融合 ──
        t6 = time.perf_counter()
        fused = _rrf_fuse([text_vec, text_kw, img_vec, img_kw], final_k)
        t7 = time.perf_counter()
        print_rows(fused, f"⑤ RRF 融合结果  ({t7-t6:.4f}s)")

    # ── Step 5: 完整流程 ──
    t_start = time.perf_counter()
    prompt_text, search_result = retrieve_context_structured(
        query,
        with_relations=relations,
        k_vector=top_k,
        k_keyword=top_k,
        k_final=final_k,
    )
    t_total = time.perf_counter() - t_start

    items = search_result.get("items", [])
    relations_list = search_result.get("relations", [])

    step_label = "⑥" if steps else "⑤"
    print_items(items, f"{step_label} 最终检索结果  主结果={sum(1 for i in items if i.get('is_main'))} 关联={sum(1 for i in items if not i.get('is_main'))}  (完整流程 {t_total:.3f}s)")

    if relations_list:
        print(f"\n{THIN_SEP}")
        print(f"  关联关系  （{len(relations_list)} 条）")
        print(THIN_SEP)
        for rel in relations_list:
            print(f"  {rel['source_id']} --[{rel['relation_type']}]--> {rel['target_id']}")

    # ── Prompt 文本预览 ──
    print(f"\n{SEP}")
    print("  Prompt 文本（注入 LLM system prompt）")
    print(SEP)
    if prompt_text:
        preview = prompt_text[:prompt_preview_chars]
        print(preview)
        if len(prompt_text) > prompt_preview_chars:
            print(f"\n  ... 省略，共 {len(prompt_text)} 字符")
    else:
        print("  (无结果)")

    print(f"\n{SEP}")
    print(f"  完成 ✓  总耗时: {t_embed + t_total:.3f}s  (embed {t_embed:.3f}s + 检索 {t_total:.3f}s)")
    print(SEP)


def main():
    parser = argparse.ArgumentParser(description="RAG 混合检索测试")
    parser.add_argument("query", nargs="?", help="检索 query，不传则交互输入")
    parser.add_argument("-k", "--top-k", type=int, default=3, help="每路检索返回数量 (默认 5)")
    parser.add_argument("-n", "--final-k", type=int, default=5, help="RRF 融合后保留数量 (默认 10)")
    parser.add_argument("-r", "--relations", action="store_true", help="是否补充关联数据")
    parser.add_argument("--steps", action="store_true", help="展示四路检索中间结果")
    args = parser.parse_args()

    query = args.query or input("请输入检索 query: ").strip()
    if not query:
        print("query 不能为空")
        sys.exit(1)

    run_single_query(
        query,
        top_k=args.top_k,
        final_k=args.final_k,
        relations=args.relations,
        steps=args.steps,
    )


if __name__ == "__main__":
    main()