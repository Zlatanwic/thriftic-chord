"""Chord DHT AI 客户端 — 分布式语义搜索 + RAG 演示。

用法：
    # 存入文档（自动计算 embedding 并路由到正确的 node）
    python ai_client.py --node 127.0.0.1:6001 store-doc "doc1" "Chord 是一种分布式哈希表协议"

    # 批量导入文档
    python ai_client.py --node 127.0.0.1:6001 load-docs docs.jsonl

    # 分布式语义搜索（scatter-gather 所有 node）
    python ai_client.py --node 127.0.0.1:6001 search "什么是一致性哈希" --top-k 3

    # RAG 问答（检索 + LLM 生成）
    python ai_client.py --node 127.0.0.1:6001 rag "Chord 如何处理节点加入？"
"""
import argparse
import json
import sys
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import thriftpy2
from thriftpy2.rpc import make_client

from config import RPC_TIMEOUT
from embedding import encode, to_json

chord_thrift = thriftpy2.load("chord.thrift", module_name="chord_ai_thrift")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ai_client")


def get_client(host: str, port: int):
    return make_client(chord_thrift.ChordService, host, port, timeout=RPC_TIMEOUT)


def parse_node(node_str: str):
    host, port = node_str.rsplit(":", 1)
    return host, int(port)


# ========================
# === 文档存储 ===
# ========================

def cmd_store_doc(host: str, port: int, doc_id: str, text: str):
    """计算文档 embedding 并存入 Chord 环。"""
    print(f"Computing embedding for doc_id={doc_id!r}...")
    vec = encode(text)
    embedding_json = to_json(vec)

    client = get_client(host, port)
    client.put_document(doc_id, text, embedding_json)
    client._iprot.trans.close()
    print(f"OK: stored doc_id={doc_id!r} (dim={len(vec)})")


def cmd_load_docs(host: str, port: int, filepath: str):
    """从 JSONL 文件批量导入文档。

    每行格式：{"doc_id": "...", "text": "..."}
    """
    with open(filepath, "r", encoding="utf-8") as f:
        docs = [json.loads(line) for line in f if line.strip()]

    print(f"Loading {len(docs)} documents...")
    for i, doc in enumerate(docs):
        doc_id = doc["doc_id"]
        text = doc["text"]
        vec = encode(text)
        embedding_json = to_json(vec)

        client = get_client(host, port)
        client.put_document(doc_id, text, embedding_json)
        client._iprot.trans.close()
        print(f"  [{i+1}/{len(docs)}] stored {doc_id!r}")

    print(f"Done: {len(docs)} documents loaded.")


# ========================
# === 分布式语义搜索 ===
# ========================

def scatter_gather_search(host: str, port: int, query: str, top_k: int) -> list:
    """Scatter-Gather 分布式语义搜索。

    1. 通过任意 node 获取环上所有 node 列表
    2. 并发向每个 node 发送 local_search RPC
    3. 合并所有结果，返回全局 top-k
    """
    # 计算查询向量
    query_vec = encode(query)
    query_json = to_json(query_vec)

    # 获取所有 node
    client = get_client(host, port)
    all_nodes = client.get_all_nodes()
    client._iprot.trans.close()
    print(f"Ring has {len(all_nodes)} nodes, searching all...")

    # 并发 scatter
    all_results = []

    def search_one_node(node):
        try:
            c = get_client(node.host, node.port)
            results = c.local_search(query_json, top_k)
            c._iprot.trans.close()
            return results
        except Exception as e:
            logger.warning(f"Search on node {node.node_id} failed: {e}")
            return []

    with ThreadPoolExecutor(max_workers=min(len(all_nodes), 10)) as pool:
        futures = {pool.submit(search_one_node, node): node for node in all_nodes}
        for future in as_completed(futures):
            all_results.extend(future.result())

    # Gather: 全局排序取 top_k
    all_results.sort(key=lambda r: r.score, reverse=True)
    return all_results[:top_k]


def cmd_search(host: str, port: int, query: str, top_k: int):
    """分布式语义搜索。"""
    results = scatter_gather_search(host, port, query, top_k)
    if not results:
        print("No results found.")
        return
    print(f"\nTop {len(results)} results for query: {query!r}\n")
    print("-" * 60)
    for i, r in enumerate(results):
        print(f"  [{i+1}] score={r.score:.4f}  doc_id={r.doc_id!r}")
        # 截断长文本
        text_preview = r.text[:200] + "..." if len(r.text) > 200 else r.text
        print(f"      {text_preview}")
        print()


# ========================
# === RAG 问答 ===
# ========================

def cmd_rag(host: str, port: int, question: str, top_k: int):
    """RAG（Retrieval-Augmented Generation）问答。

    1. 用语义搜索检索相关文档
    2. 将文档作为上下文拼入 prompt
    3. 调用 LLM 生成回答
    """
    # 检索
    print(f"Retrieving relevant documents for: {question!r}")
    results = scatter_gather_search(host, port, question, top_k)

    if not results:
        print("No relevant documents found in the distributed store.")
        return

    print(f"Found {len(results)} relevant documents, generating answer...\n")

    # 构建上下文
    context_parts = []
    for i, r in enumerate(results):
        context_parts.append(f"[Document {i+1}] (relevance: {r.score:.3f})\n{r.text}")
    context = "\n\n".join(context_parts)

    # 尝试调用 OpenAI API
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        answer = _call_openai(question, context, api_key)
    else:
        # 无 API key 时，展示 prompt 模板（用于演示）
        answer = None
        print("[INFO] OPENAI_API_KEY not set, showing RAG prompt template instead.\n")

    print("=" * 60)
    print("RAG Pipeline Result")
    print("=" * 60)
    print(f"\nQuestion: {question}")
    print(f"\nRetrieved {len(results)} documents from Chord DHT ring:")
    for i, r in enumerate(results):
        print(f"  [{i+1}] doc_id={r.doc_id!r} score={r.score:.4f}")
    print()

    if answer:
        print(f"Answer:\n{answer}")
    else:
        # 展示完整 prompt
        prompt = _build_rag_prompt(question, context)
        print(f"Generated Prompt (send to any LLM):\n")
        print("-" * 40)
        print(prompt)
        print("-" * 40)


def _build_rag_prompt(question: str, context: str) -> str:
    return (
        "You are a helpful assistant. Answer the question based ONLY on the "
        "following context retrieved from a distributed Chord DHT.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def _call_openai(question: str, context: str, api_key: str) -> str:
    """调用 OpenAI Chat API 生成回答。"""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        prompt = _build_rag_prompt(question, context)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.3,
        )
        return response.choices[0].message.content
    except ImportError:
        print("[WARN] openai package not installed, showing prompt only.")
        return None
    except Exception as e:
        print(f"[WARN] OpenAI API call failed: {e}")
        return None


# ========================
# === CLI ===
# ========================

def main():
    parser = argparse.ArgumentParser(
        description="Chord DHT AI Client — Distributed Semantic Search & RAG"
    )
    parser.add_argument("--node", required=True, help="Node address, e.g. 127.0.0.1:6001")

    sub = parser.add_subparsers(dest="cmd", required=True)

    # store-doc
    p = sub.add_parser("store-doc", help="Store a document with auto-computed embedding")
    p.add_argument("doc_id", help="Document ID")
    p.add_argument("text", help="Document text content")

    # load-docs
    p = sub.add_parser("load-docs", help="Batch load documents from JSONL file")
    p.add_argument("filepath", help="Path to JSONL file (each line: {doc_id, text})")

    # search
    p = sub.add_parser("search", help="Distributed semantic search across all nodes")
    p.add_argument("query", help="Search query (natural language)")
    p.add_argument("--top-k", type=int, default=5, help="Number of results (default: 5)")

    # rag
    p = sub.add_parser("rag", help="RAG: retrieve from DHT + generate answer via LLM")
    p.add_argument("question", help="Question to answer")
    p.add_argument("--top-k", type=int, default=3, help="Number of docs to retrieve (default: 3)")

    args = parser.parse_args()
    host, port = parse_node(args.node)

    if args.cmd == "store-doc":
        cmd_store_doc(host, port, args.doc_id, args.text)
    elif args.cmd == "load-docs":
        cmd_load_docs(host, port, args.filepath)
    elif args.cmd == "search":
        cmd_search(host, port, args.query, args.top_k)
    elif args.cmd == "rag":
        cmd_rag(host, port, args.question, args.top_k)


if __name__ == "__main__":
    main()
