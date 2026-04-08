"""文本嵌入模块。

优先使用 sentence-transformers（质量高），
fallback 到基于词频的简单向量（无需 GPU / PyTorch）。
"""
import json
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger("embedding")

# 嵌入维度
EMBED_DIM = 384  # all-MiniLM-L6-v2 输出维度

_model = None
_use_transformer = False


def _try_load_transformer():
    """尝试加载 sentence-transformers 模型。"""
    global _model, _use_transformer
    if _model is not None:
        return
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        _use_transformer = True
        logger.info("Using sentence-transformers (all-MiniLM-L6-v2)")
    except ImportError:
        _use_transformer = False
        logger.info("sentence-transformers not available, using bag-of-words fallback")


def _bow_embed(text: str) -> np.ndarray:
    """基于字符 n-gram 哈希的简单嵌入（fallback）。

    将文本的字符 trigram 映射到固定维度向量，归一化后返回。
    不需要任何 ML 依赖，仅用于演示分布式搜索架构。
    """
    vec = np.zeros(EMBED_DIM, dtype=np.float32)
    text_lower = text.lower()
    for i in range(len(text_lower) - 2):
        trigram = text_lower[i:i+3]
        idx = hash(trigram) % EMBED_DIM
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def encode(text: str) -> np.ndarray:
    """将文本编码为嵌入向量。返回 shape=(EMBED_DIM,) 的 numpy 数组。"""
    _try_load_transformer()
    if _use_transformer:
        return _model.encode(text, normalize_embeddings=True)
    return _bow_embed(text)


def encode_batch(texts: list[str]) -> np.ndarray:
    """批量编码。返回 shape=(N, EMBED_DIM) 的 numpy 数组。"""
    _try_load_transformer()
    if _use_transformer:
        return _model.encode(texts, normalize_embeddings=True)
    return np.array([_bow_embed(t) for t in texts])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个向量的余弦相似度。假设已归一化则等价于点积。"""
    return float(np.dot(a, b))


def to_json(vec: np.ndarray) -> str:
    """将向量序列化为 JSON 字符串（用于 thrift 传输）。"""
    return json.dumps(vec.tolist())


def from_json(s: str) -> np.ndarray:
    """从 JSON 字符串反序列化为向量。"""
    return np.array(json.loads(s), dtype=np.float32)
