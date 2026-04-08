"""Phase 5: 向量搜索（embedding / local_search / scatter-gather）的单元测试。"""
import json
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from embedding import encode, to_json, from_json, cosine_similarity, _bow_embed, EMBED_DIM
from chord_node import ChordNode


class TestEmbedding:
    """embedding 模块基础功能测试。"""

    def test_encode_returns_correct_dim(self):
        vec = encode("hello world")
        assert vec.shape == (EMBED_DIM,)

    def test_encode_deterministic(self):
        v1 = encode("test sentence")
        v2 = encode("test sentence")
        np.testing.assert_array_almost_equal(v1, v2)

    def test_encode_different_texts_differ(self):
        v1 = encode("distributed hash table")
        v2 = encode("banana smoothie recipe")
        # 不同文本的向量不应完全相同
        assert not np.allclose(v1, v2, atol=0.01)

    def test_bow_embed_normalized(self):
        vec = _bow_embed("test text for normalization")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5

    def test_bow_embed_empty_string(self):
        vec = _bow_embed("")
        assert vec.shape == (EMBED_DIM,)
        # 空字符串没有 trigram，全零向量
        assert np.allclose(vec, 0)


class TestJsonSerialization:
    """向量 JSON 序列化/反序列化测试。"""

    def test_roundtrip(self):
        original = np.random.randn(EMBED_DIM).astype(np.float32)
        json_str = to_json(original)
        recovered = from_json(json_str)
        np.testing.assert_array_almost_equal(original, recovered, decimal=5)

    def test_json_is_valid(self):
        vec = encode("test")
        json_str = to_json(vec)
        parsed = json.loads(json_str)
        assert isinstance(parsed, list)
        assert len(parsed) == EMBED_DIM


class TestCosineSimilarity:
    """余弦相似度计算测试。"""

    def test_identical_vectors(self):
        vec = np.array([1.0, 0.0, 0.0])
        assert abs(cosine_similarity(vec, vec) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        assert abs(cosine_similarity(v1, v2)) < 1e-6

    def test_similar_texts_higher_score(self):
        v1 = encode("Chord distributed hash table protocol")
        v2 = encode("DHT is a distributed hash table")
        v3 = encode("chocolate cake recipe ingredients")
        sim_related = cosine_similarity(v1, v2)
        sim_unrelated = cosine_similarity(v1, v3)
        assert sim_related > sim_unrelated


class TestPutDocument:
    """单节点 put_document 测试。"""

    def test_put_and_check_stored(self):
        node = ChordNode("127.0.0.1", 7201)
        vec = encode("test document content")
        embedding_json = to_json(vec)
        node.put_document("doc1", "test document content", embedding_json)
        assert "doc1" in node.vector_store
        assert node.vector_store["doc1"]["text"] == "test document content"

    def test_put_multiple_documents(self):
        node = ChordNode("127.0.0.1", 7202)
        docs = [
            ("d1", "Chord protocol overview"),
            ("d2", "Finger table acceleration"),
            ("d3", "Stabilization process"),
        ]
        for doc_id, text in docs:
            vec = encode(text)
            node.put_document(doc_id, text, to_json(vec))
        assert len(node.vector_store) == 3


class TestLocalSearch:
    """单节点 local_search 测试。"""

    def _build_node_with_docs(self, port: int) -> ChordNode:
        """构建一个包含多个文档的测试节点。"""
        node = ChordNode("127.0.0.1", port)
        docs = [
            ("chord", "Chord is a distributed hash table protocol for peer-to-peer networks"),
            ("raft", "Raft is a consensus algorithm for managing replicated logs"),
            ("vector", "Vector similarity search computes cosine distance between embeddings"),
            ("recipe", "Mix flour, sugar, eggs, and butter to make a cake"),
        ]
        for doc_id, text in docs:
            vec = encode(text)
            node.put_document(doc_id, text, to_json(vec))
        return node

    def test_local_search_returns_results(self):
        node = self._build_node_with_docs(7210)
        query_vec = encode("distributed hash table")
        results = node.local_search(to_json(query_vec), 2)
        assert len(results) == 2

    def test_local_search_ranking(self):
        node = self._build_node_with_docs(7211)
        query_vec = encode("distributed hash table")
        results = node.local_search(to_json(query_vec), 4)
        # 结果应按 score 降序排列
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_local_search_relevance(self):
        node = self._build_node_with_docs(7212)
        query_vec = encode("distributed hash table protocol")
        results = node.local_search(to_json(query_vec), 4)
        # "chord" 文档应该排在 "recipe" 文档之前
        doc_ids = [r.doc_id for r in results]
        chord_idx = doc_ids.index("chord")
        recipe_idx = doc_ids.index("recipe")
        assert chord_idx < recipe_idx

    def test_local_search_top_k_limit(self):
        node = self._build_node_with_docs(7213)
        query_vec = encode("test query")
        results = node.local_search(to_json(query_vec), 2)
        assert len(results) <= 2

    def test_local_search_empty_store(self):
        node = ChordNode("127.0.0.1", 7214)
        query_vec = encode("test query")
        results = node.local_search(to_json(query_vec), 5)
        assert len(results) == 0


class TestGetAllNodes:
    """get_all_nodes 测试（单节点）。"""

    def test_single_node_returns_self(self):
        node = ChordNode("127.0.0.1", 7220)
        nodes = node.get_all_nodes()
        assert len(nodes) == 1
        assert nodes[0].node_id == node.node_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
