"""数据存储（put/get/transfer_keys）的单元测试。"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chord_node import ChordNode, make_node_info
from utils import consistent_hash
from config import M, RING_SIZE


class TestPutGetSingleNode:
    """单节点内的 put/get 测试。"""

    def test_put_and_get_same_node(self):
        """put 一个 key，再 get 应该能取到。"""
        node = ChordNode("127.0.0.1", 7101)
        key = "mykey"
        value = "myvalue"
        node.put(key, value)
        result = node.get(key)
        assert result.found is True
        assert result.value == value

    def test_get_nonexistent_key(self):
        """get 不存在的 key 返回 found=False。"""
        node = ChordNode("127.0.0.1", 7102)
        result = node.get("nonexistent")
        assert result.found is False

    def test_put_overwrites_value(self):
        """同一 key 多次 put 会覆盖。"""
        node = ChordNode("127.0.0.1", 7103)
        node.put("key", "v1")
        node.put("key", "v2")
        result = node.get("key")
        assert result.value == "v2"


class TestTransferKeys:
    """transfer_keys 数据迁移测试。"""

    def test_transfer_keys_in_range(self):
        """迁移 (from_id, to_id] 区间内的 keys。"""
        node = ChordNode("127.0.0.1", 7110)
        with node._lock:
            node.data.clear()
            node.data["ka"] = "va"
            node.data["kb"] = "vb"

        k_a = consistent_hash("ka")
        k_b = consistent_hash("kb")

        if k_a < k_b:
            # 迁移 (k_a, k_b]，k_b 在右边界内
            result = node.transfer_keys(k_a, k_b)
            assert "kb" in result
            assert "ka" not in result
            assert "kb" not in node.data
        elif k_a > k_b:
            # 跨 0 点：迁移 (k_a, RING_SIZE) ∪ [0, k_b]
            # k_a 不在（左侧排除），k_b 在（右侧包含）
            result = node.transfer_keys(k_a, k_b)
            assert "kb" in result
            assert "ka" not in result
        else:
            # k_a == k_b: 区间覆盖整个环
            result = node.transfer_keys(k_a, k_b)
            assert "ka" in result and "kb" in result

    def test_transfer_keys_empty_range(self):
        """区间内没有 key 时返回空。"""
        node = ChordNode("127.0.0.1", 7111)
        node.put("testkey_xyz", "v1")
        k = consistent_hash("testkey_xyz")
        # 找一个小区间不跨 0 点，明确不包含 k
        from_id = (k + 1) % RING_SIZE
        to_id = (k + 5) % RING_SIZE
        # 区间 (k+1, k+5] 不包含 k 本身
        result = node.transfer_keys(from_id, to_id)
        assert len(result) == 0

    def test_transfer_keys_removes_from_source(self):
        """transfer 后，原 node 的 data 中不再包含被迁移的 key。"""
        node = ChordNode("127.0.0.1", 7112)
        node.put("unique_key_xyz", "v1")
        key_id = consistent_hash("unique_key_xyz")
        from_id = (key_id - 1) % RING_SIZE
        to_id = key_id
        result = node.transfer_keys(from_id, to_id)
        assert "unique_key_xyz" in result
        assert "unique_key_xyz" not in node.data

    def test_transfer_keys_excludes_boundary(self):
        """transfer_keys 区间是 (from_id, to_id]，左开右闭。"""
        node = ChordNode("127.0.0.1", 7113)
        key = "boundary_key"
        node.put(key, "v1")  # 先存数据
        key_id = consistent_hash(key)
        # 迁移 (key_id-1, key_id] — key_id 在右边界，应被包含
        result = node.transfer_keys((key_id - 1) % RING_SIZE, key_id)
        assert key in result


class TestConsistentHash:
    """consistent_hash 分布特性测试。"""

    def test_same_input_same_output(self):
        """相同输入总是得到相同哈希。"""
        h1 = consistent_hash("testkey")
        h2 = consistent_hash("testkey")
        assert h1 == h2

    def test_different_input_different_output(self):
        """不同输入大概率得到不同哈希（SHA-1 碰撞率低但 RING_SIZE=256 时有汇聚）。"""
        hashes = set()
        for i in range(100):
            h = consistent_hash(f"key_{i}")
            hashes.add(h)
        # RING_SIZE=256 时 100 个随机 key 有少量碰撞是正常的
        assert len(hashes) >= 70

    def test_hash_within_ring_size(self):
        """哈希值总是在 [0, RING_SIZE) 范围内。"""
        for i in range(50):
            h = consistent_hash(f"key_{i}")
            assert 0 <= h < RING_SIZE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
