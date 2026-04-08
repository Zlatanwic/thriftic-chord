"""Chord 环形成的集成测试。"""
import pytest
import sys
import os
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chord_node import ChordNode, rpc_call, make_node_info
from config import M, RING_SIZE
import thriftpy2

chord_thrift = thriftpy2.load(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chord.thrift"),
    module_name="chord_thrift"
)


class TestSingleNode:
    """单节点自环测试。"""

    def test_node_successor_is_itself(self):
        node = ChordNode("127.0.0.1", 7001)
        assert node.successor.node_id == node.node_id
        assert node.successor.host == "127.0.0.1"
        assert node.successor.port == 7001

    def test_node_predecessor_is_none_initially(self):
        node = ChordNode("127.0.0.1", 7002)
        assert node.predecessor is None

    def test_finger_table_all_point_to_self(self):
        node = ChordNode("127.0.0.1", 7003)
        for entry in node.finger_table:
            assert entry["node"].node_id == node.node_id

    def test_closest_preceding_finger_returns_self(self):
        node = ChordNode("127.0.0.1", 7004)
        result = node.closest_preceding_finger(50)
        assert result.node_id == node.node_id


class TestClosestPrecedingFinger:
    """closest_preceding_finger 正确性测试。"""

    def test_finger_table_single_node(self):
        """单节点：所有 finger 都指向自己。"""
        node = ChordNode("127.0.0.1", 7010)
        # 任意 id 都应该返回自己
        for test_id in [1, 50, 100, 200, 255]:
            result = node.closest_preceding_finger(test_id)
            assert result.node_id == node.node_id, f"id={test_id} should return self"

    def test_finger_table_middle_node(self):
        """finger table 中间节点应该跳过超过目标 id 的 finger。"""
        # 构造：node_id=5，finger table 已设置好
        node = ChordNode("127.0.0.1", 7011)
        with node._lock:
            # 手动设置 finger table 模拟已加入环的状态
            # finger[0].node = successor = 10
            node.successor = make_node_info(10, "127.0.0.1", 7012)
            node.finger_table[0]["node"] = node.successor
            # finger[1].start = (5 + 2) % 256 = 7
            node.finger_table[1]["start"] = (node.node_id + 2) % RING_SIZE
            node.finger_table[1]["node"] = make_node_info(7, "127.0.0.1", 7012)
            # finger[2].start = (5 + 4) % 256 = 9
            node.finger_table[2]["start"] = (node.node_id + 4) % RING_SIZE
            node.finger_table[2]["node"] = make_node_info(20, "127.0.0.1", 7013)

        # 查 id=8：(5, 8) 双开
        # finger[2].node=20 不在 (5, 8)
        # finger[1].node=7 在 (5, 8)，返回 7
        result = node.closest_preceding_finger(8)
        assert result.node_id == 7

    def test_closest_preceding_finger_no_match(self):
        """没有 finger 落在 (n, id) 内，返回自己。"""
        node = ChordNode("127.0.0.1", 7014)
        with node._lock:
            # node_id > id (227 > 20)，区间 (227, 20) 跨 0 点 = {228-255, 0-20}
            # finger entries = 100, 150, 200 都不在 (227, 20) 内
            node.successor = make_node_info(100, "127.0.0.1", 7015)
            node.finger_table[0]["node"] = node.successor
            node.finger_table[1]["node"] = make_node_info(100, "127.0.0.1", 7015)
            node.finger_table[2]["node"] = make_node_info(150, "127.0.0.1", 7016)
            node.finger_table[3]["node"] = make_node_info(200, "127.0.0.1", 7017)

        # 查 id=20：(227, 20) 区间 = {228-255, 0-20}，不含 100/150/200
        result = node.closest_preceding_finger(20)
        assert result.node_id == node.node_id


class TestFindSuccessor:
    """find_successor 正确性测试。"""

    def test_single_node_finds_itself_for_any_id(self):
        """单节点环：任意 id 的 successor 都是自己。"""
        node = ChordNode("127.0.0.1", 7020)
        for test_id in [0, 1, 100, 200, 255]:
            result = node.find_successor(test_id)
            assert result.node_id == node.node_id

    def test_successor_chain_simple(self):
        """两节点链：验证 find_successor 正确返回 successor。"""
        # 创建两个节点，不通过 join，而是直接设置 successor/predecessor
        node_a = ChordNode("127.0.0.1", 7030)
        node_b = ChordNode("127.0.0.1", 7031)

        # node_a.successor = node_b, node_b.predecessor = node_a
        node_a.successor = make_node_info(node_b.node_id, "127.0.0.1", 7031)
        node_a.finger_table[0]["node"] = node_a.successor
        node_b.predecessor = make_node_info(node_a.node_id, "127.0.0.1", 7030)

        # node_a 查 id 在 (a, b] 区间，应该返回 node_b
        if node_a.node_id < node_b.node_id:
            test_id = (node_a.node_id + 1) % RING_SIZE
            result = node_a.find_successor(test_id)
            assert result.node_id == node_b.node_id


class TestNodeIdUniqueness:
    """节点 ID 哈希正确性。"""

    def test_same_host_port_same_id(self):
        id1 = ChordNode("127.0.0.1", 8001).node_id
        id2 = ChordNode("127.0.0.1", 8001).node_id
        assert id1 == id2

    def test_different_port_different_id(self):
        id1 = ChordNode("127.0.0.1", 8002).node_id
        id2 = ChordNode("127.0.0.1", 8003).node_id
        assert id1 != id2

    def test_id_within_ring_size(self):
        node = ChordNode("127.0.0.1", 8004)
        assert 0 <= node.node_id < RING_SIZE

    def test_finger_table_start_values(self):
        """验证 finger[i].start = (node_id + 2^i) % RING_SIZE"""
        node = ChordNode("127.0.0.1", 8005)
        for i in range(M):
            expected_start = (node.node_id + (2 ** i)) % RING_SIZE
            assert node.finger_table[i]["start"] == expected_start


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
