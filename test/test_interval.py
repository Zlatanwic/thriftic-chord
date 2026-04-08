"""in_interval 模运算区间判断的单元测试。"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import in_interval
from config import RING_SIZE, M


class TestInIntervalNormal:
    """普通区间（a < b），不跨 0 点。"""

    def test_x_in_open_interval(self):
        # (1, 5) 开开，x=3 在内部
        assert in_interval(3, 1, 5, inclusive_left=False, inclusive_right=False) is True

    def test_x_at_left_boundary_excluded(self):
        # (1, 5) 左开，x=1 不在
        assert in_interval(1, 1, 5, inclusive_left=False, inclusive_right=False) is False

    def test_x_at_right_boundary_included(self):
        # (1, 5] 右闭，x=5 在内部
        assert in_interval(5, 1, 5, inclusive_left=False, inclusive_right=True) is True

    def test_x_outside_right(self):
        # (1, 5)，x=6 不在
        assert in_interval(6, 1, 5, inclusive_left=False, inclusive_right=False) is False

    def test_x_outside_left(self):
        # (1, 5)，x=0 不在
        assert in_interval(0, 1, 5, inclusive_left=False, inclusive_right=False) is False

    def test_left_inclusive(self):
        # [1, 5)，x=1 在内部
        assert in_interval(1, 1, 5, inclusive_left=True, inclusive_right=False) is True


class TestInIntervalWrapAround:
    """跨 0 点的区间（a > b）。"""

    def test_x_in_upper_part(self):
        # (7, 3) 跨 0 点，x=0 在 [7, 3) 的上部分
        assert in_interval(0, 7, 3, inclusive_left=False, inclusive_right=False) is True

    def test_x_in_lower_part(self):
        # (7, 3) 跨 0 点，x=2 在 [7, 3) 的下部分
        assert in_interval(2, 7, 3, inclusive_left=False, inclusive_right=False) is True

    def test_x_at_a_excluded(self):
        # (7, 3) 左开，x=7 不在
        assert in_interval(7, 7, 3, inclusive_left=False, inclusive_right=False) is False

    def test_x_at_b_excluded(self):
        # (7, 3) 右开右开，x=3 不在
        # 注意：(a, b] 右闭在跨 0 点时包含 b；(a, b) 双开不包含 b
        assert in_interval(3, 7, 3, inclusive_left=False, inclusive_right=False) is False
        # (7, 3] 右开左闭，x=3 在（因为右闭）
        assert in_interval(3, 7, 3, inclusive_left=False, inclusive_right=True) is True

    def test_x_in_wrapping_inclusive_left(self):
        # [7, 3)，x=7 在内部（左包含）
        assert in_interval(7, 7, 3, inclusive_left=True, inclusive_right=False) is True

    def test_x_in_wrapping_inclusive_right(self):
        # (7, 3]，x=3 在内部（右包含）
        assert in_interval(3, 7, 3, inclusive_left=False, inclusive_right=True) is True

    def test_x_not_in_wrapping(self):
        # (7, 3)，x=5 不在（5 在中间，不跨 0 点也不在任一端）
        assert in_interval(5, 7, 3, inclusive_left=False, inclusive_right=False) is False


class TestInIntervalEdgeCases:
    """边界情况。"""

    def test_a_equals_b_full_open(self):
        # (a, a) 开开 — 数学意义上是空集（没有数同时 > a 且 < a）
        # 实际实现返回 False
        assert in_interval(0, 0, 0, inclusive_left=False, inclusive_right=False) is False
        assert in_interval(5, 5, 5, inclusive_left=False, inclusive_right=False) is False

    def test_a_equals_b_both_inclusive(self):
        # [a, a] — 只有 a 本身
        assert in_interval(0, 0, 0, inclusive_left=True, inclusive_right=True) is True
        assert in_interval(5, 5, 5, inclusive_left=True, inclusive_right=True) is True
        assert in_interval(1, 0, 0, inclusive_left=True, inclusive_right=True) is False

    def test_a_equals_b_left_only(self):
        # [a, a) — 整个环（a 被包含端覆盖）
        assert in_interval(0, 0, 0, inclusive_left=True, inclusive_right=False) is True
        assert in_interval(5, 0, 0, inclusive_left=True, inclusive_right=False) is True

    def test_chord_find_predecessor_use_case(self):
        """验证 Chord find_predecessor 循环条件：(n', successor] 左开右闭"""
        # blog 例子：m=3, 环大小=8
        # 在 node 3 上查 id=1，n'=3, successor=0
        # 需要判断 1 ∈ (3, 0] — 即左开右闭跨 0点区间
        # 区间 (3, 0] = {4,5,6,7,0}，1 不在里面
        assert in_interval(1, 3, 0, inclusive_left=False, inclusive_right=True) is False
        # 0 ∈ (3, 0] — 正确，0 在区间内
        assert in_interval(0, 3, 0, inclusive_left=False, inclusive_right=True) is True

    def test_chord_closest_preceding_finger_use_case(self):
        """验证 Chord closest_preceding_finger 双开区间：(n, k)"""
        # blog 例子：node 3 的 finger table 中，finger[3].node=0
        # 判断 0 ∈ (3, 1) — 双开跨 0 点区间
        assert in_interval(0, 3, 1, inclusive_left=False, inclusive_right=False) is True

    def test_notify_use_case(self):
        """验证 Chord notify 判断：candidate ∈ (predecessor, n) 双开"""
        # 新 node n=5 加入，np=1, ns=7 之间
        # 判断 5 ∈ (1, 7) — 普通区间，不跨 0
        assert in_interval(5, 1, 7, inclusive_left=False, inclusive_right=False) is True
        # 3 也在 (1, 7)
        assert in_interval(3, 1, 7, inclusive_left=False, inclusive_right=False) is True
        # 1 不在（左侧排除）
        assert in_interval(1, 1, 7, inclusive_left=False, inclusive_right=False) is False

    def test_init_finger_table_optimization_interval(self):
        """验证 init_finger_table 优化分支条件：start ∈ [n, finger[i].node) 左闭右开"""
        # node n=1, finger[i].node=3，判断 2 ∈ [1, 3)
        assert in_interval(2, 1, 3, inclusive_left=True, inclusive_right=False) is True
        # 1 ∈ [1, 3)
        assert in_interval(1, 1, 3, inclusive_left=True, inclusive_right=False) is True
        # 3 ∉ [1, 3)
        assert in_interval(3, 1, 3, inclusive_left=True, inclusive_right=False) is False
