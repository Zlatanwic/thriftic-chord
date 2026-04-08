import hashlib

from config import  RING_SIZE


def consistent_hash(key: str) -> int:
    """对 key 做 SHA-1 哈希，取前 M 位作为 identifier。"""
    h = hashlib.sha1(key.encode()).hexdigest()
    return int(h, 16) % RING_SIZE


def in_interval(x: int, a: int, b: int,
                inclusive_left: bool = False,
                inclusive_right: bool = False) -> bool:
    """
    判断 x 是否在环上的区间内。

    a, b 是环上的两个点，区间方向为顺时针（a -> b）。
    当 a == b 时：
      - 如果两端都不包含，返回 False（空区间）
      - 如果任一端包含且 x == a == b，返回 True
      - 否则返回 True（整个环）
    """
    x = x % RING_SIZE
    a = a % RING_SIZE
    b = b % RING_SIZE

    if a == b:
        # a == b 时，区间覆盖整个环（可能排除端点）
        # (a, a) 开开 -> 整个环除 a 本身
        # [a, a) 或 (a, a] -> 整个环（a 被包含端覆盖）
        # [a, a] -> 仅 a 本身
        if inclusive_left and inclusive_right:
            return x == a
        if x == a:
            return inclusive_left or inclusive_right
        return True

    if a < b:
        lower = a < x if not inclusive_left else a <= x
        upper = x < b if not inclusive_right else x <= b
        return lower and upper
    else:
        # 跨 0 点: (a, RING_SIZE) ∪ [0, b)
        # 上部分 (a, RING_SIZE): x 必须 > a（不是 a < x！）
        # 下部分 [0, b): x < b（右开）或 x <= b（右闭）
        in_upper = x > a if not inclusive_left else x >= a
        in_lower = x < b if not inclusive_right else x <= b
        if in_upper:
            return True
        if in_lower:
            return True
        return False

