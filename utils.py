import hashlib

from config import M, RING_SIZE


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
        if not inclusive_left and not inclusive_right:
            return False
        if inclusive_left and x == a:
            return True
        if inclusive_right and x == b:
            return True
        if inclusive_left and inclusive_right:
            return x == a
        # 半开区间覆盖整个环（除了 a 本身在非包含端）
        return x != a if not inclusive_left else x != b if not inclusive_right else True

    if a < b:
        lower = a < x if not inclusive_left else a <= x
        upper = x < b if not inclusive_right else x <= b
        return lower and upper
    else:
        # 跨 0 点: (a, 2^m) ∪ [0, b)
        in_upper = a < x if not inclusive_left else a <= x
        in_lower = x < b if not inclusive_right else x <= b
        # x > a 或 x < b
        if in_upper and x < RING_SIZE:
            return True
        if in_lower and x >= 0:
            return True
        return False
