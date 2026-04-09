M = 8  # identifier 位数，环大小 2^8 = 256
RING_SIZE = 2 ** M

SUCCESSOR_LIST_SIZE = 3  # successor list 长度（容错冗余数）

STABILIZE_INTERVAL = 1    # 秒
FIX_FINGERS_INTERVAL = 3  # 秒
CHECK_PREDECESSOR_INTERVAL = 5  # 秒

RPC_TIMEOUT = 5000  # 毫秒
