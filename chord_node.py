import threading
import time
import logging

import thriftpy2
from thriftpy2.rpc import make_client

from config import M, RING_SIZE, STABILIZE_INTERVAL, FIX_FINGERS_INTERVAL, CHECK_PREDECESSOR_INTERVAL, RPC_TIMEOUT
from utils import consistent_hash, in_interval

chord_thrift = thriftpy2.load("chord.thrift", module_name="chord_thrift")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ChordNode")


def make_node_info(node_id: int, host: str, port: int):
    """创建一个 NodeInfo thrift 结构体。"""
    return chord_thrift.NodeInfo(node_id=node_id, host=host, port=port)


def rpc_call(host: str, port: int):
    """创建到目标 node 的 RPC client。使用方需要自行关闭。"""
    return make_client(chord_thrift.ChordService, host, port, timeout=RPC_TIMEOUT)


class ChordNode:
    """
    Chord DHT 节点，作为 thrift server 的 Dispatcher。
    实现 ChordService 中的所有 RPC 方法。
    """

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.node_id = consistent_hash(f"{host}:{port}")

        self.successor = make_node_info(self.node_id, host, port)  # 初始指向自己
        self.predecessor = None

        # finger table: 索引 0 对应论文中的 i=1
        self.finger_table = []
        for i in range(M):
            start = (self.node_id + 2 ** i) % RING_SIZE
            self.finger_table.append({
                "start": start,
                "node": make_node_info(self.node_id, host, port),
            })

        self.data = {}
        self._lock = threading.RLock()

        logger.info(f"Node created: id={self.node_id}, addr={host}:{port}")

    # ========================
    # === 核心查找接口 ===
    # ========================

    def find_successor(self, id: int) -> "chord_thrift.NodeInfo":
        """找到 identifier id 的 successor node。"""
        n_prime = self.find_predecessor(id)
        # 如果 predecessor 是自己，直接返回自己的 successor
        if n_prime.node_id == self.node_id:
            with self._lock:
                return self.successor
        # 否则 RPC 获取 n_prime 的 successor
        try:
            client = rpc_call(n_prime.host, n_prime.port)
            result = client.get_successor()
            client._iprot.trans.close()
            return result
        except Exception as e:
            logger.error(f"find_successor RPC failed: {e}")
            with self._lock:
                return self.successor

    def find_predecessor(self, id: int) -> "chord_thrift.NodeInfo":
        """找到 identifier id 的 predecessor node。"""
        with self._lock:
            n_prime_id = self.node_id
            n_prime_info = make_node_info(self.node_id, self.host, self.port)
            succ = self.successor

        # while id not in (n', n'.successor]
        while not in_interval(id, n_prime_id, succ.node_id,
                              inclusive_left=False, inclusive_right=True):
            if n_prime_id == self.node_id:
                # 本地调用
                cpf = self.closest_preceding_finger(id)
            else:
                try:
                    client = rpc_call(n_prime_info.host, n_prime_info.port)
                    cpf = client.closest_preceding_finger(id)
                    client._iprot.trans.close()
                except Exception as e:
                    logger.error(f"find_predecessor RPC failed: {e}")
                    break

            # 如果没有进展（返回了自己），跳出防止死循环
            if cpf.node_id == n_prime_id:
                break

            n_prime_id = cpf.node_id
            n_prime_info = cpf

            # 获取 n_prime 的 successor
            if n_prime_id == self.node_id:
                with self._lock:
                    succ = self.successor
            else:
                try:
                    client = rpc_call(n_prime_info.host, n_prime_info.port)
                    succ = client.get_successor()
                    client._iprot.trans.close()
                except Exception as e:
                    logger.error(f"get_successor RPC failed: {e}")
                    break

        return n_prime_info

    def closest_preceding_finger(self, id: int) -> "chord_thrift.NodeInfo":
        """在本地 finger table 中找 id 的最近前驱。"""
        with self._lock:
            for i in range(M - 1, -1, -1):
                finger_node = self.finger_table[i]["node"]
                if in_interval(finger_node.node_id, self.node_id, id,
                               inclusive_left=False, inclusive_right=False):
                    return finger_node
            return make_node_info(self.node_id, self.host, self.port)

    # ========================
    # === 状态查询接口 ===
    # ========================

    def get_successor(self) -> "chord_thrift.NodeInfo":
        with self._lock:
            return self.successor

    def get_predecessor(self) -> "chord_thrift.NodeInfo":
        with self._lock:
            if self.predecessor is None:
                return make_node_info(-1, "", 0)
            return self.predecessor

    def get_node_id(self) -> int:
        return self.node_id

    def get_finger_table(self) -> list:
        with self._lock:
            result = []
            for entry in self.finger_table:
                result.append(chord_thrift.FingerEntry(
                    start=entry["start"],
                    node=entry["node"],
                ))
            return result

    # ========================
    # === Stabilization ===
    # ========================

    def notify(self, candidate: "chord_thrift.NodeInfo") -> None:
        """candidate 告知自己可能是本 node 的新 predecessor。"""
        with self._lock:
            if self.predecessor is None or \
               in_interval(candidate.node_id, self.predecessor.node_id, self.node_id,
                           inclusive_left=False, inclusive_right=False):
                logger.info(f"Predecessor updated: {self.predecessor and self.predecessor.node_id} -> {candidate.node_id}")
                self.predecessor = candidate

    def _stabilize(self) -> None:
        """后台周期任务：检查并修复 successor。"""
        while True:
            time.sleep(STABILIZE_INTERVAL)
            try:
                with self._lock:
                    succ = self.successor

                # 获取 successor 的 predecessor
                if succ.node_id == self.node_id:
                    # successor 是自己，取本地 predecessor
                    with self._lock:
                        x = self.predecessor
                else:
                    client = rpc_call(succ.host, succ.port)
                    x = client.get_predecessor()
                    client._iprot.trans.close()
                    # get_predecessor 返回 node_id=-1 表示 None
                    if x.node_id == -1:
                        x = None

                if x is not None and \
                   in_interval(x.node_id, self.node_id, succ.node_id,
                               inclusive_left=False, inclusive_right=False):
                    with self._lock:
                        self.successor = x
                        self.finger_table[0]["node"] = x
                        succ = x
                        logger.info(f"Successor updated to {x.node_id}")

                # notify successor
                me = make_node_info(self.node_id, self.host, self.port)
                if succ.node_id == self.node_id:
                    self.notify(me)
                else:
                    client = rpc_call(succ.host, succ.port)
                    client.notify(me)
                    client._iprot.trans.close()

            except Exception as e:
                logger.debug(f"stabilize error: {e}")

    def _fix_fingers(self) -> None:
        """后台周期任务：随机修复 finger table 中的一项。"""
        import random
        while True:
            time.sleep(FIX_FINGERS_INTERVAL)
            try:
                i = random.randint(1, M - 1)  # 跳过 i=0 (successor)
                start = self.finger_table[i]["start"]
                new_node = self.find_successor(start)
                with self._lock:
                    self.finger_table[i]["node"] = new_node
            except Exception as e:
                logger.debug(f"fix_fingers error: {e}")

    def _check_predecessor(self) -> None:
        """后台周期任务：检查 predecessor 是否存活。"""
        while True:
            time.sleep(CHECK_PREDECESSOR_INTERVAL)
            with self._lock:
                pred = self.predecessor
            if pred is None or pred.node_id == self.node_id:
                continue
            try:
                client = rpc_call(pred.host, pred.port)
                client.ping()
                client._iprot.trans.close()
            except Exception:
                logger.info(f"Predecessor {pred.node_id} failed, setting to None")
                with self._lock:
                    self.predecessor = None

    # ========================
    # === Node 加入 ===
    # ========================

    def join(self, known_host: str, known_port: int) -> None:
        """通过已知 node 加入 Chord 环。"""
        self.predecessor = None
        try:
            client = rpc_call(known_host, known_port)
            self.successor = client.find_successor(self.node_id)
            self.finger_table[0]["node"] = self.successor
            client._iprot.trans.close()
            logger.info(f"Joined via {known_host}:{known_port}, successor={self.successor.node_id}")
        except Exception as e:
            logger.error(f"Join failed: {e}")
            raise

    def start_background_tasks(self) -> None:
        """启动三个后台 daemon 线程。"""
        for fn in [self._stabilize, self._fix_fingers, self._check_predecessor]:
            t = threading.Thread(target=fn, daemon=True)
            t.start()

    # ========================
    # === 数据存取（Phase 3，先留桩）===
    # ========================

    def put(self, key: str, value: str) -> None:
        pass

    def get(self, key: str) -> "chord_thrift.GetResult":
        return chord_thrift.GetResult(found=False, value="")

    def transfer_keys(self, from_id: int, to_id: int) -> dict:
        return {}

    # ========================
    # === 运维 ===
    # ========================

    def ping(self) -> bool:
        return True
