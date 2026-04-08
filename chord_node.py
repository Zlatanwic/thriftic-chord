import json
import threading
import time
import logging

import numpy as np
import thriftpy2
from thriftpy2.rpc import make_client

from config import M, RING_SIZE, STABILIZE_INTERVAL, FIX_FINGERS_INTERVAL, CHECK_PREDECESSOR_INTERVAL, RPC_TIMEOUT
from utils import consistent_hash, in_interval
from embedding import cosine_similarity, from_json

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
        # Phase 5: 向量存储 — doc_id -> {text, embedding(np.ndarray)}
        self.vector_store: dict[str, dict] = {}
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

    def init_finger_table(self) -> None:
        """初始化 finger table：每一项都通过 find_successor 确定真正的 successor。"""
        with self._lock:
            for i in range(M):
                start = self.finger_table[i]["start"]
                # 复用本地 find_successor 逻辑，不发 RPC
                n_prime = self
                while True:
                    succ = self._get_successor_internal(n_prime)
                    # 判断 id 是否在 (n_prime, successor] 区间内
                    if in_interval(start, n_prime.node_id, succ.node_id,
                                   inclusive_left=False, inclusive_right=True):
                        self.finger_table[i]["node"] = succ
                        break
                    # 否则跳到 closest preceding finger
                    cpf = self._closest_preceding_finger_internal(start, n_prime)
                    if cpf.node_id == n_prime.node_id:
                        # 无法前进，使用 successor
                        self.finger_table[i]["node"] = succ
                        break
                    n_prime = cpf
                logger.debug(f"finger[{i}] start={start} -> node={self.finger_table[i]['node'].node_id}")

    def _get_successor_internal(self, node_info: "chord_thrift.NodeInfo") -> "chord_thrift.NodeInfo":
        """获取指定 node 的 successor（内部使用，不加锁）。"""
        if node_info.node_id == self.node_id:
            return self.successor
        try:
            client = rpc_call(node_info.host, node_info.port)
            result = client.get_successor()
            client._iprot.trans.close()
            return result
        except Exception as e:
            logger.warning(f"_get_successor_internal failed: {e}")
            return self.successor

    def _closest_preceding_finger_internal(self, id: int,
                                          node_info: "chord_thrift.NodeInfo") -> "chord_thrift.NodeInfo":
        """获取指定 node 的 closest preceding finger（内部使用，不加锁）。"""
        if node_info.node_id == self.node_id:
            for i in range(M - 1, -1, -1):
                finger_node = self.finger_table[i]["node"]
                if in_interval(finger_node.node_id, self.node_id, id,
                               inclusive_left=False, inclusive_right=False):
                    return finger_node
            return make_node_info(self.node_id, self.host, self.port)
        try:
            client = rpc_call(node_info.host, node_info.port)
            result = client.closest_preceding_finger(id)
            client._iprot.trans.close()
            return result
        except Exception as e:
            logger.warning(f"_closest_preceding_finger_internal failed: {e}")
            return make_node_info(self.node_id, self.host, self.port)

    # ========================
    # === Node 加入 ===
    # ========================

    def update_finger_table(self, s: "chord_thrift.NodeInfo", i: int) -> None:
        """被其他 node 调用：尝试将 finger[i] 更新为 s，并递归向前传播。"""
        with self._lock:
            finger_i_node = self.finger_table[i]["node"]

        # 判断 s 是否落在 [self, finger_i_node) 区间（左闭右开）
        if in_interval(s.node_id, self.node_id, finger_i_node.node_id,
                       inclusive_left=True, inclusive_right=False):
            with self._lock:
                self.finger_table[i]["node"] = s
            # 递归向前传播
            if self.predecessor is not None and self.predecessor.node_id != self.node_id:
                try:
                    client = rpc_call(self.predecessor.host, self.predecessor.port)
                    client.update_finger_table(s, i)
                    client._iprot.trans.close()
                except Exception as e:
                    logger.warning(f"update_finger_table propagate failed: {e}")

    def update_others(self) -> None:
        """通知所有需要将本 node 加入 finger table 的已有 node。"""
        for i in range(M):
            p_id = (self.node_id - (2 ** i)) % RING_SIZE
            try:
                p = self.find_predecessor(p_id)
                if p.node_id != self.node_id:
                    p.update_finger_table(make_node_info(self.node_id, self.host, self.port), i)
            except Exception as e:
                logger.warning(f"update_others i={i} failed: {e}")

    def join(self, known_host: str, known_port: int) -> None:
        """通过已知 node 加入 Chord 环。"""
        self.predecessor = None
        try:
            client = rpc_call(known_host, known_port)
            self.successor = client.find_successor(self.node_id)
            self.finger_table[0]["node"] = self.successor
            client._iprot.trans.close()
            logger.info(f"Joined via {known_host}:{known_port}, successor={self.successor.node_id}")
            # 初始化完整 finger table
            self.init_finger_table()
            # 通知其他 node 更新 finger table
            self.update_others()
            # 从 successor 迁移属于本 node 的 keys
            self._migrate_from_successor()
        except Exception as e:
            logger.error(f"Join failed: {e}")
            raise

    def _migrate_from_successor(self) -> None:
        """从 successor 迁移 (predecessor, self.node_id] 区间的 keys。"""
        if self.successor.node_id == self.node_id:
            return
        try:
            client = rpc_call(self.successor.host, self.successor.port)
            transferred = client.transfer_keys(self.predecessor.node_id if self.predecessor else self.node_id,
                                              self.node_id)
            client._iprot.trans.close()
            with self._lock:
                for key, value in transferred.items():
                    self.data[key] = value
            if transferred:
                logger.info(f"Migrated {len(transferred)} keys from successor {self.successor.node_id}")
        except Exception as e:
            logger.warning(f"_migrate_from_successor failed: {e}")

    def start_background_tasks(self) -> None:
        """启动三个后台 daemon 线程。"""
        for fn in [self._stabilize, self._fix_fingers, self._check_predecessor]:
            t = threading.Thread(target=fn, daemon=True)
            t.start()

    # ========================
    # === 数据存取（Phase 3）===
    # ========================

    def put(self, key: str, value: str) -> None:
        """存储 key-value。先通过 find_successor 找到负责的 node，再转发。"""
        key_id = consistent_hash(key)
        target = self.find_successor(key_id)
        if target.node_id == self.node_id:
            with self._lock:
                self.data[key] = value
            logger.info(f"put key={key} (id={key_id}) -> local store")
        else:
            try:
                client = rpc_call(target.host, target.port)
                client.put(key, value)
                client._iprot.trans.close()
                logger.info(f"put key={key} (id={key_id}) -> forward to node {target.node_id}")
            except Exception as e:
                logger.error(f"put forward failed: {e}")
                raise

    def get(self, key: str) -> "chord_thrift.GetResult":
        """读取 key。先通过 find_successor 找到负责的 node，再转发。"""
        key_id = consistent_hash(key)
        target = self.find_successor(key_id)
        if target.node_id == self.node_id:
            with self._lock:
                if key in self.data:
                    return chord_thrift.GetResult(found=True, value=self.data[key])
                return chord_thrift.GetResult(found=False, value="")
        else:
            try:
                client = rpc_call(target.host, target.port)
                result = client.get(key)
                client._iprot.trans.close()
                return result
            except Exception as e:
                logger.error(f"get forward failed: {e}")
                return chord_thrift.GetResult(found=False, value="")

    def transfer_keys(self, from_id: int, to_id: int) -> dict:
        """将 (from_id, to_id] 区间内的所有 key 迁移出去，返回待迁移的 kv 对。"""
        to_transfer = {}
        with self._lock:
            keys_to_remove = []
            for key in self.data:
                key_id = consistent_hash(key)
                if in_interval(key_id, from_id, to_id,
                               inclusive_left=False, inclusive_right=True):
                    to_transfer[key] = self.data[key]
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del self.data[key]
        if to_transfer:
            logger.info(f"transfer_keys ({from_id}, {to_id}]: {len(to_transfer)} keys")
        return to_transfer

    # ========================
    # === Phase 5: AI 向量搜索 ===
    # ========================

    def put_document(self, doc_id: str, text: str, embedding_json: str) -> None:
        """存储文档及其 embedding 向量。

        文档按 doc_id 的哈希路由到负责的 node（与 put 相同逻辑）。
        embedding_json 是客户端预计算好的向量的 JSON 序列化。
        """
        key_id = consistent_hash(doc_id)
        target = self.find_successor(key_id)
        if target.node_id == self.node_id:
            embedding = from_json(embedding_json)
            with self._lock:
                self.vector_store[doc_id] = {"text": text, "embedding": embedding}
            logger.info(f"put_document doc_id={doc_id!r} (id={key_id}) -> local, "
                        f"total docs={len(self.vector_store)}")
        else:
            try:
                client = rpc_call(target.host, target.port)
                client.put_document(doc_id, text, embedding_json)
                client._iprot.trans.close()
                logger.info(f"put_document doc_id={doc_id!r} -> forward to node {target.node_id}")
            except Exception as e:
                logger.error(f"put_document forward failed: {e}")
                raise

    def local_search(self, query_embedding_json: str, top_k: int) -> list:
        """在本 node 存储的文档中，按余弦相似度返回 top_k 个最相似的结果。

        这是 scatter-gather 的 scatter 侧：
        客户端并发调用所有 node 的 local_search，然后在客户端 merge。
        """
        query_vec = from_json(query_embedding_json)
        results = []
        with self._lock:
            for doc_id, entry in self.vector_store.items():
                score = cosine_similarity(query_vec, entry["embedding"])
                results.append((doc_id, entry["text"], score))

        # 按相似度降序排序，取 top_k
        results.sort(key=lambda x: x[2], reverse=True)
        results = results[:top_k]

        return [
            chord_thrift.SearchResult(doc_id=doc_id, text=text, score=score)
            for doc_id, text, score in results
        ]

    def get_all_nodes(self) -> list:
        """遍历 successor 链，返回环上所有 node 的信息。

        用于 scatter-gather 搜索时，客户端需要知道所有 node 地址。
        """
        nodes = []
        with self._lock:
            start_id = self.node_id
            nodes.append(make_node_info(self.node_id, self.host, self.port))
            current = self.successor

        # 沿 successor 链遍历直到回到起点
        visited = {start_id}
        while current.node_id not in visited:
            visited.add(current.node_id)
            nodes.append(current)
            try:
                client = rpc_call(current.host, current.port)
                next_node = client.get_successor()
                client._iprot.trans.close()
                current = next_node
            except Exception as e:
                logger.warning(f"get_all_nodes: failed to reach {current.node_id}: {e}")
                break

        return nodes

    # ========================
    # === 运维 ===
    # ========================

    def ping(self) -> bool:
        return True
