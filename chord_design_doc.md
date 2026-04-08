# 基于 Thriftpy2 的 Chord DHT 设计文档

## 1. 项目概述

### 1.1 目标

基于现有的 pingpong thrift RPC 框架，实现一个完整的 Chord 分布式哈希表（DHT）。每个 node 作为独立进程运行，通过 thriftpy2 提供的 RPC 机制互相通信，共同维护一个一致性哈希环，对外提供分布式的 key-value 存储服务。

### 1.2 参考

- Chord 论文：Ion Stoica et al., *Chord: A Scalable Peer-to-peer Lookup Service for Internet Applications*
- 博客解析：k-ye，《Chord 详解》（知乎）
- 现有代码：`pingpong.thrift`、`pingpongserver.py`、`pingpongclient.py`

### 1.3 术语约定

| 术语 | 含义 |
|------|------|
| identifier | 通过哈希算法对 node 地址或 key 计算得到的 m 位整数 |
| identifier space | 所有 identifier 构成的取模 2^m 的环形空间 |
| successor(k) | 从 k 顺时针方向遇到的第一个 node |
| predecessor(k) | 从 k 逆时针方向遇到的第一个 node |
| finger table | 每个 node 维护的长度为 m 的路由加速表 |
| stabilization | 后台周期性地修复 successor/predecessor/finger table 的过程 |

---

## 2. 系统架构

### 2.1 整体拓扑

```
         Node 0 (port 6001)
        /                   \
       /                     \
Node 5 (port 6006)     Node 1 (port 6002)
      |                       |
Node 4 (port 6005)     Node 3 (port 6004)
       \                     /
        \                   /
         Node 2 (port 6003)
```

每个 node 是一个独立的 Python 进程，同时扮演两个角色：

- **Server**：监听固定端口，通过 `make_server` 对外提供 `ChordService` 的 RPC 接口
- **Client**：在执行 Chord 逻辑时，临时通过 `make_client` 连接到其他 node 发起 RPC 调用

### 2.2 从 PingPong 到 Chord 的映射

现有框架中的概念与 Chord 的对应关系：

| PingPong 框架 | Chord 系统 |
|---------------|-----------|
| `pingpong.thrift` | `chord.thrift`（重新定义 Chord 所需的全部 RPC 接口） |
| `Dispatcher` 类 | `ChordNode` 类（实现所有 Chord 逻辑 + 本地状态） |
| `make_server(service, handler, host, port)` | 每个 node 启动时用自己的端口创建 server |
| `make_client(service, host, port)` | node 之间互相调用时临时创建 client |
| `ping()` → 返回 `"pong"` | `find_successor(key)` → 返回负责该 key 的 node 信息 |
| `oneway sleep()` | stabilize / fix_fingers 等后台异步任务 |

---

## 3. 接口设计

### 3.1 Thrift IDL 定义

```thrift
// chord.thrift

// 表示一个 Chord node 的网络位置和身份
struct NodeInfo {
    1: i64 node_id,      // identifier（哈希值）
    2: string host,       // IP 地址
    3: i32 port,          // 端口号
}

// 表示一条 finger table 记录
struct FingerEntry {
    1: i64 start,         // (n + 2^(i-1)) mod 2^m
    2: NodeInfo node,     // successor(start)
}

// 存储操作的结果
struct GetResult {
    1: bool found,
    2: string value,
}

service ChordService {

    // === 核心查找 ===
    // 给定 identifier，返回其 successor node
    NodeInfo find_successor(1: i64 id),

    // 给定 identifier，返回其 predecessor node
    NodeInfo find_predecessor(1: i64 id),

    // 在本地 finger table 中找 id 的最近前驱
    NodeInfo closest_preceding_finger(1: i64 id),

    // === 状态查询 ===
    // 获取本 node 的 successor
    NodeInfo get_successor(),

    // 获取本 node 的 predecessor
    NodeInfo get_predecessor(),

    // 获取本 node 的 identifier
    i64 get_node_id(),

    // 获取完整 finger table（调试用）
    list<FingerEntry> get_finger_table(),

    // === Stabilization 协议 ===
    // 通知本 node：调用方可能是本 node 的新 predecessor
    void notify(1: NodeInfo candidate),

    // === 数据存取 ===
    // 存储 key-value
    void put(1: string key, 2: string value),

    // 读取 key 对应的 value
    GetResult get(1: string key),

    // 数据迁移：新 node 加入时，successor 将部分数据转移给它
    map<string, string> transfer_keys(1: i64 from_id, 2: i64 to_id),

    // === 运维 ===
    // 健康检查
    bool ping(),
}
```

### 3.2 接口职责说明

**核心查找接口**（对应博客中带 `n.` 前缀的 RPC 函数）：

- `find_successor(id)`：入口函数。内部调用 `find_predecessor(id)`，再取其 successor 返回。这是整个 Chord 对外最核心的操作。
- `find_predecessor(id)`：沿着 finger table 链式跳转，直到找到满足 `id ∈ (n', n'.successor]` 的 node n'。
- `closest_preceding_finger(id)`：纯本地操作，从 finger table 的最远项开始倒序扫描，找到落在 `(n, id)` 开区间内的最近 node。

**Stabilization 接口**：

- `notify(candidate)`：被调用方检查 candidate 是否应成为自己的新 predecessor。对应博客中 stabilize 过程的 `successor.notify(n)` 步骤。

**数据接口**：

- `put/get`：标准 KV 操作。内部先通过 `find_successor(hash(key))` 定位目标 node，再对该 node 发起实际的存取 RPC。
- `transfer_keys`：node 加入时的数据迁移。新 node 告诉 successor："把 identifier 在 `(from_id, to_id]` 范围内的 key 都给我。"

---

## 4. 核心数据结构

### 4.1 ChordNode 类

每个进程维护一个 `ChordNode` 实例，作为 thrift server 的 Dispatcher：

```
ChordNode:
    ├── node_id: int            # 本 node 的 identifier
    ├── host: str               # 本 node 的 IP
    ├── port: int               # 本 node 的端口
    ├── m: int                  # identifier 位数（如 m=8 则环大小为 256）
    ├── successor: NodeInfo     # 环上的直接后继
    ├── predecessor: NodeInfo   # 环上的直接前驱（可为 None）
    ├── finger_table: list[FingerEntry]  # 长度为 m 的路由表
    ├── data: dict[str, str]    # 本 node 负责的 KV 数据
    └── _lock: threading.Lock   # 保护并发访问
```

### 4.2 Finger Table 结构

对于 identifier 为 n 的 node，finger table 共 m 行：

```
第 i 行 (1 ≤ i ≤ m):
    start    = (n + 2^(i-1)) mod 2^m
    interval = [finger[i].start, finger[i+1].start)
    node     = successor(start)
```

关键性质：第 1 行的 node 就是本 node 的 successor。每行覆盖的 identifier 范围呈指数增长，因此 finger table 实现了类似跳表的加速效果。

### 4.3 模运算区间判断

这是实现中最容易出错的地方。需要一个通用工具函数来处理环上的区间判断：

```
def in_interval(x, a, b, inclusive_left=False, inclusive_right=False):
    """
    判断 x 是否在环上的区间内。
    
    当 a < b 时：普通区间判断
    当 a ≥ b 时：区间跨过了 0 点，等价于 x ∈ [a, 2^m) ∪ [0, b]
    
    示例 (m=3, 环大小=8):
        2 ∈ [7, 3)  → True   （跨 0 点）
        5 ∉ [7, 3)  → False
        0 ∈ (7, 1]  → True
    """
```

博客中出现的所有区间类型：

| 使用场景 | 区间类型 | 示例 |
|---------|---------|------|
| `find_predecessor` 的 while 条件 | 左开右闭 `(n', n'.successor]` | `k ∈ (3, 0]` |
| `closest_preceding_finger` 的判断 | 双开 `(n, k)` | `finger[i].node ∈ (3, 1)` |
| `notify` 的判断 | 双开 `(predecessor, n)` | `n' ∈ (0, 3)` |
| `init_finger_table` 的优化 | 左闭右开 `[n, finger[i].node)` | `start ∈ [1, 3)` |

---

## 5. 核心流程

### 5.1 Node 启动与加入

```
启动流程:
    1. 计算 node_id = hash(host:port) mod 2^m
    2. 启动 thrift server，监听指定端口
    3. 如果是第一个 node:
         - successor = 自己
         - predecessor = None
         - finger table 全部指向自己
    4. 如果不是第一个 node（已知某个 existing_node 的地址）:
         - 调用 existing_node.find_successor(self.node_id) 得到自己的 successor
         - predecessor = None（等 stabilize 填充）
         - 启动后台任务，等待 stabilize 自动修复完整的环结构
    5. 启动后台周期任务: stabilize(), fix_fingers(), check_predecessor()
```

### 5.2 Stabilization 流程

三个后台定时任务协同工作，确保环结构的最终一致性：

**stabilize()** — 每隔 T1 秒运行一次：

```
1. x = successor.get_predecessor()     // RPC
2. if x 在 (self, successor) 区间内:
       successor = x                   // 发现了更近的 successor
3. successor.notify(self)              // 告诉 successor "我存在"
```

**notify(candidate)** — 被动调用：

```
1. if predecessor 为空，或 candidate 在 (predecessor, self) 区间内:
       predecessor = candidate
       触发数据迁移：将属于 candidate 负责范围的 key 转移给它
```

**fix_fingers()** — 每隔 T2 秒运行一次：

```
1. 随机选择 i (2 ≤ i ≤ m)     // i=1 即 successor，由 stabilize 负责
2. finger[i].node = find_successor(finger[i].start)
```

**check_predecessor()** — 每隔 T3 秒运行一次：

```
1. 尝试 predecessor.ping()
2. 如果失败（超时或连接拒绝）:
       predecessor = None
```

### 5.3 Key Lookup 流程

```
客户端请求 get(key) 或 put(key, value):

    1. key_id = hash(key) mod 2^m
    2. target = find_successor(key_id)
    3. if target 是自己:
           本地读写 data[key]
       else:
           RPC 调用 target.get(key) 或 target.put(key, value)
```

`find_successor` 的内部跳转过程（对应博客核心算法）：

```
find_successor(id):
    n' = find_predecessor(id)
    return n'.get_successor()

find_predecessor(id):
    n' = self
    while id ∉ (n'.node_id, n'.successor.node_id]:
        n' = n'.closest_preceding_finger(id)   // RPC 跳转
    return n'

closest_preceding_finger(id):
    for i = m downto 1:
        if finger[i].node_id ∈ (self.node_id, id):   // 双开区间
            return finger[i].node
    return self
```

### 5.4 Node 加入的完整时序

以 node N 加入 Np 和 Ns 之间为例：

```
时刻 0:  环状态 ... → Np → Ns → ...
         N.join(existing_node)
         N.successor = Ns, N.predecessor = None

时刻 1:  N 执行 stabilize()
         x = Ns.get_predecessor() → 返回 Np
         Np ∉ (N, Ns)，不更新 successor
         Ns.notify(N) → Ns 发现 N ∈ (Np, Ns)，更新 predecessor = N

时刻 2:  Np 执行 stabilize()
         x = Ns.get_predecessor() → 返回 N
         N ∈ (Np, Ns)，更新 successor = N
         N.notify(Np) → N 发现 predecessor 为 None，更新 predecessor = Np

时刻 3:  环状态 ... → Np → N → Ns → ...  ✓ 完成
         后续 fix_fingers() 逐步修复各 node 的 finger table
```

---

## 6. 项目文件结构

```
chord-dht/
├── chord.thrift              # Thrift IDL 接口定义
├── chord_node.py             # ChordNode 类：所有 Chord 逻辑 + 状态
├── server.py                 # 启动入口：解析参数、创建 node、启动 thrift server
├── client.py                 # 客户端 CLI：连接任意 node 执行 put/get/lookup
├── utils.py                  # 工具函数：hash、in_interval、make_rpc_client
├── config.py                 # 全局配置：m 值、stabilize 周期、超时时间
└── test/
    ├── test_interval.py      # 模运算区间判断的单元测试
    ├── test_single_node.py   # 单 node 自环测试
    ├── test_join.py          # 多 node 加入的集成测试
    └── test_stabilize.py     # stabilization 收敛性测试
```

### 6.1 各模块职责

**`chord_node.py`** — 系统核心，实现 `ChordService` 的所有方法，同时管理本地状态（finger table、predecessor、data）。作为 thrift server 的 Dispatcher。

**`server.py`** — 启动脚本。接收命令行参数（自身端口、已知 node 地址），创建 `ChordNode` 实例，启动后台线程（stabilize/fix_fingers/check_predecessor），最后调用 `make_server(...).serve()` 进入事件循环。

**`client.py`** — 用户交互入口。连接到环上任意一个 node，提供 `put`、`get`、`lookup`、`status` 等命令。

**`utils.py`** — 包含 `consistent_hash(key, m)`、`in_interval(x, a, b, ...)`、以及封装了连接复用或错误处理的 `rpc_call(host, port, method, *args)` 等工具。

---

## 7. 关键设计决策

### 7.1 哈希函数与 m 值选择

开发和测试阶段建议使用较小的 m（如 m=8，环大小 256），便于手动验证。生产环境可以使用 m=160（SHA-1）。哈希函数使用 `hashlib.sha1`，取前 m 位。

### 7.2 RPC 连接管理

`make_client` 每次创建新连接。在 `find_predecessor` 的链式跳转中可能频繁创建短连接。两种策略：

- **简单版**：每次 RPC 创建新 client，用完即关。实现简单，适合开发阶段。
- **优化版**：维护一个连接池 `{(host, port): client}`，复用已有连接。注意处理连接失效的情况。

### 7.3 并发与线程安全

thriftpy2 的 `make_server` 默认使用 `TThreadedServer`，每个请求一个线程。`ChordNode` 的状态（finger table、predecessor、data）会被多个线程并发访问。需要用 `threading.Lock` 或 `threading.RLock` 保护关键操作。

需要加锁的操作包括：读写 `predecessor`、读写 `successor`（即 `finger[0]`）、修改 finger table 任意项、读写 `data` 字典。

### 7.4 后台任务的实现方式

使用 `threading.Thread` + `daemon=True` 运行后台周期任务。每个任务在独立线程中以固定间隔循环执行，外层包裹 try-except 防止单次失败导致线程退出。

建议的周期参数：

| 任务 | 周期 | 说明 |
|------|------|------|
| stabilize | 1-2 秒 | 频率最高，保证 successor 链的及时更新 |
| fix_fingers | 2-5 秒 | 每次随机修复一行，m 轮后全部更新 |
| check_predecessor | 5-10 秒 | 检测 predecessor 是否存活 |

### 7.5 错误处理策略

在分布式环境中，RPC 调用随时可能因为目标 node 宕机、网络分区等原因失败。处理原则：

- **查找类操作**（find_successor 等）：捕获异常后返回错误或重试。上层 client 可用 exponential backoff 策略。
- **stabilize 类操作**：捕获异常后静默跳过本轮，等待下一轮执行。
- **predecessor 检测**：RPC 失败时将 predecessor 置为 None，等待 notify 重新填充。

---

## 8. 开发计划

### Phase 1：基础环结构（正确性）

**目标**：多个 node 能组成环，successor 链表完整闭合。

- 实现 `chord.thrift` 和基础 `ChordNode`（仅含 successor/predecessor）
- 实现简化版 `join()`：新 node 找到自己的 successor
- 实现 `stabilize()` + `notify()`：后台修复 successor/predecessor
- 实现 `check_predecessor()`
- 验证：启动 3-5 个 node，确认它们在若干轮 stabilize 后形成完整的环

### Phase 2：Finger Table 加速（性能）

**目标**：lookup 复杂度从 O(N) 降到 O(log N)。

- 实现完整的 finger table 初始化
- 实现 `fix_fingers()` 后台任务
- 实现 `find_predecessor()` + `closest_preceding_finger()`
- 验证：在 16+ 个 node 的环上执行 lookup，统计 hop 次数应接近 log₂(N)

### Phase 3：数据存储（功能）

**目标**：支持分布式 key-value 读写。

- 实现 `put()` / `get()` 接口
- 实现 `transfer_keys()`：新 node 加入时从 successor 迁移数据
- 实现 client CLI 工具
- 验证：写入数据后增删 node，数据仍可正确读取

### Phase 4：容错与健壮性（可选进阶）

**目标**：应对 node 故障。

- 维护 successor list（多个备选 successor），而非只存一个
- node 离开时的优雅退出（主动迁移数据 + 通知邻居）
- 数据副本：在 successor list 的 node 上冗余存储

### Phase 5：分布式语义向量搜索 + RAG（AI 扩展）

**目标**：在 Chord DHT 上构建分布式向量存储，支持语义搜索和 RAG 问答。

**核心思路**：Chord 环不仅存储 KV 数据，还存储文档的嵌入向量（embedding）。语义搜索采用 Scatter-Gather 模式：

```
                    ┌───────────┐
                    │ AI Client │
                    │ (query)   │
                    └─────┬─────┘
                          │ 1. encode(query) → vector
                          │ 2. get_all_nodes()
                    ┌─────┴─────┐
           scatter  │           │  scatter
         ┌─────────┼───────────┼─────────┐
         ▼         ▼           ▼         ▼
    ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
    │ Node A  │ │ Node B  │ │ Node C  │ │ Node D  │
    │local_   │ │local_   │ │local_   │ │local_   │
    │search() │ │search() │ │search() │ │search() │
    └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘
         │           │           │           │
         └───────────┴─────┬─────┴───────────┘
                           │ gather: merge top-k
                    ┌──────┴──────┐
                    │  AI Client  │
                    │ (results)   │
                    │ + LLM call  │
                    └─────────────┘
```

**新增接口**：

| RPC | 说明 |
|-----|------|
| `put_document(doc_id, text, embedding_json)` | 存储文档及其向量，按 hash(doc_id) 路由 |
| `local_search(query_embedding_json, top_k)` | 本 node 上的局部向量相似搜索 |
| `get_all_nodes()` | 遍历 successor 链，返回环上所有 node |

**新增文件**：

| 文件 | 职责 |
|------|------|
| `embedding.py` | 文本嵌入：sentence-transformers (高质量) 或 trigram fallback (零依赖) |
| `ai_client.py` | AI 客户端 CLI：store-doc / search / rag 三个命令 |
| `sample_docs.jsonl` | 示例文档集（Chord/DHT/RAG 相关知识） |

**RAG 流程**：

1. 用户提问 → 计算 query embedding
2. Scatter：并发调用所有 node 的 `local_search()`
3. Gather：合并所有 node 返回的结果，取全局 top-k
4. 将检索到的文档拼入 prompt，调用 LLM 生成回答

---

## 9. 测试策略

### 9.1 单元测试

- `in_interval` 的各种边界情况（跨 0 点、端点包含/不包含、a == b）
- `consistent_hash` 的分布均匀性
- `closest_preceding_finger` 在各种 finger table 状态下的正确性

### 9.2 集成测试

- **环形成测试**：启动 N 个 node（按随机顺序 join），等待 stabilize 收敛，验证沿 successor 链遍历能回到起点且恰好经过 N 个 node。
- **Lookup 正确性测试**：在环稳定后，对所有可能的 identifier 执行 find_successor，验证结果与暴力遍历一致。
- **数据迁移测试**：写入 100 个 key，新增一个 node，验证 key 的归属自动调整且数据不丢失。
- **Hop 次数测试**：统计 lookup 的平均 hop 次数，验证接近 O(log N)。

### 9.3 故障注入测试

- 随机 kill 一个 node，验证其他 node 的 check_predecessor 能检测到并将其 predecessor 置空。
- 在 stabilize 尚未完成时发起 lookup，验证系统不会死循环（需设置 TTL 或最大跳转次数）。

---

## 10. 启动示例

```bash
# 终端 1：启动第一个 node（种子节点）
python server.py --host 127.0.0.1 --port 6001

# 终端 2：第二个 node 加入
python server.py --host 127.0.0.1 --port 6002 --join 127.0.0.1:6001

# 终端 3：第三个 node 加入
python server.py --host 127.0.0.1 --port 6003 --join 127.0.0.1:6001

# 终端 4：客户端操作
python client.py --node 127.0.0.1:6001 put mykey myvalue
python client.py --node 127.0.0.1:6002 get mykey
# 输出: myvalue

python client.py --node 127.0.0.1:6003 status
# 输出: 环上所有 node 的 id、successor、predecessor 信息
```
