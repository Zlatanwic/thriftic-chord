struct NodeInfo {
    1: i64 node_id,  //identifier（哈希值）
    2: string host,
    3: i32 port,
}

//finger表中的一项
struct FingerEntry {
    1: i64 start,   //(n+2^(i-1))mod 2^m
    2: NodeInfo node,  //successor(start)
}

//存储操作的结果
struct GetResult {
    1: bool found,
    2: string value,
}

service ChordService {
    NodeInfo find_successor(1: i64 id),
    NodeInfo find_predecessor(1: i64 id),
    NodeInfo closest_preceding_finger(1: i64 id),

    NodeInfo get_successor(),
    NodeInfo get_predecessor(),
    i64 get_node_id(),
    list<FingerEntry> get_finger_table(),

    void notify(1: NodeInfo candidate),

    void put(1: string key, 2: string value),
    GetResult get(1: string key),
    map<string,string> transfer_keys(1: i64 from_id, 2: i64 to_id),

    bool ping(),
}