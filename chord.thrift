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

// === Phase 5: AI 向量搜索 ===

//语义搜索结果
struct SearchResult {
    1: string doc_id,
    2: string text,
    3: double score,
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

    // === Phase 5: AI 向量搜索 ===
    // 存储文档及其 embedding 向量（embedding 序列化为 JSON 字符串）
    void put_document(1: string doc_id, 2: string text, 3: string embedding_json),

    // 本地向量搜索：在本 node 存储的文档中找 top_k 个最相似的
    list<SearchResult> local_search(1: string query_embedding_json, 2: i32 top_k),

    // 获取环上所有 node 的地址列表（用于 scatter-gather）
    list<NodeInfo> get_all_nodes(),
}