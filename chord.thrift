struct NodeInfo {
    1: i64 node_id,  //identifier（哈希值）
    2: string host,
    3: port,  
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

    
}