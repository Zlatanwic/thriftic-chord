"""Chord DHT 客户端 CLI。"""
import argparse
import sys

import thriftpy2
from thriftpy2.rpc import make_client

from config import RPC_TIMEOUT
from utils import consistent_hash

chord_thrift = thriftpy2.load("chord.thrift", module_name="chord_client_thrift")


def parse_node(node_str: str):
    """解析 'host:port' 格式的 node 地址。"""
    host, port = node_str.rsplit(":", 1)
    return host, int(port)


def get_client(host: str, port: int):
    return make_client(chord_thrift.ChordService, host, port, timeout=RPC_TIMEOUT)


def cmd_put(host: str, port: int, key: str, value: str):
    client = get_client(host, port)
    client.put(key, value)
    client._iprot.trans.close()
    print(f"OK: put key={key!r} value={value!r}")


def cmd_get(host: str, port: int, key: str):
    client = get_client(host, port)
    result = client.get(key)
    client._iprot.trans.close()
    if result.found:
        print(f"OK: get key={key!r} -> value={result.value!r}")
    else:
        print(f"NOT FOUND: key={key!r}")


def cmd_status(host: str, port: int):
    client = get_client(host, port)
    node_id = client.get_node_id()
    successor = client.get_successor()
    predecessor = client.get_predecessor()
    finger_table = client.get_finger_table()
    client._iprot.trans.close()

    print(f"Node ID: {node_id}")
    print(f"Address: {host}:{port}")
    print(f"Successor: {successor.node_id} ({successor.host}:{successor.port})")
    if predecessor.node_id == -1:
        print(f"Predecessor: None")
    else:
        print(f"Predecessor: {predecessor.node_id} ({predecessor.host}:{predecessor.port})")
    print(f"Finger table:")
    for i, entry in enumerate(finger_table):
        print(f"  [{i}] start={entry.start} -> node={entry.node.node_id} ({entry.node.host}:{entry.node.port})")


def cmd_lookup(host: str, port: int, key: str):
    """查询某个 key 会被哪个 node 负责。"""
    key_id = consistent_hash(key)
    client = get_client(host, port)
    target = client.find_successor(key_id)
    client._iprot.trans.close()
    print(f"key={key!r} (id={key_id}) -> node={target.node_id} ({target.host}:{target.port})")


def main():
    parser = argparse.ArgumentParser(description="Chord DHT Client")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("put", help="Store a key-value pair")
    p.add_argument("key", help="Key")
    p.add_argument("value", help="Value")

    g = sub.add_parser("get", help="Retrieve a value by key")
    g.add_argument("key", help="Key")

    sub.add_parser("status", help="Show node info and finger table")

    l = sub.add_parser("lookup", help="Show which node is responsible for a key")
    l.add_argument("key", help="Key")

    parser.add_argument("--node", required=True, help="Node address, e.g. 127.0.0.1:6001")
    args = parser.parse_args()

    host, port = parse_node(args.node)

    if args.cmd == "put":
        cmd_put(host, port, args.key, args.value)
    elif args.cmd == "get":
        cmd_get(host, port, args.key)
    elif args.cmd == "status":
        cmd_status(host, port)
    elif args.cmd == "lookup":
        cmd_lookup(host, port, args.key)


if __name__ == "__main__":
    main()
