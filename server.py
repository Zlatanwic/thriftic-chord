import argparse
import logging

import thriftpy2
from thriftpy2.rpc import make_server

from chord_node import ChordNode

chord_thrift = thriftpy2.load("chord.thrift", module_name="chord_server_thrift")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("server")


def main():
    parser = argparse.ArgumentParser(description="Start a Chord DHT node")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--join", type=str, default=None,
                        help="Address of existing node to join, e.g. 127.0.0.1:6001")
    args = parser.parse_args()

    node = ChordNode(args.host, args.port)

    if args.join:
        join_host, join_port = args.join.split(":")
        node.join(join_host, int(join_port))

    node.start_background_tasks()

    logger.info(f"Starting Chord node {node.node_id} on {args.host}:{args.port}")
    server = make_server(chord_thrift.ChordService, node, args.host, args.port)
    server.serve()


if __name__ == "__main__":
    main()
