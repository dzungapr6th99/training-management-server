"""Tiny length-prefixed pickle protocol shared by the policy server and the Isaac Sim client.

Stdlib + numpy only, so it runs unchanged in both the SmolVLA conda env and Isaac Sim's bundled python.
Arrays are sent as (dtype, shape, bytes) triples so numpy versions on each side don't have to match.
"""

import pickle
import socket
import struct

import numpy as np

HOST = "127.0.0.1"
PORT = 5555


def _pack(obj):
    if isinstance(obj, np.ndarray):
        return {"__nd__": (str(obj.dtype), obj.shape, np.ascontiguousarray(obj).tobytes())}
    if isinstance(obj, dict):
        return {k: _pack(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_pack(v) for v in obj)
    return obj


def _unpack(obj):
    if isinstance(obj, dict):
        if "__nd__" in obj:
            dtype, shape, buf = obj["__nd__"]
            return np.frombuffer(buf, dtype=dtype).reshape(shape).copy()
        return {k: _unpack(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_unpack(v) for v in obj)
    return obj


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed")
        buf.extend(chunk)
    return bytes(buf)


def send_msg(sock: socket.socket, obj) -> None:
    data = pickle.dumps(_pack(obj), protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack("!Q", len(data)) + data)


def recv_msg(sock: socket.socket):
    (n,) = struct.unpack("!Q", _recv_exact(sock, 8))
    return _unpack(pickle.loads(_recv_exact(sock, n)))
