from typing import Protocol, runtime_checkable


@runtime_checkable
class SupportsRecvInto(Protocol):
    def recv_into(self, buffer: memoryview, nbytes: int = ...) -> int: ...


@runtime_checkable
class SupportsSendAll(Protocol):
    def sendall(self, data: bytes | bytearray | memoryview, flags: int = ..., /) -> None: ...


def recv_all(sock: SupportsRecvInto, n: int) -> bytearray:
    """
    Receives exactly `n` bytes from a socket-like object.

    This function blocks until exactly `n` bytes have been received. If the socket
    closes before the full amount is read, a ConnectionError is raised.

    The data is written directly into a pre-allocated buffer using a `memoryview`
    passed to `recv_into()` for efficient in-place writing without extra copies.

    Args:
        sock (SupportsRecvInto):
            A socket-like object that supports the `recv_into()` method.

        n (int):
            The number of bytes to receive.

    Returns:
        bytearray:
            A bytearray containing exactly `n` bytes received from the socket.

    Raises:
        ConnectionError:
            If the socket is closed before all `n` bytes are received.
    """
    buf = bytearray(n)
    view = memoryview(buf)
    total_recv = 0
    while total_recv < n:
        if (num_recv := sock.recv_into(view[total_recv:], n - total_recv)) == 0:
            raise ConnectionError("Socket connection closed")
        total_recv += num_recv
    return buf
