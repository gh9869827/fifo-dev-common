import pytest
from fifo_dev_common.socket.socket_utils import recv_all

class DummySocket:
    """A dummy socket for testing recv_all."""
    def __init__(self, data_chunks: list[bytes]):
        self._chunks = [memoryview(chunk) for chunk in data_chunks]
        self._closed = False

    def recv_into(self, buffer: memoryview, nbytes: int = 0):
        if not self._chunks:
            return 0  # Simulate closed socket
        chunk = self._chunks.pop(0)
        to_copy = min(len(chunk), nbytes, len(buffer))
        buffer[:to_copy] = chunk[:to_copy]
        if to_copy < len(chunk):
            # Put back the rest for next call
            self._chunks.insert(0, chunk[to_copy:])
        return to_copy

def test_recv_all_exact_bytes():
    data = b"hello world"
    sock = DummySocket([data])
    result = recv_all(sock, len(data))
    assert result == data

def test_recv_all_multiple_chunks():
    data = [b"foo", b"bar", b"baz"]
    expected = b"foobarbaz"
    sock = DummySocket(data)
    result = recv_all(sock, len(expected))
    assert result == expected

def test_recv_all_partial_chunks():
    # Data comes in partial chunks, not aligned to requested size
    data = [b"a", b"bcd", b"efg", b"hij"]
    expected = b"abcdefghij"
    sock = DummySocket(data)
    result = recv_all(sock, 10)
    assert result == expected

def test_recv_all_socket_closed_early():
    data = [b"abc", b"de"]
    sock = DummySocket(data)  # Only 5 bytes available
    with pytest.raises(ConnectionError):
        recv_all(sock, 10)

def test_recv_all_zero_bytes():
    sock = DummySocket([])
    result = recv_all(sock, 0)
    assert result == b""

def test_recv_all_socket_closed_immediately():
    sock = DummySocket([])
    with pytest.raises(ConnectionError):
        recv_all(sock, 1)
