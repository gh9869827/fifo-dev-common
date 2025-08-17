"""
Asyncio-based TCP transport for `FifoEvent` objects, with optional TLS 1.3 encryption.

> **Security Note**
> - Certificate revocation (OCSP/CRL) is out of scope and **not** implemented.
> - Not audited or penetration-tested; not for safety-critical use.
> - Intended for personal use on an internal network with **non-sensitive data** (e.g., hobby
>   robotics / experimentation).
> - TLS 1.3 and optional mTLS are supported, but there is **no application-layer authentication**.
> - Do **not** expose this service to the public internet or untrusted networks.
"""

import asyncio
import threading
import contextlib
import ssl
import hashlib
from typing import Optional, Sequence, cast
from fifo_dev_common.event.fifo_event import FifoEvent, FifoEventShutdown
from fifo_dev_common.event.fifo_event_protocols import SupportsFifoEventPut
from fifo_dev_common.logging.logger import get_logger

logger = get_logger(__name__)


# --------------------------
# TLS helpers (TLS 1.3 only)
# --------------------------

def make_server_tls_context(certfile: str,
                            keyfile: str,
                            *,
                            cafile: Optional[str] = None,
                            require_client_cert: bool = False) -> ssl.SSLContext:
    """
    Build a TLS 1.3-only server SSL context.

    Args:
        certfile (str):
            Path to the PEM-encoded server certificate.

        keyfile (str):
            Path to the PEM-encoded private key for the server certificate.

        cafile (str | None, optional):
            Path to a CA bundle used to validate client certificates when
            `require_client_cert=True`. Ignored otherwise.

        require_client_cert (bool, optional):
            If True, require a client certificate (mutual TLS). Defaults to False.

    Returns:
        ssl.SSLContext:
            Configured SSL context restricted to TLS 1.3.

    Raises:
        FileNotFoundError: If provided certificate/key files cannot be read.
        ssl.SSLError: If certificates are invalid or context initialization fails.
    """
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_3
    ctx.maximum_version = ssl.TLSVersion.TLSv1_3  # enforce TLS 1.3 only
    ctx.load_cert_chain(certfile=certfile, keyfile=keyfile)

    if require_client_cert:
        if not cafile:
            raise ValueError("cafile is required when require_client_cert=True")
        ctx.verify_mode = ssl.CERT_REQUIRED
        ctx.load_verify_locations(cafile=cafile)
    else:
        ctx.verify_mode = ssl.CERT_NONE

    # TLS compression and renegotiation are disabled by default on modern OpenSSL.
    return ctx


def make_client_tls_context(cafile: str,
                            *,
                            certfile: Optional[str] = None,
                            keyfile: Optional[str] = None,
                            check_hostname: bool = True) -> ssl.SSLContext:
    """
    Build a TLS 1.3-only client SSL context.

    Args:
        cafile (str):
            Path to a CA bundle used to validate the server certificate.

        certfile (str | None, optional):
            Path to a PEM-encoded client certificate (for mutual TLS). Defaults to None.

        keyfile (str | None, optional):
            Path to a PEM-encoded private key for the client certificate. Defaults to None.

        check_hostname (bool, optional):
            If True, validate the server's certificate hostname (recommended). Defaults to True.

    Returns:
        ssl.SSLContext:
            Configured SSL context restricted to TLS 1.3.

    Raises:
        FileNotFoundError: If provided certificate/key/CA files cannot be read.
        ssl.SSLError: If certificates are invalid or context initialization fails.
    """
    ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=cafile)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_3
    ctx.maximum_version = ssl.TLSVersion.TLSv1_3  # enforce TLS 1.3 only
    ctx.check_hostname = check_hostname

    if certfile and keyfile:
        ctx.load_cert_chain(certfile=certfile, keyfile=keyfile)

    return ctx


# ---------- helpers (close/wait bounded, no broad except) ----------

async def _bounded_close_and_wait_closed_writer(writer: asyncio.StreamWriter,
                                                *,
                                                timeout: float,
                                                label: str) -> None:
    """
    Close an asyncio StreamWriter and wait for it to finish closing with timeout protection.

    This function performs a graceful shutdown of a StreamWriter by calling close() and then
    waiting for wait_closed() to complete. It includes timeout protection and logs only
    benign connection errors that commonly occur during shutdown.

    Args:
        writer (asyncio.StreamWriter):
            The asyncio StreamWriter to close and wait for.

        timeout (float):
            Maximum time in seconds to wait for the writer to close completely.

        label (str):
            A label for logging purposes to identify which connection is being closed.

    Raises:
        No exceptions are raised; all errors are logged and the function proceeds.
    """
    peer: tuple[str, int] | None = writer.get_extra_info("peername")
    writer.close()
    try:
        await asyncio.wait_for(writer.wait_closed(), timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(
            "[%s] wait_closed() timed out after %.1fs (peer=%s); proceeding.", label, timeout, peer
        )
    except (BrokenPipeError, ConnectionResetError) as e:
        logger.debug(
            "[%s] wait_closed() benign socket error: %r (peer=%s); proceeding.", label, e, peer
        )


async def _bounded_wait_closed_server(server: asyncio.base_events.Server,
                                      *,
                                      timeout: float) -> None:
    """
    Wait for an asyncio Server to finish closing with timeout protection.

    This function waits for an asyncio Server's wait_closed() method to complete
    with a timeout. It handles common runtime errors that occur during shutdown
    gracefully by logging them at appropriate levels.

    Args:
        server (asyncio.base_events.Server):
            The asyncio Server to wait for closing.

        timeout (float):
            Maximum time in seconds to wait for the server to close completely.

    Raises:
        No exceptions are raised; all errors are logged and the function proceeds.
    """
    try:
        await asyncio.wait_for(server.wait_closed(), timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(
            "[server] server.wait_closed() timed out after %.1fs; proceeding.", timeout
        )
    except RuntimeError as e:
        # e.g., loop shutting down or already-closed race; not harmful during teardown
        logger.debug(
            "[server] server.wait_closed() runtime note: %r; proceeding.", e
        )

def _get_cert_dict_key(cert_dict: dict[str,   str
                                            | tuple[tuple[tuple[str, str], ...], ...]
                                            | tuple[tuple[str, str], ...]],
                       key: str,
                       values: tuple[str, ...]) -> str | None:
    """
    Extract a specific field value from an X.509 certificate dictionary.

    This function searches through the complex nested structure of certificate fields
    (like subject or subjectAltName) to find a value matching any of the specified
    field names. It handles various formatting inconsistencies across different
    SSL implementations.

    Args:
        cert_dict (dict):
            Certificate dictionary as returned by ssl.SSLSocket.getpeercert().
            Contains parsed certificate fields with potentially complex nested structures.

        key (str):
            The top-level key to search in the certificate dictionary
            (e.g., "subject", "subjectAltName", "issuer").

        values (tuple[str, ...]):
            Tuple of field names to search for within the specified key.
            Multiple names allow matching different formats (e.g., ("commonName", "CN")).

    Returns:
        str | None:
            The first matching field value found, or None if no match is found.

    Example:
        # Extract common name from subject
        cn = _get_cert_dict_key(cert_dict, "subject", ("commonName", "CN"))
        
        # Extract DNS name from subject alternative names
        dns = _get_cert_dict_key(cert_dict, "subjectAltName", ("DNS",))
    """
    rdns = cert_dict.get(key, ())
    values_lower = tuple(v.lower() for v in values)
    pairs: Sequence[tuple[str, str]]
    for rdn in rdns:
        # Normalize RDN shape so linters don't complain and odd shapes won't break
        if isinstance(rdn, tuple):
            if len(rdn) == 2 and isinstance(rdn[0], str):
                pairs = (cast(tuple[str, str], rdn),)  # e.g., ('commonName', 'example.com')
            else:
                pairs = cast(Sequence[tuple[str, str]], rdn) # e.g., (('commonName','example.com'),)
        else: # isinstance(rdn, str):
            # Some platforms may hand a single 'CN=example.com' or 'commonName=example.com' string
            kv = rdn.split('=', 1)
            if len(kv) == 2 and kv[0].strip().lower() in values_lower:
                return kv[1].strip()

            continue  # unrecognized string shape; skip

        for k, v in pairs:
            if k.lower() in values_lower:
                return v

    return None

def _log_tls_peer(writer: asyncio.StreamWriter, role: str) -> None:
    """
    Log a brief, non-sensitive summary of the TLS peer certificate (if TLS is in use).

    This function extracts basic certificate information from an active TLS connection
    and logs it for debugging and monitoring purposes. It avoids logging sensitive
    information while providing useful details for certificate verification and troubleshooting.

    Args:
        writer (asyncio.StreamWriter):
            The asyncio stream writer for the TLS connection. Used to access the underlying
            SSL object and certificate information.

        role (str):
            A label identifying the role in the connection (e.g., "client", "server")
            for logging purposes.

    Note:
        This function silently returns if the connection is not using TLS or if
        certificate information cannot be retrieved. No exceptions are raised.
    """
    sslobj: ssl.SSLObject | None = writer.get_extra_info("ssl_object")
    if not sslobj:
        return  # plaintext connection
    try:
        cert_dict = sslobj.getpeercert()                  # parsed fields (may be partial)
        der = sslobj.getpeercert(binary_form=True)        # for fingerprint
    except ssl.SSLError:
        return

    if cert_dict is None:
        return

    # Extract CN from subject (tuple of RDNs)
    cn = (   _get_cert_dict_key(cert_dict, "subject", ("commonName", "CN"))
          or _get_cert_dict_key(cert_dict, "subjectAltName", ("DNS",))
    )

    not_after = cert_dict.get("notAfter")
    issuer = cert_dict.get("issuer")
    fp = hashlib.sha256(der).hexdigest() if der else None

    logger.info("[%s] TLS peer: CN=%s, fpSHA256=%s, notAfter=%s, issuer=%s",
                role, cn or "?", fp or "?", not_after or "?", issuer or "?")


# -----------------------------------
# Base mixin for common functionality
# -----------------------------------


class _FifoEventQueueNetworkAsyncMixin:
    """
    Mixin class providing common functionality for both client and server network classes.
    
    This mixin contains shared methods for sending events, receiving events, and stopping
    the network connection. It assumes the inheriting class has _reader, _writer, _out_queue,
    and _task attributes.
    """
    _reader: asyncio.StreamReader
    _writer: asyncio.StreamWriter
    _out_queue: asyncio.PriorityQueue[FifoEvent]
    _task: asyncio.Task[None]

    async def stop(self):
        """
        Signal the connection to stop by sending a shutdown event.

        This method sends a FifoEventShutdown event which will cause the background
        network-to-queue task to terminate after processing the shutdown signal.
        Call join() after this method to wait for the actual shutdown to complete.

        Raises:
            ConnectionError: If the connection is already closed or there's a network error.
        """
        await self.send(FifoEventShutdown())

    async def _network_to_queue(self):
        """
        Background task that continuously receives events from the network and queues them.

        This method runs in a background asyncio task and continuously deserializes events
        from the network stream, placing them into the output queue. It terminates when
        a FifoEventShutdown event is received.

        Raises:
            ConnectionError: If the network connection is lost during operation.
            ValueError: If an invalid event is received and cannot be deserialized.
        """
        while True:
            try:
                event = await FifoEvent.deserialize_from_socket_async(self._reader)
            except (RuntimeError, TypeError, ValueError) as e:
                role = "client" if "Client" in type(self).__name__ else "server"
                logger.error("[%s] Error receiving event: %r", role, type(e))
                continue

            await self._out_queue.put(event)
            if isinstance(event, FifoEventShutdown):
                break

    async def send(self, event: FifoEvent):
        """
        Send a FifoEvent over the network connection.

        This method immediately serializes and sends the provided event over the
        network connection. The event is automatically flushed to ensure delivery.

        Args:
            event (FifoEvent):
                The event to send over the network connection.

        Raises:
            ConnectionError: If the connection is closed or there's a network error.
        """
        await event.serialize_to_socket_async(self._writer)  # drain handled by serializer

    async def put(self, item: FifoEvent) -> None:
        """
        Queue-like alias for send.

        Provides compatibility with asyncio.PriorityQueue.put so that network
        clients and servers can be used interchangeably with asyncio queues
        expecting a put method.

        Args:
            item (FifoEvent):
                Event to send over the network connection.
        """

        await self.send(item)


class FifoEventQueueNetworkAsyncClient(_FifoEventQueueNetworkAsyncMixin, SupportsFifoEventPut):
    """
    Asyncio-based network client for sending and receiving FifoEvent objects over TCP.

    This class establishes a TCP connection to a server and provides bidirectional event
    communication. Outbound events are sent immediately when send() is called, while
    inbound events are automatically received in a background task and queued in a
    priority queue for consumption by the application.

    The client handles connection management, serialization/deserialization of events,
    and graceful shutdown procedures. It is designed to work with asyncio and provides
    proper resource cleanup through stop() and join() methods.

    Optional TLS:
        If an `ssl_ctx` is provided to `connect()`, the connection is encrypted with TLS 1.3.

    Attributes:
        _reader (asyncio.StreamReader):
            The asyncio stream reader for receiving data from the network connection.

        _writer (asyncio.StreamWriter):
            The asyncio stream writer for sending data over the network connection.

        _out_queue (asyncio.PriorityQueue[FifoEvent]):
            Priority queue containing received events from the network, ready for
            application consumption.

        _thread (threading.Thread):
            Currently unused thread attribute (legacy).

        _task (asyncio.Task[None]):
            Background asyncio task that continuously receives events from the network
            and places them in the output queue.
    """
    _reader: asyncio.StreamReader
    _writer: asyncio.StreamWriter
    _out_queue: asyncio.PriorityQueue[FifoEvent]
    _thread: threading.Thread
    _task: asyncio.Task[None]

    def __init__(self,
                 reader: asyncio.StreamReader,
                 writer: asyncio.StreamWriter,
                 out_queue: asyncio.PriorityQueue[FifoEvent] | None):
        """
        Initialize a FifoEventQueueNetworkAsyncClient with existing connection streams.

        Args:
            reader (asyncio.StreamReader):
                The asyncio stream reader for receiving data from the network connection.

            writer (asyncio.StreamWriter):
                The asyncio stream writer for sending data over the network connection.

            out_queue (asyncio.PriorityQueue[FifoEvent] | None):
                Optional priority queue for received events. If None, a new queue is created.
        """
        self._reader = reader
        self._writer = writer
        self._out_queue = out_queue or asyncio.PriorityQueue()
        self._task = asyncio.create_task(self._network_to_queue())

    @classmethod
    async def connect(cls,
                      host: str,
                      port: int,
                      out_queue: asyncio.PriorityQueue[FifoEvent] | None = None,
                      *,
                      ssl_ctx: ssl.SSLContext | None = None,
                      server_hostname: str | None = None,
                      connect_timeout: float | None = None,
                      require_tls: bool = False):
        """
        Establish a connection to a remote server and create a client instance.

        This factory method creates an asyncio TCP connection to the specified host and port,
        then returns a FifoEventQueueNetworkAsyncClient instance that uses this connection.

        If `ssl_ctx` is provided, the connection is encrypted with TLS 1.3.

        Args:
            host (str):
                The hostname or IP address to connect to.

            port (int):
                The port number to connect to.

            out_queue (asyncio.PriorityQueue[FifoEvent] | None, optional):
                Optional priority queue for received events. If None, a new queue is created.

            ssl_ctx (ssl.SSLContext | None, optional):
                SSL context configured for TLS 1.3. If provided, enables TLS.

            server_hostname (str | None, optional):
                The expected server hostname for certificate verification (SNI). If not provided,
                `host` is used. Ignored when `ssl_ctx` is None.

            connect_timeout (float | None, optional):
                Maximum time in seconds to wait for the TCP (and TLS, if enabled) connection to be
                established. If None, no timeout is applied and the call may block indefinitely on
                network or handshake issues.

            require_tls (bool, optional):
                If True, require TLS (ssl_ctx must be provided). Defaults to False.

        Returns:
            FifoEventQueueNetworkAsyncClient:
                A new client instance connected to the specified server.

        Raises:
            ConnectionError: If the connection to the server fails.
            OSError: If there are network-related issues during connection.
            ssl.SSLError: If TLS handshake or verification fails (when `ssl_ctx` is used).
            ValueError: If require_tls=True but ssl_ctx is None.
        """
        # Add explicit security policy
        if require_tls and ssl_ctx is None:
            raise ValueError("TLS is required but no SSL context provided")

        if ssl_ctx is not None:
            logger.warning("[client] Establishing TLS 1.3 encrypted connection to %s:%s",
                           host, port)
        else:
            logger.warning("[client] Establishing NOT encrypted, "
                           "NOT authenticated connection to %s:%s",
                           host, port)

        async def _do_open():
            return await asyncio.open_connection(
                host, port,
                ssl=ssl_ctx,
                server_hostname=(server_hostname or host) if ssl_ctx else None
            )

        try:
            if connect_timeout:
                reader, writer = await asyncio.wait_for(_do_open(), timeout=connect_timeout)
            else:
                reader, writer = await _do_open()
        except ssl.SSLCertVerificationError as e:
            logger.error("[client] TLS certificate verification failed: %s", e)
            raise
        except ssl.SSLError as e:
            logger.error("[client] TLS handshake failed: %s", e)
            raise
        except (ConnectionRefusedError, TimeoutError, OSError) as e:
            logger.error("[client] TCP connection failed: %s", e)
            raise

        if ssl_ctx is not None:
            _log_tls_peer(writer, "client")

        return cls(reader, writer, out_queue)

    async def join(self, timeout: float = 5.0):
        """
        Wait for the client to finish shutting down and clean up resources.

        This method waits for the background network-to-queue task to finish, then
        properly closes the network connection. It includes timeout protection to
        prevent hanging during shutdown.

        Args:
            timeout (float, optional):
                Maximum time in seconds to wait for the background task to finish.
                Defaults to 5.0 seconds.

        Raises:
            No exceptions are raised; timeouts and errors are logged and handled gracefully.
        """
        # Wait for `_network_to_queue` task to finish, cancel on timeout
        try:
            await asyncio.wait_for(self._task, timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("[client] join() timed out after %.1fs; cancelling background task.",
                           timeout)
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

        # Close writer with a bounded wait
        await _bounded_close_and_wait_closed_writer(self._writer, timeout=3.0, label="client")


class FifoEventQueueNetworkAsyncServer(_FifoEventQueueNetworkAsyncMixin, SupportsFifoEventPut):
    """
    Asyncio-based network server for sending and receiving FifoEvent objects over TCP.

    This class creates a TCP server that accepts exactly one client connection and provides
    bidirectional event communication. Outbound events are sent immediately when send() is called,
    while inbound events are automatically received in a background task and queued in a
    priority queue for consumption by the application.

    The server enforces a "one client at a time" policy, rejecting additional connection attempts
    after the first client connects. It handles connection management, serialization/deserialization
    of events, and graceful shutdown procedures.

    Optional TLS:
        If an `ssl_ctx` is provided to `accept()`, the listening socket and client connection
        are encrypted with TLS 1.3.

    Attributes:
        _reader (asyncio.StreamReader):
            The asyncio stream reader for receiving data from the connected client.

        _writer (asyncio.StreamWriter):
            The asyncio stream writer for sending data to the connected client.

        _server (asyncio.Server):
            The asyncio server instance that manages the listening socket.

        _out_queue (asyncio.PriorityQueue[FifoEvent]):
            Priority queue containing received events from the client, ready for
            application consumption.

        _task (asyncio.Task[None]):
            Background asyncio task that continuously receives events from the client
            and places them in the output queue.
    """
    _reader: asyncio.StreamReader
    _writer: asyncio.StreamWriter
    _server: asyncio.Server
    _out_queue: asyncio.PriorityQueue[FifoEvent]
    _task: asyncio.Task[None]

    def __init__(self,
                 reader: asyncio.StreamReader,
                 writer: asyncio.StreamWriter,
                 server: asyncio.Server,
                 out_queue: asyncio.PriorityQueue[FifoEvent] | None):
        """
        Initialize a FifoEventQueueNetworkAsyncServer with existing connection and server.

        Args:
            reader (asyncio.StreamReader):
                The asyncio stream reader for receiving data from the connected client.

            writer (asyncio.StreamWriter):
                The asyncio stream writer for sending data to the connected client.

            server (asyncio.Server):
                The asyncio server instance that manages the listening socket.

            out_queue (asyncio.PriorityQueue[FifoEvent] | None):
                Optional priority queue for received events. If None, a new queue is created.
        """
        self._reader = reader
        self._writer = writer
        self._server = server
        self._out_queue = out_queue or asyncio.PriorityQueue()
        self._task = asyncio.create_task(self._network_to_queue())

    @classmethod
    async def accept(cls,
                     host: str,
                     port: int,
                     out_queue: asyncio.PriorityQueue[FifoEvent] | None = None,
                     *,
                     ssl_ctx: ssl.SSLContext | None = None,
                     require_tls: bool = False):
        """
        Create a server that accepts exactly one client connection.

        This factory method creates an asyncio TCP server on the specified host and port,
        waits for exactly one client to connect, then stops listening for additional connections.

        If `ssl_ctx` is provided, the listener and accepted connection are encrypted
        with TLS 1.3.

        Args:
            host (str):
                The hostname or IP address to bind the server to.

            port (int):
                The port number to bind the server to.

            out_queue (asyncio.PriorityQueue[FifoEvent] | None, optional):
                Optional priority queue for received events. If None, a new queue is created.

            ssl_ctx (ssl.SSLContext | None, optional):
                SSL context configured for TLS 1.3. If provided, enables TLS for the server.

            require_tls (bool, optional):
                If True, require TLS (ssl_ctx must be provided). Defaults to False.

        Returns:
            FifoEventQueueNetworkAsyncServer:
                A new server instance with one connected client.

        Raises:
            OSError: If there are network-related issues during server creation or binding.
            ssl.SSLError: If TLS setup fails (when `ssl_ctx` is used).
            ValueError: If require_tls=True but ssl_ctx is None.
        """
        # Add explicit security policy
        if require_tls and ssl_ctx is None:
            raise ValueError("TLS is required but no SSL context provided")

        if ssl_ctx is not None:
            logger.warning("[server] Accepting TLS 1.3 encrypted connection on %s:%s", host, port)
        else:
            logger.warning("[server] Accepting NOT encrypted, "
                           "NOT authenticated connection on %s:%s",
                           host, port)

        loop = asyncio.get_running_loop()
        conn_future: asyncio.Future[tuple[asyncio.StreamReader, asyncio.StreamWriter]]
        conn_future = loop.create_future()

        async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
            if not conn_future.done():
                if ssl_ctx is not None:
                    logger.warning("[server] Opening TLS 1.3 encrypted connection on %s:%s",
                                   host, port)
                    _log_tls_peer(writer, "server")
                else:
                    logger.warning("[server] Opening NOT encrypted, "
                                   "NOT authenticated connection on %s:%s",
                                   host, port)
                conn_future.set_result((reader, writer))
            else:
                # Explicitly reject extra clients
                logger.warning("[server] rejecting extra client on %s:%s", host, port)
                await _bounded_close_and_wait_closed_writer(writer, timeout=3.0, label="server")

        server = await asyncio.start_server(handle_client, host, port, ssl=ssl_ctx)

        # Wait for the first connection to arrive
        reader, writer = await conn_future

        # Enforce "one client at a time": stop listening once connected (defer wait to join)
        server.close()
        return cls(reader, writer, server, out_queue)

    async def join(self, timeout: float = 5.0):
        """
        Wait for the server to finish shutting down and clean up resources.

        This method waits for the background network-to-queue task to finish, then
        properly closes the network connection and server. It includes timeout protection
        to prevent hanging during shutdown.

        Args:
            timeout (float, optional):
                Maximum time in seconds to wait for the background task to finish.
                Defaults to 5.0 seconds.

        Raises:
            No exceptions are raised; timeouts and errors are logged and handled gracefully.
        """
        # Wait for `_network_to_queue` task to finish, cancel on timeout
        try:
            await asyncio.wait_for(self._task, timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("[server] join() timed out after %.1fs; cancelling background task.",
                           timeout)
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

        # Close connected writer
        await _bounded_close_and_wait_closed_writer(self._writer, timeout=3.0, label="server")

        # Listener was closed in accept(); now wait for it to finish closing
        await _bounded_wait_closed_server(self._server, timeout=3.0)
