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
import contextlib
import ssl
import hashlib
import itertools
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, cast

from fifo_dev_common.event.fifo_event import FifoEvent, FifoEventShutdown
from fifo_dev_common.event.fifo_event_protocols import SendStatus, SupportsFifoEventPut
from fifo_dev_common.event.fifo_event_queue_connector import (
    FifoEventQueueConnectorAsyncClient,
    FifoEventQueueConnectorAsyncMixin,
    bounded_close_and_wait_closed_writer,
)
from fifo_dev_common.event.fifo_event_queue_connector_handler import (
    FifoEventQueueConnectorAsyncHandlerBase,
)
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

class FifoEventQueueNetworkAsyncClient(FifoEventQueueConnectorAsyncClient):
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

        _task (asyncio.Task[None]):
            Background asyncio task that continuously receives events from the network
            and places them in the output queue.

        _handler (FifoEventQueueConnectorAsyncHandlerBase | None):
            Handler used to intercept and process events. If None, events are
            queued and sent directly without additional processing. When provided,
            the handler can:
            - process incoming events before they are enqueued,
            - process outgoing events before they are sent, and
            - observe sent events after they have been successfully sent.
    """
    _reader: asyncio.StreamReader
    _writer: asyncio.StreamWriter
    _out_queue: asyncio.PriorityQueue[FifoEvent]
    _task: asyncio.Task[None]
    _handler: FifoEventQueueConnectorAsyncHandlerBase | None

    @classmethod
    async def connect(cls,
                      host: str,
                      port: int,
                      out_queue: asyncio.PriorityQueue[FifoEvent] | None = None,
                      handler: FifoEventQueueConnectorAsyncHandlerBase | None = None,
                      *,
                      ssl_ctx: ssl.SSLContext | None = None,
                      server_hostname: str | None = None,
                      connect_timeout: float | None = None,
                      ensure_ssl_ctx: bool = False):
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

            handler (FifoEventQueueConnectorAsyncHandlerBase | None, optional):
                Handler used to intercept and process events. If None, events are
                queued and sent directly without additional processing. When provided,
                the handler can:
                - process incoming events before they are enqueued,
                - process outgoing events before they are sent, and
                - observe sent events after they have been successfully sent.

            ssl_ctx (ssl.SSLContext | None, optional):
                SSL context configured for TLS 1.3. If provided, enables TLS.

            server_hostname (str | None, optional):
                The expected server hostname for certificate verification (SNI). If not provided,
                `host` is used. Ignored when `ssl_ctx` is None.

            connect_timeout (float | None, optional):
                Maximum time in seconds to wait for the TCP (and TLS, if enabled) connection to be
                established. If None, no timeout is applied and the call may block indefinitely on
                network or handshake issues.

            ensure_ssl_ctx (bool, optional):
                If True, ensures an SSL context is provided and raises a ValueError
                if `ssl_ctx` is None. If False, TLS is used when `ssl_ctx` is
                supplied; otherwise the connection is plaintext. Defaults to
                False. Allows implementing an explicit policy verifying that
                `ssl_ctx` is not None.

        Returns:
            FifoEventQueueNetworkAsyncClient:
                A new client instance connected to the specified server.

        Raises:
            ConnectionError: If the connection to the server fails.
            OSError: If there are network-related issues during connection.
            ssl.SSLError: If TLS handshake or verification fails (when `ssl_ctx` is used).
            ValueError: If ensure_ssl_ctx=True but ssl_ctx is None.
        """
        # Add explicit security policy
        if ensure_ssl_ctx and ssl_ctx is None:
            raise ValueError("SSL context is required but none was provided")

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

        return cls(reader, writer, out_queue, handler)


class FifoEventQueueNetworkAsyncServer(
    FifoEventQueueConnectorAsyncMixin,
    SupportsFifoEventPut,
):
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

        _handler (FifoEventQueueConnectorAsyncHandlerBase | None):
            Handler used to intercept and process events. If None, events are
            queued and sent directly without additional processing. When provided,
            the handler can:
            - process incoming events before they are enqueued,
            - process outgoing events before they are sent, and
            - observe sent events after they have been successfully sent.
    """
    _reader: asyncio.StreamReader
    _writer: asyncio.StreamWriter
    _server: asyncio.Server
    _out_queue: asyncio.PriorityQueue[FifoEvent]
    _task: asyncio.Task[None]
    _handler: FifoEventQueueConnectorAsyncHandlerBase | None

    def __init__(self,
                 reader: asyncio.StreamReader,
                 writer: asyncio.StreamWriter,
                 server: asyncio.Server,
                 out_queue: asyncio.PriorityQueue[FifoEvent] | None,
                 handler: FifoEventQueueConnectorAsyncHandlerBase | None = None):
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

            handler (FifoEventQueueConnectorAsyncHandlerBase | None, optional):
                Handler used to intercept and process events. If None, events are
                queued and sent directly without additional processing. When provided,
                the handler can:
                - process incoming events before they are enqueued,
                - process outgoing events before they are sent, and
                - observe sent events after they have been successfully sent.
        """
        self._reader = reader
        self._writer = writer
        self._server = server
        self._out_queue = out_queue or asyncio.PriorityQueue()
        self._handler = handler
        self._task = asyncio.create_task(self._network_to_queue())

    @classmethod
    async def accept(cls,
                     host: str,
                     port: int,
                     out_queue: asyncio.PriorityQueue[FifoEvent] | None = None,
                     handler: FifoEventQueueConnectorAsyncHandlerBase | None = None,
                     *,
                     ssl_ctx: ssl.SSLContext | None = None,
                     ensure_ssl_ctx: bool = False):
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

            handler (FifoEventQueueConnectorAsyncHandlerBase | None, optional):
                Handler used to intercept and process events. If None, events are
                queued and sent directly without additional processing. When provided,
                the handler can:
                - process incoming events before they are enqueued,
                - process outgoing events before they are sent, and
                - observe sent events after they have been successfully sent.

            ssl_ctx (ssl.SSLContext | None, optional):
                SSL context configured for TLS 1.3. If provided, enables TLS for the server.

            ensure_ssl_ctx (bool, optional):
                If True, ensures an SSL context is provided and raises a ValueError
                if `ssl_ctx` is None. If False, TLS is used when `ssl_ctx` is
                supplied; otherwise the connection is plaintext. Defaults to
                False. Allows implementing an explicit policy verifying that
                `ssl_ctx` is not None.

        Returns:
            FifoEventQueueNetworkAsyncServer:
                A new server instance with one connected client.

        Raises:
            OSError: If there are network-related issues during server creation or binding.
            ssl.SSLError: If TLS setup fails (when `ssl_ctx` is used).
            ValueError: If ensure_ssl_ctx=True but ssl_ctx is None.
        """
        # Add explicit security policy
        if ensure_ssl_ctx and ssl_ctx is None:
            raise ValueError("SSL context is required but none was provided")

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
                await bounded_close_and_wait_closed_writer(writer, timeout=3.0, label="server")

        server = await asyncio.start_server(handle_client, host, port, ssl=ssl_ctx)

        # Wait for the first connection to arrive
        reader, writer = await conn_future

        # Enforce "one client at a time": stop listening once connected (defer wait to join)
        server.close()
        return cls(reader, writer, server, out_queue, handler)

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
        await bounded_close_and_wait_closed_writer(self._writer, timeout=3.0, label="server")

        # Listener was closed in accept(); now wait for it to finish closing
        await _bounded_wait_closed_server(self._server, timeout=3.0)


@dataclass(slots=True)
class _HubClientState:
    """Internal bookkeeping for hub client connections."""

    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    context: "FifoEventQueueNetworkAsyncHubClientContext"
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    task: asyncio.Task[None] | None = field(init=False, default=None)


class FifoEventQueueNetworkAsyncHubClientContext:
    """
    Runtime context handed to hub callbacks for each connected client.

    The context exposes helpers to reply directly to the originating client,
    broadcast events to every active client, and maintain per-client state via
    the mutable `data` dictionary. Instances are created and managed by the
    hub; applications should not instantiate them manually.

    Attributes:
        client_id (int):
            Monotonic identifier assigned by the hub when the client connects.

        data (dict[str, Any]):
            Mutable dictionary for storing arbitrary per-client state across
            multiple callbacks. Cleared only when the client disconnects.
    """

    def __init__(self,
                 hub: "FifoEventQueueNetworkAsyncHub",
                 client_id: int,
                 reader: asyncio.StreamReader,
                 writer: asyncio.StreamWriter):
        self._hub = hub
        self._reader = reader
        self._writer = writer
        self._closed = False
        self.client_id = client_id
        self.data: dict[str, Any] = {}

    @property
    def peername(self) -> tuple[str, int] | None:
        """Return the TCP peername of the connected client, if available."""

        return cast("tuple[str, int] | None", self._writer.get_extra_info("peername"))

    @property
    def is_active(self) -> bool:
        """Whether the client connection is still active."""

        return not self._closed

    async def send(self, event: FifoEvent) -> SendStatus:
        """
        Send an event back to the originating client using the hub transport.

        Args:
            event (FifoEvent):
                Event instance to serialize and deliver to this client.

        Returns:
            SendStatus:
                `SendStatus.SENT` when the event is written to the stream, or
                `SendStatus.SUPPRESSED` if a handler suppressed the event
                before it reached the transport.

        Raises:
            ConnectionError:
                Raised when the client connection is already closed or closing.
        """

        if self._closed:
            raise ConnectionError("Client connection is closed")
        return await self._hub._send_to_client(self.client_id, event)

    async def broadcast(self,
                        event: FifoEvent,
                        *,
                        include_self: bool = True) -> dict[int, SendStatus]:
        """
        Broadcast an event to every currently connected client.

        Args:
            event (FifoEvent):
                Event instance to deliver to all active clients.

            include_self (bool, optional):
                If `True` (default), also send the event back to the
                originating client. When `False` the caller is excluded from
                the broadcast.

        Returns:
            dict[int, SendStatus]:
                Mapping of client identifiers to the `SendStatus` outcome for
                each targeted client. Returned only when every send succeeds.

        Raises:
            ConnectionError:
                Raised when one or more client connections fail while sending.
                The broadcast is aborted and partial results are not returned.
        """

        if self._closed and include_self:
            raise ConnectionError("Client connection is closed")

        excluded = None if include_self else {self.client_id}
        return await self._hub._broadcast(event, excluded_client_ids=excluded)

    async def stop(self) -> None:
        """Close this client connection gracefully."""

        await self._hub._close_client(self.client_id)

    def _mark_closed(self) -> None:
        """Mark the context as closed to prevent further sends."""

        self._closed = True


HubEventCallback = Callable[
    [FifoEvent, FifoEventQueueNetworkAsyncHubClientContext],
    Awaitable[None] | None,
]


class FifoEventQueueNetworkAsyncHub:
    """
    Asyncio-based multi-client hub that dispatches events to application callbacks.

    The hub accepts any number of TCP clients (optionally protected with TLS
    1.3), deserializes incoming `FifoEvent` objects, and delivers them to the
    user-provided `event_callback` together with a
    `FifoEventQueueNetworkAsyncHubClientContext`. The context enables the
    callback to reply directly to the originating client, broadcast events to
    all currently connected clients, and maintain arbitrary per-client state.

    Outgoing events flow through the same optional handler pipeline used by the
    single-client client/server pair, allowing interception and suppression of
    traffic when desired.

    Attributes:
        _event_callback (HubEventCallback):
            Callable invoked for every incoming event. May be synchronous or
            ``async``; coroutine results are awaited before the next event from
            the same client is read.

        _handler (FifoEventQueueConnectorAsyncHandlerBase | None):
            Optional async handler used to process incoming, outgoing, and
            successfully sent events.

        _tls_enabled (bool):
            Indicates whether the listening socket is wrapped in TLS 1.3.

        _server (asyncio.Server | None):
            Asyncio server instance backing the listener. `None` until
            `listen()` completes.

        _clients (dict[int, _HubClientState]):
            Active client registry keyed by the hub-assigned identifier.

        _client_tasks (set[asyncio.Task[None]]):
            Tasks currently handling client receive loops.

        _client_ids (itertools.count):
            Monotonic counter used to assign new client identifiers.

        _stopping (bool):
            Flag indicating whether ``stop()`` has been invoked.
    """

    def __init__(self,
                 event_callback: HubEventCallback,
                 handler: FifoEventQueueConnectorAsyncHandlerBase | None,
                 *,
                 tls_enabled: bool):
        self._event_callback = event_callback
        self._handler = handler
        self._tls_enabled = tls_enabled
        self._server: asyncio.Server | None = None
        self._clients: dict[int, _HubClientState] = {}
        self._client_tasks: set[asyncio.Task[None]] = set()
        self._client_ids = itertools.count(1)
        self._stopping = False

    @classmethod
    async def listen(cls,
                     host: str,
                     port: int,
                     event_callback: HubEventCallback,
                     *,
                     handler: FifoEventQueueConnectorAsyncHandlerBase | None = None,
                     ssl_ctx: ssl.SSLContext | None = None,
                     ensure_ssl_ctx: bool = False) -> "FifoEventQueueNetworkAsyncHub":
        """
        Start listening for multiple clients and dispatch events to a callback.

        Args:
            host (str):
                Interface address passed to `asyncio.start_server`.

            port (int):
                TCP port to bind.

            event_callback (HubEventCallback):
                Callable invoked for every incoming event. May be synchronous
                or asynchronous.

            handler (FifoEventQueueConnectorAsyncHandlerBase | None, optional):
                Optional connector handler used to intercept incoming, outgoing,
                and sent events.

            ssl_ctx (ssl.SSLContext | None, optional):
                SSL context enabling TLS 1.3. When `None` the hub listens in
                cleartext.

            ensure_ssl_ctx (bool, optional):
                When `True` and `ssl_ctx` is `None`, raise `ValueError`
                instead of allowing an unencrypted listener. Defaults to `False`.

        Returns:
            FifoEventQueueNetworkAsyncHub:
                Hub instance bound to the requested endpoint.

        Raises:
            ValueError:
                Raised when `ensure_ssl_ctx` is `True` but `ssl_ctx` is not
                provided.
        """

        if ensure_ssl_ctx and ssl_ctx is None:
            raise ValueError("SSL context is required but none was provided")

        hub = cls(event_callback, handler, tls_enabled=ssl_ctx is not None)

        async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            await hub._handle_client(reader, writer)

        if ssl_ctx is not None:
            logger.warning(
                "[hub] Listening for TLS 1.3 encrypted connections on %s:%s",
                host,
                port,
            )
        else:
            logger.warning(
                "[hub] Listening for NOT encrypted, NOT authenticated connections on %s:%s",
                host,
                port,
            )

        hub._server = await asyncio.start_server(handle_client, host, port, ssl=ssl_ctx)
        return hub

    async def stop(self) -> None:
        """Signal the hub to stop accepting clients and shut down existing ones."""

        if self._stopping:
            return
        self._stopping = True

        if self._server is not None:
            self._server.close()

        clients = list(self._clients.keys())
        for client_id in clients:
            await self._close_client(client_id)

    async def join(self, timeout: float = 5.0) -> None:
        """Wait for all client tasks to finish and the server socket to close."""

        server = self._server
        if server is not None:
            await _bounded_wait_closed_server(server, timeout=timeout)

        tasks = list(self._client_tasks)
        for task in tasks:
            if task.done():
                continue
            try:
                await asyncio.wait_for(task, timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning(
                    "[hub] join() timed out waiting for client task; cancelling.",
                )
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    async def _handle_client(self,
                             reader: asyncio.StreamReader,
                             writer: asyncio.StreamWriter) -> None:
        if self._stopping:
            logger.debug("[hub] Rejecting client connection while stopping.")
            await bounded_close_and_wait_closed_writer(writer, timeout=3.0, label="hub")
            return

        client_id = next(self._client_ids)

        if self._tls_enabled:
            logger.warning(
                "[hub] Opening TLS 1.3 encrypted connection for client %s.",
                client_id,
            )
            _log_tls_peer(writer, "hub")
        else:
            logger.warning(
                "[hub] Opening NOT encrypted, NOT authenticated connection for client %s.",
                client_id,
            )

        context = FifoEventQueueNetworkAsyncHubClientContext(self, client_id, reader, writer)
        state = _HubClientState(reader=reader, writer=writer, context=context)
        self._clients[client_id] = state
        task = asyncio.create_task(self._client_loop(client_id, state))
        state.task = task
        self._client_tasks.add(task)
        task.add_done_callback(
            lambda fut, cid=client_id, st=state: asyncio.create_task(
                self._on_client_done(cid, st, fut)
            )
        )

    async def _client_loop(self,
                           client_id: int,
                           state: _HubClientState) -> None:
        reader = state.reader
        context = state.context

        try:
            while True:
                event = await FifoEvent.deserialize_from_stream_async(reader)
                event_to_dispatch = event

                if self._handler is not None:
                    try:
                        candidate = await self._handler.process_incoming_event(event)
                    except Exception:  # pylint: disable=broad-exception-caught
                        logger.error(
                            "[hub] process_incoming_event handler failed. Discarding event.",
                        )
                        candidate = None

                    if not isinstance(event, FifoEventShutdown):
                        if candidate is None:
                            continue
                        event_to_dispatch = candidate

                await self._invoke_callback(event_to_dispatch, context)

                if isinstance(event, FifoEventShutdown):
                    break

        except asyncio.CancelledError:
            raise
        except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError) as exc:
            logger.debug(
                "[hub] Client %s disconnected: %r",
                client_id,
                exc,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error(
                "[hub] Unexpected error while handling client %s: %r",
                client_id,
                exc,
            )

    async def _invoke_callback(self,
                               event: FifoEvent,
                               context: FifoEventQueueNetworkAsyncHubClientContext) -> None:
        try:
            result = self._event_callback(event, context)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("[hub] Event callback raised: %r", exc)
            return

        if asyncio.iscoroutine(result):
            try:
                await result
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.error("[hub] Event callback coroutine raised: %r", exc)

    async def _send_to_client(self, client_id: int, event: FifoEvent) -> SendStatus:
        state = self._clients.get(client_id)
        if state is None:
            raise ConnectionError("Client is not connected")

        async with state.send_lock:
            if state.writer.is_closing():
                raise ConnectionError("Client connection is closing")

            event_to_send = event
            if self._handler is not None:
                try:
                    candidate = await self._handler.process_outgoing_event(event)
                except Exception:  # pylint: disable=broad-exception-caught
                    logger.error(
                        "[hub] process_outgoing_event handler failed. Discarding event.",
                    )
                    candidate = None

                if not isinstance(event, FifoEventShutdown):
                    if candidate is None:
                        return SendStatus.SUPPRESSED
                    event_to_send = candidate

            try:
                await event_to_send.serialize_to_stream_async(state.writer)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.error(
                    "[hub] Failed to send event to client %s: %r",
                    client_id,
                    exc,
                )
                raise ConnectionError("Failed to send event") from exc

            if self._handler is not None:
                try:
                    await self._handler.process_sent_event(event_to_send)
                except Exception:  # pylint: disable=broad-exception-caught
                    logger.error("[hub] process_sent_event handler failed.")

            return SendStatus.SENT

    async def _broadcast(self,
                         event: FifoEvent,
                         *,
                         excluded_client_ids: set[int] | None = None) -> dict[int, SendStatus]:
        targets = [
            client_id
            for client_id in list(self._clients.keys())
            if excluded_client_ids is None or client_id not in excluded_client_ids
        ]

        results: dict[int, SendStatus] = {}
        failures: dict[int, ConnectionError] = {}

        for client_id in targets:
            try:
                results[client_id] = await self._send_to_client(client_id, event)
            except ConnectionError as exc:
                logger.debug(
                    "[hub] Broadcast send failed for client %s: %r",
                    client_id,
                    exc,
                )
                failures[client_id] = exc

        if failures:
            summary = ", ".join(f"{cid}: {err}" for cid, err in failures.items())
            first_error = next(iter(failures.values()))
            raise ConnectionError(
                f"Failed to broadcast event to clients: {summary}"
            ) from first_error

        return results

    async def _close_client(self,
                            client_id: int,
                            *,
                            send_shutdown: bool = True,
                            timeout: float = 3.0) -> None:
        state = self._clients.get(client_id)
        if state is None:
            return

        if send_shutdown:
            try:
                await self._send_to_client(client_id, FifoEventShutdown())
            except ConnectionError:
                pass

        task = state.task
        if task is not None and not task.done():
            try:
                await asyncio.wait_for(task, timeout=timeout)
            except asyncio.TimeoutError:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    async def _on_client_done(self,
                              client_id: int,
                              state: _HubClientState,
                              fut: asyncio.Future[None]) -> None:
        if self._clients.get(client_id) is not state:
            return

        self._clients.pop(client_id, None)

        self._client_tasks.discard(state.task)
        state.context._mark_closed()

        await bounded_close_and_wait_closed_writer(state.writer, timeout=3.0, label="hub")

        if not fut.cancelled():
            exc = fut.exception()
            if exc is not None:
                logger.error(
                    "[hub] Client task for %s exited with error: %r",
                    client_id,
                    exc,
                )
