"""One reusable HTTPS connection for the fixed RunPod API host.

There is deliberately no redirect handling or automatic retry here. In
particular, a failed POST may already have created a job on the server.
"""

import http.client
import socket
import threading
import time


MAX_RESPONSE_BYTES = 20_000_000
MAX_CONNECTION_IDLE_SECONDS = 60


class TransportError(OSError):
    """A network failure with safe, numeric request timing details."""

    def __init__(self, timing):
        super().__init__("RunPod transport failed.")
        self.timing = timing


class PersistentHTTPS:
    """Serialize requests on a single TLS connection to one known host."""

    def __init__(self, host="api.runpod.ai", *, connection_factory=None):
        self._host = host
        self._factory = connection_factory or http.client.HTTPSConnection
        self._connection = None
        self._last_used = None
        self._lock = threading.Lock()

    def close(self):
        with self._lock:
            self._discard()

    def _discard(self):
        connection, self._connection = self._connection, None
        self._last_used = None
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass

    @staticmethod
    def _milliseconds(start, end):
        return round(max(0.0, end - start) * 1000, 2)

    @staticmethod
    def _remaining(sock, deadline):
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError("RunPod request timed out.")
        sock.settimeout(left)

    def request(self, method, path, headers, body, *, timeout,
                max_response_bytes=MAX_RESPONSE_BYTES):
        """Return (HTTP status, bounded bytes, timings); never retry a request.

        connect_ms combines DNS, TCP, and TLS on a fresh connection. All other
        phases use a single absolute deadline, including response body reads.
        """
        if not path.startswith("/") or path.startswith("//"):
            raise ValueError("RunPod request path must be relative to the API host.")
        started = time.monotonic()
        deadline = started + timeout
        timing = {"connection": "reused", "connect_ms": 0.0}
        with self._lock:
            watchdog = None
            try:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError("RunPod request timed out.")
                connection = self._connection
                if (connection is None or connection.sock is None or
                        self._last_used is not None and
                        time.monotonic() - self._last_used >= MAX_CONNECTION_IDLE_SECONDS):
                    # Refresh idle sockets before sending a new job, never by replaying its POST.
                    self._discard()
                    connection = self._factory(self._host, timeout=left)
                    self._connection = connection
                    timing["connection"] = "new"
                    connecting = time.monotonic()
                    connection.connect()
                    timing["connect_ms"] = self._milliseconds(connecting, time.monotonic())

                # getresponse() detaches connection.sock on Connection: close,
                # while the response still owns this socket for its body.
                sock = connection.sock
                expired = threading.Event()

                def expire():
                    expired.set()
                    try:
                        sock.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass

                watchdog = threading.Timer(max(0, deadline - time.monotonic()), expire)
                watchdog.daemon = True
                watchdog.start()

                self._remaining(sock, deadline)
                uploading = time.monotonic()
                connection.putrequest(method, path, skip_accept_encoding=True)
                for name, value in headers.items():
                    connection.putheader(name, value)
                if body is not None:
                    connection.putheader("Content-Length", str(len(body)))
                connection.endheaders()
                if body is not None:
                    self._remaining(sock, deadline)
                    connection.send(body)
                timing["upload_ms"] = self._milliseconds(uploading, time.monotonic())

                self._remaining(sock, deadline)
                waiting = time.monotonic()
                response = connection.getresponse()
                timing["response_headers_ms"] = self._milliseconds(waiting, time.monotonic())
                status = response.status

                reading = time.monotonic()
                raw = bytearray()
                while not response.isclosed() and len(raw) <= max_response_bytes:
                    self._remaining(sock, deadline)
                    chunk = response.read1(min(64 * 1024, max_response_bytes + 1 - len(raw)))
                    if not chunk:
                        break
                    raw.extend(chunk)
                timing["response_body_ms"] = self._milliseconds(reading, time.monotonic())
                if len(raw) > max_response_bytes:
                    self._discard()
                elif (response.will_close or connection.sock is None or
                      getattr(response, "length", None) not in (None, 0)):
                    self._discard()
                watchdog.cancel()
                watchdog.join()
                watchdog = None
                if expired.is_set() or time.monotonic() > deadline:
                    raise TimeoutError("RunPod request timed out.")
                if self._connection is connection:
                    self._last_used = time.monotonic()
                timing["total_ms"] = self._milliseconds(started, time.monotonic())
                return status, bytes(raw), timing
            except (OSError, TimeoutError, http.client.HTTPException):
                self._discard()
                timing["total_ms"] = self._milliseconds(started, time.monotonic())
                raise TransportError(timing) from None
            finally:
                if watchdog is not None:
                    watchdog.cancel()
                    watchdog.join()
