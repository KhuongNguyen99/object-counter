"""Persistent RunPod HTTPS protocol without external network or GPU jobs."""

from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
from types import SimpleNamespace

import pytest

from app import runpod_client as remote
from app import runpod_transport
from app.runpod_transport import PersistentHTTPS, TransportError


class _Socket:
    def __init__(self):
        self.timeouts = []

    def settimeout(self, seconds):
        self.timeouts.append(seconds)

    def shutdown(self, how):
        pass


class _Response:
    def __init__(self, status, payload, *, will_close=False):
        self.status = status
        self.payload = payload
        self.will_close = will_close
        self.offset = 0

    def isclosed(self):
        return self.offset >= len(self.payload)

    def read1(self, limit):
        chunk = self.payload[self.offset:self.offset + limit]
        self.offset += len(chunk)
        return chunk


class _Connection:
    def __init__(self, host, timeout, responses, *, fail_second_send=False):
        self.host = host
        self.timeout = timeout
        self.responses = responses
        self.sock = None
        self.requests = []
        self.closed = 0
        self.fail_second_send = fail_second_send
        self.sends = 0

    def connect(self):
        self.sock = _Socket()

    def putrequest(self, method, path, *, skip_accept_encoding):
        assert skip_accept_encoding
        self.requests.append({"method": method, "path": path, "headers": {}})

    def putheader(self, name, value):
        self.requests[-1]["headers"][name] = value

    def endheaders(self):
        pass

    def send(self, body):
        self.sends += 1
        if self.fail_second_send and self.sends == 2:
            raise BrokenPipeError("stale connection")
        self.requests[-1]["body"] = body

    def getresponse(self):
        return self.responses.pop(0)

    def close(self):
        self.closed += 1
        self.sock = None


def _client(responses, *, fail_second_send=False):
    connections = []

    def factory(host, timeout):
        connection = _Connection(host, timeout, responses,
                                 fail_second_send=fail_second_send and not connections)
        connections.append(connection)
        return connection

    client = remote.RunPodClient("top-secret-key", "endpoint-1")
    client._transport = PersistentHTTPS(connection_factory=factory)
    return client, connections


def _json_response(job_id):
    return _Response(200, json.dumps({"id": job_id, "status": "COMPLETED"}).encode())


def test_successive_jobs_reuse_tls_connection_and_report_request_phases():
    client, connections = _client([_json_response("first"), _json_response("second")])
    assert client.run({"image_base64": "QUJD"})["id"] == "first"
    first = client.last_job["transport_requests"]
    assert client.run({"image_base64": "REVG"})["id"] == "second"
    second = client.last_job["transport_requests"]

    assert len(connections) == 1
    assert connections[0].host == "api.runpod.ai"
    assert [item["connection"] for item in (first[0], second[0])] == ["new", "reused"]
    assert first[0]["request"] == second[0]["request"] == "runsync"
    for timing in (first[0], second[0]):
        for phase in ("connect_ms", "upload_ms", "response_headers_ms",
                      "response_body_ms", "total_ms"):
            assert isinstance(timing[phase], float) and timing[phase] >= 0
        assert "top-secret-key" not in repr(timing)
        assert "QUJD" not in repr(timing)
    assert all(request["headers"]["Authorization"] == "Bearer top-secret-key"
               for request in connections[0].requests)
    assert [json.loads(request["body"])["input"]["image_base64"]
            for request in connections[0].requests] == ["QUJD", "REVG"]
    remote.RunPodInferenceEngine(client=client).close()
    assert connections[0].closed == 1


def test_stale_reused_connection_is_discarded_without_replaying_post():
    client, connections = _client([_json_response("first"), _json_response("third")],
                                  fail_second_send=True)
    assert client.run({"action": "count"})["id"] == "first"
    with pytest.raises(remote.RunPodError, match="network error"):
        client.run({"action": "count"})
    assert len(connections) == 1
    assert len(connections[0].requests) == 2
    assert connections[0].closed == 1
    assert client.last_transport["connection"] == "reused"

    assert client.run({"action": "count"})["id"] == "third"
    assert len(connections) == 2
    assert len(connections[1].requests) == 1
    assert client.last_job["transport_requests"][0]["connection"] == "new"


def test_redirect_is_an_error_and_response_limit_discards_connection(monkeypatch):
    client, connections = _client([_Response(302, b'{}'), _Response(200, b"123456")])
    with pytest.raises(remote.RunPodError, match="302"):
        client.submit({"action": "count"})
    assert len(connections) == 1
    assert len(connections[0].requests) == 1

    monkeypatch.setattr(remote, "MAX_RESPONSE_BYTES", 5)
    with pytest.raises(remote.RunPodError, match="exceeded"):
        client.submit({"action": "count"})
    assert connections[0].closed == 1
    assert len(connections[0].requests) == 2


def test_absolute_deadline_stops_slow_body_read(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(runpod_transport, "time",
                        SimpleNamespace(monotonic=lambda: clock[0]))

    class SlowResponse(_Response):
        def read1(self, limit):
            clock[0] += .6
            return super().read1(1)

    connection = _Connection("api.runpod.ai", 1, [SlowResponse(200, b"abcd")])
    transport = PersistentHTTPS(connection_factory=lambda host, timeout: connection)
    with pytest.raises(TransportError) as exc:
        transport.request("GET", "/v2/endpoint-1/status/job-1", {}, None,
                          timeout=1)
    assert exc.value.timing["total_ms"] == 1200.0
    assert connection.closed == 1
    assert len(connection.requests) == 1


def test_real_http_connection_reuses_socket_for_complete_responses():
    seen = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            seen.append((self.client_address[1], self.path, body))
            payload = b'{"ok":true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    transport = PersistentHTTPS(
        host="127.0.0.1",
        connection_factory=lambda host, timeout: HTTPConnection(
            host, server.server_port, timeout=timeout))
    try:
        first = transport.request("POST", "/v2/endpoint-1/runsync", {}, b"first", timeout=2)
        second = transport.request("POST", "/v2/endpoint-1/runsync", {}, b"second", timeout=2)
        assert first[:2] == second[:2] == (200, b'{"ok":true}')
        assert [item[2] for item in seen] == [b"first", b"second"]
        assert seen[0][0] == seen[1][0]
        assert [first[2]["connection"], second[2]["connection"]] == ["new", "reused"]
    finally:
        transport.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_connection_close_response_body_is_read_before_reconnect():
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            payload = b'{"ok":true}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(payload)
            self.close_connection = True

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    transport = PersistentHTTPS(host="127.0.0.1", connection_factory=lambda host, timeout:
                                HTTPConnection(host, server.server_port, timeout=timeout))
    try:
        first = transport.request("GET", "/status", {}, None, timeout=2)
        second = transport.request("GET", "/status", {}, None, timeout=2)
        assert first[:2] == second[:2] == (200, b'{"ok":true}')
        assert [first[2]["connection"], second[2]["connection"]] == ["new", "new"]
    finally:
        transport.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_drip_fed_headers_obey_absolute_deadline():
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            try:
                for byte in b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}":
                    self.wfile.write(bytes((byte,)))
                    self.wfile.flush()
                    time.sleep(.02)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    transport = PersistentHTTPS(host="127.0.0.1", connection_factory=lambda host, timeout:
                                HTTPConnection(host, server.server_port, timeout=timeout))
    started = time.monotonic()
    try:
        with pytest.raises(TransportError):
            transport.request("GET", "/status", {}, None, timeout=.15)
        assert time.monotonic() - started < .5
    finally:
        transport.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
