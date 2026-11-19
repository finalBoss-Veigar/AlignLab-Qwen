import json
import threading
import urllib.error
import urllib.request
import pytest
from alignlab.serving.server import LocalModel, make_server, serve
from alignlab.serving.client import open_url, request


def test_no_weights_is_not_ready_and_never_generates(monkeypatch):
    monkeypatch.setenv('HTTP_PROXY', 'http://127.0.0.1:1')
    monkeypatch.setenv('HTTPS_PROXY', 'http://127.0.0.1:1')
    monkeypatch.setenv('NO_PROXY', '')
    backend = LocalModel()
    server = make_server(backend, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        with open_url(url + "/health", timeout=3) as r:
            assert json.load(r)["ready"] is False
        with pytest.raises(urllib.error.HTTPError) as exc:
            open_url(url + "/ready", timeout=3)
        assert exc.value.code == 503
        req = urllib.request.Request(url + "/generate", data=b'{"messages":[]}', headers={"Content-Type":"application/json"})
        with pytest.raises(urllib.error.HTTPError) as exc:
            open_url(req, timeout=3)
        assert exc.value.code == 503
        assert "model_not_ready" in exc.value.read().decode()
        with pytest.raises(RuntimeError, match='HTTP 503'):
            request('A local test prompt', url=url, timeout=3)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_dry_run_does_not_load():
    result = serve("missing-model", dry_run=True)
    assert result["status"] == "dry_run_passed"
    assert result["model_loaded"] is False


def test_external_bind_rejected():
    with pytest.raises(ValueError):
        make_server(LocalModel(), host="0.0.0.0")


def test_failed_partial_model_load_never_becomes_ready(monkeypatch):
    def fail_during_device_move(self, path):
        self.model = object()
        self.tokenizer = object()
        raise RuntimeError("CUDA out of memory during device move")
    monkeypatch.setattr(LocalModel, "load", fail_during_device_move)
    backend = LocalModel("local-model", "sha")
    assert backend.ready is False
    assert "out of memory" in backend.load_error


def test_dry_run_rejects_invalid_port():
    with pytest.raises(ValueError):
        serve(port=-1, dry_run=True)
