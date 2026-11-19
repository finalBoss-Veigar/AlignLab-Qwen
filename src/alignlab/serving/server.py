import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time


class LocalModel:
    def __init__(self, model_path=None, revision=None, max_input_tokens=1024, max_output_tokens=256):
        self.model = None
        self.tokenizer = None
        self.model_path = str(model_path) if model_path else None
        self.revision = revision
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.lock = threading.Lock()
        self.load_error = None
        if model_path:
            try:
                self.load(model_path)
            except Exception as e:
                self.model = None
                self.tokenizer = None
                self.load_error = str(e)

    @property
    def ready(self):
        return self.model is not None and self.tokenizer is not None

    def load(self, model_path):
        path = Path(model_path)
        if not path.is_dir():
            raise ValueError("Model must be an existing local directory; automatic downloads are disabled")
        if not self.revision:
            raise ValueError("A model revision or exported artifact fingerprint is required")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
        if not tokenizer.chat_template:
            raise ValueError("A real chat template is required")
        model = AutoModelForCausalLM.from_pretrained(str(path), local_files_only=True,
            trust_remote_code=False, torch_dtype="auto")
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        self.tokenizer, self.model = tokenizer, model

    def info(self):
        return {"ready": self.ready, "model_path": self.model_path, "model_revision": self.revision,
                "load_error": self.load_error, "backend": "transformers", "mock": False}

    def generate(self, request):
        if not self.ready:
            raise RuntimeError("model_not_ready: no usable local weights loaded")
        if not isinstance(request, dict) or not isinstance(request.get("messages"), list):
            raise ValueError("messages must be a list")
        messages = request["messages"]
        if not messages or len(messages) > 64:
            raise ValueError("messages must contain 1..64 items")
        expected = "user"
        for i, item in enumerate(messages):
            if not isinstance(item, dict) or not isinstance(item.get("content"), str) or not item["content"].strip():
                raise ValueError("Every message must have nonempty string content")
            if i == 0 and item.get("role") == "system":
                continue
            if item.get("role") != expected:
                raise ValueError("Invalid role order")
            expected = "assistant" if expected == "user" else "user"
        if messages[-1]["role"] != "user":
            raise ValueError("Generation context must end with user")
        limit = request.get("max_new_tokens", self.max_output_tokens)
        if type(limit) is not int or not 1 <= limit <= self.max_output_tokens:
            raise ValueError(f"max_new_tokens must be 1..{self.max_output_tokens}")
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        if ids.shape[1] > self.max_input_tokens:
            raise ValueError("input_token_limit_exceeded")
        if not self.lock.acquire(blocking=False):
            raise BlockingIOError("server_busy: retry after the current generation")
        try:
            import torch
            ids = ids.to(self.model.device)
            if self.model.device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                out = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                    do_sample=False, max_new_tokens=limit, pad_token_id=self.tokenizer.eos_token_id,
                    max_time=60.0)
            if self.model.device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if elapsed >= 60.0:
                raise TimeoutError("generation_deadline_exceeded; partial output discarded")
            tokens = out[0, ids.shape[1]:]
            return {"response": self.tokenizer.decode(tokens, skip_special_tokens=True),
                "model_revision": self.revision, "input_tokens": ids.shape[1], "output_tokens": len(tokens),
                "elapsed_seconds": elapsed, "tokens_per_second": len(tokens) / elapsed,
                "ttft_seconds": None, "ttft_reason": "non_streaming_interface",
                "decoding": {"do_sample": False, "max_new_tokens": limit},
                "chat_template_sha256": hashlib.sha256(self.tokenizer.chat_template.encode()).hexdigest()}
        finally:
            self.lock.release()


def make_server(backend, host="127.0.0.1", port=8000):
    if host not in {"localhost", "127.0.0.1"}:
        raise ValueError("Only localhost binding is supported")

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def send(self, status, value):
            raw = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path == "/health":
                self.send(200, {"alive": True, **backend.info()})
            elif self.path == "/ready":
                self.send(200 if backend.ready else 503, backend.info())
            else:
                self.send(404, {"error": "not_found"})

        def do_POST(self):
            if self.path != "/generate":
                return self.send(404, {"error": "not_found"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size <= 0 or size > 65536:
                    return self.send(413, {"error": "body_size_limit", "max_bytes": 65536})
                body = json.loads(self.rfile.read(size))
                self.send(200, backend.generate(body))
            except (ValueError, TypeError) as e:
                self.send(400, {"error": str(e)})
            except BlockingIOError as e:
                self.send(429, {"error": str(e)})
            except TimeoutError as e:
                self.send(504, {"error": str(e)})
            except RuntimeError as e:
                self.send(503, {"error": str(e), **backend.info()})
            except Exception as e:
                self.send(500, {"error": type(e).__name__, "detail": str(e)})

        def log_message(self, format, *args):
            return  # Never print request bodies or user content.

    return ThreadingHTTPServer((host, port), Handler)


def serve(model_path=None, revision=None, host="127.0.0.1", port=8000, dry_run=False):
    if host not in {"localhost", "127.0.0.1"} or type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("A localhost address and port in 1..65535 are required")
    if dry_run:
        return {"status": "dry_run_passed", "model_loaded": False, "training_completed": False,
                "host": host, "port": port, "model_path": str(model_path) if model_path else None,
                "note": "Only command configuration checked; no weights loaded or generated"}
    backend = LocalModel(model_path, revision)
    server = make_server(backend, host, port)
    print(json.dumps({"url": f"http://{host}:{port}", **backend.info()}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
