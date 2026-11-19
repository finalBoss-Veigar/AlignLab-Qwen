"""Explicit timeout client; reports HTTP errors instead of synthesizing answers."""
import json
import urllib.error
import urllib.request
from urllib.parse import urlparse


def open_url(req, timeout):
    url = req.full_url if isinstance(req, urllib.request.Request) else req
    if urlparse(url).hostname in {'127.0.0.1', 'localhost', '::1'}:
        return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=timeout)
    return urllib.request.urlopen(req, timeout=timeout)


def request(prompt, url="http://127.0.0.1:8000", timeout=70):
    if len(prompt) > 16000:
        raise ValueError("Input exceeds client character limit")
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "max_new_tokens": 128}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/generate", data=body, headers={"Content-Type": "application/json"})
    try:
        with open_url(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8')}") from e
