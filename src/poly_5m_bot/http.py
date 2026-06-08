from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class HttpError(RuntimeError):
    pass


def get_json(url: str, timeout: float = 10.0) -> Any:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "poly-5m-paper-trader/0.1",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise HttpError(f"GET {url} failed with HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise HttpError(f"GET {url} failed: {exc.reason}") from exc
    return json.loads(payload)


def post_json(url: str, body: Any, timeout: float = 10.0) -> Any:
    payload = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={
            "User-Agent": "poly-5m-paper-trader/0.1",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise HttpError(f"POST {url} failed with HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise HttpError(f"POST {url} failed: {exc.reason}") from exc
    return json.loads(data)

