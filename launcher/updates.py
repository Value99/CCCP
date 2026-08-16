"""Non-blocking launcher update checks with independent network fallbacks."""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import httpx


UPDATE_SCHEMA = "cccp-launcher-update-v1"


@dataclass(frozen=True)
class UpdateSource:
    id: str
    label: str
    manifest_url: str
    download_url: str


UPDATE_SOURCES = (
    UpdateSource(
        id="visionsic",
        label="VisionSic 官网",
        manifest_url="https://www.visionsic.com/cccp/latest.json",
        download_url="https://www.visionsic.com/cccp/",
    ),
    UpdateSource(
        id="github",
        label="GitHub",
        manifest_url="https://raw.githubusercontent.com/Value99/CCCP/main/latest.json",
        download_url="https://github.com/Value99/CCCP",
    ),
)
UPDATE_DOWNLOAD_URLS = {source.id: source.download_url for source in UPDATE_SOURCES}

_VERSION_RE = re.compile(r"^v?(0|[1-9]\d*)(?:\.(0|[1-9]\d*)){1,3}$")


def version_key(value: str) -> tuple[int, int, int, int]:
    text = str(value or "").strip()
    if not _VERSION_RE.fullmatch(text):
        raise ValueError(f"无效版本号: {text or '空'}")
    parts = [int(part) for part in text.removeprefix("v").split(".")]
    return tuple((parts + [0] * 4)[:4])  # type: ignore[return-value]


def fetch_manifest(source: UpdateSource) -> dict[str, Any]:
    timeout = httpx.Timeout(connect=3.0, read=4.0, write=4.0, pool=2.0)
    headers = {
        "Accept": "application/json",
        "User-Agent": "CCCP-Launcher-Update-Check/1",
    }
    with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
        response = client.get(source.manifest_url)
        response.raise_for_status()
        payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("更新清单顶层必须是对象")
    return payload


class UpdateChecker:
    """Runs all network I/O on a daemon thread and exposes snapshot status."""

    def __init__(
        self,
        current_version: str,
        ignored_version: Callable[[], str],
        *,
        fetcher: Callable[[UpdateSource], dict[str, Any]] = fetch_manifest,
    ) -> None:
        version_key(current_version)
        self.current_version = current_version
        self._ignored_version = ignored_version
        self._fetcher = fetcher
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._state: dict[str, Any] = {
            "status": "idle",
            "current_version": current_version,
            "latest_version": "",
            "source": "",
            "source_label": "",
            "download_url": "",
            "title": "",
            "summary": "",
            "release_notes": [],
            "checked_at": 0.0,
            "errors": {},
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            result = dict(self._state)
            result["release_notes"] = list(self._state.get("release_notes") or [])
            result["errors"] = dict(self._state.get("errors") or {})
            return result

    def start(self, *, force: bool = False) -> dict[str, Any]:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return dict(self._state)
            if not force and self._state["status"] not in {"idle", "unavailable"}:
                return dict(self._state)
            self._state.update({"status": "checking", "errors": {}})
            self._thread = threading.Thread(
                target=self._run, daemon=True, name="cccp-update-check"
            )
            self._thread.start()
            return dict(self._state)

    def wait(self, timeout: float = 10.0) -> dict[str, Any]:
        thread = self._thread
        if thread:
            thread.join(timeout)
        return self.snapshot()

    def refresh_ignored_state(self) -> dict[str, Any]:
        with self._lock:
            latest = str(self._state.get("latest_version") or "")
            if latest and latest == self._ignored_version():
                self._state["status"] = "ignored"
            return dict(self._state)

    def _run(self) -> None:
        errors: dict[str, str] = {}
        for source in UPDATE_SOURCES:
            try:
                payload = self._fetcher(source)
                if payload.get("schema") != UPDATE_SCHEMA:
                    raise ValueError("更新清单 schema 不匹配")
                latest = str(payload.get("version") or "").strip()
                newest = version_key(latest) > version_key(self.current_version)
                notes = payload.get("release_notes") or []
                if not isinstance(notes, list):
                    notes = []
                status = "available" if newest else "current"
                if newest and latest == self._ignored_version():
                    status = "ignored"
                result = {
                    "status": status,
                    "current_version": self.current_version,
                    "latest_version": latest,
                    "source": source.id,
                    "source_label": source.label,
                    "download_url": source.download_url,
                    "title": str(payload.get("title") or f"CCCP 启动器 {latest}"),
                    "summary": str(payload.get("summary") or ""),
                    "release_notes": [str(item) for item in notes[:8]],
                    "checked_at": time.time(),
                    "errors": errors,
                }
                with self._lock:
                    self._state = result
                return
            except Exception as exc:  # each source is an independent fallback
                errors[source.id] = f"{type(exc).__name__}: {exc}"
        with self._lock:
            self._state.update({
                "status": "unavailable",
                "checked_at": time.time(),
                "errors": errors,
            })
