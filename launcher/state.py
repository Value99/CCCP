"""data/ 下的轻量持久化:当前 profile 组合、最近发动记录、聊天会话。

全部为 JSON 文件,原子写(先写 .tmp 再替换),保证崩溃不出半截文件。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .settings import DATA_DIR

STATE_FILE = DATA_DIR / "state.json"


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


class AppState:
    """应用级持久状态(单例,由 app.py 持有)。"""

    def __init__(self) -> None:
        self.data: dict[str, Any] = {
            "selected_profiles": [],   # 当前勾选的 profile id
            "last_launch": None,       # 最近一次发动 {model, profiles, port, at}
            "chat_sessions": {},       # id -> {title, messages:[{role, content}], updated_at}
        }
        self._load()

    def _load(self) -> None:
        if STATE_FILE.exists():
            try:
                raw = json.loads(STATE_FILE.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    self.data.update(raw)
            except (json.JSONDecodeError, OSError):
                pass

    def save(self) -> None:
        _atomic_write(STATE_FILE, self.data)

    # -- profile 组合 --
    def set_selected_profiles(self, ids: list[str]) -> None:
        self.data["selected_profiles"] = list(ids)
        self.save()

    # -- 发动记录 --
    def record_launch(self, model: str, profiles: list[str], port: int) -> None:
        self.data["last_launch"] = {
            "model": model, "profiles": list(profiles),
            "port": port, "at": time.time(),
        }
        self.save()

    # -- 聊天会话 --
    def save_session(self, sid: str, title: str, messages: list[dict]) -> str:
        sessions = self.data.setdefault("chat_sessions", {})
        sessions[sid] = {
            "title": title, "messages": messages, "updated_at": time.time(),
        }
        # 只保留最近 50 个会话
        if len(sessions) > 50:
            for old in sorted(sessions, key=lambda k: sessions[k]["updated_at"])[:-50]:
                sessions.pop(old, None)
        self.save()
        return sid

    def list_sessions(self) -> list[dict]:
        sessions = self.data.get("chat_sessions", {})
        return [
            {"id": sid, "title": s.get("title", sid), "updated_at": s.get("updated_at", 0)}
            for sid, s in sorted(sessions.items(), key=lambda kv: -kv[1].get("updated_at", 0))
        ]

    def get_session(self, sid: str) -> dict | None:
        return self.data.get("chat_sessions", {}).get(sid)

    def delete_session(self, sid: str) -> bool:
        sessions = self.data.get("chat_sessions", {})
        if sid in sessions:
            sessions.pop(sid)
            self.save()
            return True
        return False
