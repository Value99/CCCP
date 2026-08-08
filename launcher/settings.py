"""WINUI-EXE 自身的运行配置(与 TPQ-Final 解耦,仅存 data/)。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

from .resources import data_dir, detect_tpq_path

DATA_DIR = data_dir()
SETTINGS_FILE = DATA_DIR / "settings.json"


@dataclass
class Settings:
    """可被前端 /api/settings 修改并持久化的配置。"""

    tpq_path: str = ""  # TPQ-Final 根目录;空则自动探测 ../TPQ-Final
    model_roots: list[str] = field(default_factory=list)  # 扫描 cccp.json 的根目录
    api_port_alloc_start: int = 8801  # 为 TPQ serve 分配的端口起点
    tpq_api_key: str = ""
    # 模型专家规格(校准前为估算默认值;INTERFACE I-2 落地后自动更新)
    model_layers: int = 60
    model_experts_per_layer: int = 256
    default_context: int = 8192

    def save(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8"
        )


def load_settings() -> Settings:
    s = Settings()
    if SETTINGS_FILE.exists():
        try:
            raw = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            for k, v in raw.items():
                if hasattr(s, k):
                    setattr(s, k, v)
        except (json.JSONDecodeError, OSError):
            pass
    if not s.tpq_path:
        det = detect_tpq_path()
        if det:
            s.tpq_path = str(det)
    return s
