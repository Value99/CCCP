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
    # 社区(v0.3):首页与配置库页展示;均为可选,未配置时前端给提示
    discord_url: str = ""  # Discord 社区邀请链接
    community_index_url: str = ""  # 社区 profile 索引 JSON URL
    # 模型下载(v0.3)
    hf_endpoint: str = "https://hf-mirror.com"  # HuggingFace 端点(可改官方 https://huggingface.co)
    model_download_dir: str = ""  # 模型默认下载落盘目录(空 → model_roots[0] 或 data/models)
    default_device: str = "cuda"  # 默认推理设备(cuda|cpu;前端两处下拉默认同步)
    theme_mode: str = "system"  # 主题:system(跟随系统) | light | dark
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
