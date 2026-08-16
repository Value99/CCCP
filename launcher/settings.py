"""WINUI-EXE 自身的运行配置(与 CCCP-Engine 解耦,仅存 data/)。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from .resources import (
    data_dir,
    default_models_dir,
    detect_model_roots,
    detect_python_path,
    detect_engine_path,
)
from .io_utils import atomic_write_text

DATA_DIR = data_dir()
SETTINGS_FILE = DATA_DIR / "settings.json"
INTERNAL_PATH_FIELDS = {
    "cccp_engine_path", "python_path", "cpu_python_path",
    "cuda_python_path", "amd_python_path",
}
INTERNAL_TUNING_FIELDS = {
    "memory_limit_gb", "expert_cache_gb", "cpu_threads",
    "cpu_compile_mode", "default_context",
}
INTERNAL_COMMUNITY_DOWNLOAD_FIELDS = {
    "discord_url", "modelscope_profile_url", "community_index_url",
    "default_download_source", "hf_endpoint", "model_download_dir",
}


@dataclass
class Settings:
    """可被前端 /api/settings 修改并持久化的配置。"""

    settings_version: int = 11
    cccp_engine_path: str = ""  # CCCP-Engine 根目录;空则自动探测 ../CCCP-Engine
    python_path: str = ""  # 随包本地 Python；冻结 EXE 不可拿自身充当解释器
    cpu_python_path: str = ""  # CPU 推理环境；python_path 保留为兼容别名
    cuda_python_path: str = ""  # NVIDIA CUDA 独立环境
    amd_python_path: str = ""  # AMD ROCm/HIP 独立环境
    model_roots: list[str] = field(default_factory=list)  # 扫描 cccp.json 的根目录
    api_port_alloc_start: int = 8801  # 为 CCCP serve 分配的端口起点
    cccp_api_key: str = ""
    # 社区(v0.3):首页与配置库页展示;均为可选,未配置时前端给提示
    discord_url: str = "https://discord.gg/eNnwmAUY4M"  # Discord 社区邀请链接
    modelscope_profile_url: str = "https://www.modelscope.cn/profile/ValueFX"
    community_index_url: str = ""  # 社区 profile 索引 JSON URL
    # 模型下载(v0.3)
    default_download_source: str = "modelscope"  # modelscope | hf
    hf_endpoint: str = "https://hf-mirror.com"  # HuggingFace 端点(可改官方 https://huggingface.co)
    model_download_dir: str = ""  # 模型默认下载落盘目录(空 → model_roots[0] 或 data/models)
    default_device: str = "cpu"  # 本发行版面向无 GPU 的低内存电脑
    default_profile_mode: str = "auto"
    memory_limit_gb: float = 32.0  # 设备 RAM/VRAM 风险提示阈值；不是配置生成预算
    expert_cache_gb: float = 8.0
    cpu_threads: int = 0  # 0=自动选择物理核心
    cpu_compile_mode: str = "q4"  # 进程内 Q4 执行映像；不改模型文件，CPU 最快
    startup_timeout_s: int = 1800
    theme_mode: str = "system"  # 主题:system(跟随系统) | light | dark
    skipped_update_version: str = ""  # 用户选择“本版本不升级”的远端版本
    # 旧设置槽位仅用于设置迁移；模型规格始终从 cccp.json/接口契约读取。
    model_layers: int = 43
    model_experts_per_layer: int = 256
    default_expert_size_mb: float = 6.4
    default_context: int = 512

    def validate(self) -> None:
        self.cccp_api_key = str(self.cccp_api_key or "").strip()
        if len(self.cccp_api_key) > 512 or any(ch in self.cccp_api_key for ch in "\r\n"):
            raise ValueError("CCCP API Key 长度不能超过 512 个字符，且不能包含换行")
        if self.default_device not in {"cpu", "cuda", "amd"}:
            raise ValueError("default_device 必须是 cpu、cuda 或 amd")
        self.theme_mode = str(self.theme_mode or "system").strip().lower()
        if self.theme_mode not in {"system", "light", "dark"}:
            raise ValueError("theme_mode 必须是 system、light 或 dark")
        self.skipped_update_version = str(self.skipped_update_version or "").strip()
        if len(self.skipped_update_version) > 64 or any(
            char in self.skipped_update_version for char in "\r\n"
        ):
            raise ValueError("忽略的更新版本号无效")
        if self.default_download_source not in {"modelscope", "hf"}:
            raise ValueError("default_download_source 必须是 modelscope 或 hf")
        if self.default_profile_mode not in {"auto", "mapped"}:
            raise ValueError("default_profile_mode 非法")
        if not 4.0 <= float(self.memory_limit_gb) <= 2048.0:
            raise ValueError("memory_limit_gb 必须在 4 到 2048 GiB 之间")
        if not 0.25 <= float(self.expert_cache_gb) <= 2048.0:
            raise ValueError("expert_cache_gb 必须在 0.25 到 2048 GiB 之间")
        if not 1 <= int(self.api_port_alloc_start) <= 65535:
            raise ValueError("api_port_alloc_start 必须在 1 到 65535 之间")
        if not 64 <= int(self.default_context) <= 32768:
            raise ValueError("default_context 必须在 64 到 32768 之间")
        if not 0 <= int(self.cpu_threads) <= 512:
            raise ValueError("cpu_threads 必须在 0 到 512 之间")
        if self.cpu_compile_mode not in {"auto", "off", "q4"}:
            raise ValueError("cpu_compile_mode 必须是 auto、off 或 q4")
        if not 30 <= int(self.startup_timeout_s) <= 86400:
            raise ValueError("startup_timeout_s 必须在 30 到 86400 秒之间")
        self.model_roots = _normalise_paths(self.model_roots)
        if self.cccp_engine_path:
            self.cccp_engine_path = str(Path(self.cccp_engine_path).expanduser().resolve())
        for name in ("python_path", "cpu_python_path", "cuda_python_path", "amd_python_path"):
            value = getattr(self, name)
            if value:
                setattr(self, name, str(Path(value).expanduser().resolve()))

    def update(self, values: dict[str, Any]) -> None:
        # 引擎和三套 Python 环境必须来自发行目录，REST 设置接口不能覆盖。
        allowed = (
            set(asdict(self)) - INTERNAL_PATH_FIELDS - INTERNAL_TUNING_FIELDS
            - INTERNAL_COMMUNITY_DOWNLOAD_FIELDS
        )
        previous = asdict(self)
        try:
            for key, value in values.items():
                if key in allowed:
                    setattr(self, key, value)
            self.validate()
        except (TypeError, ValueError):
            # REST 校验失败不能把进程内设置留在半更新状态，否则下一次
            # 合法保存也会被前一次非法主题/设备值污染。
            for key, value in previous.items():
                setattr(self, key, value)
            raise

    def save(self) -> None:
        self.validate()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        persisted = {
            key: value for key, value in asdict(self).items()
            if key not in (
                INTERNAL_PATH_FIELDS | INTERNAL_TUNING_FIELDS
                | INTERNAL_COMMUNITY_DOWNLOAD_FIELDS
            )
        }
        atomic_write_text(
            SETTINGS_FILE, json.dumps(persisted, ensure_ascii=False, indent=2)
        )


def load_settings() -> Settings:
    s = Settings()
    loaded_version = 0
    if SETTINGS_FILE.exists():
        try:
            raw = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            loaded_version = int(raw.get("settings_version") or 0)
            s.update(raw)
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass
    if loaded_version < 2:
        # v1 默认面向 CUDA/大内存；v2 是本次 32GB 无显卡离线发行的安全迁移。
        s.settings_version = 2
        s.default_device = "cpu"
        s.default_profile_mode = "mapped"
        s.memory_limit_gb = 24.0
        s.expert_cache_gb = 8.0
        s.cpu_threads = 0
        s.model_layers = 43
        s.model_experts_per_layer = 256
        s.default_expert_size_mb = 6.4
        s.default_context = 512
    if loaded_version < 3:
        s.settings_version = 3
        s.cpu_compile_mode = "auto"
    if loaded_version < 4:
        s.settings_version = 4
        s.cpu_compile_mode = "q4"
    if loaded_version < 5:
        s.settings_version = 5
        s.memory_limit_gb = 32.0
    if loaded_version < 6:
        # v6 发行版改用项目内 Miniconda 环境，旧设置中的系统 Python 不再复用。
        s.settings_version = 6
        det_py = detect_python_path()
        if det_py:
            s.python_path = str(det_py)
    if loaded_version < 7:
        # v7 默认使用随包 ModelScope SDK，并补齐首页社区/模型主页入口。
        s.settings_version = 7
        s.default_download_source = "modelscope"
        s.discord_url = "https://discord.gg/eNnwmAUY4M"
        s.modelscope_profile_url = "https://www.modelscope.cn/profile/ValueFX"
    if loaded_version < 8:
        # v8 将互相冲突的 PyTorch 构建拆为 CPU/NVIDIA/AMD 三个环境。
        s.settings_version = 8
        s.cpu_python_path = s.python_path or str(detect_python_path("cpu") or "")
        s.cuda_python_path = str(detect_python_path("cuda") or "")
        s.amd_python_path = str(detect_python_path("amd") or "")
    if loaded_version < 9:
        # v9 完成 CCCP 引擎命名与协议字段迁移。旧目录字段不再写回；
        # 由下方的随包引擎探测获得 engine/CCCP-Engine，避免硬编码机器路径。
        s.settings_version = 9
    if loaded_version < 10:
        # 用户只需选择“自动高速”或显式磁盘；自动模式由引擎按显存/RAM
        # 从 resident/ram 逐级降到 mapped，不能一开始就固定为硬盘映射。
        s.settings_version = 10
        s.default_profile_mode = "auto"
    if loaded_version < 11:
        s.settings_version = 11
        s.skipped_update_version = ""
    # 不信任旧设置中的绝对路径：引擎及 Python 环境始终按当前发行目录重探测。
    detected_engine = detect_engine_path()
    detected_cpu = detect_python_path("cpu")
    s.cccp_engine_path = str(detected_engine or "")
    s.cpu_python_path = str(detected_cpu or "")
    s.python_path = s.cpu_python_path  # 下载后端继续复用内置 CPU/工具环境。
    s.cuda_python_path = str(detect_python_path("cuda") or "")
    s.amd_python_path = str(detect_python_path("amd") or "")
    default_root = str(default_models_dir())
    s.model_roots = [default_root, *[p for p in s.model_roots if p != default_root]]
    s.validate()
    return s


def _normalise_paths(values: list[str] | tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        try:
            text = str(Path(text).expanduser().resolve())
        except OSError:
            pass
        if text not in out:
            out.append(text)
    return out
