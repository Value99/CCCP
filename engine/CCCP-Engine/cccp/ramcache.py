"""一次性模型 RAM 镜像：先顺序读取 NAS，再从内存装载 GPU。

该镜像只服务启动阶段。全显存 Expert Parallel 完成后，Engine 会解除所有
SafeFile 对镜像的引用并释放 bytearray，推理期不保留 routed expert 的主机副本。
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path


_ACTIVE_LOCK = threading.RLock()
_ACTIVE_FILES: dict[str, bytearray] = {}


def _file_key(path: str | os.PathLike[str]) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def active_ram_file(
    path: str | os.PathLike[str],
) -> bytearray | None:
    """Return the active in-process RAM image for ``path``, if any."""
    with _ACTIVE_LOCK:
        return _ACTIVE_FILES.get(_file_key(path))


def _weight_sort_key(path: Path) -> tuple[int, str]:
    name = path.name
    if name == "dense.safetensors":
        return (0, name)
    if name.startswith("experts.L"):
        try:
            layer = int(name.split(".L", 1)[1].split(".", 1)[0])
        except (IndexError, ValueError):
            layer = 1_000_000
        return (1, f"{layer:07d}")
    if name == "mtp.safetensors":
        return (2, name)
    return (3, name)


class ModelRamMirror:
    """Read all model weight files on one background thread.

    The main thread calls :meth:`wait_and_activate` before constructing the
    model.  Activation is atomic: SafeFile either sees the complete mirror or
    reads the original file, never a partially filled bytearray.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        reserve_gb: float | None = None,
        available_bytes: int | None = None,
        exclude_paths: tuple[str | os.PathLike[str], ...] = (),
    ):
        self.root = os.path.abspath(os.fspath(root))
        configured_reserve = (
            float(os.environ.get("CCCP_RAM_RESERVE_GB", "2"))
            if reserve_gb is None
            else float(reserve_gb)
        )
        self.reserve_bytes = int(max(3.0, configured_reserve) * 2**30)
        self._available_bytes = available_bytes
        self._exclude_paths = {
            _file_key(path) for path in exclude_paths
        }
        self._files: dict[str, bytearray] = {}
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._skip_reason: str | None = None
        self._active = False
        self.total_bytes = 0
        self.loaded_bytes = 0
        self.started_at = 0.0

    @property
    def active(self) -> bool:
        return self._active

    @property
    def resident_bytes(self) -> int:
        return sum(len(blob) for blob in self._files.values())

    def _weight_files(self) -> list[Path]:
        files = [
            path
            for path in Path(self.root).iterdir()
            if (
                path.is_file()
                and path.suffix == ".safetensors"
                and _file_key(path) not in self._exclude_paths
            )
        ]
        return sorted(files, key=_weight_sort_key)

    def _available_ram(self) -> int:
        if self._available_bytes is not None:
            return int(self._available_bytes)
        import psutil

        return int(psutil.virtual_memory().available)

    @staticmethod
    def _read_file(path: Path, size: int) -> bytearray:
        blob = bytearray(size)
        view = memoryview(blob)
        offset = 0
        with path.open("rb", buffering=8 * 2**20) as source:
            while offset < size:
                count = source.readinto(view[offset:])
                if not count:
                    raise EOFError(
                        f"{path} ended at {offset} bytes, expected {size}"
                    )
                offset += count
        return blob

    def _load(self) -> None:
        try:
            paths = self._weight_files()
            sizes = [(path, path.stat().st_size) for path in paths]
            self.total_bytes = sum(size for _path, size in sizes)
            available = self._available_ram()
            if self.total_bytes + self.reserve_bytes > available:
                self._skip_reason = (
                    f"权重 {self.total_bytes / 2**30:.2f}GB + "
                    f"预留 {self.reserve_bytes / 2**30:.2f}GB > "
                    f"可用RAM {available / 2**30:.2f}GB"
                )
                return
            print(
                f"[cccp] RAM 镜像线程启动：{len(paths)} 个权重文件 / "
                f"{self.total_bytes / 2**30:.2f}GB，"
                f"至少预留 {self.reserve_bytes / 2**30:.2f}GB",
                flush=True,
            )
            for path, size in sizes:
                blob = self._read_file(path, size)
                self._files[_file_key(path)] = blob
                self.loaded_bytes += size
                print(
                    f"[cccp] RAM 镜像 "
                    f"{self.loaded_bytes / 2**30:.2f}/"
                    f"{self.total_bytes / 2**30:.2f}GB "
                    f"({path.name})",
                    flush=True,
                )
        except BaseException as error:
            self._error = error
            self._files.clear()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("RAM mirror thread already started")
        self.started_at = time.time()
        self._thread = threading.Thread(
            target=self._load,
            name="cccp-model-ram-mirror",
            daemon=True,
        )
        self._thread.start()

    def wait_and_activate(self) -> bool:
        if self._thread is None:
            raise RuntimeError("RAM mirror thread was not started")
        self._thread.join()
        if self._error is not None:
            raise RuntimeError("模型 RAM 镜像读取失败") from self._error
        if self._skip_reason is not None:
            print(
                f"[cccp] RAM 镜像未启用：{self._skip_reason}；"
                "回退直接文件读取",
                flush=True,
            )
            return False
        with _ACTIVE_LOCK:
            overlap = set(self._files).intersection(_ACTIVE_FILES)
            if overlap:
                raise RuntimeError(
                    f"RAM mirror already active for {len(overlap)} files"
                )
            _ACTIVE_FILES.update(self._files)
        self._active = True
        print(
            f"[cccp] RAM 镜像同步完成："
            f"{self.loaded_bytes / 2**30:.2f}GB，"
            f"{time.time() - self.started_at:.1f}s；"
            "后续权重读取不再访问 NAS",
            flush=True,
        )
        return True

    def release(self) -> int:
        """Deactivate the mirror and return the number of released bytes."""
        released = self.resident_bytes
        with _ACTIVE_LOCK:
            for key, blob in self._files.items():
                if _ACTIVE_FILES.get(key) is blob:
                    del _ACTIVE_FILES[key]
        self._active = False
        self._files.clear()
        return released

    def release_paths(
        self,
        paths: tuple[str | os.PathLike[str], ...],
    ) -> int:
        """Release selected files while keeping packed expert blobs active."""
        keys = {_file_key(path) for path in paths}
        released = 0
        with _ACTIVE_LOCK:
            for key in keys:
                blob = self._files.pop(key, None)
                if blob is None:
                    continue
                released += len(blob)
                if _ACTIVE_FILES.get(key) is blob:
                    del _ACTIVE_FILES[key]
        if not self._files:
            self._active = False
        return released
