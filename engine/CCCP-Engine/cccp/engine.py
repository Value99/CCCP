"""CCCP 生成引擎：tokenizer 封装 + 自回归生成循环（贪心 / top-p 采样）。

默认 EOS 取自 generation_config（GLM-5.2: [154820, 154827, 154829]），
<|user|>/<|observation|> 命中即停（对话模板的安全边界）。
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

from .model import GLMModel
from .prefill import begin_prefill_block, end_prefill_block
from .speculative import (
    DraftAcceptancePolicy,
    provider_attachment_available,
    provider_for_architecture,
)

DEFAULT_EOS = [154820, 154827, 154829]
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def _token_lcp(
    left: list[int] | None,
    right: list[int],
) -> int:
    """Return the exact token-ID longest common prefix."""
    if not left:
        return 0
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


@dataclass(frozen=True)
class KVPrefillStats:
    mode: str
    reason: str
    prompt_tokens: int
    baseline_tokens: int
    lcp_tokens: int
    replay_tokens: int
    suffix_tokens: int
    processed_tokens: int
    prefill_ms: float
    snapshot_bytes: int
    vision_load_ms: float | None = None
    vision_forward_ms: float | None = None
    language_prefill_ms: float | None = None


@dataclass
class _DSV4Baseline:
    ids: list[int]
    snapshot: object


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE_VALUES


def _generation_open(
    generated: int,
    max_new: int | None,
    position: int,
    max_ctx: int | None,
) -> bool:
    """Whether another output token may be committed."""
    if max_new is not None and generated >= max_new:
        return False
    return max_ctx is None or position < max_ctx


def _apply_token_penalties(
    logits: torch.Tensor,
    previous: list[int],
    *,
    repetition_penalty: float = 1.0,
    presence_penalty: float = 0.0,
) -> torch.Tensor:
    """Apply repetition and OpenAI-style presence penalties once per seen token."""
    if not previous or (
        repetition_penalty == 1.0 and presence_penalty == 0.0
    ):
        return logits
    adjusted = logits.clone()
    seen = torch.tensor(sorted(set(previous)), device=adjusted.device)
    if repetition_penalty != 1.0:
        values = adjusted[seen]
        adjusted[seen] = torch.where(
            values > 0,
            values / repetition_penalty,
            values * repetition_penalty,
        )
    if presence_penalty != 0.0:
        adjusted[seen] -= presence_penalty
    return adjusted


def _make_model(
    model_dir: str,
    cache_gb: float,
    max_ctx: int,
    device: str,
    vram_cache_gb: float,
    tp_size: int = 1,
    extreme_fixed_gpu_bytes: int = 0,
    dense_residency: str = "auto",
):
    """按 cccp.json 能力字段分派架构适配器。"""
    with open(os.path.join(model_dir, "cccp.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)
    cfg = manifest["config"]
    adapter = _model_adapter_architecture(manifest)
    if adapter == "glm5_next":
        from .glm5_next_model import GLM5NextCCCPModel

        return GLM5NextCCCPModel(
            model_dir,
            cache_gb=cache_gb,
            max_ctx=max_ctx,
            device=device,
            vram_cache_gb=vram_cache_gb,
            tp_size=tp_size,
        ), adapter
    dense_vq = bool(manifest.get("tensor_vq")) and not bool(
        manifest.get("expert_files")
        or (manifest.get("routed_experts") or {}).get("layer_files")
    )
    if dense_vq and str(
        cfg.get("text_model_type")
        or cfg.get("outer_model_type")
        or manifest.get("architecture")
        or ""
    ).startswith("qwen3_5"):
        from .qwen35_model import Qwen35DenseVQModel

        return Qwen35DenseVQModel(
            model_dir,
            cache_gb=cache_gb,
            max_ctx=max_ctx,
            device=device,
            vram_cache_gb=vram_cache_gb,
            tp_size=tp_size,
        ), "qwen3_5_dense"
    if (
        manifest.get("model_family") == "kimi_k3"
        or ("kda_layers" in cfg and "routed_hidden" in cfg)
    ):
        from .kimi_model import KimiK3CCCPModel

        return KimiK3CCCPModel(
            model_dir,
            cache_gb=cache_gb,
            max_ctx=max_ctx,
            device=device,
            vram_cache_gb=vram_cache_gb,
            tp_size=tp_size,
            extreme_fixed_gpu_bytes=extreme_fixed_gpu_bytes,
            dense_residency=dense_residency,
        ), "kimi_k3"
    if "hc_mult" in cfg or "compress_ratios" in cfg:
        # Reuse the canonical manifest parser: projection-VQ may be declared
        # per layer or by a heterogeneous per-expert precision map.
        from .store import Manifest

        projection_vq = Manifest(model_dir).projection_vq
        if tp_size != 1 and not projection_vq:
            raise ValueError(
                "--tp > 1 requires a projection-VQ DeepSeek-V4 archive"
            )
        from .dsv4model import DSV4CCCPModel
        return DSV4CCCPModel(model_dir, cache_gb=cache_gb, max_ctx=max_ctx,
                             device=device, vram_cache_gb=vram_cache_gb,
                             tp_size=tp_size,
                             extreme_fixed_gpu_bytes=(
                                 extreme_fixed_gpu_bytes
                             )), "dsv4"
    return GLMModel(model_dir, cache_gb=cache_gb, max_ctx=max_ctx,
                    device=device, vram_cache_gb=vram_cache_gb,
                    tp_size=tp_size), "glm"


def _model_adapter_architecture(manifest: dict) -> str:
    """Resolve the model adapter only from manifest capabilities."""

    config = manifest.get("config") or {}
    if str(manifest.get("architecture") or "").lower() == "glm5_next":
        return "glm5_next"
    dense_vq = bool(manifest.get("tensor_vq")) and not bool(
        manifest.get("expert_files")
        or (manifest.get("routed_experts") or {}).get("layer_files")
    )
    if dense_vq and str(
        config.get("text_model_type")
        or config.get("outer_model_type")
        or manifest.get("architecture")
        or ""
    ).startswith("qwen3_5"):
        return "qwen3_5_dense"
    if (
        manifest.get("model_family") == "kimi_k3"
        or ("kda_layers" in config and "routed_hidden" in config)
    ):
        return "kimi_k3"
    if "hc_mult" in config or "compress_ratios" in config:
        return "dsv4"
    return "glm"


def _dense_need_gb(model_dir: str, arch_hint: str, kv_gb: float) -> float:
    """按产物清单与 config 实际计算 dense 常驻需求（替代按架构硬编码）：
    DSV4 全 BF16 路径按 safetensors 头部精确计算展开常驻量；其他路径按
    dense.safetensors 实际大小 + head f32（vocab×hidden×4）+
    mtp/DSpark 附件 + 瞬时缓冲 1.5GB + KV。读取失败回退架构经验值。
    清单驱动使得任意档位产物（S/M/L）与任意显卡（16GB 起）都能正确自适应。"""
    fallback = (
        8.2 if arch_hint == "dsv4"
        else 60.0 if arch_hint == "kimi_k3"
        else 18.0 if arch_hint == "qwen3_5_dense"
        else 16.0 if arch_hint == "glm5_next"
        else 13.5
    ) + kv_gb
    try:
        with open(os.path.join(model_dir, "cccp.json"), "r", encoding="utf-8") as f:
            man = json.load(f)
        cfg = man["config"]
        if arch_hint == "glm5_next":
            from .store import CCCPStore

            store = CCCPStore(model_dir)
            text_names = [
                name
                for name in store.dense_names()
                if (
                    name == "lm_head.weight"
                    or name.startswith("model.language_model.")
                )
                and not name.startswith(
                    "model.language_model.layers.45."
                )
            ]
            fixed_bytes = 0
            for name in text_names:
                fixed_bytes += store.dense_nbytes(name)
                scale_name = name + "_scale_inv"
                if store.has(scale_name):
                    fixed_bytes += store.dense_nbytes(scale_name)
            store.close()
            # Model buffers, KDA recurrent state and allocator-private blocks
            # are not safetensors payloads. Keep one measured fixed margin;
            # the routed expert arena is still planned independently.
            return fixed_bytes / 2**30 + 1.5 + kv_gb
        if arch_hint == "qwen3_5_dense":
            files = [str(man.get("dense_file") or "dense.safetensors")]
            files.extend(
                str(name) for name in (man.get("tensor_files") or [])
            )
            fixed_bytes = sum(
                os.path.getsize(os.path.join(model_dir, name))
                for name in dict.fromkeys(files)
            )
            return fixed_bytes / 2**30 + kv_gb
        if arch_hint == "kimi_k3":
            audit_name = man.get("dense_audit_file")
            if not audit_name:
                raise ValueError("Kimi dense audit is required")
            with open(
                os.path.join(model_dir, audit_name),
                "r",
                encoding="utf-8",
            ) as handle:
                audit = json.load(handle)
            fixed_bytes = int(audit.get("fixed_bytes") or 0)
            if not fixed_bytes:
                fixed_bytes = sum(
                    os.path.getsize(os.path.join(model_dir, str(name)))
                    for name in man.get("dense_files", ())
                    if os.path.isfile(os.path.join(model_dir, str(name)))
                )
            if not fixed_bytes:
                raise ValueError("Kimi fixed dense bytes are missing")
            kda_state_gb = (
                len(cfg.get("kda_layers", []))
                * int(cfg["n_heads"])
                * int(cfg["head_dim"])
                * int(cfg["head_dim"])
                * 4
                / 2**30
            )
            return (
                fixed_bytes / 2**30
                + kda_state_gb
                + 1.5
                + kv_gb
            )
        dense_path = os.path.join(
            model_dir,
            man.get("dense_file", "dense.safetensors"),
        )
        dense_gb = os.path.getsize(dense_path) / 2**30
        head_gb = cfg["vocab"] * cfg["hidden"] * 4 / 2**30
        mtp_gb = 0.0
        dsv4_bf16_resident = False
        if arch_hint == "dsv4":
            from .capacity import dsv4_dense_runtime_bytes

            runtime_bytes = dsv4_dense_runtime_bytes(
                dense_path,
                os.environ.get("CCCP_DENSE_BF16"),
            )
            # The header calculation covers both compact and BF16-resident
            # modes and already includes lm_head exactly once.
            dense_gb = runtime_bytes / 2**30
            head_gb = 0.0
            dsv4_bf16_resident = str(
                os.environ.get("CCCP_DENSE_BF16", "")
            ).strip().lower() in {"1", "true", "all"}
        fn = man.get("dspark_file") or man.get("mtp_file")  # 清单指引（产物自包含）
        if fn and os.path.exists(os.path.join(model_dir, fn)):
            if man.get("dspark_file"):
                # DSpark：文件主体是草稿专家权重（驻 RAM/独立 LRU，不占 dense 显存），
                # dense 显存只需 stage bf16 ≈1.4GB + markov + VQ LRU ≈ 2.5GB
                if (
                    not dsv4_bf16_resident
                    or os.environ.get("CCCP_SPEC", "0") == "1"
                ):
                    mtp_gb = 2.5
            else:  # GLM MTP：dense 附件整体驻显存，按文件实际大小计
                mtp_gb = os.path.getsize(os.path.join(model_dir, fn)) / 2**30
        return dense_gb + head_gb + mtp_gb + 1.5 + kv_gb
    except Exception:
        return fallback


def _glm_startup_kv_gb(max_ctx: int, *, latent: bool) -> float:
    """Estimate only GLM's initial dynamic-KV working set.

    ``max_ctx`` is a logical admission ceiling; GLMModel grows KV tensors on
    demand instead of allocating that ceiling at startup.  Reserving the full
    model limit here would make the declared 1M context look like 90+ GiB of
    fixed VRAM and incorrectly force CUDA startup to CPU.  Expert arenas are
    physically shrunk later by the existing runtime VRAM monitor as KV grows.
    """
    logical_ctx = max(0, int(max_ctx))
    if latent:
        initial_ctx = min(logical_ctx, 4096)
        return 2.3 + 0.09 * initial_ctx / 1024
    initial_ctx = min(logical_ctx, 1024)
    return 5.0 * initial_ctx / 1024


def _dsv4_prefill_workspace_reserve_gb(max_ctx: int) -> float:
    """Return the initial DSV4 admission reserve, not the logical KV ceiling.

    The 4096-token Prefill executor allocates its current layer workspace from
    live free VRAM and temporarily shrinks/reuses the expert arena.  Treating
    that transient peak as permanent startup residency double-counted the same
    memory and rejected 20-GiB GPUs before Prefill could run.  KV and larger
    batches still expand dynamically after admission.
    """

    block = min(512, max(1, int(max_ctx)))
    return max(0.25, 8.75 * block / 4096.0)


def _gpu_startup_admission_margin_gb(
    *,
    architecture: str,
    reserve_gb: float,
    working_set_margin_gb: float,
    extreme_mode: bool,
) -> float:
    """Return the one fixed margin used by CUDA startup admission.

    Dynamic-workspace architectures grow KV and reuse their transient Prefill
    buffers after startup, so their hard admission floor is the configured
    physical reserve rather than a second copy of the working-set estimate.
    Extreme mode keeps its existing capacity policy.
    """

    if extreme_mode:
        return max(0.0, float(working_set_margin_gb))
    if architecture in {"dsv4", "qwen3_5_dense", "glm5_next"}:
        return max(0.0, float(reserve_gb))
    return max(float(reserve_gb), float(working_set_margin_gb), 0.0)


def _safe_expert_budget(*, limit_bytes: int, allocated_bytes: int,
                        expert_bytes: int, requested_bytes: int,
                        reserve_bytes: int, min_bytes: int = 2**29) -> int:
    """Cap expert VRAM from actual fixed allocations, not model-size estimates."""
    fixed_bytes = max(0, int(allocated_bytes) - int(expert_bytes))
    room = max(0, int(limit_bytes) - fixed_bytes - int(reserve_bytes))
    return max(int(min_bytes), min(int(requested_bytes), room))


def _use_short_reset_decode(pool, token_count: int) -> bool:
    """Use exact Decode when a pool declares batch Prefill disruptive.

    Bounded hybrid pools keep a much larger hot-expert arena for Decode than
    for long batch Prefill.  Repartitioning that arena for a tiny prompt costs
    more than executing the prompt through the ordinary exact Decode kernel
    and also destroys the cache needed by generation.  Full-resident HIP
    retains its established 16-token rule without changing CUDA full-resident
    or CPU scheduling.
    """
    limit = max(0, int(getattr(pool, "short_reset_decode_tokens", 0)))
    if (
        limit == 0
        and torch.version.hip is not None
        and bool(getattr(pool, "full_resident", False))
    ):
        limit = 16
    return 0 < int(token_count) <= limit


def _initial_expert_vram_request_gb(
    *,
    architecture: str,
    planning_free_gb: float,
    dense_estimate_gb: float,
    runtime_margin_gb: float,
    extreme: bool,
) -> float:
    """Give fixed expert pools the live post-Dense allocator ceiling.

    Every current CUDA packed pool builds its fixed arena *after* Dense,
    masks and static graphs are resident.  The pool then reads the allocator's
    real bytes/free space and subtracts the single public runtime reserve.
    Pre-subtracting a manifest Dense estimate and a second architecture margin
    here capped a 20-GiB card at a 2.44-GiB arena even though several GiB were
    still physically free.  That was especially destructive for DSV4: the
    arena held about 244 experts while one decode round needs 43*6=258, making
    the cyclic LRU miss every expert on every token.
    """

    del architecture, dense_estimate_gb, runtime_margin_gb, extreme
    return max(0.5, float(planning_free_gb))


def _dense_file_paths(
    model_dir: str,
    manifest: dict,
) -> tuple[str, ...]:
    """Resolve manifest-declared Dense files without opening tensor bodies."""
    files = manifest.get("dense_files")
    if files is None:
        files = [manifest.get("dense_file", "dense.safetensors")]
    dense_root = str(
        (manifest.get("nonexpert") or {}).get("path", "dense")
    ).strip("/\\")
    resolved = []
    for filename in files:
        value = str(filename).replace("/", os.sep)
        direct = os.path.join(model_dir, value)
        nested = os.path.join(model_dir, dense_root, value)
        resolved.append(direct if os.path.exists(direct) else nested)
    return tuple(os.path.abspath(path) for path in resolved)


def _trim_process_heap() -> None:
    """Return large transient Dense read buffers to the host OS when possible."""
    if os.name != "posix":
        return
    try:
        import ctypes

        libc = ctypes.CDLL(None)
        malloc_trim = getattr(libc, "malloc_trim", None)
        if malloc_trim is not None:
            malloc_trim(0)
    except (OSError, TypeError):
        pass


class _PrefillProgressMonitor:
    """长 prefill 周期性进度日志（vLLM 风格），避免静默期看起来像卡死。

    后台 daemon 线程每 ``CCCP_PREFILL_PROGRESS_INTERVAL`` 秒（默认 10）
    打印一次 model.pos 进度与瞬时速度；仅在非 quiet 且本次处理量不少于
    ``CCCP_PREFILL_PROGRESS_MIN``（默认 2048）时启动；
    ``CCCP_PREFILL_PROGRESS=0`` 整体关闭。
    """

    def __init__(self, engine: "Engine", total: int):
        self._engine = engine
        self._total = max(1, int(total))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "_PrefillProgressMonitor":
        try:
            interval = float(
                os.environ.get("CCCP_PREFILL_PROGRESS_INTERVAL", "10")
            )
            minimum = int(
                os.environ.get("CCCP_PREFILL_PROGRESS_MIN", "2048")
            )
        except ValueError:
            interval, minimum = 10.0, 2048
        if (
            getattr(self._engine, "quiet", False)
            or os.environ.get("CCCP_PREFILL_PROGRESS", "1") == "0"
            or self._total < minimum
            or interval <= 0
        ):
            return self
        model = self._engine.model
        started = time.perf_counter()
        last = [0, started, 0]  # pos, time, max_pos

        def progress_pos() -> int:
            # kimi 批量 prefill 按层更新 _prefill_progress（块级 pos 太粗）；
            # 其余模型回退 model.pos。只增不减，避免多块/重置产生负速度。
            current = max(
                int(getattr(model, "pos", 0)),
                int(getattr(model, "_prefill_progress", 0)),
            )
            last[2] = max(last[2], current)
            return last[2]

        last[0] = last[2] = progress_pos()
        total = self._total
        stop = self._stop

        def watch() -> None:
            while not stop.wait(interval):
                now = time.perf_counter()
                pos = progress_pos()
                delta_t = max(now - last[1], 1e-9)
                rate = (pos - last[0]) / delta_t
                last[0], last[1] = pos, now
                elapsed = now - started
                print(
                    f"[KV] prefill 进行中 {pos}/{total} tok "
                    f"({100.0 * pos / total:.0f}%)，"
                    f"已用 {elapsed:.0f}s，近段 {rate:.1f} tok/s",
                    flush=True,
                )

        self._thread = threading.Thread(
            target=watch,
            name="cccp-prefill-progress",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, *exc) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        return False


class Engine:
    """GLM-5.2-CCCP 的生成引擎（CPU / CUDA，内存显存自动适配）。

    cache_gb / vram_cache_gb 传 None 时自动计算：
      RAM 预算  = 可用内存 − (运行时 2GB + f32 常驻 4.5GB + KV cache + 安全 3GB)
      VRAM 预算 = 空闲显存 − (dense 常驻 ≈13.5GB + KV cache + 安全 1GB)
    显存不足以常驻 dense 时自动回退 CPU 模式并提示。
    共享显存防线（WDDM）：初始化时按真实空闲显存给 torch 分配器设置
    per-process 物理上限；专家池在该上限内保留 2GB 公共安全线，并根据
    架构与自动上下文额外留下 Attention、KV、Prefill 的真实计算工作集。
    """

    def __init__(
        self,
        model_dir: str,
        cache_gb: float | None = None,
        max_ctx: int = 2048,
        quiet: bool = False,
        device: str = "cpu",
        vram_cache_gb: float | None = None,
        tp_size: int = 1,
        dense_residency: str = "auto",
        extreme_mode: bool | None = None,
    ):
        import psutil
        t0 = time.time()
        self.quiet = quiet
        requested_extreme = extreme_mode
        if extreme_mode is True:
            from .extreme import configure_extreme_environment

            configure_extreme_environment()
        self.extreme_mode = bool(
            extreme_mode is True
            or os.environ.get("CCCP_EXTREME_MODE", "0") != "0"
        )
        self.auto_extreme_decision = None
        self.extreme_strategy = "disabled"
        extreme_archive = None
        if tp_size <= 0:
            raise ValueError("tp_size must be positive")
        self.tp_size = int(tp_size)
        if self.extreme_mode and (self.tp_size != 1 or device != "cuda"):
            raise ValueError("极限模式要求单卡 device='cuda', tp_size=1")
        if self.extreme_mode:
            dense_residency = "gpu"
        dense_residency = str(dense_residency).strip().lower()
        if dense_residency not in {"auto", "gpu", "ram"}:
            raise ValueError(
                "dense_residency must be 'auto', 'gpu', or 'ram'"
            )
        if dense_residency == "gpu" and device != "cuda":
            raise ValueError("dense_residency='gpu' requires device='cuda'")
        self.dense_residency = {
            "requested": dense_residency,
            "actual": "host",
            "host_mirror_bytes": 0,
        }
        ram_mirror = None
        self._vram_limit_bytes = 0
        self._vram_runtime_reserve_gb = 0.0
        # 架构判定（先读一次 cccp.json，供 RAM/VRAM 开销与模型分派共用）
        arch_hint = "glm"
        _manifest: dict = {}
        cccp_j = os.path.join(model_dir, "cccp.json")
        if os.path.exists(cccp_j):
            with open(cccp_j, "r", encoding="utf-8") as _f:
                _manifest = json.load(_f)
                _cfg = _manifest["config"]
                arch_hint = _model_adapter_architecture(_manifest)
        if (
            not self.extreme_mode
            and arch_hint != "qwen3_5_dense"
            and dense_residency != "ram"
            and requested_extreme is None
            and os.environ.get("CCCP_AUTO_EXTREME", "1") != "0"
        ):
            from .extreme import (
                configure_extreme_environment,
                detect_auto_extreme,
            )

            normal_reserve = float(
                os.environ.get("CCCP_RESIDENT_RESERVE_GB", "2")
            )
            self.auto_extreme_decision = detect_auto_extreme(
                model_dir,
                max_ctx=max_ctx,
                device=device,
                tp_size=self.tp_size,
                normal_ram_reserve_gib=normal_reserve,
                environment=os.environ,
            )
            if self.auto_extreme_decision.activate:
                configure_extreme_environment()
                self.extreme_mode = True
                dense_residency = "gpu"
                self.dense_residency["requested"] = "gpu"
                if not quiet:
                    decision = self.auto_extreme_decision
                    print(
                        "[cccp-auto] RAM 单侧安全容量不足，自动切换极限模式："
                        f"专家 {decision.expert_bytes / 2**30:.2f}GiB；"
                        f"转入 GPU {decision.spill_bytes / 2**30:.2f}GiB；"
                        f"GPU 专家余量 "
                        f"{decision.gpu_expert_capacity / 2**30:.2f}GiB",
                        flush=True,
                    )
        if self.extreme_mode:
            from .extreme import inspect_compact_projection_archive

            extreme_archive = inspect_compact_projection_archive(model_dir)
            self.extreme_strategy = "layered"
        # Tokenizer 是运行时硬依赖，必须在数百 GiB 权重加载之前验证并初始化。
        # 旧顺序在模型完整 preload 后才 import ``tokenizers``，一旦 Python
        # 环境缺包，会白白消耗数分钟加载时间和大量磁盘读。Kimi 继续使用其
        # 自身 tokenizer 适配，GLM/DeepSeek 使用标准 tokenizer.json。
        if arch_hint == "kimi_k3":
            from .kimi_tokenizer import KimiTokenizer

            prepared_tokenizer = KimiTokenizer(model_dir)
        else:
            from tokenizers import Tokenizer

            prepared_tokenizer = Tokenizer.from_file(
                os.path.join(model_dir, "tokenizer.json")
            )
        # RAM 开销按架构。普通 DSV4 允许 paged KV 按需增长；极限模式的
        # GPU-only 专家不可再收缩，所以必须在放置专家前为完整声明上下文预留。
        if arch_hint == "dsv4":
            if self.extreme_mode:
                from .capacity import dsv4_context_runtime_bytes

                kv_gb = (
                    dsv4_context_runtime_bytes(_cfg, max_ctx).total_bytes
                    / 2**30
                )
            else:
                kv_gb = 0.2
            ram_overhead = 2.0 + 2.1 + kv_gb + 3.0   # f32 2.1 + 安全 3（用户实测调优）
        elif arch_hint == "qwen3_5_dense":
            initial_ctx = min(max(1, int(max_ctx)), 4096)
            kv_gb = 0.6 + 0.00055 * initial_ctx
            ram_overhead = 2.0 + kv_gb + 3.0
        else:
            # GLM：MLA 潜变量 KV（默认开）≈0.09MB/token + 吸收矩阵 2.3GB；
            # CCCP_LATENT_KV=0 回退逐头全量 K/V ≈5MB/token。KV 按需增长，
            # 启动预算不能把模型声明的逻辑上限当作已分配显存。
            kv_gb = _glm_startup_kv_gb(
                max_ctx,
                latent=os.environ.get("CCCP_LATENT_KV", "1") != "0",
            )
            ram_overhead = 2.0 + 4.5 + kv_gb + 6.0  # 安全余量 6GB
        if arch_hint == "kimi_k3":
            initial_ctx = min(max(0, int(max_ctx)), 4096)
            kv_gb = 0.5 + 0.027 * initial_ctx / 1024
            ram_overhead = 2.0 + kv_gb + 6.0
        if self.extreme_mode:
            from .extreme import effective_available_memory_bytes

            avail_ram = effective_available_memory_bytes() / 2**30
        else:
            avail_ram = psutil.virtual_memory().available / 2**30
        auto_ram = (
            max(0.0, avail_ram - 1.0)
            if self.extreme_mode
            else max(2.0, avail_ram - ram_overhead)
        )

        dev = device
        auto_vram = vram_cache_gb
        extreme_fixed_gpu_bytes = 0
        if device == "cuda":
            if not torch.cuda.is_available():
                if dense_residency == "gpu":
                    raise RuntimeError(
                        "Dense 要求 GPU 常驻，但当前 CUDA 不可用"
                    )
                print("[cccp] 无 CUDA，回退 CPU 模式", flush=True)
                dev = "cpu"
            else:
                if self.tp_size > torch.cuda.device_count():
                    raise RuntimeError(
                        f"tp={self.tp_size} but only "
                        f"{torch.cuda.device_count()} CUDA devices are visible"
                    )
                if (
                    self.tp_size > 1
                    and (
                        arch_hint == "glm"
                        or (
                            arch_hint == "kimi_k3"
                            and os.environ.get(
                                "CCCP_TP_PACKED_HYBRID",
                                "0",
                            )
                            != "0"
                        )
                    )
                    and os.environ.get("CCCP_RAM_MIRROR", "0") == "1"
                ):
                    from .ramcache import ModelRamMirror

                    ram_mirror = ModelRamMirror(
                        model_dir,
                        exclude_paths=_dense_file_paths(
                            model_dir,
                            _manifest,
                        ),
                    )
                    ram_mirror.start()
                visible_ranks = min(
                    max(1, self.tp_size),
                    torch.cuda.device_count(),
                )
                rank_memory = []
                for rank in range(visible_ranks):
                    with torch.cuda.device(rank):
                        rank_memory.append(torch.cuda.mem_get_info(rank))
                free_v = min(item[0] for item in rank_memory) / 2**30
                total_v = min(item[1] for item in rank_memory) / 2**30
                # 单一显存预留：WDDM 下专家槽、Attention/KV 与临时工作区
                # 必须共同留在当前物理可用显存内。分配器硬上限取启动时的
                # 实际空闲显存；专家池再在这个上限内只扣一次 reserve_gb。
                # 旧实现先在进程上限外扣一次、专家池内又扣一次，3 GiB
                # 配置实际空出了约 6 GiB，显著降低热专家命中率。
                reserve_gb = float(os.environ.get("CCCP_VRAM_RESERVE_GB", "1"))
                explicit_vram_limit_gb = max(
                    0.0,
                    float(os.environ.get("CCCP_VRAM_LIMIT_GB", "0")),
                )
                extreme_vram_cap_gb = max(
                    0.0,
                    float(os.environ.get("CCCP_EXTREME_VRAM_CAP_GB", "0")),
                )
                vram_cap_gb = (
                    explicit_vram_limit_gb
                    if explicit_vram_limit_gb > 0
                    else extreme_vram_cap_gb
                )
                planning_free_v = (
                    min(free_v, vram_cap_gb)
                    if vram_cap_gb > 0
                    else free_v
                )
                fractions = []
                limits = []
                for rank, (free_bytes, total_bytes) in enumerate(
                    rank_memory
                ):
                    process_available = free_bytes
                    minimum_fraction = 0.10
                    if vram_cap_gb > 0:
                        process_available = min(
                            process_available,
                            int(vram_cap_gb * 2**30),
                        )
                        minimum_fraction = 0.01
                    fraction = max(
                        minimum_fraction,
                        min(
                            0.99,
                            (
                                process_available
                            )
                            / total_bytes,
                        ),
                    )
                    torch.cuda.set_per_process_memory_fraction(
                        fraction,
                        rank,
                    )
                    fractions.append(fraction)
                    limits.append(int(fraction * total_bytes))
                frac = min(fractions)
                self._vram_limit_bytes = min(limits)
                if not quiet:
                    print(f"[cccp] 显存适配: 物理 {total_v:.1f}GB / 空闲 {free_v:.1f}GB → "
                          f"本进程物理上限 {frac * total_v:.1f}GB"
                          f"（运行时总预留 {reserve_gb:.2f}GB，仅扣一次）",
                          flush=True)
                    if vram_cap_gb > 0:
                        print(
                            "[cccp] 进程显存硬上限："
                            f"{vram_cap_gb:.2f}GiB",
                            flush=True,
                        )
                # dense 常驻需求按架构：GLM ≈13.5GB（int4 9.2 + lm_head 3.8 + router 0.5），
                # DSV4 ≈10.5GB（dense 一次性反量化 bf16 常驻 ≈7.2 + head bf16 ≈1.1 + DSpark ≈2.2；
                # bf16 消除逐调用反量化，是 attn 段的关键提速；+ KV + 安全 2GB（悬崖余量）
                dense_need = _dense_need_gb(model_dir, arch_hint, kv_gb)
                if self.extreme_mode and extreme_archive is not None:
                    from .extreme import (
                        EXTREME_GPU_LOAD_WORKSPACE_GIB,
                        GIB,
                        choose_extreme_strategy,
                    )

                    dense_resident_need = max(
                        0.0,
                        dense_need - EXTREME_GPU_LOAD_WORKSPACE_GIB,
                    )

                    self.extreme_strategy = choose_extreme_strategy(
                        compact_expert_bytes=extreme_archive.expert_bytes,
                        fixed_gpu_bytes=int(dense_resident_need * GIB),
                        gpu_limit_bytes=self._vram_limit_bytes,
                    )
                    if self.extreme_strategy == "layered":
                        # The hybrid pool turns this estimate into a real CUDA
                        # allocation before placing any expert. Dense later
                        # replaces that allocation, so capacity is proven
                        # without keeping a second full weight copy.
                        extreme_fixed_gpu_bytes = int(
                            dense_resident_need * GIB
                        )
                    if self.extreme_strategy == "full-gpu":
                        # Reuse the same model-independent packed resident
                        # pool as profile=resident.  Extreme remains a capacity
                        # policy; it must not force RAM/H2D when the complete
                        # compact archive already fits one GPU.
                        os.environ["CCCP_PACKED_FULL_GPU"] = "1"
                    if not quiet:
                        print(
                            "[cccp-extreme] 公共紧凑归档："
                            f"{len(extreme_archive.layers)} 层 / "
                            f"{extreme_archive.expert_bytes / GIB:.2f}GiB；"
                            f"策略={self.extreme_strategy}",
                            flush=True,
                        )
                # 余量按架构：GLM 的 dense 常驻 + 吸收矩阵(2.1GB) + 瞬态反量化块贴近
                # 分配器硬上限，实测 1-2GB 边缘余量仍会在 decode 中 OOM；
                # 公共物理安全线为 2GB，另保留不可分块架构的真实工作集。
                #（GLM 专家本就走 RAM/磁盘流式，显存缓存价值低）
                margin = 0.0 if self.extreme_mode else (
                    3.0 if arch_hint == "glm"
                    else (
                        2.0 + _dsv4_prefill_workspace_reserve_gb(max_ctx)
                        if arch_hint == "dsv4"
                        else 2.0
                    )
                )
                if (
                    arch_hint == "kimi_k3"
                    and dense_residency == "auto"
                    and planning_free_v < dense_need + margin
                ):
                    # Kimi has a real RAM-Dense/CUDA-packed-MoE path.  Keep
                    # CUDA active on consumer cards instead of silently
                    # changing the entire launch to CPU inference.  This
                    # reserve remains outside the expert arena for attention,
                    # batch expansion and allocator/private overhead.
                    dense_residency = "ram"
                    margin = 10.0
                    if not quiet:
                        print(
                            "[cccp] Kimi Dense 无法常驻当前显存；自动使用 "
                            "RAM Dense + CUDA packed MoE，完整专家集保持不变",
                            flush=True,
                        )
                # Admission still verifies the architecture/context working
                # set below, but the fixed expert arena must deduct only the
                # launcher's one public physical safety line.  Charging
                # ``margin`` here a second time left a 20-GiB 3090 with only a
                # 3.10-GiB/310-slot arena even though Prefill already sizes its
                # expanded-expert scratch from live free VRAM and splits it
                # when necessary.  Keep the requested one-GiB total reserve;
                # dynamic workspaces remain bounded by their live planners.
                live_chunked_workspace = arch_hint in {
                    "dsv4", "qwen3_5_dense", "glm5_next"
                }
                runtime_headroom_gb = (
                    reserve_gb
                    if live_chunked_workspace
                    else max(reserve_gb, margin)
                )
                os.environ["CCCP_VRAM_HEADROOM_GB"] = str(
                    runtime_headroom_gb
                )
                self._vram_safety_reserve_gb = reserve_gb
                self._vram_runtime_headroom_gb = runtime_headroom_gb
                admission_margin_gb = _gpu_startup_admission_margin_gb(
                    architecture=arch_hint,
                    reserve_gb=reserve_gb,
                    working_set_margin_gb=margin,
                    extreme_mode=self.extreme_mode,
                )
                if not quiet:
                    print(
                        "[cccp-vram-plan] phase=runtime-headroom "
                        f"safety_reserve={reserve_gb:.2f}GiB "
                        f"dynamic_workspace={'live-chunked' if live_chunked_workspace else 'fixed-headroom'} "
                        f"total_headroom={runtime_headroom_gb:.2f}GiB "
                        f"architecture={arch_hint} max_ctx={max_ctx}",
                        flush=True,
                    )
                dense_gpu_need = (
                    0.0 if dense_residency == "ram" else dense_need
                )
                if planning_free_v < dense_gpu_need + admission_margin_gb:
                    if dense_residency == "gpu":
                        requirement = (
                            "GPU Dense 与完整高速 Prefill"
                            if arch_hint == "dsv4"
                            else "Dense"
                        )
                        raise RuntimeError(
                            f"{requirement} 要求 GPU 常驻，但空闲显存 "
                            f"{planning_free_v:.1f}GB < 需要 "
                            f"{dense_gpu_need + admission_margin_gb:.1f}GB"
                        )
                    print(f"[cccp] 显存不足（空闲 {planning_free_v:.1f}GB < 需要 {dense_gpu_need + admission_margin_gb:.1f}GB），"
                          f"回退 CPU 模式", flush=True)
                    dev = "cpu"
                elif vram_cache_gb is None:
                    # 显存余量 2GB（DSV4）：顶到 100% 会触发分配器 cudaFree+同步回收（悬崖 ×4）
                    # 极限模式把完整可用余量交给 packed pool；pool 会先用
                    # extreme_fixed_gpu_bytes 建立真实 CUDA 占位，再从剩余空间
                    # 放置专家。Dense 流式加载时直接复用占位块，不会重复扣除。
                    auto_vram = _initial_expert_vram_request_gb(
                        architecture=arch_hint,
                        planning_free_gb=planning_free_v,
                        dense_estimate_gb=dense_gpu_need,
                        runtime_margin_gb=admission_margin_gb,
                        extreme=self.extreme_mode,
                    )
        if cache_gb is None:
            cache_gb = auto_ram
        if vram_cache_gb is None:
            vram_cache_gb = auto_vram if dev == "cuda" else 0.0
        if not quiet:
            if dev == "cuda" and self.tp_size > 1 and arch_hint == "glm":
                print(
                    f"[cccp] 内存适配: 可用RAM {avail_ram:.1f}GB；"
                    f"TP={self.tp_size} 优先全显存专家（运行期专家 RAM/H2D=0）；"
                    f"容量不足时自动回退 RAM {cache_gb:.1f}GB / "
                    f"主卡显存 {vram_cache_gb:.1f}GB",
                    flush=True,
                )
            else:
                print(
                    f"[cccp] 内存适配: 可用RAM {avail_ram:.1f}GB → "
                    f"专家缓存 {cache_gb:.1f}GB"
                    + (
                        f"；显存缓存 {vram_cache_gb:.1f}GB"
                        if dev == "cuda"
                        else ""
                    ),
                    flush=True,
                )

        if ram_mirror is not None and not ram_mirror.wait_and_activate():
            ram_mirror = None

        retry_vram_cache_gb = None
        try:
            self.model, self.arch = _make_model(
                model_dir,
                cache_gb=cache_gb,
                max_ctx=max_ctx,
                device=dev,
                vram_cache_gb=vram_cache_gb or 4.0,
                tp_size=self.tp_size,
                extreme_fixed_gpu_bytes=extreme_fixed_gpu_bytes,
                dense_residency=dense_residency,
            )
            self.model.preload()
            if self.arch == "kimi_k3" and hasattr(self.model, "preload_vision"):
                self.model.preload_vision()
        except torch.cuda.OutOfMemoryError as oom_error:
            if self.extreme_mode:
                raise RuntimeError(
                    "极限模式显存不足：Dense、GPU 专家层、Top-K staging 与当前"
                    f" max_ctx={max_ctx} 无法同时容纳。请降低 --max-ctx、"
                    "关闭其他显存进程或换用更小模型。底层分配："
                    f"{oom_error}"
                ) from oom_error
            if dev != "cuda" or (vram_cache_gb or 4.0) <= 1.0:
                raise
            # Leave the exception handler before retrying.  Its traceback keeps
            # preload frames alive; retrying inside this block used to retain
            # the failed model's dense weights and made the second attempt OOM
            # as well.
            self.model = None
            retry_vram_cache_gb = max(0.5, (vram_cache_gb or 4.0) / 2)
            print(f"[cccp] 显存触顶（硬上限保护），显存缓存降至 "
                  f"{retry_vram_cache_gb:.1f}GB 重试",
                  flush=True)
        if retry_vram_cache_gb is not None:
            import gc as _gc
            _gc.collect()
            torch.cuda.empty_cache()
            self.model, self.arch = _make_model(
                model_dir,
                cache_gb=cache_gb,
                max_ctx=max_ctx,
                device=dev,
                vram_cache_gb=retry_vram_cache_gb,
                tp_size=self.tp_size,
                extreme_fixed_gpu_bytes=0,
                dense_residency=dense_residency,
            )
            self.model.preload()
            if self.arch == "kimi_k3" and hasattr(self.model, "preload_vision"):
                self.model.preload_vision()
        routed_vq = getattr(self.model, "routed_vq", None)
        dense_codebook_stats = getattr(
            self.model,
            "codebook_stats",
            None,
        )
        full_resident = bool(
            routed_vq.full_resident
            if routed_vq is not None
            else getattr(dense_codebook_stats, "full_resident", False)
        )
        compact_cpu_resident = bool(
            dev == "cpu"
            and (
                routed_vq.compact_full_resident
                if routed_vq is not None
                else getattr(
                    dense_codebook_stats,
                    "compact_full_resident",
                    False,
                )
            )
        )
        if dev == "cuda" and dense_residency != "ram":
            released, dense_paths = (
                self.model.store.release_dense_ram_blob()
            )
            mirror_released = (
                ram_mirror.release_paths(dense_paths)
                if ram_mirror is not None
                else 0
            )
            import gc as _dense_gc

            _dense_gc.collect()
            _trim_process_heap()
            self.dense_residency = {
                "requested": dense_residency,
                "actual": "gpu-only",
                "host_mirror_bytes": max(released, mirror_released),
            }
            if not quiet:
                print(
                    "[cccp] Dense 驻留：GPU-only；"
                    "CPU 仅保留启动期流式缓冲，运行期源镜像已释放"
                    + (
                        f" {max(released, mirror_released) / 2**30:.2f}GB"
                        if max(released, mirror_released)
                        else ""
                    ),
                    flush=True,
                )
        if (
            ram_mirror is not None
            and routed_vq is not None
            and routed_vq.retains_store_ram_blobs
        ):
            self._ram_mirror = ram_mirror
            ram_mirror = None
            if not quiet:
                print(
                    "[cccp] RAM 镜像直接作为 packed 专家常驻存储；"
                    "不建立第二份专家索引",
                    flush=True,
                )
        if ram_mirror is not None:
            self.model.store.release_ram_blobs()
            released = ram_mirror.release()
            import gc as _gc

            _gc.collect()
            if not quiet:
                print(
                    f"[cccp] RAM staging 已释放 "
                    f"{released / 2**30:.2f}GB；推理期不保留模型文件镜像",
                    flush=True,
                )
        if dev == "cuda":
            self._vram_runtime_reserve_gb = float(os.environ.get(
                "CCCP_VRAM_HEADROOM_GB",
                os.environ.get("CCCP_VRAM_RESERVE_GB", "1"),
            ))
        if dev == "cuda" and not full_resident:
            self._cap_expert_cache(
                self._vram_runtime_reserve_gb,
                "Attention/KV/Prefill 实际工作区与统一安全线",
            )
        # 动态显存监测：滞回调节专家显存缓存预算（防其他进程抢占/碎片化/小显卡
        # 顶满物理显存触发共享显存换页）；CCCP_VRAM_WATCH=0 关闭
        if (
            dev == "cuda"
            and not full_resident
            and not self.extreme_mode
            and os.environ.get("CCCP_VRAM_WATCH", "1") != "0"
        ):
            if routed_vq is not None:
                self._vwatch = routed_vq.start_vram_watch(
                    low_gb=self._vram_runtime_reserve_gb,
                    high_gb=self._vram_runtime_reserve_gb + 1.0,
                    quiet=quiet,
                )
            if routed_vq is not None and self._vwatch is None and not quiet:
                runtime_stats = routed_vq.stats()
                print(
                    "[cccp-vram-plan] monitor=disabled "
                    "reason=fixed-address-arena "
                    f"arena={runtime_stats.gpu_arena_bytes / 2**30:.2f}GiB；"
                    "容量由进程物理上限与单一总预留固定，运行期不再破坏性缩容",
                    flush=True,
                )
        self.tok = prepared_tokenizer
        if self.arch == "glm5_next" and os.environ.get("CCCP_PRELOAD_VISION", "0") == "1":
            seconds = self.model.preload_vision()
            if not quiet:
                print(
                    f"[cccp-glm5-next] 视觉塔按配置预载完成：{seconds:.3f}s",
                    flush=True,
                )
        gc = os.path.join(model_dir, "generation_config.json")
        self.eos = DEFAULT_EOS
        if os.path.exists(gc):
            with open(gc, "r", encoding="utf-8") as f:
                e = json.load(f).get("eos_token_id", DEFAULT_EOS)
                self.eos = [e] if isinstance(e, int) else list(e)
        self.quiet = quiet
        self._cache_ids: list[int] | None = None   # KV 中已缓存的 token 前缀（多轮复用）
        self._cache_media_digest: str | None = None
        self._cache_media_slots: tuple[dict[str, object], ...] = ()
        self._active_media_slots: tuple[dict[str, object], ...] = ()
        self._cache_via_spec = False   # 缓存是否由投机路径建立（DSpark 环覆盖一致才可直接复用）
        self._kv_baseline: _DSV4Baseline | None = None
        self.last_kv_stats: KVPrefillStats | None = None
        self._kv_prefill_events = None
        if dev == "cuda" and dense_residency == "ram":
            self.dense_residency = {
                "requested": "ram",
                "actual": "ram+cuda",
                "host_mirror_bytes": 0,
            }
        if not quiet:
            if dev == "cuda":
                runtime_stats = (
                    routed_vq.stats()
                    if routed_vq is not None
                    else dense_codebook_stats
                )
                operator = str(
                    getattr(self.model, "packed_operator_name", "") or
                    "架构公共 CUDA 路径"
                )
                prefetch_enabled = False
                prefetch_probe = getattr(self.model, "_prefetch_enabled", None)
                if callable(prefetch_probe):
                    prefetch_enabled = bool(prefetch_probe())
                print(
                    "[cccp-cuda-audit] "
                    f"Dense={self.dense_residency['actual']}；"
                    f"专家计算={operator}；"
                    f"RAM专家={getattr(runtime_stats, 'host_expert_bytes', 0) / 2**30:.2f}GiB；"
                    f"锁页RAM={getattr(runtime_stats, 'host_pinned_bytes', getattr(runtime_stats, '_host_pinned_bytes', 0)) / 2**30:.2f}GiB；"
                    f"显存热缓存={getattr(runtime_stats, 'gpu_arena_bytes', 0) / 2**30:.2f}GiB；"
                    f"跨层预取={'启用' if prefetch_enabled else '关闭'}",
                    flush=True,
                )
                # Windows Task Manager reports CUDA page-locked host pages
                # in the "shared GPU memory" counter.  Those pages are the
                # DMA source for RAM-resident packed experts; they are not
                # WDDM-evicted VRAM and are not disk offload.  Print the CUDA
                # allocator figures beside the pin count so a launcher user
                # can distinguish the two without guessing from Task Manager.
                try:
                    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
                    allocated_bytes = torch.cuda.memory_allocated(0)
                    reserved_bytes = torch.cuda.memory_reserved(0)
                    print(
                        "[cccp-vram] "
                        f"CUDA专用显存：已分配 {allocated_bytes / 2**30:.2f}GiB；"
                        f"分配器保留 {reserved_bytes / 2**30:.2f}GiB；"
                        f"驱动可用 {free_bytes / 2**30:.2f}/"
                        f"{total_bytes / 2**30:.2f}GiB；"
                        "未占满部分保留给 Attention、KV、Prefill 与 WDDM",
                        flush=True,
                    )
                except (RuntimeError, TypeError, ValueError):
                    pass
                host_pinned_bytes = int(
                    getattr(
                        runtime_stats,
                        "host_pinned_bytes",
                        getattr(runtime_stats, "_host_pinned_bytes", 0),
                    )
                    or 0
                )
                if os.name == "nt" and host_pinned_bytes > 0:
                    print(
                        "[cccp-vram] Windows 的“共享 GPU 内存”可能包含 "
                        f"{host_pinned_bytes / 2**30:.2f}GiB 已锁页专家 RAM；"
                        "这是异步 DMA 缓存，不代表专用显存溢出，也不是磁盘卸载",
                        flush=True,
                    )
            if full_resident and self.arch == "qwen3_5_dense":
                print(
                    f"[cccp] 模型加载完成（{time.time() - t0:.1f}s）："
                    f"Qwen3.5 Dense VQ 全显存常驻 "
                    f"{dense_codebook_stats.gpu_storage_bytes / 2**30:.2f}GB；"
                    "动态专家=无；运行期权重 H2D=0",
                    flush=True,
                )
            elif full_resident:
                runtime_stats = routed_vq.stats()
                print(
                    f"[cccp] 模型加载完成（{time.time() - t0:.1f}s）："
                    f"TP={self.model.effective_tp_size} routed experts "
                    f"{runtime_stats.gpu_storage_bytes / 2**30:.2f}GB 全显存常驻，"
                    f"主机专家 {runtime_stats.host_expert_bytes / 2**30:.2f}GB",
                    flush=True,
                )
            elif compact_cpu_resident and self.arch == "qwen3_5_dense":
                print(
                    f"[cccp] 模型加载完成（{time.time() - t0:.1f}s）："
                    "Qwen3.5 Dense VQ CPU Q4 执行映像已常驻；动态专家=无",
                    flush=True,
                )
            elif compact_cpu_resident:
                runtime_stats = routed_vq.stats()
                print(
                    f"[cccp] 模型加载完成（{time.time() - t0:.1f}s）："
                    f"CPU 专家执行镜像 {runtime_stats.host_expert_bytes / 2**30:.2f}GB "
                    f"全量常驻；cpu_compile="
                    f"{runtime_stats.cpu_compile_mode}；"
                    f"expanded_index_bytes="
                    f"{runtime_stats.expanded_index_bytes}",
                    flush=True,
                )
            elif dev == "cuda" and dense_residency == "ram":
                print(
                    "[cccp] Dense 驻留：RAM 紧凑权重直接计算；"
                    "CUDA 仅运行已注册的异构算子与固定工作区",
                    flush=True,
                )
            else:
                extreme_detail = ""
                if self.extreme_mode and routed_vq is not None:
                    ram_layers = len(routed_vq.extreme_ram_layers)
                    gpu_layers = len(routed_vq.extreme_gpu_layers)
                    ratio = routed_vq.extreme_storage_ratio
                    extreme_detail = (
                        f"；极限常驻 RAM={ram_layers}层/GPU={gpu_layers}层"
                        f"/紧凑开销={ratio:.3f}x"
                    )
                print(
                    f"[cccp] 模型加载完成（{time.time() - t0:.1f}s）"
                    f"专家缓存预算 {cache_gb:.0f}GB"
                    f"{extreme_detail}",
                    flush=True,
                )

    def _cap_expert_cache(self, reserve_gb: float, reason: str) -> int | None:
        """Immediately enforce a cache ceiling within the allocator hard limit."""
        routed_vq = getattr(getattr(self, "model", None), "routed_vq", None)
        if (
            routed_vq is None
            or routed_vq.full_resident
            or routed_vq.fixed_extreme_residency
            or routed_vq.manages_per_rank_budget
            or routed_vq.cache_budget is None
            or not self._vram_limit_bytes
        ):
            return None
        allocated = torch.cuda.memory_allocated()
        runtime_stats = routed_vq.stats()
        expert_storage = (
            runtime_stats.gpu_storage_bytes or runtime_stats.bytes
        )
        new_budget = _safe_expert_budget(
            limit_bytes=self._vram_limit_bytes,
            allocated_bytes=allocated,
            expert_bytes=expert_storage,
            requested_bytes=routed_vq.cache_budget,
            reserve_bytes=int(reserve_gb * 2**30),
        )
        old_budget = routed_vq.cache_budget
        fixed_gb = max(0, allocated - expert_storage) / 2**30
        if not self.quiet:
            print(
                "[cccp-vram-plan] phase=post-load-cap "
                f"process_limit={self._vram_limit_bytes / 2**30:.2f}GiB "
                f"allocated={allocated / 2**30:.2f}GiB "
                f"fixed_without_experts={fixed_gb:.2f}GiB "
                f"arena={expert_storage / 2**30:.2f}GiB "
                f"requested={old_budget / 2**30:.2f}GiB "
                f"safety_reserve={getattr(self, '_vram_safety_reserve_gb', reserve_gb):.2f}GiB "
                f"runtime_headroom={reserve_gb:.2f}GiB "
                f"selected={new_budget / 2**30:.2f}GiB",
                flush=True,
            )
        if new_budget < old_budget:
            arena_bytes = runtime_stats.gpu_arena_bytes
            allocated_before = allocated
            resized_pair = (
                routed_vq.resize_gpu_arenas(new_budget)
                if arena_bytes > new_budget
                else None
            )
            resized = resized_pair is not None
            if resized_pair is not None:
                old_arena, new_arena = resized_pair
            else:
                routed_vq.trim_to(new_budget)
                old_arena = new_arena = arena_bytes
            torch.cuda.empty_cache()
            allocated_after = torch.cuda.memory_allocated()
            if (
                resized
                and old_arena > new_arena
                and allocated_after >= allocated_before
            ):
                raise RuntimeError(
                    "expert arena budget shrank without releasing CUDA allocations: "
                    f"arena {old_arena / 2**30:.2f}->{new_arena / 2**30:.2f} GiB, "
                    f"allocated {allocated_before / 2**30:.2f}->"
                    f"{allocated_after / 2**30:.2f} GiB"
                )
            if not self.quiet:
                detail = (
                    f"；arena {old_arena / 2**30:.1f}→{new_arena / 2**30:.1f}GB"
                    f"；allocated {allocated_before / 2**30:.1f}→"
                    f"{allocated_after / 2**30:.1f}GB"
                    if resized
                    else ""
                )
                print(
                    f"[cccp] 显存缓存安全封顶: {old_budget / 2**30:.1f}GB"
                    f" → {new_budget / 2**30:.1f}GB"
                    f"（常驻 {fixed_gb:.1f}GB + {reason} {reserve_gb:.1f}GB）",
                    f"{detail}",
                    flush=True,
                )
        watcher = getattr(self, "_vwatch", None)
        if watcher is not None:
            watcher.max_budget = min(watcher.max_budget, new_budget)
        return new_budget

    def _with_kv_capacity_retry(self, fn, *args, committed: int = 0, **kwargs):
        """Free expert VRAM and retry one transactional DSV4 page reservation."""
        from .dsv4cache import ContextCapacityError

        try:
            return fn(*args, **kwargs)
        except ContextCapacityError:
            self._cap_expert_cache(
                max(
                    1.0,
                    float(getattr(self, "_vram_runtime_reserve_gb", 2.0)),
                ),
                "KV cache 扩容",
            )
            try:
                return fn(*args, **kwargs)
            except ContextCapacityError as final:
                final.committed = committed
                raise

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)

    def new_decode_stream(self, *, skip_special_tokens: bool = False):
        """Create a stateful tokenizer stream for exact incremental decoding."""
        factory = getattr(self.tok, "new_decode_stream", None)
        if factory is not None:
            return factory(
                skip_special_tokens=skip_special_tokens,
            )
        from tokenizers.decoders import DecodeStream

        return DecodeStream(skip_special_tokens=skip_special_tokens)

    @torch.no_grad()
    def warmup_multimodal(self) -> dict[str, float]:
        """Explicitly warm GLM-5.3 image and resident Prefill fast paths.

        The method is never called by default. It uses an in-memory synthetic
        image, then clears every request/KV identity before serving traffic.
        """
        if self.arch != "glm5_next":
            raise RuntimeError("multimodal warmup is only available for glm5_next")
        from PIL import Image
        from .glm5_next_multimodal import (
            expand_image_token_ids,
            prepare_image,
        )

        prepared = prepare_image(Image.new("RGB", (1664, 933), (127, 127, 127)))
        prompt = (
            "[gMASK]<sop><|user|>"
            "<|begin_of_image|><|image|><|end_of_image|>"
            "请描述图片。\n<|assistant|><think></think>"
        )
        ids = expand_image_token_ids(
            self.encode(prompt),
            image_token_id=int(self.model._outer_config.image_token_id),
            token_counts=(prepared.token_count,),
        )
        self.reset()
        started = time.perf_counter()
        self.model.forward_multimodal_prefill(
            ids,
            pixel_values=prepared.pixel_values,
            image_grid_thw=torch.tensor(
                [prepared.grid_thw],
                dtype=torch.long,
            ),
        )
        if self.model.device.type == "cuda":
            torch.cuda.synchronize(self.model.device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        timings = dict(getattr(self.model, "last_multimodal_timings", None) or {})
        timings["warmup_total_ms"] = elapsed_ms
        timings["warmup_tokens"] = float(len(ids))
        self.reset()
        if not self.quiet:
            print(
                "[cccp-glm5-next] 多模态热身完成："
                f"tokens={len(ids)} total={elapsed_ms:.1f}ms "
                f"vision={timings.get('vision_forward_ms', 0.0):.1f}ms "
                f"language={timings.get('language_prefill_ms', 0.0):.1f}ms",
                flush=True,
            )
        return timings

    def reset(self) -> None:
        if self.arch == "glm5_next":
            # GLM owns an additional request-local media state. Its full reset
            # clears that state and restores the fixed text TokenGraph after an
            # image turn or multimodal startup warmup.
            self.model.reset()
        else:
            self.model.reset_kv()
        self._cache_ids = None
        self._cache_media_digest = None
        self._cache_media_slots = ()
        self._active_media_slots = ()
        self._cache_via_spec = False
        self._kv_baseline = None
        self.last_kv_stats = None
        self._kv_prefill_events = None
        dsp = getattr(self, "_dsp", None)
        if dsp is not None:
            dsp.reset()

    # ---- 多轮 KV 复用（省掉历史重 prefill）----
    def _kv_prefix_len(self, ids: list[int]) -> int:
        """上轮缓存的 token 序列（prompt+回复）仍是本轮 prompt 的严格前缀时返回其长度，
        调用方只需增量 prefill 后缀；否则 0（全量重置重跑）。
        逐 token id 精确比对，正确性不依赖 tokenizer decode→encode 往返稳定性；
        think 开启时思维链不回喂、前缀必然失配 → 自动回退全量（符合官方模板）。"""
        cached = getattr(self, "_cache_ids", None)
        if cached and len(cached) < len(ids) and ids[:len(cached)] == cached:
            return len(cached)
        return 0

    def _prefill_glm_suffix(
        self,
        ids: list[int],
        skip: int,
        *,
        media_state: object = None,
        media_slots: object = (),
    ) -> torch.Tensor:
        """Prefill a GLM/Kimi suffix through the public batched executor."""
        suffix = ids[skip:]
        if media_state is not None and getattr(self, "arch", "glm") == "kimi_k3":
            hidden = self.model.forward_hidden(
                suffix,
                media_state=media_state,
                media_slots=media_slots,
            )
            return self.model.logits_of(hidden[-1:]).squeeze(0)
        if media_state is not None and getattr(self, "arch", "glm") == "glm5_next":
            if skip != 0:
                # The matching image KV is already live; only replay the new
                # text suffix. Do not run the visual tower a second time.
                return self.model.forward(suffix)
            return self.model.forward_multimodal_prefill(
                ids,
                pixel_values=media_state["pixel_values"],
                image_grid_thw=media_state["image_grid_thw"],
            )
        return self.model.forward(suffix)

    def _media_cache_matches(
        self,
        media_digest: str | None,
        media_slots: object,
        media_state: object = None,
    ) -> bool:
        # A newly prepared GLM request naturally carries a fresh media_state.
        # GLM image KV reuse is keyed by stable ordered media identity/layout,
        # not Python object identity of decoded tensors. Kimi's media cache
        # still requires the historical no-fresh-state contract: its suffix
        # path uses full-prompt media slots and cannot replay them on suffix IDs.
        cached_digest = getattr(self, "_cache_media_digest", None)
        cached_slots = getattr(self, "_cache_media_slots", ())
        identity_matches = (
            media_digest == cached_digest
            and tuple(media_slots or ()) == tuple(cached_slots or ())
        )
        if getattr(self, "arch", "glm") == "glm5_next":
            return identity_matches
        return identity_matches and media_state is None

    def _prepare_glm_prompt(
        self,
        ids: list[int],
        *,
        media_digest: str | None = None,
        media_slots: object = (),
        media_state: object = None,
    ) -> torch.Tensor:
        """Prepare GLM/Kimi prompts and expose exact-prefix reuse metrics.

        Kimi's KDA state is recurrent rather than a conventional prefix-cache
        object.  When the previous canonical token sequence is an exact
        prefix, retaining the live model state and evaluating only the suffix
        is the cache reuse operation.  Report that path through the same
        ``KVPrefillStats`` contract used by DSV4 without changing its math.
        """
        started = time.perf_counter()
        live = getattr(self, "_cache_ids", None)
        lcp = _token_lcp(live, ids)
        media_matches = self._media_cache_matches(
            media_digest, media_slots, media_state
        )
        skip = self._kv_prefix_len(ids) if media_matches else 0
        if (
            skip
            and skip >= len(ids)
            and getattr(self, "arch", "glm") == "glm"
        ):
            # KV 与 prompt 完全一致(同进程二次 generate 同一 prompt):
            # 后缀为空会让层栈/argmax 空批连环崩溃——回退一个 token
            # 重放以恢复 next-token logits(Kimi 的 KDA 状态无安全
            # 截断,不适用;第三十轮实证)。
            truncate = getattr(self.model, "truncate_kv", None)
            if callable(truncate):
                self.model.truncate_kv(skip - 1)
                skip -= 1
        if skip:
            mode = "exact-prefix"
            reason = (
                "live-kda-kv-prefix"
                if getattr(self, "arch", "glm") == "kimi_k3"
                else "live-prefix"
            )
        elif (
            getattr(self, "arch", "glm") == "glm"
            and live
            and media_matches
            and 0 < lcp < len(live)
            and callable(getattr(self.model, "truncate_kv", None))
        ):
            # Re-applying a GLM chat template can retokenize the boundary
            # between the previous assistant reply and the next user turn.
            # The resulting prompt is still identical through ``lcp`` even
            # though the complete cached token sequence is no longer a strict
            # prefix.  Preserve that valid KV prefix and replay only the
            # divergent suffix.  Kimi is deliberately excluded: its KDA
            # recurrent state has no equivalent safe truncation operation.
            if lcp >= len(ids):
                # 新 prompt 是缓存序列的完整前缀(lcp==len(ids)):回退
                # 一个 token 重放,避免空后缀的空批连环崩溃(第三十轮)。
                lcp -= 1
            self.model.truncate_kv(lcp)
            skip = lcp
            mode = "lcp-replay"
            reason = "live-prefix-diverged"
        else:
            mode = "full-prefill"
            if live and not media_matches:
                cached_digest = getattr(self, "_cache_media_digest", None)
                reason = (
                    "media-digest-mismatch"
                    if cached_digest != media_digest
                    else "media-layout-mismatch"
                )
            else:
                reason = "no-live-prefix" if live else "empty-cache"
            self.reset()
        self._active_media_digest = media_digest
        self._active_media_slots = tuple(media_slots or ())
        self._active_media_state = media_state
        with _PrefillProgressMonitor(self, len(ids) - skip):
            logits = self._prefill_glm_suffix(
                ids,
                skip,
                media_state=media_state,
                media_slots=media_slots,
            )
        multimodal_timings = (
            getattr(self.model, "last_multimodal_timings", None)
            if media_state is not None and skip == 0
            else None
        ) or {}
        stats = KVPrefillStats(
            mode=mode,
            reason=reason,
            prompt_tokens=len(ids),
            baseline_tokens=skip,
            lcp_tokens=lcp,
            replay_tokens=0,
            suffix_tokens=len(ids) - skip,
            processed_tokens=len(ids) - skip,
            prefill_ms=(time.perf_counter() - started) * 1000.0,
            snapshot_bytes=0,
            vision_load_ms=multimodal_timings.get("vision_load_ms"),
            vision_forward_ms=multimodal_timings.get("vision_forward_ms"),
            language_prefill_ms=multimodal_timings.get("language_prefill_ms"),
        )
        self.last_kv_stats = stats
        if not getattr(self, "quiet", False):
            print(
                f"[KV] mode={stats.mode} reason={stats.reason} "
                f"baseline={stats.baseline_tokens} "
                f"lcp={stats.lcp_tokens} "
                f"suffix={stats.suffix_tokens} "
                f"prefill={stats.prefill_ms:.1f}ms",
                flush=True,
            )
        return logits

    @torch.no_grad()
    def _dsv4_prefill_suffix(self, ids: list[int], skip: int) -> tuple[torch.Tensor, torch.Tensor]:
        """DSV4 增量批量 prefill：复用 forward_verify 通道（快照不回滚，状态自然前进），
        64 token 分块（< sliding_window=128，避免环槽同批回绕）。
        返回 (末位 logits [vocab], 后缀全部位置 main_hidden [T, 3·hidden])。"""
        m = self.model
        mhs = []
        lg = None
        pos = skip
        for i in range(skip, len(ids), 64):
            chunk = ids[i:i + 64]
            lg2, mh2 = self._with_kv_capacity_retry(
                m.forward_verify, chunk, pos
            )
            m._spec = None            # 增量 prefill 不回滚，丢弃快照
            pos += len(chunk)
            mhs.append(mh2)
            lg = lg2
        m.pos = len(ids)
        return lg[-1], torch.cat(mhs, dim=0)

    @torch.no_grad()
    def _dsv4_prefill_range(
        self,
        ids: list[int],
        start: int,
        stop: int,
        *,
        manage_arena: bool = True,
    ) -> torch.Tensor:
        """Append only a new suffix with the normal exact Decode executor.

        Existing DSV4 KV is an ordered recurrent state.  It must not be fed
        back through the layer-first full-Prefill executor, and canonical
        replay must use the same Attention/MoE Graph path as ordinary token
        generation.  Full prompts remain batched in
        :meth:`_prefill_from_reset_to_boundary`; this method processes only
        the suffix missing from an already reusable KV snapshot.
        """
        del manage_arena
        if not 0 < start < stop <= len(ids):
            raise ValueError(
                f"invalid DSV4 prefill range "
                f"{start}:{stop}/{len(ids)}"
            )
        model = self.model
        if model.pos != start:
            raise RuntimeError(
                f"DSV4 live position {model.pos} != range start {start}"
            )
        logits = None
        for position in range(start, stop):
            logits = self._with_kv_capacity_retry(
                model.forward,
                [ids[position]],
            )
            if model.pos != position + 1:
                raise RuntimeError(
                    f"DSV4 live position {model.pos} "
                    f"!= committed position {position + 1}"
                )
            finite = torch.isfinite(logits)
            if not bool(finite.all().item()):
                finite_count = int(finite.sum().item())
                raise RuntimeError(
                    "DSV4 exact suffix Decode produced non-finite logits: "
                    f"finite={finite_count}/{finite.numel()}, "
                    f"token={position - start + 1}/{stop - start}, "
                    f"position={position}"
                )
        assert logits is not None
        model._last_prefill_scheduler = "incremental-exact-decode"
        if not getattr(self, "quiet", False):
            print(
                "[KV] scheduler=incremental-exact-decode "
                f"tokens={stop - start}",
                flush=True,
            )
        return logits

    def _save_dsv4_baseline(
        self,
        ids: list[int],
        baseline_len: int,
    ) -> int:
        snapshot = self.model.snapshot_kv()
        if snapshot.pos != baseline_len:
            raise RuntimeError(
                f"snapshot position {snapshot.pos} "
                f"!= baseline {baseline_len}"
            )
        self._kv_baseline = _DSV4Baseline(
            ids=list(ids[:baseline_len]),
            snapshot=snapshot,
        )
        return int(snapshot.nbytes)

    def commit_canonical_history(self, ids: list[int]) -> None:
        """Promote the adapter's exact DSV4 history to the reusable KV point.

        Thinking mode intentionally removes private reasoning from subsequent
        requests.  The generated live token stream therefore differs from the
        adapter's committed history.  When token IDs and model position already
        match, the live KV is the exact next-token state and must be reused
        directly.  Only a genuinely different public history is rebuilt from
        the request's stable snapshot.
        """
        if getattr(self, "arch", "glm") != "dsv4":
            return
        canonical = list(ids)
        if not canonical:
            self.reset()
            return

        started = time.perf_counter()
        live = list(getattr(self, "_cache_ids", None) or ())
        live_pos = int(getattr(self.model, "pos", 0))
        logits = None
        mode = "live"
        if live == canonical and live_pos <= len(canonical):
            if live_pos < len(canonical):
                logits = self._dsv4_prefill_range(
                    canonical,
                    live_pos,
                    len(canonical),
                )
                mode = "live-tail"
        else:
            baseline = getattr(self, "_kv_baseline", None)
            lcp = _token_lcp(
                canonical,
                getattr(baseline, "ids", None),
            )
            if (
                baseline is not None
                and lcp == len(baseline.ids)
                and getattr(baseline.snapshot, "pos", None)
                == len(baseline.ids)
                and len(baseline.ids) < len(canonical)
            ):
                self.model.restore_kv(baseline.snapshot)
                logits = self._dsv4_prefill_range(
                    canonical,
                    len(baseline.ids),
                    len(canonical),
                )
                mode = "canonical-branch"
            else:
                logits = self._prefill_from_reset_to_boundary(
                    canonical,
                    len(canonical),
                )
                mode = "canonical-rebuild"

        if logits is not None:
            finite = torch.isfinite(logits)
            if not bool(finite.all().item()):
                finite_count = int(finite.sum().item())
                raise RuntimeError(
                    "canonical DSV4 replay produced non-finite logits: "
                    f"finite={finite_count}/{finite.numel()}, mode={mode}"
                )

        if int(getattr(self.model, "pos", 0)) != len(canonical):
            raise RuntimeError(
                "committed DSV4 KV position does not match canonical history"
            )
        snapshot_bytes = self._save_dsv4_baseline(
            canonical,
            len(canonical),
        )
        self._cache_ids = canonical
        if not getattr(self, "quiet", False):
            print(
                "[KV] canonical-commit "
                f"mode={mode} tokens={len(canonical)} "
                f"snapshot={snapshot_bytes / 2**20:.1f}MiB "
                f"elapsed={(time.perf_counter() - started) * 1000:.1f}ms",
                flush=True,
            )

    def _prefill_from_reset_to_boundary(
        self,
        ids: list[int],
        baseline_len: int,
        *,
        manage_arena: bool = True,
    ) -> torch.Tensor:
        """Build canonical DSV4 state independently of request boundaries."""
        self.reset()
        routed_vq = getattr(self.model, "routed_vq", None)
        short_exact_decode = _use_short_reset_decode(
            routed_vq,
            baseline_len,
        )
        arena_active = False
        if short_exact_decode:
            # A freshly loaded bounded pool starts in its long-Prefill layout.
            # Initialize the Decode layout once before the exact short path;
            # activate_decode_arena is idempotent once the runtime directory
            # is valid, so subsequent short prompts retain their hot slots.
            end_prefill_block(routed_vq)
        elif manage_arena:
            begin_prefill_block(routed_vq)
            arena_active = True
        # A reset prompt is canonical batch prefill, not incremental replay.
        # The old one-token seed followed by ``_dsv4_prefill_range`` forced
        # every remaining prompt token through a complete 43-layer decode and
        # made even a five-token request look like a stalled Prefill.  DSV4's
        # native ``forward`` already dispatches an uninitialized state to the
        # exact layer-first batch implementation (4096-token outer blocks for
        # long prompts), while the conservative sequential primitive remains
        # in place for live/rollback suffixes whose KV begins at pos > 0.
        try:
            if short_exact_decode:
                # ``reset`` above deliberately removed the KV state. Allocate
                # it once, then let the normal compiled Decode path commit
                # every short-prompt token in order. This is a scheduler, not
                # a numerical fallback: all resident experts stay GPU-only.
                self.model._alloc(1)
                print(
                    "[cccp-prefill] scheduler=short-exact-decode "
                    f"tokens={baseline_len}; arena=decode-preserved",
                    flush=True,
                )
                logits = self._with_kv_capacity_retry(
                    self.model.forward,
                    ids[:baseline_len],
                )
            else:
                logits = self._with_kv_capacity_retry(
                    self.model.forward,
                    ids[:baseline_len],
                )
        finally:
            # Full batch Prefill and autoregressive Decode need opposite VRAM
            # layouts.  Restore the large heat/previous-route arena only after
            # the expanded BF16 expert scratch has been released.  A failed
            # Prefill also restores it so the next request starts cleanly.
            if arena_active:
                end_prefill_block(routed_vq)
        if self.model.pos != baseline_len:
            raise RuntimeError(
                f"DSV4 batched prefill position {self.model.pos} "
                f"!= baseline {baseline_len}"
            )
        if not getattr(self, "quiet", False):
            block = int(
                getattr(
                    self.model,
                    "last_prefill_block_size",
                    baseline_len,
                )
            )
            scheduler = (
                "short-exact-decode"
                if short_exact_decode
                else "batched-layer-first"
            )
            print(
                f"[KV] scheduler={scheduler} "
                f"block={block} baseline={baseline_len}",
                flush=True,
            )
        self.model._last_prefill_scheduler = (
            "short-exact-decode"
            if short_exact_decode
            else "batched-layer-first"
        )
        return logits

    def _trace_kv_divergence(
        self,
        live: list[int] | None,
        reencoded: list[int],
        lcp: int,
        radius: int = 8,
    ) -> None:
        """Print the first token mismatch without touching model state."""
        if (
            not _env_enabled("CCCP_KV_TRACE")
            or not live
            or lcp >= len(live)
            or lcp >= len(reencoded)
        ):
            return

        start = max(0, lcp - radius)
        stop = lcp + radius + 1
        live_tokens = live[start:min(len(live), stop)]
        reencoded_tokens = reencoded[
            start:min(len(reencoded), stop)
        ]

        def token_piece(token_id: int) -> str | None:
            try:
                return self.tok.id_to_token(token_id)
            except Exception as error:
                return f"<id_to_token-error:{type(error).__name__}>"

        def decoded(
            token_ids: list[int],
            *,
            skip_special_tokens: bool,
        ) -> str:
            try:
                return self.tok.decode(
                    token_ids,
                    skip_special_tokens=skip_special_tokens,
                )
            except Exception as error:
                return f"<decode-error:{type(error).__name__}>"

        false_text = {
            "live": decoded(
                live_tokens,
                skip_special_tokens=False,
            ),
            "reencoded": decoded(
                reencoded_tokens,
                skip_special_tokens=False,
            ),
        }
        true_text = {
            "live": decoded(
                live_tokens,
                skip_special_tokens=True,
            ),
            "reencoded": decoded(
                reencoded_tokens,
                skip_special_tokens=True,
            ),
        }
        lines = [
            (
                f"[KV-DIVERGE] pos={lcp} "
                f"window={start}:{stop} "
                f"live_len={len(live)} "
                f"reencoded_len={len(reencoded)}"
            ),
            f"live_id={live[lcp]}",
            f"reencoded_id={reencoded[lcp]}",
            f"live_tokens={json.dumps(live_tokens)}",
            f"reencoded_tokens={json.dumps(reencoded_tokens)}",
            (
                "live_piece="
                + json.dumps(
                    token_piece(live[lcp]),
                    ensure_ascii=False,
                )
            ),
            (
                "reencoded_piece="
                + json.dumps(
                    token_piece(reencoded[lcp]),
                    ensure_ascii=False,
                )
            ),
            (
                "skip_special_false_text="
                + json.dumps(false_text, ensure_ascii=False)
            ),
            (
                "skip_special_true_text="
                + json.dumps(true_text, ensure_ascii=False)
            ),
        ]
        print("\n".join(lines), flush=True)

    @torch.no_grad()
    def _prepare_dsv4_prompt(
        self,
        ids: list[int],
        baseline_len: int | None,
    ) -> torch.Tensor:
        """Prepare one prompt while retaining a pre-think rollback point."""
        started = time.perf_counter()
        cuda_events = None
        model_device = getattr(self.model, "device", None)
        if (
            model_device is not None
            and torch.device(model_device).type == "cuda"
        ):
            begin_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            begin_event.record()
            cuda_events = (begin_event, end_event)
        self._kv_prefill_events = None
        baseline_len = (
            len(ids) if baseline_len is None else baseline_len
        )
        if not 0 < baseline_len <= len(ids):
            raise ValueError(
                f"invalid DSV4 baseline_len={baseline_len} "
                f"for prompt length {len(ids)}"
            )

        live = getattr(self, "_cache_ids", None)
        baseline = getattr(self, "_kv_baseline", None)
        lcp = _token_lcp(live, ids)
        self._trace_kv_divergence(live, ids, lcp)
        strategy = "full"
        mode = "full-prefill"
        reason = "no-valid-baseline"
        baseline_tokens = 0
        replay_tokens = 0
        suffix_tokens = len(ids)
        processed_tokens = len(ids)
        start = 0

        if (
            live
            and lcp == len(live)
            and lcp <= baseline_len
            and lcp < len(ids)
            and self.model.pos == len(live)
        ):
            strategy = "live"
            mode = "lcp-replay"
            reason = "live-prefix"
            start = lcp
            baseline_tokens = lcp
            suffix_tokens = len(ids) - lcp
            processed_tokens = suffix_tokens
        elif (
            baseline is not None
            and getattr(baseline.snapshot, "pos", None)
            == len(baseline.ids)
            and lcp >= len(baseline.ids)
            and len(baseline.ids) <= baseline_len
            and lcp < len(ids)
        ):
            strategy = "rollback"
            mode = "lcp-replay"
            reason = "canonical-rollback"
            start = len(baseline.ids)
            baseline_tokens = start
            replay_tokens = lcp - start
            suffix_tokens = len(ids) - lcp
            processed_tokens = replay_tokens + suffix_tokens
        elif baseline is not None and lcp < len(baseline.ids):
            reason = "lcp-before-baseline"
        elif lcp == len(ids):
            reason = "no-new-suffix"

        def prepare_selected() -> tuple[torch.Tensor, int]:
            if strategy == "live":
                logits = (
                    self._dsv4_prefill_range(
                        ids,
                        start,
                        baseline_len,
                        manage_arena=False,
                    )
                    if start < baseline_len
                    else None
                )
            elif strategy == "rollback":
                self.model.restore_kv(baseline.snapshot)
                logits = (
                    self._dsv4_prefill_range(
                        ids,
                        start,
                        baseline_len,
                        manage_arena=False,
                    )
                    if start < baseline_len
                    else None
                )
            else:
                logits = self._prefill_from_reset_to_boundary(
                    ids,
                    baseline_len,
                    manage_arena=False,
                )

            snapshot_bytes = self._save_dsv4_baseline(
                ids,
                baseline_len,
            )
            if baseline_len < len(ids):
                logits = self._dsv4_prefill_range(
                    ids,
                    baseline_len,
                    len(ids),
                    manage_arena=False,
                )
            if logits is None:
                raise RuntimeError(
                    "DSV4 prompt preparation produced no logits"
                )
            finite = torch.isfinite(logits)
            if not bool(finite.all().item()):
                finite_count = int(finite.sum().item())
                scheduler = str(getattr(
                    self.model,
                    "_last_prefill_scheduler",
                    "unknown",
                ))
                raise RuntimeError(
                    "DSV4 prompt produced non-finite logits before sampling: "
                    f"finite={finite_count}/{finite.numel()}, "
                    f"strategy={strategy}, scheduler={scheduler}"
                )
            return logits, snapshot_bytes

        routed_vq = getattr(self.model, "routed_vq", None)
        # Only a fresh long prompt uses the batch-Prefill arena. Live and
        # rollback suffixes always use the exact Decode executor, while a
        # short reset can explicitly keep the existing Decode layout through
        # the pool capability above. No environment tuning switch is needed.
        batch_arena_required = bool(
            strategy == "full"
            and not _use_short_reset_decode(routed_vq, baseline_len)
        )
        arena_active = False
        try:
            if batch_arena_required:
                arena_active = begin_prefill_block(routed_vq)
            with _PrefillProgressMonitor(self, len(ids)):
                logits, snapshot_bytes = prepare_selected()
        finally:
            if arena_active:
                end_prefill_block(routed_vq)

        stats = KVPrefillStats(
            mode=mode,
            reason=reason,
            prompt_tokens=len(ids),
            baseline_tokens=baseline_tokens,
            lcp_tokens=lcp,
            replay_tokens=replay_tokens,
            suffix_tokens=suffix_tokens,
            processed_tokens=processed_tokens,
            prefill_ms=(
                time.perf_counter() - started
            ) * 1000.0,
            snapshot_bytes=snapshot_bytes,
        )
        if cuda_events is not None:
            cuda_events[1].record()
            self._kv_prefill_events = cuda_events
        self.last_kv_stats = stats
        if not getattr(self, "quiet", False):
            print(
                f"[KV] mode={stats.mode} reason={stats.reason} "
                f"baseline={stats.baseline_tokens} "
                f"lcp={stats.lcp_tokens} "
                f"replay={stats.replay_tokens} "
                f"suffix={stats.suffix_tokens} "
                f"prefill={stats.prefill_ms:.1f}ms",
                flush=True,
            )
        return logits

    def kv_prefill_cuda_ms(self) -> float | None:
        """Return completed CUDA-event preparation time without syncing."""
        events = getattr(self, "_kv_prefill_events", None)
        if events is None:
            return None
        begin_event, end_event = events
        if not end_event.query():
            return None
        return float(begin_event.elapsed_time(end_event))

    def _glm_device_greedy_window(
        self,
        *,
        temp: float,
        rep_penalty: float,
        presence_penalty: float,
        no_repeat_ngram: int,
    ) -> int:
        """Return the safe GPU-token feedback window for greedy TP decode."""
        routed_vq = getattr(self.model, "routed_vq", None)
        public_fixed_graph = bool(
            getattr(self.model, "device_greedy_supported", False)
        )
        if (
            temp > 1e-6
            or rep_penalty != 1.0
            or presence_penalty != 0.0
            or no_repeat_ngram != 0
            or getattr(self.model, "device", torch.device("cpu")).type
            != "cuda"
        ):
            return 0
        legacy_resident_graph = bool(
            getattr(self, "arch", "glm") == "glm"
            and routed_vq is not None
            and routed_vq.resident_parallel_supported
            and routed_vq.full_resident
        )
        if not public_fixed_graph and not legacy_resident_graph:
            return 0
        try:
            window = max(
                0,
                int(os.environ.get("CCCP_GREEDY_DEVICE_WINDOW", "8")),
            )
        except ValueError:
            return 0
        if (
            legacy_resident_graph
            and window > 1
            and os.environ.get(
                "CCCP_FLASHINFER_MLA",
                "1",
            )
            != "0"
        ):
            from .fusedext import available as fused_available

            if (
                os.environ.get(
                    "CCCP_FLASHINFER_GPU_PLAN",
                    "1",
                )
                == "0"
                or not fused_available()
                or int(
                    getattr(
                        self.model,
                        "cfg",
                        {},
                    ).get("n_heads", 0)
                )
                != 64
            ):
                return 1
        return window

    def _generate_glm_device_greedy(
        self,
        *,
        ids: list[int],
        logits: torch.Tensor,
        out: list[int],
        max_new: int | None,
        max_ctx: int | None,
        window: int,
        callback,
        should_stop: Callable[[], bool] | None,
    ) -> list[int]:
        """Greedy decode with several GPU-resident token decisions per sync."""
        previous_static = os.environ.get("CCCP_STATIC_LM_OUTPUT")
        os.environ["CCCP_STATIC_LM_OUTPUT"] = "1"
        try:
            graph_target = max(
                0,
                int(
                    getattr(
                        self.model,
                        "cfg",
                        {},
                    ).get("n_layers", 4)
                )
                - 4,
            )
            graph_needs_capture = (
                os.environ.get(
                    "CCCP_ATTENTION_GRAPH",
                    "1",
                )
                != "0"
                and os.environ.get(
                    "CCCP_QB_SPLIT",
                    "1",
                )
                != "0"
                and not getattr(
                    self.model,
                    "_attention_graph_failed",
                    False,
                )
                and len(
                    getattr(
                        self.model,
                        "_attention_graphs",
                        {},
                    )
                )
                < graph_target
                and getattr(
                    self.model,
                    "_flashinfer_mla_state",
                    None,
                )
                is not None
            )
            if graph_needs_capture:
                # The first captured replay still shares temporary buffers
                # with graph construction.  Capture with one sacrificial
                # token, synchronize, then roll KV back and start generation
                # from the untouched prompt logits.  This is paid once per
                # model lifetime and keeps the first real device window exact.
                prompt_logits = logits.clone()
                capture_token = torch.argmax(
                    prompt_logits
                ).reshape(1)
                self.model.forward(capture_token)
                torch.cuda.synchronize(logits.device)
                # FlashInfer/Attention graph construction mutates fixed decode
                # workspaces beyond the single captured layer output.  A KV
                # truncation alone is insufficient; rebuild the prompt once
                # after capture while retaining the now-stable graph objects.
                self.reset()
                logits = self.model.forward(ids)

            while _generation_open(
                len(out),
                max_new,
                len(ids) + len(out),
                max_ctx,
            ):
                remaining = window
                attention_graph_warmup = (
                    os.environ.get(
                        "CCCP_ATTENTION_GRAPH",
                        "1",
                    )
                    != "0"
                    and os.environ.get(
                        "CCCP_QB_SPLIT",
                        "1",
                    )
                    != "0"
                    and not getattr(
                        self.model,
                        "_attention_graph_failed",
                        False,
                    )
                    and len(
                        getattr(
                            self.model,
                            "_attention_graphs",
                            {},
                        )
                    )
                    < graph_target
                )
                # Graph capture uses a side stream and must finish before the
                # next token reuses its fixed metadata/output buffers.  Only
                # the very first decode token is serialized; steady-state
                # generation immediately returns to the configured window.
                if attention_graph_warmup:
                    remaining = 1
                if max_new is not None:
                    remaining = min(remaining, max_new - len(out))
                if max_ctx is not None:
                    remaining = min(
                        remaining,
                        max_ctx - len(ids) - len(out),
                    )
                if remaining <= 0:
                    break

                base_position = self.model.pos
                device_tokens = torch.empty(
                    remaining,
                    dtype=torch.long,
                    device=logits.device,
                )
                for index in range(remaining):
                    torch.argmax(
                        logits,
                        out=device_tokens[index],
                    )
                    logits = self.model.forward(
                        device_tokens[index:index + 1]
                    )

                accepted = 0
                stop = False
                for next_token in device_tokens.cpu().tolist():
                    if next_token in self.eos:
                        stop = True
                        break
                    out.append(next_token)
                    accepted += 1
                    if callback:
                        callback(
                            next_token,
                            self.decode([next_token]),
                        )
                    if should_stop is not None and should_stop():
                        stop = True
                        break

                if accepted != remaining:
                    self.model.truncate_kv(
                        base_position + accepted
                    )
                if stop:
                    break
        finally:
            if previous_static is None:
                os.environ.pop("CCCP_STATIC_LM_OUTPUT", None)
            else:
                os.environ[
                    "CCCP_STATIC_LM_OUTPUT"
                ] = previous_static

        self._cache_ids = list(ids) + out
        self._cache_media_digest = getattr(self, "_active_media_digest", None)
        self._cache_media_slots = getattr(self, "_active_media_slots", ())
        self._cache_via_spec = False
        return out

    @torch.no_grad()
    def generate(self, ids: list[int], max_new: int | None = 128, temp: float = 0.0,
                 top_p: float = 1.0, callback=None, rep_penalty: float = 1.0,
                 presence_penalty: float = 0.0,
                 no_repeat_ngram: int = 0,
                 should_stop: Callable[[], bool] | None = None,
                 kv_baseline_len: int | None = None,
                 media_digest: str | None = None,
                 media_slots: object = (),
                 media_state: object = None) -> list[int]:
        """自回归生成。temp=0 贪心；callback(tok_id, 增量文本) 逐 token 回调。

        rep_penalty>1：对已出现 token 的 logits 施加重复惩罚（正除负乘），
        压制 PTQ 模型自由文本/长生成的复读循环（knee 档已知倾向）。
        presence_penalty：对已出现 token 固定减去该值（OpenAI 语义，-2~2）。
        no_repeat_ngram>0：禁止会复现已生成 n-gram 的候选 token。
        """
        out: list[int] = []
        mc = getattr(self.model, "max_ctx", None)
        if mc and len(ids) >= mc:
            print(f"[cccp] prompt 已达到 max_ctx={mc}，无法继续生成", flush=True)
            return out
        if max_new is not None and mc and len(ids) + max_new > mc:
            # 提前友好报错：越界会在 KV 压缩槽/rope 索引处抛 cryptic IndexError
            max_new = max(0, mc - len(ids))
            kv_hint = (
                "MLA latent KV 约 0.09MB/token"
                if getattr(self, "arch", "glm") == "glm"
                else "DSV4 使用环形窗+压缩槽"
            )
            print(f"[cccp] 警告：prompt {len(ids)} + max_new 超过 max_ctx={mc}，"
                  f"本次最多生成 {max_new} token"
                  f"（--max-ctx 可调大，{kv_hint}）",
                  flush=True)
            if max_new == 0:
                return out
        if getattr(self, "arch", "glm") == "dsv4":
            logits = self._prepare_dsv4_prompt(
                ids,
                kv_baseline_len,
            )
        else:
            logits = self._prepare_glm_prompt(
                ids,
                media_digest=media_digest,
                media_slots=media_slots,
                media_state=media_state,
            )
        device_window = self._glm_device_greedy_window(
            temp=temp,
            rep_penalty=rep_penalty,
            presence_penalty=presence_penalty,
            no_repeat_ngram=no_repeat_ngram,
        )
        if device_window:
            return self._generate_glm_device_greedy(
                ids=ids,
                logits=logits,
                out=out,
                max_new=max_new,
                max_ctx=mc,
                window=device_window,
                callback=callback,
                should_stop=should_stop,
            )
        prev = list(ids)
        ngram_ban: dict[tuple, set] = {}
        decode_stage_probe_pending = True
        # GLM can defer committing the last sampled token into KV.  A forward
        # pass after the requested output limit only computes logits that the
        # caller will never consume (one very expensive full-model decode on
        # CPU).  Keep the cache ledger honest so the next request replays that
        # one token before it reuses the prefix.
        glm_final_token_deferred = False
        while _generation_open(
            len(out), max_new, len(ids) + len(out), mc
        ):
            lg = _apply_token_penalties(
                logits,
                prev,
                repetition_penalty=rep_penalty,
                presence_penalty=presence_penalty,
            )
            if no_repeat_ngram > 0 and len(prev) >= no_repeat_ngram:
                if lg is logits:
                    lg = logits.clone()
                key = tuple(prev[-(no_repeat_ngram - 1):]) if no_repeat_ngram > 1 else ()
                for tok in ngram_ban.get(key, ()):  # 禁掉会复现 n-gram 的 token
                    lg[tok] = float("-inf")
            if temp <= 1e-6:
                nxt = int(lg.argmax().item())
            else:
                nxt = _sample_top_p(lg, temp, top_p)
            if nxt in self.eos:
                # Kimi 的 <|end_of_msg|>（163586）既是 eos 又是 XTML 结构性
                # token：它必须保留在 _cache_ids 里，否则下一轮渲染的 assistant
                # 消息序列（含 <|end_of_msg|>）无法命中前缀缓存。callback 也要
                # 收到它以保证 callback_ids 与 generated_ids 一致（decode_stream
                # 的 skip_special_tokens 会过滤它，不泄露给客户端）。
                # DSV4 and Kimi adapters consume their structural EOS marker
                # to finish XML/reasoning state.  GLM role tokens such as
                # ``<|user|>`` are plain stops; exposing one to the GLM text
                # parser leaks it into the assistant response.
                if getattr(self, "arch", "glm") != "glm":
                    out.append(nxt)
                    prev.append(nxt)
                    if callback:
                        callback(nxt, self.decode([nxt]))
                    if getattr(self, "arch", "glm") == "qwen3_5_dense":
                        # Qwen's <|im_end|> is both EOS and a structural chat
                        # token.  Commit it into the recurrent/KV state so the
                        # exact next-turn suffix can reuse the live cache.
                        self.model.forward([nxt])
                break
            out.append(nxt)
            if no_repeat_ngram > 0:
                seq = prev + [nxt]
                if len(seq) >= no_repeat_ngram:
                    k = tuple(seq[-no_repeat_ngram:-1]) if no_repeat_ngram > 1 else ()
                    ngram_ban.setdefault(k, set()).add(nxt)
            prev.append(nxt)
            if callback:
                callback(nxt, self.decode([nxt]))
            stop_requested = should_stop is not None and should_stop()
            output_limit_reached = not _generation_open(
                len(out), max_new, len(ids) + len(out), mc
            )
            if output_limit_reached and getattr(self, "arch", "glm") == "glm":
                glm_final_token_deferred = True
                break
            if getattr(self, "arch", "glm") == "dsv4":
                if decode_stage_probe_pending:
                    logits = _profile_dsv4_stage_call(
                        self.model,
                        "decode-token",
                        1,
                        self._with_kv_capacity_retry,
                        self.model.forward,
                        [nxt],
                        committed=len(out),
                    )
                    decode_stage_probe_pending = False
                else:
                    logits = self._with_kv_capacity_retry(
                        self.model.forward, [nxt], committed=len(out)
                    )
            else:
                logits = self.model.forward([nxt])
            if stop_requested:
                break
        cached_output = out[:-1] if glm_final_token_deferred else out
        self._cache_ids = list(ids) + cached_output
        self._cache_media_digest = getattr(self, "_active_media_digest", None)
        self._cache_media_slots = getattr(self, "_active_media_slots", ())
        self._cache_via_spec = False   # 非投机路径不写 DSpark 环
        return out

    @torch.no_grad()
    def generate_speculative(
        self,
        ids: list[int],
        max_new: int | None = 128,
        k: int = 3,
        callback=None,
        should_stop: Callable[[], bool] | None = None,
        kv_baseline_len: int | None = None,
        media_digest: str | None = None,
        media_slots: object = (),
        media_state: object = None,
    ) -> list[int]:
        """Run the registered MTP/DSpark drafter with bounded verification."""
        try:
            provider_spec = provider_for_architecture(
                getattr(self, "arch", "glm")
            )
        except ValueError:
            provider_spec = None
        if (
            provider_spec is None
            or not provider_attachment_available(provider_spec, self.model)
        ):
            self.spec_stats = {
                "mode": "draft-provider-unavailable",
                "rounds": 0,
                "accepted": 0,
                "drafted": 0,
            }
            print(
                "[cccp-mtp] 模型配置未声明可用草稿附件，"
                "自动使用主模型贪心解码",
                flush=True,
            )
            return self.generate(
                ids,
                max_new=max_new,
                temp=0.0,
                callback=callback,
                should_stop=should_stop,
                media_digest=media_digest,
                media_slots=media_slots,
                media_state=media_state,
            )
        provider = provider_spec.provider
        policy = provider_spec.policy
        if provider == "kimi_prompt_lookup":
            if (
                getattr(self.model, "device", torch.device("cpu")).type
                == "cpu"
                and hasattr(self.model, "forward_hidden_block_cpu")
                and hasattr(self.model, "snapshot_decode_state")
            ):
                return self._generate_kimi_prompt_lookup(
                    ids,
                    max_new=max_new,
                    k=k,
                    callback=callback,
                    should_stop=should_stop,
                    policy=policy,
                    media_digest=media_digest,
                    media_slots=media_slots,
                    media_state=media_state,
                )
            return self.generate(
                ids,
                max_new=max_new,
                temp=0.0,
                callback=callback,
                should_stop=should_stop,
                media_digest=media_digest,
                media_slots=media_slots,
                media_state=media_state,
            )
        if provider == "qwen35_mtp":
            if not bool(getattr(self.model, "supports_mtp", False)):
                self.spec_stats = {
                    "mode": "mtp-unavailable",
                    "rounds": 0,
                    "accepted": 0,
                    "drafted": 0,
                }
                return self.generate(
                    ids,
                    max_new=max_new,
                    temp=0.0,
                    callback=callback,
                    should_stop=should_stop,
                    media_digest=media_digest,
                    media_slots=media_slots,
                    media_state=media_state,
                )
            return self._generate_qwen35_mtp(
                ids,
                max_new=max_new,
                k=k,
                callback=callback,
                should_stop=should_stop,
                policy=policy,
            )
        if provider == "dsv4_dspark":
            self._kv_baseline = None
            self.last_kv_stats = None
            mc = getattr(self.model, "max_ctx", None)
            if mc and len(ids) >= mc:
                print(f"[cccp] prompt 已达到 max_ctx={mc}，无法继续生成", flush=True)
                return []
            if max_new is not None and mc and len(ids) + max_new > mc:
                max_new = max(0, mc - len(ids))
                print(f"[cccp] 警告：超出 max_ctx={mc}，本次最多生成 {max_new} token",
                      flush=True)
            return self._generate_dspark(
                ids,
                max_new=max_new,
                k=k,
                callback=callback,
                should_stop=should_stop,
                policy=policy,
            )
        if provider != "glm_mtp":
            print(
                "[cccp-mtp] 当前架构未声明草稿能力，回退主模型贪心",
                flush=True,
            )
            return self.generate(
                ids,
                max_new=max_new,
                temp=0.0,
                callback=callback,
                should_stop=should_stop,
                media_digest=media_digest,
                media_slots=media_slots,
                media_state=media_state,
            )
        if media_state is not None:
            print(
                "[cccp-mtp] GLM 多模态 prompt 使用标准主模型解码，"
                "当前 MTP 不重放视觉 prefill",
                flush=True,
            )
            return self.generate(
                ids,
                max_new=max_new,
                temp=0.0,
                callback=callback,
                should_stop=should_stop,
                kv_baseline_len=kv_baseline_len,
                media_digest=media_digest,
                media_slots=media_slots,
                media_state=media_state,
            )
        mc = getattr(self.model, "max_ctx", None)
        if mc and len(ids) >= mc:
            print(f"[cccp] prompt 已达到 max_ctx={mc}，无法继续生成", flush=True)
            return []
        if max_new is not None and mc and len(ids) + max_new > mc:
            max_new = max(0, mc - len(ids))
        from .mtp import MTPHead

        self.reset()           # GLM-MTP 路径不支持增量 prefill，每轮全量重建
        mtp = MTPHead(self.model)
        mtp.reset()
        out: list[int] = []
        h_all = self.model.forward_hidden(ids)
        logits = self.model.logits_of(h_all[-1:]).squeeze(0)
        # MTP prefill 建立第 78 层上下文 KV；草稿首步用主模型 hidden（DeepSeek 流程），
        # 链式步才回喂 MTP 自身输出 h78
        mtp.prefill(h_all, ids)
        h_main_last = h_all[-1:]
        next_pos = len(ids)          # 下一个 MTP 步的 RoPE 位置
        next_t1 = int(logits.argmax())
        stats = {
            "mode": policy.mode,
            "rounds": 0,
            "accepted": 0,
            "drafted": 0,
        }
        stop_requested = False
        while (
            _generation_open(len(out), max_new, len(ids) + len(out), mc)
            and next_t1 not in self.eos
            and not stop_requested
        ):
            t1 = next_t1
            out.append(t1)
            if callback:
                callback(t1, self.decode([t1]))
            stop_requested = should_stop is not None and should_stop()
            if stop_requested or not _generation_open(
                len(out), max_new, len(ids) + len(out), mc
            ):
                break
            # 1) 起草：首步输入 = (主模型 hidden, emb(t1))；其后回喂 h78
            kv0 = mtp.kv[0].shape[1] if mtp.kv is not None else 0
            h, drafts = h_main_last, []
            draft_count = k
            if max_new is not None:
                draft_count = min(draft_count, max_new - len(out))
            if mc is not None:
                draft_count = min(draft_count, mc - len(ids) - len(out))
            for j in range(max(0, draft_count)):
                h, lg = mtp.step(h, t1 if not drafts else drafts[-1], next_pos + j)
                drafts.append(int(lg.argmax()))
            stats["drafted"] += len(drafts)
            # 2) 主模型一次前向验证 [t1, d1..dk]
            pos0 = self.model.pos
            h2 = self.model.forward_hidden([t1] + drafts)
            lg2 = self.model.logits_of(h2)
            accepted = 0
            for i in range(len(drafts)):
                if not _generation_open(
                    len(out), max_new, len(ids) + len(out), mc
                ):
                    break
                if (
                    policy.accepts(lg2[i], drafts[i])
                    and drafts[i] not in self.eos
                ):
                    accepted += 1
                    out.append(drafts[i])
                    if callback:
                        callback(drafts[i], self.decode([drafts[i]]))
                    if should_stop is not None and should_stop():
                        stop_requested = True
                        break
                else:
                    break
            stats["accepted"] += accepted
            stats["rounds"] += 1
            next_t1 = int(lg2[accepted].argmax())
            # 3) 主 KV 截断：被拒草稿不得留在上下文（只保留 t1 + 接受的前缀）
            keep = pos0 + 1 + accepted
            self.model.truncate_kv(keep)
            # 4) MTP 状态推进：KV 截断（保留 t1 + 接受前缀，t1 步恒有效）；
            #    下一轮首步的 hidden = 主模型在最后接受位的 hidden（h2[accepted]）
            L = kv0 + 1 + accepted
            mtp.kv = (mtp.kv[0][:, :L], mtp.kv[1][:, :L])
            h_main_last = h2[accepted:accepted + 1]
            next_pos += 1 + accepted
        self.spec_stats = stats
        return out

    @torch.no_grad()
    def _generate_qwen35_mtp(
        self,
        ids: list[int],
        *,
        max_new: int | None,
        k: int,
        callback=None,
        should_stop: Callable[[], bool] | None = None,
        policy: DraftAcceptancePolicy,
    ) -> list[int]:
        """Qwen3.5 MTP with main-model Top-3 block verification.

        The main hybrid cache is snapshotted only around a verification
        block.  A full hit keeps the already-computed state; a partial hit
        restores the recurrent state and replays only the committed prefix.
        """
        overall_started = time.perf_counter()
        model = self.model
        mtp = model.mtp
        max_ctx = getattr(model, "max_ctx", None)
        if max_ctx and len(ids) >= max_ctx:
            return []
        if max_new is not None and max_ctx:
            max_new = min(max_new, max(0, max_ctx - len(ids)))

        # The Qwen drafter owns a shifted full-attention cache.  Rebuilding it
        # together with the main prompt keeps the two state machines exact;
        # later rounds reuse both caches without another prompt pass.
        self.reset()
        main_hidden_all = model.forward_hidden(ids)
        logits = model.logits_of(main_hidden_all[-1:]).squeeze(0)
        mtp.prefill(main_hidden_all, ids)
        main_hidden_last = main_hidden_all[-1:]
        next_token = int(logits.argmax().item())
        if getattr(model, "device", torch.device("cpu")).type == "cuda":
            torch.cuda.synchronize(model.device)
        prefill_finished = time.perf_counter()
        next_position = len(ids)
        out: list[int] = []
        stats = {
            "mode": policy.mode,
            "rounds": 0,
            "drafted": 0,
            "accepted": 0,
            "replayed": 0,
            "max_draft": policy.draft_count(k),
            "prefill_ms": (prefill_finished - overall_started) * 1000.0,
        }
        stop_requested = False

        # CCCP_MTP_PROFILE=1 时按段累计轮时间(draft/verify/logits/accept/
        # commit/argmax/callback),定位投机解码的隐藏开销。默认关闭零开销。
        profile = os.environ.get("CCCP_MTP_PROFILE", "0") == "1"
        phase_ms: dict[str, float] = {}

        def _phase(name: str, t0: float) -> None:
            if not profile:
                return
            if getattr(model, "device", torch.device("cpu")).type == "cuda":
                torch.cuda.synchronize(model.device)
            phase_ms[name] = phase_ms.get(name, 0.0) + (
                time.perf_counter() - t0
            ) * 1000.0

        while (
            not stop_requested
            and _generation_open(
                len(out), max_new, len(ids) + len(out), max_ctx
            )
        ):
            first = int(next_token)
            if first in self.eos:
                out.append(first)
                if callback:
                    callback(first, self.decode([first]))
                model.forward_hidden([first])
                break

            out.append(first)
            if callback:
                callback(first, self.decode([first]))
            stop_requested = should_stop is not None and should_stop()

            room = policy.draft_count(k)
            if max_new is not None:
                room = min(room, max_new - len(out))
            if max_ctx is not None:
                room = min(room, max_ctx - len(ids) - len(out))
            if stop_requested or room <= 0:
                hidden = model.forward_hidden([first])
                main_hidden_last = hidden[-1:]
                break

            mtp_base = mtp.cache_length
            draft_hidden = main_hidden_last
            draft_token = first
            _t = time.perf_counter()
            if hasattr(mtp, "draft_block"):
                draft_hidden, drafts = mtp.draft_block(
                    draft_hidden,
                    draft_token,
                    next_position,
                    room,
                )
            else:
                drafts = []
                for offset in range(room):
                    draft_hidden, draft_logits = mtp.step(
                        draft_hidden,
                        draft_token,
                        next_position + offset,
                    )
                    draft_token = int(draft_logits.argmax().item())
                    drafts.append(draft_token)
            stats["drafted"] += len(drafts)
            if profile:
                _phase("draft", _t)

            direct_commit = bool(
                getattr(model, "supports_direct_verify_commit", False)
            )
            _t = time.perf_counter()
            snapshot = (
                None if direct_commit else model.snapshot_decode_state()
            )
            if profile:
                _phase("snapshot", _t)
            verify = getattr(model, "forward_hidden_verify", model.forward_hidden)
            _t = time.perf_counter()
            verified_hidden = verify([first] + drafts)
            if profile:
                _phase("verify", _t)
            _t = time.perf_counter()
            verified_logits = model.logits_of(verified_hidden)
            if profile:
                _phase("logits", _t)
            _t = time.perf_counter()
            accepted = policy.accepted_prefix_batched(
                verified_logits,
                drafts,
            )
            if profile:
                _phase("accept", _t)

            # EOS is structural for Qwen.  It may be accepted, but no token
            # after it can enter the committed prefix.
            _t = time.perf_counter()
            emit_accepted = accepted
            eos_seen = False
            for index in range(accepted):
                draft = drafts[index]
                out.append(draft)
                if callback:
                    callback(draft, self.decode([draft]))
                if draft in self.eos:
                    emit_accepted = index + 1
                    eos_seen = True
                    stop_requested = True
                    break
                if should_stop is not None and should_stop():
                    emit_accepted = index + 1
                    stop_requested = True
                    break
                if not _generation_open(
                    len(out), max_new, len(ids) + len(out), max_ctx
                ):
                    emit_accepted = index + 1
                    stop_requested = True
                    break

            stats["accepted"] += emit_accepted
            if profile:
                _phase("emit", _t)
            _t = time.perf_counter()
            committed = 1 + emit_accepted
            full_block = emit_accepted == len(drafts) and not eos_seen
            if full_block:
                main_hidden_last = verified_hidden[-1:]
                next_token = int(verified_logits[-1].argmax().item())
            elif direct_commit:
                model.commit_verified_prefix(committed)
                main_hidden_last = verified_hidden[
                    committed - 1:committed
                ]
                next_token = int(
                    verified_logits[committed - 1].argmax().item()
                )
            else:
                model.restore_decode_state(snapshot)
                canonical = [first] + drafts[:emit_accepted]
                replay_hidden = model.forward_hidden(canonical)
                replay_logits = model.logits_of(replay_hidden[-1:]).squeeze(0)
                main_hidden_last = replay_hidden[-1:]
                next_token = int(replay_logits.argmax().item())
                stats["replayed"] += len(canonical)
            mtp.crop(min(mtp.cache_length, mtp_base + committed))
            next_position += committed
            stats["rounds"] += 1
            if profile:
                _phase("commit", _t)

        if getattr(model, "device", torch.device("cpu")).type == "cuda":
            torch.cuda.synchronize(model.device)
        finished = time.perf_counter()
        decode_tokens = max(0, len(out) - 1)
        decode_seconds = max(0.0, finished - prefill_finished)
        stats.update({
            "decode_tokens": decode_tokens,
            "decode_seconds": decode_seconds,
            "decode_tokens_per_second": (
                decode_tokens / decode_seconds if decode_seconds > 0.0 else 0.0
            ),
            "total_seconds": finished - overall_started,
        })
        self.spec_stats = stats
        self._cache_ids = list(ids) + out
        self._cache_media_digest = getattr(self, "_active_media_digest", None)
        self._cache_media_slots = getattr(self, "_active_media_slots", ())
        self._cache_via_spec = False
        if profile and stats.get("rounds"):
            print(
                "[cccp-mtp-profile] rounds={} ".format(stats["rounds"])
                + " ".join(
                    f"{name}={ms:.1f}ms" for name, ms in phase_ms.items()
                ),
                flush=True,
            )
        print(
            "[cccp-mtp] architecture=qwen3.5-dense; "
            f"mode={stats['mode']}; rounds={stats['rounds']}; "
            f"accepted={stats['accepted']}/{stats['drafted']}; "
            f"replayed={stats['replayed']}; "
            f"prefill={stats['prefill_ms']:.1f}ms; "
            f"decode={stats['decode_tokens_per_second']:.3f}tok/s "
            f"({stats['decode_tokens']} tokens/{stats['decode_seconds']:.3f}s)",
            flush=True,
        )
        return out

    @staticmethod
    def _prompt_lookup_draft(
        history: list[int],
        maximum: int,
        *,
        minimum_ngram: int = 3,
        maximum_ngram: int = 16,
    ) -> list[int]:
        """Copy a continuation after the longest previous suffix match."""
        if maximum <= 0 or len(history) <= minimum_ngram:
            return []
        upper = min(maximum_ngram, len(history) - 1)
        for width in range(upper, minimum_ngram - 1, -1):
            suffix = history[-width:]
            latest = len(history) - width - 1
            for start in range(latest, -1, -1):
                if history[start:start + width] != suffix:
                    continue
                draft = history[start + width:start + width + maximum]
                if draft:
                    return list(draft)
        return []

    @torch.no_grad()
    def _generate_kimi_prompt_lookup(
        self,
        ids: list[int],
        *,
        max_new: int | None,
        k: int,
        callback=None,
        should_stop: Callable[[], bool] | None = None,
        policy: DraftAcceptancePolicy,
        media_digest: str | None = None,
        media_slots: object = (),
        media_state: object = None,
    ) -> list[int]:
        """Prompt-lookup drafts using Kimi's CPU block verifier."""
        maximum_draft = policy.draft_count(k)
        max_ctx = getattr(self.model, "max_ctx", None)
        if max_ctx and len(ids) >= max_ctx:
            return []
        if max_new is not None and max_ctx:
            max_new = min(max_new, max(0, max_ctx - len(ids)))
        logits = self._prepare_glm_prompt(
            ids,
            media_digest=media_digest,
            media_slots=media_slots,
            media_state=media_state,
        )
        history = list(ids)
        out: list[int] = []
        stats = {
            "mode": f"kimi-prompt-lookup-{policy.mode}",
            "rounds": 0,
            "block_rounds": 0,
            "fallback_rounds": 0,
            "drafted": 0,
            "accepted": 0,
            "replayed": 0,
        }
        stop = False
        while (
            not stop
            and _generation_open(
                len(out), max_new, len(ids) + len(out), max_ctx
            )
        ):
            first = int(logits.argmax().item())
            if first in self.eos:
                break
            out.append(first)
            history.append(first)
            if callback:
                callback(first, self.decode([first]))
            stop = should_stop is not None and should_stop()
            stats["rounds"] += 1
            room = maximum_draft
            if max_new is not None:
                room = min(room, max_new - len(out))
            if max_ctx is not None:
                room = min(room, max_ctx - len(ids) - len(out))
            if stop or room <= 0:
                logits = self.model.forward([first])
                stats["fallback_rounds"] += 1
                break
            drafts = self._prompt_lookup_draft(history, room)
            if not drafts:
                logits = self.model.forward([first])
                stats["fallback_rounds"] += 1
                continue
            snapshot = self.model.snapshot_decode_state()
            hidden = self.model.forward_hidden_block_cpu([first] + drafts)
            block_logits = self.model.logits_of(hidden)
            stats["block_rounds"] += 1
            stats["drafted"] += len(drafts)
            accepted = 0
            for index, draft in enumerate(drafts):
                if not policy.accepts(block_logits[index], draft):
                    break
                if draft in self.eos:
                    stop = True
                    break
                out.append(draft)
                history.append(draft)
                accepted += 1
                if callback:
                    callback(draft, self.decode([draft]))
                if should_stop is not None and should_stop():
                    stop = True
                    break
            stats["accepted"] += accepted
            fully_committed = accepted == len(drafts) and not stop
            if fully_committed:
                logits = block_logits[accepted]
                continue
            self.model.restore_decode_state(snapshot)
            committed = [first] + drafts[:accepted]
            canonical_hidden = self.model.forward_hidden(committed)
            logits = self.model.logits_of(canonical_hidden[-1:]).squeeze(0)
            stats["replayed"] += len(committed)
        self.spec_stats = stats
        self._cache_ids = list(ids) + out
        self._cache_media_digest = getattr(self, "_active_media_digest", None)
        self._cache_media_slots = getattr(self, "_active_media_slots", ())
        self._cache_via_spec = False
        return out

    @torch.no_grad()
    def _generate_dspark(
        self,
        ids: list[int],
        policy: DraftAcceptancePolicy,
        max_new: int | None = 128,
        k: int = 5,
        callback=None,
        should_stop: Callable[[], bool] | None = None,
    ) -> list[int]:
        """DSV4 DSpark block drafting with registered Top-N acceptance.

        每轮：1 次 DSpark 前向并行产出 block_size(=5) 个草稿 → 主模型 1 次批量
        前向验证 [t1, d1..dk] → 接受最长连续前缀（argmax 比对），首位不匹配的
        argmax 为奖励 token → 主 KV 按接受前缀截断（spec_commit），DSpark 环
        写入接受位置的 main_kv。
        """
        from .dspark import DSparkHead
        model = self.model
        dsp = getattr(self, "_dsp", None)
        if dsp is None:
            dspark_gb = float(os.environ.get("CCCP_DSPARK_VRAM_GB", "2.75"))
            self._cap_expert_cache(
                self._vram_runtime_reserve_gb + dspark_gb,
                "运行时+DSpark 余量",
            )
            dsp = self._dsp = DSparkHead(model)
        overall_started = time.perf_counter()
        k = policy.draft_count(k, dsp.block_size)
        out: list[int] = []
        skip = self._kv_prefix_len(ids) if self._cache_via_spec else 0
        if skip:
            # 多轮 KV 复用：主模型只增量 prefill 新后缀，DSpark 环补写新位置的 main_kv
            lg_last, mh_suf = self._dsv4_prefill_suffix(ids, skip)
            dsp.update_kv(mh_suf, skip)
            t1 = int(lg_last.argmax())
            mh_last = mh_suf[-1]                    # 最末位置 main_hidden [3D]
        else:
            self.reset()
            dsp.reset()
            logits_last, mh = self._with_kv_capacity_retry(
                model.prefill_mh,
                torch.tensor([ids], device=model.device),
            )
            dsp.prefill_kv(mh[0])                # DSpark 环：positions 0..T-1
            t1 = int(logits_last[0].argmax())
            mh_last = mh[0, -1]                  # position p 的 main_hidden [3D]
        if getattr(model, "device", torch.device("cpu")).type == "cuda":
            torch.cuda.synchronize(model.device)
        prefill_finished = time.perf_counter()
        p = len(ids) - 1                         # 最末已处理位置
        stats = {
            "mode": policy.mode,
            "rounds": 0,
            "accepted": 0,
            "drafted": 0,
            "prefill_ms": (prefill_finished - overall_started) * 1000.0,
        }
        mc = getattr(model, "max_ctx", None)
        stop_requested = False
        while (
            _generation_open(len(out), max_new, len(ids) + len(out), mc)
            and t1 not in self.eos
            and not stop_requested
        ):
            drafts = dsp.draft(t1, mh_last, p)   # 5 草稿；main_kv@p 写入各层环
            out.append(t1)
            if callback:
                callback(t1, self.decode([t1]))
            stop_requested = should_stop is not None and should_stop()
            draft_count = k
            if max_new is not None:
                draft_count = min(draft_count, max_new - len(out))
            if mc is not None:
                draft_count = min(draft_count, mc - len(ids) - len(out))
            if stop_requested:
                draft_count = 0
            block = [t1] + drafts[:max(0, draft_count)]
            pos0 = model.pos                      # = p+1
            lg2, mh2 = self._with_kv_capacity_retry(
                model.forward_verify,
                block,
                pos0,
                committed=len(out),
            )
            accepted = 0
            for i in range(max(0, draft_count)):
                if not _generation_open(
                    len(out), max_new, len(ids) + len(out), mc
                ):
                    break
                if (
                    policy.accepts(lg2[i], drafts[i])
                    and drafts[i] not in self.eos
                ):
                    accepted += 1
                    out.append(drafts[i])
                    if callback:
                        callback(drafts[i], self.decode([drafts[i]]))
                    if should_stop is not None and should_stop():
                        stop_requested = True
                        break
                else:
                    break
            stats["accepted"] += accepted
            stats["drafted"] += max(0, draft_count)
            stats["rounds"] += 1
            next_t1 = int(lg2[accepted].argmax())
            keep = pos0 + 1 + accepted
            model.spec_commit(keep)               # 主 KV 按接受前缀截断
            dsp.update_kv(mh2[:accepted], pos0)   # 接受前缀入 DSpark 环（末位下轮 draft 写）
            mh_last = mh2[accepted]
            p = keep - 1
            t1 = next_t1
        if getattr(model, "device", torch.device("cpu")).type == "cuda":
            torch.cuda.synchronize(model.device)
        finished = time.perf_counter()
        decode_tokens = max(0, len(out) - 1)
        decode_seconds = max(0.0, finished - prefill_finished)
        stats.update({
            "decode_tokens": decode_tokens,
            "decode_seconds": decode_seconds,
            "decode_tokens_per_second": (
                decode_tokens / decode_seconds if decode_seconds > 0.0 else 0.0
            ),
            "total_seconds": finished - overall_started,
        })
        self.spec_stats = stats
        self._cache_ids = list(ids) + out
        self._cache_via_spec = True    # DSpark 环已覆盖 prompt+回复全部位置
        print(
            "[cccp-mtp] architecture=dsv4; provider=dspark; "
            f"mode={stats['mode']}; rounds={stats['rounds']}; "
            f"accepted={stats['accepted']}/{stats['drafted']}; "
            f"prefill={stats['prefill_ms']:.1f}ms; "
            f"decode={stats['decode_tokens_per_second']:.3f}tok/s "
            f"({stats['decode_tokens']} tokens/{stats['decode_seconds']:.3f}s)",
            flush=True,
        )
        return out


def _profile_dsv4_stage_call(
    model,
    phase: str,
    tokens: int,
    function,
    *args,
    **kwargs,
):
    """Profile one real DSV4 CUDA call without changing its executor."""

    start_profile = getattr(model, "start_profile", None)
    finish_profile = getattr(model, "finish_profile", None)
    device = torch.device(getattr(model, "device", "cpu"))
    tp_contexts = getattr(model, "_tp_attention_contexts", None)
    hip_tp1_probe = bool(
        torch.version.hip is not None
        and tp_contexts is not None
        and int(getattr(model, "tp_size", 1)) == 1
        and not bool(getattr(model, "_packed_full_gpu", False))
        and not bool(getattr(model, "_cccp_hip_stage_probe_done", False))
    )
    enabled = (
        os.environ.get("CCCP_STAGE_TIMING", "1") != "0"
        and device.type == "cuda"
        and callable(start_profile)
        and callable(finish_profile)
        and not bool(getattr(model, "_profile_enabled", False))
        and (tp_contexts is None or hip_tp1_probe)
    )
    if not enabled:
        return function(*args, **kwargs)

    started = time.perf_counter()
    start_profile()
    profile = None
    try:
        return function(*args, **kwargs)
    finally:
        if hip_tp1_probe:
            # One real eager TP token identifies the slow GPU stage. Every
            # later token/request returns to the full TokenGraph fast path.
            model._cccp_hip_stage_probe_done = True
        try:
            profile = finish_profile()
        except Exception as profile_error:
            print(
                "[cccp-stage] "
                f"phase={phase} profile_error="
                f"{type(profile_error).__name__}:{profile_error}",
                flush=True,
            )
        if isinstance(profile, dict):
            wall_ms = (time.perf_counter() - started) * 1000.0
            tp_profile = profile.get("tensor_parallel", {})
            if isinstance(tp_profile, dict) and tp_profile:
                tp_totals = tp_profile.get("totals", {})
                print(
                    "[cccp-stage-tp] "
                    f"phase={phase} tokens={int(tokens)} "
                    f"wall={wall_ms:.1f}ms "
                    f"gpu_critical="
                    f"{float(tp_profile.get('critical_path_ms', 0.0)):.1f}ms "
                    f"attention="
                    f"{float(tp_totals.get('attention_ms', 0.0)):.1f}ms "
                    f"moe={float(tp_totals.get('moe_ms', 0.0)):.1f}ms "
                    f"ffn_post="
                    f"{float(tp_totals.get('ffn_post_ms', 0.0)):.1f}ms "
                    "probe=eager-component-graphs; "
                    "subsequent=TP1-TokenGraph",
                    flush=True,
                )
            totals = profile.get("totals_ms", {})
            moe = profile.get("moe_totals_ms", {})
            covered_ms = float(profile.get("covered_ms", 0.0) or 0.0)
            top_layers = profile.get("top_layers", [])
            slow = ",".join(
                f"L{int(item.get('layer', -1))}:"
                f"{float(item.get('total_ms', 0.0)):.1f}ms"
                for item in list(top_layers)[:3]
                if isinstance(item, dict)
            ) or "none"
            print(
                "[cccp-stage] "
                f"phase={phase} tokens={int(tokens)} wall={wall_ms:.1f}ms "
                f"covered={covered_ms:.1f}ms "
                f"attn_norm={float(totals.get('attn_hc_norm', 0.0)):.1f}ms "
                f"attention={float(totals.get('attention', 0.0)):.1f}ms "
                f"ffn_norm={float(totals.get('ffn_hc_norm', 0.0)):.1f}ms "
                f"moe={float(totals.get('moe', 0.0)):.1f}ms "
                f"ffn_post={float(totals.get('ffn_hc_post', 0.0)):.1f}ms "
                f"route={float(moe.get('route', 0.0)):.1f}ms "
                f"shared={float(moe.get('shared', 0.0)):.1f}ms "
                f"routed={float(moe.get('routed', 0.0)):.1f}ms "
                f"merge={float(moe.get('merge', 0.0)):.1f}ms "
                f"other={max(0.0, wall_ms - covered_ms):.1f}ms "
                f"slow_layers={slow}",
                flush=True,
            )


def _sample_top_p(logits: torch.Tensor, temp: float, top_p: float) -> int:
    """top-p（核）采样。"""
    probs = torch.softmax(logits.float() / max(temp, 1e-6), dim=-1)
    sp, si = torch.sort(probs, descending=True)
    cum = torch.cumsum(sp, 0)
    keep = (cum - sp) < top_p
    cand = si[keep]
    cp = sp[keep] / sp[keep].sum()
    return int(cand[torch.multinomial(cp, 1)].item())
