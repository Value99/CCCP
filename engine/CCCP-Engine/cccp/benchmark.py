"""可复核的 CCCP 单请求 decode 基准。

该入口只测模型加载完成后的自回归 decode；模型加载、prefill 和 warmup 单独记录，
不混入 token/s。生成固定步数且忽略 EOS，避免不同量化模型提前结束导致样本长度
不一致。结果可打印并保存为 JSON。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any

from .prefill import begin_prefill_block, end_prefill_block
from .presets import resolve_capacity_profile, resolve_preset


def _source_state() -> dict[str, Any]:
    root = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return {"git_commit": commit, "git_dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        commit = os.environ.get("CCCP_SOURCE_COMMIT")
        return {
            "git_commit": commit,
            "git_dirty": None if commit is None else False,
        }


def _cpu_hardware(torch: Any) -> dict[str, Any]:
    name = platform.processor() or platform.machine()
    cpuinfo = Path("/proc/cpuinfo")
    cpuinfo_text = ""
    instruction_sets: list[str] = []
    if cpuinfo.is_file():
        cpuinfo_text = cpuinfo.read_text(
            encoding="utf-8", errors="replace"
        )
        for line in cpuinfo_text.splitlines():
            if line.lower().startswith("model name"):
                name = line.split(":", 1)[-1].strip()
            if line.lower().startswith(("flags", "features")):
                flags = set(line.split(":", 1)[-1].strip().split())
                instruction_sets = [
                    feature
                    for feature in (
                        "avx2",
                        "avx512f",
                        "avx512bw",
                        "avx512vbmi",
                        "avx512_vnni",
                    )
                    if feature in flags
                ]
            if name and instruction_sets:
                break
    logical = os.cpu_count()
    physical = None
    memory_gib = None
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
        memory_gib = psutil.virtual_memory().total / 2**30
    except ImportError:
        pass
    if physical is None and cpuinfo_text:
        packages_and_cores: set[tuple[str, str]] = set()
        for block in cpuinfo_text.split("\n\n"):
            fields = {}
            for line in block.splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    fields[key.strip()] = value.strip()
            if "physical id" in fields and "core id" in fields:
                packages_and_cores.add(
                    (fields["physical id"], fields["core id"])
                )
        physical = len(packages_and_cores) or None
    numa_nodes = None
    numa_online = Path("/sys/devices/system/node/online")
    if numa_online.is_file():
        numa_nodes = numa_online.read_text(encoding="ascii").strip()
    return {
        "name": name,
        "architecture": platform.machine(),
        "physical_cores": physical,
        "logical_cpus": logical,
        "inference_threads": torch.get_num_threads(),
        "numa_nodes_online": numa_nodes,
        "memory_gib": memory_gib,
        "instruction_sets": instruction_sets,
        "torch": torch.__version__,
    }


def _process_memory() -> dict[str, float | None]:
    """返回当前进程的常驻内存与历史峰值，单位 GiB。"""
    rss_gib = None
    peak_rss_gib = None
    try:
        import psutil

        rss_gib = psutil.Process().memory_info().rss / 2**30
    except ImportError:
        pass
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            if line.startswith("VmHWM:"):
                peak_rss_gib = int(line.split()[1]) / 2**20
                break
    return {
        "rss_gib": rss_gib,
        "peak_rss_gib": peak_rss_gib,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="测量 CCCP 稳态单请求 decode token/s",
    )
    parser.add_argument("--model", help="CCCP 模型目录")
    parser.add_argument(
        "--kernel",
        choices=(
            "block-fp8",
            "block-fp8-batch",
            "block-fp8-grouped",
            "packed-moe-three",
            "kda",
        ),
        help="不加载模型，使用公共 CLI 对指定算子做 A/B 微基准",
    )
    parser.add_argument("--kernel-rows", type=int, default=8192)
    parser.add_argument("--kernel-cols", type=int, default=8192)
    parser.add_argument("--kernel-iterations", type=int, default=20)
    parser.add_argument("--kernel-batch", type=int, default=8)
    parser.add_argument(
        "--kernel-gate-bits", type=int, choices=range(8, 17), default=10
    )
    parser.add_argument(
        "--kernel-gate-vector", type=int, choices=(4, 8, 16), default=8
    )
    parser.add_argument(
        "--kernel-down-bits", type=int, choices=range(8, 17), default=8
    )
    parser.add_argument(
        "--kernel-down-vector", type=int, choices=(4, 8, 16), default=4
    )
    parser.add_argument(
        "--kernel-dtype",
        choices=("fp32", "bf16"),
        default="bf16",
        help="公共 CPU 算子微基准的激活/输出精度（默认 bf16）",
    )
    parser.add_argument(
        "--profile",
        choices=("auto", "ram", "resident", "mapped", "parallel"),
        default="auto",
    )
    parser.add_argument("--tp", type=int)
    parser.add_argument("--gpus", help="CUDA_VISIBLE_DEVICES，例如 7 或 0,1")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-ctx", type=int, default=4096)
    parser.add_argument("--prompt", default="请用中文简要介绍量化推理。")
    parser.add_argument(
        "--redact-prompt",
        action="store_true",
        help="报告只记录 prompt SHA-256 与 token 数，不写入原文",
    )
    parser.add_argument(
        "--prompt-repeat",
        type=int,
        default=1,
        help=(
            "将 --prompt 在 CLI 内重复指定次数，用于可复现的长上下文测试；"
            "不需要临时脚本或 shell 字符串拼接"
        ),
    )
    parser.add_argument(
        "--prompt-separator",
        default=" ",
        help="--prompt-repeat 各段之间的分隔符（默认一个空格）",
    )
    parser.add_argument(
        "--prefill-block-tokens",
        type=int,
        help=(
            "prefill token block; resident three-projection packed models "
            "default to 8192"
        ),
    )
    parser.add_argument(
        "--prefill-moe-batch",
        type=int,
        help=(
            "public packed-MoE prefill micro-batch in 1..8192; controls "
            "fixed activation scratch only"
        ),
    )
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument(
        "--speculative-ab",
        type=int,
        default=0,
        help=(
            "同一已加载模型依次运行贪心与无损块验证；"
            "值为最大草稿数，token 序列必须完全一致"
        ),
    )
    parser.add_argument(
        "--cpu-thread-sweep",
        help=(
            "CPU 模式在模型只加载一次的前提下扫描线程数，"
            "例如 48,96,192；每档都会独立 reset/prefill/warmup"
        ),
    )
    parser.add_argument(
        "--cpu-compile",
        choices=("auto", "off", "u16", "q4"),
        help=(
            "CPU packed 专家执行镜像：auto 按可用 RAM 自动选择，"
            "off 始终保持紧凑索引，u16 在 RAM 中在线编译为原生索引布局"
        ),
    )
    parser.add_argument(
        "--cpu-wait-idle-percent",
        type=float,
        help=(
            "CPU 模型加载后等待整机空闲率达到该百分比，"
            "避免并发任务污染正式跑分；例如 95"
        ),
    )
    parser.add_argument(
        "--cpu-wait-idle-samples",
        type=int,
        default=5,
        help="连续满足空闲率阈值的 1 秒采样数（默认 5）",
    )
    parser.add_argument(
        "--cpu-wait-timeout",
        type=int,
        default=1800,
        help="等待 CPU 空闲的最长秒数（默认 1800）",
    )
    parser.add_argument(
        "--cpu-contaminated-retries",
        type=int,
        default=3,
        help=(
            "CPU 正式计时期间若检测到外部负载，丢弃并重测的最大次数"
            "（默认 3）"
        ),
    )
    parser.add_argument("--cache-gb", type=float)
    parser.add_argument("--vram-gb", type=float)
    parser.add_argument(
        "--pin-gb",
        type=float,
        help=(
            "锁页 RAM 预算（GiB）；同时作用于普通 staged cache 与 "
            "packed archive 原地注册"
        ),
    )
    parser.add_argument(
        "--vram-limit-gb",
        type=float,
        help="整进程 CUDA allocator 硬上限（GiB）",
    )
    parser.add_argument(
        "--vram-reserve-gb",
        type=float,
        help=(
            "专家 arena、上下文与临时 workspace 共用的唯一显存总预留；"
            "用于 CLI 容量验收，不改变模型配置"
        ),
    )
    parser.add_argument(
        "--extreme",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="默认自动检测；可用 --extreme 强制或 --no-extreme 禁用",
    )
    parser.add_argument(
        "--extreme-placement",
        choices=("auto", "layer", "precision"),
        default="auto",
        help=(
            "极限模式专家放置：auto 对异构归档按量化精度预算选热点，"
            "layer 保留连续整层，precision 强制精度加权"
        ),
    )
    parser.add_argument(
        "--extreme-score-file",
        help=(
            "可选的公共专家常驻分数 JSON；支持 CCCP expert-preference "
            "审计，不提供时按 packed bit 预算推断"
        ),
    )
    parser.add_argument(
        "--save-route-scores",
        help=(
            "将本次 CLI 实测的逐层专家路由次数写成公共常驻分数 JSON；"
            "下次可通过 --extreme-score-file 复用"
        ),
    )
    parser.add_argument(
        "--extreme-load-workspace-gb",
        type=float,
        help=(
            "极限模式为加载峰值额外保留的 RAM；默认自动，严格容量"
            "验收后可缩小以增加动态专家槽"
        ),
    )
    parser.add_argument(
        "--dense-residency",
        choices=("auto", "gpu", "ram"),
        default="auto",
        help="auto 自动尝试 Dense GPU-only；gpu 容量不足即失败",
    )
    parser.add_argument(
        "--dense-bf16",
        help=(
            "Dense 常驻精度组：none、all，或以逗号分隔的 attention/"
            "compressor/embed/head/hyper/indexer/norm/shared；覆盖模型预设"
        ),
    )
    parser.add_argument(
        "--single-gpu-layer-graph",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="显式启用/禁用单卡固定地址层 Graph",
    )
    parser.add_argument("--json", help="保存完整结果的 JSON 路径")
    parser.add_argument(
        "--h2d-batch",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="显式启用/禁用 CUDA packed 专家批量 H2D",
    )
    parser.add_argument(
        "--probe-stages",
        action="store_true",
        help="测量结束后额外执行 1 token 分阶段探针；不计入吞吐",
    )
    parser.add_argument(
        "--probe-prefill",
        action="store_true",
        help="记录初始完整 Prefill 的逐层阶段耗时；仅用于性能审计",
    )
    return parser


def _device_steps(model: Any, logits: Any, steps: int, window: int):
    """GLM 贪心 decode：token 选择留在 GPU，按窗口回收结果。"""
    import torch

    output: list[int] = []
    for begin in range(0, steps, window):
        count = min(window, steps - begin)
        tokens = torch.empty(count, dtype=torch.long, device=logits.device)
        for index in range(count):
            torch.argmax(logits, out=tokens[index])
            logits = model.forward(tokens[index:index + 1])
        output.extend(tokens.cpu().tolist())
    return logits, output


def _host_steps(model: Any, logits: Any, steps: int):
    """DeepSeek decode：与当前生产生成循环一致，每步在主机取得 token id。"""
    output: list[int] = []
    for _ in range(steps):
        token = int(logits.argmax().item())
        output.append(token)
        logits = model.forward([token])
    return logits, output


def _steps(
    architecture: str,
    model: Any,
    logits: Any,
    count: int,
    window: int,
):
    if architecture == "glm":
        return _device_steps(model, logits, count, window)
    return _host_steps(model, logits, count)


def _model_prefill(model: Any, tokens: list[int]):
    """Run a benchmark Prefill through the production arena lifecycle."""

    pool = getattr(model, "pool", None)
    arena_active = begin_prefill_block(pool)
    try:
        return model.forward(tokens)
    finally:
        if arena_active:
            end_prefill_block(pool)


def _save_route_scores(pool: Any, output: str | Path) -> dict[str, Any]:
    """Persist measured Prefill routes before Decode graph construction."""

    store = getattr(pool, "store", None)
    counts = getattr(pool, "route_counts", None)
    if store is None or counts is None:
        raise RuntimeError(
            "the selected public expert pool cannot export route scores"
        )
    layers = sorted(int(layer) for layer in store.man.expert_files)
    expert_count = int(store.cfg["n_experts"])
    score_payload = {
        "format": "cccp-expert-residency-scores-v1",
        "scores": {
            f"{layer}:{expert}": float(counts.get((layer, expert), 0))
            for layer in layers
            for expert in range(expert_count)
        },
        "observations": int(sum(counts.values())),
        "source": "cccp benchmark CLI measured routes",
    }
    score_path = Path(output)
    score_path.parent.mkdir(parents=True, exist_ok=True)
    score_path.write_text(
        json.dumps(score_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "path": str(score_path),
        "observations": score_payload["observations"],
        "nonzero_experts": sum(
            value > 0 for value in score_payload["scores"].values()
        ),
    }


def _parse_cpu_thread_sweep(value: str | None) -> tuple[int, ...]:
    if not value:
        return ()
    counts: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            count = int(part)
        except ValueError as exc:
            raise ValueError(
                f"无效 CPU 线程数：{part!r}"
            ) from exc
        if count < 1:
            raise ValueError("CPU 线程数必须大于 0")
        if count not in counts:
            counts.append(count)
    if not counts:
        raise ValueError("--cpu-thread-sweep 至少需要一个线程数")
    return tuple(counts)


def _wait_for_cpu_idle(
    minimum_idle: float | None,
    stable_samples: int,
    timeout_seconds: int,
    *,
    sample_idle: Any | None = None,
) -> None:
    """Wait without unloading the model until the shared host is quiet."""
    if minimum_idle is None:
        return
    if not 0.0 < minimum_idle <= 100.0:
        raise ValueError("--cpu-wait-idle-percent 必须在 (0,100] 内")
    if stable_samples < 1:
        raise ValueError("--cpu-wait-idle-samples 必须大于 0")
    if timeout_seconds < 1:
        raise ValueError("--cpu-wait-timeout 必须大于 0")
    if sample_idle is None:
        try:
            import psutil
        except ImportError as exc:
            raise RuntimeError(
                "--cpu-wait-idle-percent 需要 psutil"
            ) from exc

        def sample_idle() -> float:
            return 100.0 - float(psutil.cpu_percent(interval=1.0))

    deadline = time.monotonic() + timeout_seconds
    consecutive = 0
    while True:
        idle = float(sample_idle())
        consecutive = consecutive + 1 if idle >= minimum_idle else 0
        print(
            f"cpu-idle={idle:.1f}% threshold={minimum_idle:.1f}% "
            f"stable={consecutive}/{stable_samples}",
            flush=True,
        )
        if consecutive >= stable_samples:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "等待整机 CPU 空闲超时；不生成污染跑分"
            )


def _cpu_activity_snapshot(enabled: bool) -> tuple[float, float, float] | None:
    """Capture aggregate, idle and current-process CPU seconds."""
    if not enabled:
        return None
    try:
        import psutil
    except ImportError as exc:
        raise RuntimeError("CPU 计时污染检测需要 psutil") from exc
    system = psutil.cpu_times()
    values = system._asdict()
    # Linux reports guest time inside user/nice as well; do not double count.
    total = sum(
        float(value)
        for name, value in values.items()
        if name not in {"guest", "guest_nice"}
    )
    idle = float(values.get("idle", 0.0)) + float(
        values.get("iowait", 0.0)
    )
    process = psutil.Process().cpu_times()
    process_seconds = sum(
        float(getattr(process, name, 0.0))
        for name in ("user", "system", "children_user", "children_system")
    )
    return total, idle, process_seconds


def _external_cpu_busy_percent(
    before: tuple[float, float, float] | None,
    after: tuple[float, float, float] | None,
) -> float | None:
    """Estimate CPU busy time not consumed by this benchmark process."""
    if before is None or after is None:
        return None
    total = after[0] - before[0]
    if total <= 0.0:
        return 0.0
    busy = total - (after[1] - before[1])
    own = max(0.0, after[2] - before[2])
    return max(0.0, busy - own) / total * 100.0


def _cpu_run_is_contaminated(
    external_busy_percent: float | None,
    minimum_idle_percent: float | None,
) -> bool:
    if external_busy_percent is None or minimum_idle_percent is None:
        return False
    return external_busy_percent > 100.0 - minimum_idle_percent


def _measure_stage_probe(
    *,
    torch: Any,
    engine: Any,
    architecture: str,
    logits: Any,
    window: int,
    device: str,
    wait_idle_percent: float | None,
    wait_idle_samples: int,
    wait_idle_timeout: int,
    contaminated_retries: int,
    replay_ids: list[int] | None = None,
) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    """Measure one accepted stage profile and reject external CPU overlap."""
    start_profile = getattr(engine.model, "start_profile", None)
    finish_profile = getattr(engine.model, "finish_profile", None)
    if not callable(start_profile) or not callable(finish_profile):
        raise RuntimeError(
            f"{architecture} 当前没有 CLI 分阶段探针"
        )
    discarded: list[dict[str, Any]] = []
    while True:
        if device == "cpu":
            _wait_for_cpu_idle(
                wait_idle_percent,
                wait_idle_samples,
                wait_idle_timeout,
            )
        start_profile()
        activity_before = _cpu_activity_snapshot(
            device == "cpu" and wait_idle_percent is not None
        )
        probe_started = time.perf_counter()
        logits, probe_tokens = _steps(
            architecture,
            engine.model,
            logits,
            1,
            window,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - probe_started) * 1000.0
        external_busy_percent = _external_cpu_busy_percent(
            activity_before,
            _cpu_activity_snapshot(
                device == "cpu" and wait_idle_percent is not None
            ),
        )
        probe = finish_profile()
        probe["wall_ms"] = wall_ms
        probe["tokens"] = probe_tokens
        probe["external_cpu_busy_percent"] = external_busy_percent
        if _cpu_run_is_contaminated(
            external_busy_percent,
            wait_idle_percent,
        ):
            probe["discard_reason"] = "external_cpu_busy"
            discarded.append(probe)
            print(
                "discard cpu stage probe "
                f"external_busy={external_busy_percent:.2f}%",
                flush=True,
            )
            if len(discarded) > contaminated_retries:
                raise RuntimeError(
                    "CPU 分阶段探针连续受到外部负载污染；"
                    "已拒绝生成误导性探针"
                )
            if replay_ids is not None:
                engine.reset()
                logits = _model_prefill(engine.model, replay_ids)
            continue
        return logits, probe, discarded


def _measure_cpu_thread_count(
    *,
    torch: Any,
    engine: Any,
    architecture: str,
    prompt_ids: list[int],
    threads: int,
    warmup: int,
    steps: int,
    repeat: int,
    window: int,
    probe_stages: bool = False,
    wait_idle_percent: float | None = None,
    wait_idle_samples: int = 5,
    wait_idle_timeout: int = 1800,
    contaminated_retries: int = 3,
) -> dict[str, Any]:
    """Measure one CPU thread setting without reloading model weights."""
    torch.set_num_threads(int(threads))
    effective_threads = int(torch.get_num_threads())
    engine.reset()
    prefill_started = time.perf_counter()
    logits = _model_prefill(engine.model, prompt_ids)
    prefill_ms = (time.perf_counter() - prefill_started) * 1000.0
    logits, warmup_tokens = _steps(
        architecture,
        engine.model,
        logits,
        warmup,
        window,
    )
    runs: list[dict[str, Any]] = []
    discarded_runs: list[dict[str, Any]] = []
    decoded_tokens: list[int] = []
    while len(runs) < repeat:
        _wait_for_cpu_idle(
            wait_idle_percent,
            wait_idle_samples,
            wait_idle_timeout,
        )
        measured_position = int(getattr(engine.model, "pos", 0))
        activity_before = _cpu_activity_snapshot(
            wait_idle_percent is not None
        )
        started = time.perf_counter()
        logits, tokens = _steps(
            architecture,
            engine.model,
            logits,
            steps,
            window,
        )
        wall_ms = (time.perf_counter() - started) * 1000.0
        external_busy_percent = _external_cpu_busy_percent(
            activity_before,
            _cpu_activity_snapshot(wait_idle_percent is not None),
        )
        run = {
            "repeat": len(runs) + 1,
            "measured_position": measured_position,
            "steps": steps,
            "wall_ms": wall_ms,
            "external_cpu_busy_percent": external_busy_percent,
            "throughput_tok_s": steps / (wall_ms / 1000.0),
            "tokens": tokens,
        }
        if _cpu_run_is_contaminated(
            external_busy_percent,
            wait_idle_percent,
        ):
            run["discard_reason"] = "external_cpu_busy"
            discarded_runs.append(run)
            print(
                "discard cpu thread-sweep run "
                f"threads={effective_threads} "
                f"position={measured_position} "
                f"external_busy={external_busy_percent:.2f}%",
                flush=True,
            )
            if len(discarded_runs) > contaminated_retries:
                raise RuntimeError(
                    "CPU 线程扫描连续受到外部负载污染；"
                    "已拒绝生成误导性跑分"
                )
            engine.reset()
            logits = _model_prefill(
                engine.model,
                prompt_ids + warmup_tokens + decoded_tokens
            )
            continue
        decoded_tokens.extend(tokens)
        runs.append(run)
    throughputs = [float(run["throughput_tok_s"]) for run in runs]
    result = {
        "requested_threads": int(threads),
        "effective_threads": effective_threads,
        "prefill_ms": prefill_ms,
        "warmup_tokens": warmup_tokens,
        "throughput_tok_s_median": statistics.median(throughputs),
        "throughput_tok_s_min": min(throughputs),
        "throughput_tok_s_max": max(throughputs),
        "decoded_measured_text": engine.decode(decoded_tokens),
        "runs": runs,
    }
    if discarded_runs:
        result["discarded_cpu_runs"] = discarded_runs
    if probe_stages:
        logits, probe, discarded_probes = _measure_stage_probe(
            torch=torch,
            engine=engine,
            architecture=architecture,
            logits=logits,
            window=window,
            device="cpu",
            wait_idle_percent=wait_idle_percent,
            wait_idle_samples=wait_idle_samples,
            wait_idle_timeout=wait_idle_timeout,
            contaminated_retries=contaminated_retries,
            replay_ids=prompt_ids + warmup_tokens + decoded_tokens,
        )
        result["stage_probe"] = probe
        if discarded_probes:
            result["discarded_cpu_stage_probes"] = discarded_probes
    print(
        f"cpu-measure threads={effective_threads} "
        f"throughput={result['throughput_tok_s_median']:.3f} token/s",
        flush=True,
    )
    return result


def _apply_preset_environment(args: argparse.Namespace, preset: Any) -> None:
    if args.gpus:
        devices = [part.strip() for part in args.gpus.split(",") if part.strip()]
        if len(devices) < preset.tp:
            raise ValueError(
                f"--gpus 只给出 {len(devices)} 张卡，但 tp={preset.tp}"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    for key, value in preset.environment.items():
        os.environ.setdefault(key, value)
    if args.dense_bf16 is not None:
        os.environ["CCCP_DENSE_BF16"] = args.dense_bf16
    if args.vram_reserve_gb is not None:
        if args.vram_reserve_gb <= 0:
            raise ValueError("--vram-reserve-gb 必须大于 0")
        os.environ["CCCP_VRAM_RESERVE_GB"] = str(args.vram_reserve_gb)
    if args.single_gpu_layer_graph is not None:
        os.environ["CCCP_SINGLE_GPU_LAYER_GRAPH"] = (
            "1" if args.single_gpu_layer_graph else "0"
        )
    if args.h2d_batch is not None:
        os.environ["CCCP_H2D_BATCH"] = "1" if args.h2d_batch else "0"
    if preset.ep_layout is not None:
        os.environ.setdefault("CCCP_EP_LAYOUT", preset.ep_layout)
    # 固定输出缓冲可减少 GLM decode 中不必要的临时分配。
    os.environ.setdefault("CCCP_STATIC_LM_OUTPUT", "1")


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.kernel:
        if args.device == "cuda":
            if args.kernel not in (
                "block-fp8",
                "block-fp8-grouped",
                "packed-moe-three",
            ):
                raise SystemExit(
                    "CUDA kernel CLI currently supports block-fp8 and "
                    "block-fp8-grouped and packed-moe-three"
                )
            if args.gpus:
                os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
            import torch

            if args.kernel == "packed-moe-three":
                from .ops import packed_moe_topk

                hidden = int(args.kernel_cols)
                intermediate = int(args.kernel_rows)
                top_k = int(args.kernel_batch)
                if hidden % 8 or intermediate % 4:
                    raise SystemExit(
                        "CUDA packed-moe-three requires cols%8=0 and rows%4=0"
                    )
                if not 1 <= top_k <= 16:
                    raise SystemExit(
                        "CUDA packed-moe-three requires 1 <= batch/top-k <= 16"
                    )

                def pack_indices(
                    indices: torch.Tensor, bits: int
                ) -> torch.Tensor:
                    from math import gcd

                    group_size = 8 // gcd(bits, 8)
                    groups = indices.reshape(-1, group_size).to(torch.int64)
                    bytes_per_group = bits * group_size // 8
                    packed = torch.empty(
                        groups.shape[0], bytes_per_group, dtype=torch.uint8
                    )
                    for byte in range(bytes_per_group):
                        bit_start = byte * 8
                        item = bit_start // bits
                        offset = bit_start % bits
                        value = groups[:, item] >> offset
                        if offset + 8 > bits and item + 1 < group_size:
                            value.bitwise_or_(
                                groups[:, item + 1] << (bits - offset)
                            )
                        packed[:, byte] = (value & 0xFF).to(torch.uint8)
                    return packed.reshape(-1)

                torch.manual_seed(8202)
                device = torch.device("cuda:0")
                value = torch.randn(
                    1, hidden, dtype=torch.bfloat16, device=device
                )
                route_ids = torch.arange(
                    top_k, dtype=torch.long, device=device
                )
                route_weights = torch.rand(top_k, device=device)
                route_weights = (
                    route_weights / route_weights.sum()
                ).float()
                gate_bits = int(args.kernel_gate_bits)
                gate_vector = int(args.kernel_gate_vector)
                down_bits = int(args.kernel_down_bits)
                down_vector = int(args.kernel_down_vector)
                if hidden % gate_vector or intermediate % down_vector:
                    raise SystemExit(
                        "packed-moe-three dimensions must be divisible by "
                        "their VQ vectors"
                    )
                gate_cb = torch.randn(
                    1 << gate_bits,
                    gate_vector,
                    dtype=torch.bfloat16,
                    device=device,
                )
                up_cb = torch.randn(
                    1 << gate_bits,
                    gate_vector,
                    dtype=torch.bfloat16,
                    device=device,
                )
                down_cb = torch.randn(
                    1 << down_bits,
                    down_vector,
                    dtype=torch.bfloat16,
                    device=device,
                )
                metadata = torch.zeros(15, top_k, dtype=torch.long)
                retained: list[torch.Tensor] = [gate_cb, up_cb, down_cb]
                definitions = (
                    (
                        gate_bits, gate_vector, 1 << gate_bits,
                        intermediate, hidden, gate_cb,
                    ),
                    (
                        gate_bits, gate_vector, 1 << gate_bits,
                        intermediate, hidden, up_cb,
                    ),
                    (
                        down_bits, down_vector, 1 << down_bits,
                        hidden, intermediate, down_cb,
                    ),
                )
                dtype_tags = {
                    8: 0, 16: 1, 12: 2, 14: 3, 10: 4,
                    9: 5, 11: 6, 13: 7, 15: 8,
                }
                for expert in range(top_k):
                    for projection, (
                        bits,
                        dimension,
                        codebook_size,
                        rows,
                        columns,
                        codebook,
                    ) in enumerate(definitions):
                        indices = torch.randint(
                            codebook_size,
                            (rows, columns // dimension),
                            dtype=torch.int64,
                        )
                        packed_cpu = pack_indices(indices, bits).contiguous()
                        packed = packed_cpu.to(device)
                        retained.append(packed)
                        base = 5 * projection
                        metadata[base : base + 5, expert] = torch.tensor(
                            [
                                packed.data_ptr(),
                                codebook.data_ptr(),
                                columns // dimension,
                                dimension,
                                dtype_tags[bits],
                            ],
                            dtype=torch.long,
                        )
                metadata = metadata.to(device)
                hidden_workspace = torch.empty(
                    top_k,
                    2 * intermediate,
                    dtype=torch.bfloat16,
                    device=device,
                )
                output_workspace = torch.empty(
                    top_k,
                    hidden,
                    dtype=torch.bfloat16,
                    device=device,
                )
                result = torch.empty(
                    hidden, dtype=torch.float32, device=device
                )

                def run() -> torch.Tensor:
                    return packed_moe_topk(
                        value,
                        route_ids,
                        route_weights,
                        metadata,
                        activation="situ",
                        activation_beta=4.0,
                        activation_linear_beta=25.0,
                        hidden_workspace=hidden_workspace,
                        output_workspace=output_workspace,
                        result=result,
                        grouped_prefix=-1,
                        packed_formats=(
                            f"p{gate_bits}", f"p{gate_bits}",
                            f"p{down_bits}",
                        ),
                        code_dims=(gate_vector, gate_vector, down_vector),
                        codebook_sizes=(
                            1 << gate_bits, 1 << gate_bits, 1 << down_bits,
                        ),
                    )

                original_down = os.environ.get(
                    "CCCP_PROJECTION_DOWN_REDUCE"
                )

                def measure(down_reduce: bool) -> tuple[float, torch.Tensor]:
                    os.environ["CCCP_PROJECTION_DOWN_REDUCE"] = (
                        "1" if down_reduce else "0"
                    )
                    for _ in range(5):
                        run()
                    torch.cuda.synchronize(device)
                    started = torch.cuda.Event(enable_timing=True)
                    finished = torch.cuda.Event(enable_timing=True)
                    started.record()
                    for _ in range(args.kernel_iterations):
                        run()
                    finished.record()
                    finished.synchronize()
                    return (
                        started.elapsed_time(finished)
                        / args.kernel_iterations,
                        result.clone(),
                    )

                try:
                    split_ms, split_result = measure(False)
                    reduce_ms, reduce_result = measure(True)
                    os.environ["CCCP_PROJECTION_DOWN_REDUCE"] = "0"
                    from .ops.packed_graph import (
                        FixedPackedMoEGraph,
                        PackedMoEGraphSpec,
                    )

                    graph = FixedPackedMoEGraph(
                        value,
                        route_ids,
                        route_weights,
                        metadata,
                        hidden_workspace,
                        output_workspace,
                        result,
                        PackedMoEGraphSpec(
                            activation="situ",
                            activation_beta=4.0,
                            activation_linear_beta=25.0,
                            limit=0.0,
                            top_k=top_k,
                            grouped_prefix=-1,
                            packed_formats=(
                                f"p{gate_bits}", f"p{gate_bits}",
                                f"p{down_bits}",
                            ),
                            code_dims=(
                                gate_vector, gate_vector, down_vector,
                            ),
                            codebook_sizes=(
                                1 << gate_bits,
                                1 << gate_bits,
                                1 << down_bits,
                            ),
                        ),
                    )
                    graph.capture()
                    for _ in range(5):
                        graph.run()
                    torch.cuda.synchronize(device)
                    graph_started = torch.cuda.Event(enable_timing=True)
                    graph_finished = torch.cuda.Event(enable_timing=True)
                    graph_started.record()
                    for _ in range(args.kernel_iterations):
                        graph.run()
                    graph_finished.record()
                    graph_finished.synchronize()
                    graph_ms = (
                        graph_started.elapsed_time(graph_finished)
                        / args.kernel_iterations
                    )
                    graph_result = result.clone()
                finally:
                    if original_down is None:
                        os.environ.pop(
                            "CCCP_PROJECTION_DOWN_REDUCE", None
                        )
                    else:
                        os.environ[
                            "CCCP_PROJECTION_DOWN_REDUCE"
                        ] = original_down
                difference = (split_result - reduce_result).abs()
                selected_ms = min(split_ms, reduce_ms, graph_ms)
                payload_bytes = sum(item.nbytes for item in retained[3:])
                projection_bytes = tuple(
                    rows * (columns // dimension) * bits // 8
                    for bits, dimension, _size, rows, columns, _cb
                    in definitions
                )
                expert_bytes = sum(projection_bytes)
                transfer_source = torch.empty(
                    top_k,
                    expert_bytes,
                    dtype=torch.uint8,
                    pin_memory=True,
                )
                transfer_target = torch.empty(
                    top_k,
                    expert_bytes,
                    dtype=torch.uint8,
                    device=device,
                )
                transfer_stream = torch.cuda.Stream(device=device)
                transfer_iterations = min(
                    50, int(args.kernel_iterations)
                )

                def measure_transfer(coalesced: bool) -> float:
                    with torch.cuda.stream(transfer_stream):
                        started = torch.cuda.Event(enable_timing=True)
                        finished = torch.cuda.Event(enable_timing=True)
                        started.record(transfer_stream)
                        for _ in range(transfer_iterations):
                            for expert in range(top_k):
                                if coalesced:
                                    transfer_target[expert].copy_(
                                        transfer_source[expert],
                                        non_blocking=True,
                                    )
                                    continue
                                offset = 0
                                for count in projection_bytes:
                                    transfer_target[
                                        expert, offset : offset + count
                                    ].copy_(
                                        transfer_source[
                                            expert, offset : offset + count
                                        ],
                                        non_blocking=True,
                                    )
                                    offset += count
                        finished.record(transfer_stream)
                    finished.synchronize()
                    return (
                        started.elapsed_time(finished)
                        / transfer_iterations
                    )

                split_transfer_ms = measure_transfer(False)
                coalesced_transfer_ms = measure_transfer(True)
                output = {
                    "kernel": "packed-moe-three",
                    "device": "cuda",
                    "layout": (
                        f"p{gate_bits}d{gate_vector}/"
                        f"p{gate_bits}d{gate_vector}/"
                        f"p{down_bits}d{down_vector}"
                    ),
                    "hidden": hidden,
                    "intermediate": intermediate,
                    "top_k": top_k,
                    "iterations": int(args.kernel_iterations),
                    "warps": int(
                        os.environ.get("CCCP_PROJECTION_WARPS", "16")
                    ),
                    "p10_shared": os.environ.get(
                        "CCCP_P10_SHARED", "1"
                    ) != "0",
                    "p10_rows_per_warp": int(
                        os.environ.get("CCCP_P10_ROWS", "4")
                    ),
                    "split_down_ms": split_ms,
                    "reduce_down_ms": reduce_ms,
                    "graph_ms": graph_ms,
                    "selected_ms": selected_ms,
                    "selected": (
                        "graph"
                        if graph_ms == selected_ms
                        else (
                            "split" if split_ms <= reduce_ms else "reduce"
                        )
                    ),
                    "speedup": min(split_ms, reduce_ms) / selected_ms,
                    "payload_gib": payload_bytes / 2**30,
                    "effective_gib_s": (
                        (payload_bytes / 2**30) / (selected_ms / 1000.0)
                    ),
                    "split_transfer_ms": split_transfer_ms,
                    "coalesced_transfer_ms": coalesced_transfer_ms,
                    "transfer_speedup": (
                        split_transfer_ms / coalesced_transfer_ms
                    ),
                    "coalesced_transfer_gib_s": (
                        (payload_bytes / 2**30)
                        / (coalesced_transfer_ms / 1000.0)
                    ),
                    "max_abs_between_paths": float(difference.max()),
                    "mean_abs_between_paths": float(difference.mean()),
                    "max_abs_graph_vs_split": float(
                        (graph_result - split_result).abs().max()
                    ),
                    "finite": bool(torch.isfinite(split_result).all()),
                    "expanded_weight_bytes": 0,
                }
                print(json.dumps(output, ensure_ascii=False, indent=2))
                if args.json:
                    output_path = Path(args.json)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_text(
                        json.dumps(output, ensure_ascii=False, indent=2)
                        + "\n",
                        encoding="utf-8",
                    )
                return

            from .kernels import BlockFP8Weight, ProjectionGroup
            from .ops import linear

            rows = int(args.kernel_rows)
            cols = int(args.kernel_cols)
            block = 128
            if rows % block or cols % block:
                raise SystemExit(
                    "CUDA block-FP8 rows/cols must be multiples of 128"
                )
            encoded = torch.randn(
                rows, cols, device="cuda", dtype=torch.bfloat16
            ).to(torch.float8_e4m3fn)
            weight = BlockFP8Weight(
                encoded.view(torch.uint8),
                torch.ones(
                    rows // block,
                    cols // block,
                    dtype=torch.float32,
                    device="cuda",
                ),
                cols,
                block,
            )
            value = torch.randn(
                1, cols, dtype=torch.bfloat16, device="cuda"
            )
            output = torch.empty(
                1, rows, dtype=torch.float32, device="cuda"
            )
            group = None
            if args.kernel == "block-fp8-grouped":
                split = (rows // 2 // block) * block
                if split == 0 or split == rows:
                    raise SystemExit(
                        "grouped CUDA benchmark requires rows >= 256"
                    )
                group = ProjectionGroup((
                    weight.row_view(0, split),
                    weight.row_view(split, rows),
                ))

            def measure(target) -> float:
                for _ in range(5):
                    linear(value, target, output=output)
                torch.cuda.synchronize()
                started = torch.cuda.Event(enable_timing=True)
                finished = torch.cuda.Event(enable_timing=True)
                started.record()
                for _ in range(args.kernel_iterations):
                    linear(value, target, output=output)
                finished.record()
                finished.synchronize()
                return (
                    started.elapsed_time(finished)
                    / args.kernel_iterations
                )

            single_ms = measure(weight)
            grouped_ms = measure(group) if group is not None else None
            selected_ms = grouped_ms if grouped_ms is not None else single_ms
            payload_gib = (weight.q.nbytes + weight.s.nbytes) / 2**30
            result = {
                "kernel": args.kernel,
                "device": "cuda",
                "rows": rows,
                "cols": cols,
                "iterations": int(args.kernel_iterations),
                "warps": int(os.environ.get("CCCP_FP8_GEMV_WARPS", "8")),
                "single_ms": single_ms,
                "grouped_ms": grouped_ms,
                "speedup": (
                    single_ms / grouped_ms
                    if grouped_ms is not None
                    else 1.0
                ),
                "payload_gib": payload_gib,
                "effective_gib_s": payload_gib / (selected_ms / 1000.0),
                "expanded_weight_bytes": 0,
            }
            print(json.dumps(result, ensure_ascii=False, indent=2))
            if args.json:
                output_path = Path(args.json)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            return
        if args.device != "cpu":
            raise SystemExit("--kernel block-fp8 当前只支持 --device cpu")
        if (
            args.kernel_rows < 1
            or args.kernel_cols < 1
            or args.kernel_iterations < 1
        ):
            raise SystemExit("kernel rows/cols/iterations 必须大于 0")
        import torch

        from .cpuext import (
            block_fp8_gemv_profile,
            configure_cpu_threads,
            prebuild,
            reset_block_fp8_gemv_profile,
        )
        from .kernels import BlockFP8Weight, ProjectionGroup
        from .ops import linear

        configure_cpu_threads()
        prebuild()
        if args.kernel == "kda":
            from .cpuext import gated_rmsnorm_cpu, kda_recurrent_cpu

            heads, dimension = 96, 128
            values = [
                torch.randn(heads, dimension, dtype=torch.bfloat16)
                for _ in range(5)
            ]
            beta = torch.randn(heads)
            a_log = torch.randn(heads)
            dt_bias = torch.randn(heads * dimension)
            norm_weight = torch.randn(dimension, dtype=torch.bfloat16)
            workspace = torch.empty(3 * heads * dimension)
            separate_state = torch.zeros(heads, dimension, dimension)
            fused_state = torch.zeros_like(separate_state)
            separate_output = torch.empty_like(values[0])
            fused_output = torch.empty_like(values[0])

            def measure_kda(fused: bool) -> float:
                samples = []
                for index in range(args.kernel_iterations + 3):
                    started = time.perf_counter()
                    kda_recurrent_cpu(
                        values[0], values[1], values[2], values[3],
                        beta, a_log, dt_bias,
                        fused_state if fused else separate_state,
                        workspace,
                        fused_output if fused else separate_output,
                        -5.0,
                        output_gate=values[4] if fused else None,
                        norm_weight=norm_weight if fused else None,
                        norm_eps=1.0e-5 if fused else 0.0,
                    )
                    if not fused:
                        gated_rmsnorm_cpu(
                            separate_output,
                            values[4],
                            norm_weight,
                            separate_output,
                            1.0e-5,
                        )
                    elapsed = (time.perf_counter() - started) * 1000.0
                    if index >= 3:
                        samples.append(elapsed)
                return statistics.median(samples)

            separate_ms = measure_kda(False)
            fused_ms = measure_kda(True)
            result = {
                "kernel": "kda",
                "heads": heads,
                "dimension": dimension,
                "iterations": int(args.kernel_iterations),
                "threads": int(torch.get_num_threads()),
                "separate_ms": separate_ms,
                "fused_ms": fused_ms,
                "speedup": separate_ms / fused_ms,
            }
            print(json.dumps(result, ensure_ascii=False, indent=2))
            if args.json:
                output_path = Path(args.json)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            return
        if args.kernel == "block-fp8-grouped":
            cols = int(args.kernel_cols)
            block = 128
            row_counts = (12288, 12288, 12288, 12288, 128, 96)
            weights = []
            for index, rows in enumerate(row_counts):
                raw = torch.randint(
                    0, 255, (rows, cols), dtype=torch.uint8
                )
                scales = torch.ones(
                    (rows + block - 1) // block,
                    (cols + block - 1) // block,
                    dtype=torch.float32,
                )
                weight = BlockFP8Weight(raw, scales, cols, block)
                weights.append(
                    weight.to_block_major() if index < 4 else weight
                )
            group = ProjectionGroup(weights)
            value = torch.randn(1, cols, dtype=torch.bfloat16)
            output = torch.empty(
                1, sum(row_counts), dtype=torch.bfloat16
            )

            def run_individual() -> None:
                offset = 0
                for weight, rows in zip(weights, row_counts):
                    linear(
                        value,
                        weight,
                        output=output.narrow(-1, offset, rows),
                    )
                    offset += rows

            def run_grouped() -> None:
                linear(value, group, output=output)

            def measure_call(callable_) -> float:
                for _ in range(3):
                    callable_()
                samples = []
                for _ in range(args.kernel_iterations):
                    started = time.perf_counter()
                    callable_()
                    samples.append(
                        (time.perf_counter() - started) * 1000.0
                    )
                return statistics.median(samples)

            individual_ms = measure_call(run_individual)
            from .cpuext import reset_block_fp8_gemv_profile

            reset_block_fp8_gemv_profile()
            grouped_ms = measure_call(run_grouped)
            profile = block_fp8_gemv_profile()
            result = {
                "kernel": "block-fp8-grouped",
                "rows": list(row_counts),
                "cols": cols,
                "iterations": int(args.kernel_iterations),
                "threads": int(torch.get_num_threads()),
                "individual_ms": individual_ms,
                "mixed_grouped_ms": grouped_ms,
                "speedup": individual_ms / grouped_ms,
                "grouped_native_calls_per_iteration": (
                    profile["calls"] / (args.kernel_iterations + 3)
                ),
                "expanded_weight_bytes": 0,
            }
            print(json.dumps(result, ensure_ascii=False, indent=2))
            if args.json:
                output_path = Path(args.json)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            return
        if args.kernel == "packed-moe-three":
            from .ops import packed_moe_selected_topk
            from .store import PackedVQWeight

            torch.manual_seed(8202)
            hidden = int(args.kernel_cols)
            intermediate = int(args.kernel_rows)
            if hidden % 8 or intermediate % 4:
                raise ValueError(
                    "packed-moe-three requires cols%8=0 and rows%4=0"
                )
            top_k = int(args.kernel_batch)
            gate_codebook = torch.randn(1024, 8, dtype=torch.float32)
            up_codebook = torch.randn(1024, 8, dtype=torch.float32)
            down_codebook = torch.randn(256, 4, dtype=torch.float32)

            def packed_weight(
                rows: int,
                cols: int,
                bits: int,
                codebook: torch.Tensor,
            ) -> PackedVQWeight:
                byte_count = rows * (cols // codebook.shape[1]) * bits // 8
                return PackedVQWeight(
                    torch.randint(
                        0,
                        256,
                        (byte_count,),
                        dtype=torch.uint8,
                    ),
                    codebook,
                    rows,
                    cols,
                    bits,
                )

            experts = tuple(
                (
                    packed_weight(
                        intermediate, hidden, 10, gate_codebook
                    ),
                    packed_weight(
                        intermediate, hidden, 10, up_codebook
                    ),
                    packed_weight(
                        hidden, intermediate, 8, down_codebook
                    ),
                )
                for _ in range(top_k)
            )
            value = torch.randn(1, hidden, dtype=torch.float32)
            route_weights = torch.full(
                (top_k,), 1.0 / top_k, dtype=torch.float32
            )

            def run(selected_experts) -> torch.Tensor:
                output = packed_moe_selected_topk(
                    value,
                    selected_experts,
                    route_weights,
                    activation="situ",
                    activation_beta=4.0,
                    activation_linear_beta=25.0,
                )
                if output is None:
                    raise RuntimeError(
                        "registered packed three-projection CPU op refused input"
                    )
                return output

            original_mode = os.environ.get("CCCP_CPU_PACKED_SINGLE_TEAM")
            original_rows16 = os.environ.get("CCCP_CPU_PACKED_ROWS16")

            def measure(
                mode: str,
                rows16: str,
                selected_experts,
            ) -> tuple[list[float], torch.Tensor]:
                os.environ["CCCP_CPU_PACKED_SINGLE_TEAM"] = mode
                os.environ["CCCP_CPU_PACKED_ROWS16"] = rows16
                for _ in range(3):
                    run(selected_experts)
                samples = []
                output = run(selected_experts)
                for _ in range(args.kernel_iterations):
                    started = time.perf_counter()
                    output = run(selected_experts)
                    samples.append(
                        (time.perf_counter() - started) * 1000.0
                    )
                return samples, output.clone()

            try:
                baseline_samples, baseline_output = measure("0", "0", experts)
                rows1_samples, rows1_output = measure("1", "0", experts)
                fused_samples, fused_output = measure("1", "1", experts)
            finally:
                if original_mode is None:
                    os.environ.pop("CCCP_CPU_PACKED_SINGLE_TEAM", None)
                else:
                    os.environ["CCCP_CPU_PACKED_SINGLE_TEAM"] = original_mode
                if original_rows16 is None:
                    os.environ.pop("CCCP_CPU_PACKED_ROWS16", None)
                else:
                    os.environ["CCCP_CPU_PACKED_ROWS16"] = original_rows16
            baseline_ms = statistics.median(baseline_samples)
            rows1_ms = statistics.median(rows1_samples)
            fused_ms = statistics.median(fused_samples)
            max_abs_diff = float(
                (baseline_output - fused_output).abs().max()
            )
            rows1_max_abs_diff = float(
                (baseline_output - rows1_output).abs().max()
            )
            from .cpuext import make_packed_three_layer_cpu

            resident = make_packed_three_layer_cpu(tuple(experts * 4))
            if resident is None:
                raise RuntimeError("resident packed layer refused input")
            resident_ids = torch.arange(top_k, dtype=torch.long)

            def run_resident() -> torch.Tensor:
                output = resident.forward(
                    value.float().contiguous(),
                    resident_ids,
                    route_weights.float().contiguous(),
                    0.0,
                    "situ",
                    4.0,
                    25.0,
                )
                if output.numel() == 0:
                    raise RuntimeError(
                        "resident packed layer rejected a shared-codebook route"
                    )
                return output

            for _ in range(3):
                run_resident()
            resident_samples = []
            resident_output = run_resident().clone()
            for _ in range(args.kernel_iterations):
                started = time.perf_counter()
                resident_output = run_resident()
                resident_samples.append(
                    (time.perf_counter() - started) * 1000.0
                )
            resident_output = resident_output.clone()
            resident_ms = statistics.median(resident_samples)
            resident_max_abs_diff = float(
                (baseline_output - resident_output).abs().max()
            )
            result = {
                "kernel": "packed-moe-three",
                "layout": "p10/p10/p8",
                "top_k": top_k,
                "hidden": hidden,
                "intermediate": intermediate,
                "iterations": int(args.kernel_iterations),
                "threads": int(torch.get_num_threads()),
                "baseline_ms": baseline_ms,
                "single_team_rows1_ms": rows1_ms,
                "single_team_ms": fused_ms,
                "single_team_speedup": baseline_ms / fused_ms,
                "rows16_speedup": rows1_ms / fused_ms,
                "resident_ms": resident_ms,
                "resident_speedup": baseline_ms / resident_ms,
                "resident_over_selected_speedup": fused_ms / resident_ms,
                "baseline_min_ms": min(baseline_samples),
                "single_team_min_ms": min(fused_samples),
                "max_abs_diff": max_abs_diff,
                "rows1_max_abs_diff": rows1_max_abs_diff,
                "resident_max_abs_diff": resident_max_abs_diff,
                "checksum": float(fused_output.double().sum()),
                "expanded_index_bytes": 0,
            }
            print(json.dumps(result, ensure_ascii=False, indent=2))
            if args.json:
                output_path = Path(args.json)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            return
        rows = int(args.kernel_rows)
        cols = int(args.kernel_cols)
        block = 128
        raw = torch.randint(0, 255, (rows, cols), dtype=torch.uint8)
        # E4M3FN reserves magnitude 0x7f for NaN.  Kernel throughput tests
        # need deterministic finite outputs so the sequential/block equality
        # gate remains meaningful.
        raw.masked_fill_(raw.bitwise_and(0x7F) == 0x7F, 0)
        scales = torch.ones(
            (rows + block - 1) // block,
            (cols + block - 1) // block,
            dtype=torch.float32,
        )
        row_major = BlockFP8Weight(raw, scales, cols, block)
        block_major = row_major.to_block_major()
        batch = (
            int(args.kernel_batch)
            if args.kernel == "block-fp8-batch"
            else 1
        )
        if not 1 <= batch <= 16:
            raise SystemExit("--kernel-batch 必须在 1..16")
        kernel_dtype = (
            torch.float32
            if args.kernel_dtype == "fp32"
            else torch.bfloat16
        )
        value = torch.randn(batch, cols, dtype=kernel_dtype)
        output = torch.empty(batch, rows, dtype=kernel_dtype)

        if args.kernel == "block-fp8-batch":
            sequential_output = torch.empty_like(output)

            def run_sequential() -> None:
                for token in range(batch):
                    linear(
                        value[token:token + 1],
                        block_major,
                        output=sequential_output[token:token + 1],
                    )

            def run_block() -> None:
                linear(value, block_major, output=output)

            def measure_call(callable_) -> float:
                for _ in range(3):
                    callable_()
                samples = []
                for _ in range(args.kernel_iterations):
                    started = time.perf_counter()
                    callable_()
                    samples.append(
                        (time.perf_counter() - started) * 1000.0
                    )
                return statistics.median(samples)

            sequential_ms = measure_call(run_sequential)
            reset_block_fp8_gemv_profile()
            block_ms = measure_call(run_block)
            max_abs_diff = float(
                (sequential_output.float() - output.float()).abs().max()
            )
            result = {
                "kernel": "block-fp8-batch",
                "rows": rows,
                "cols": cols,
                "batch": batch,
                "iterations": int(args.kernel_iterations),
                "threads": int(torch.get_num_threads()),
                "sequential_ms": sequential_ms,
                "block_ms": block_ms,
                "speedup": sequential_ms / block_ms,
                "accepted_token_equivalent_tok_s": batch * 1000.0 / block_ms,
                "max_abs_diff": max_abs_diff,
                "compact_bytes": int(block_major.nbytes),
                "expanded_weight_bytes": 0,
                "layout_profile": block_fp8_gemv_profile(),
            }
            print(json.dumps(result, ensure_ascii=False, indent=2))
            if args.json:
                output_path = Path(args.json)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            return

        def measure(weight) -> float:
            for _ in range(3):
                linear(value, weight, output=output)
            samples = []
            for _ in range(args.kernel_iterations):
                started = time.perf_counter()
                linear(value, weight, output=output)
                samples.append((time.perf_counter() - started) * 1000.0)
            return statistics.median(samples)

        row_ms = measure(row_major)
        previous_rows8 = os.environ.get("CCCP_CPU_BLOCK_FP8_ROWS8")
        try:
            os.environ["CCCP_CPU_BLOCK_FP8_ROWS8"] = "0"
            block_rows4_ms = measure(block_major)
            os.environ["CCCP_CPU_BLOCK_FP8_ROWS8"] = "1"
            block_rows8_ms = measure(block_major)
        finally:
            if previous_rows8 is None:
                os.environ.pop("CCCP_CPU_BLOCK_FP8_ROWS8", None)
            else:
                os.environ["CCCP_CPU_BLOCK_FP8_ROWS8"] = previous_rows8
        result = {
            "kernel": "block-fp8",
            "rows": rows,
            "cols": cols,
            "iterations": int(args.kernel_iterations),
            "threads": int(torch.get_num_threads()),
            "dtype": args.kernel_dtype,
            "row_major_ms": row_ms,
            "block_major32_rows4_ms": block_rows4_ms,
            "block_major32_rows8_ms": block_rows8_ms,
            "rows8_speedup": block_rows4_ms / block_rows8_ms,
            "row_to_rows8_speedup": row_ms / block_rows8_ms,
            "compact_bytes": int(block_major.nbytes),
            "expanded_weight_bytes": 0,
            "layout_profile": block_fp8_gemv_profile(),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.json:
            output_path = Path(args.json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        return
    if not args.model:
        raise SystemExit("模型基准需要 --model；算子基准可使用 --kernel")
    if args.gpus:
        # 自动容量探测会首次导入 torch，必须先限定可见物理卡。
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    if (
        args.extreme is not True
        and args.profile in {"auto", "ram"}
        and args.device == "cuda"
        and args.tp in (None, 1)
        and args.cache_gb is None
        and args.vram_gb is None
        and args.dense_residency != "ram"
    ):
        from .extreme import detect_auto_extreme

        preview = resolve_preset(
            args.model,
            profile=args.profile,
            tp=args.tp,
        )
        decision = detect_auto_extreme(
            str(preview.model_dir),
            max_ctx=args.max_ctx,
            device="cuda",
            tp_size=1,
            normal_ram_reserve_gib=float(
                preview.environment.get("CCCP_RESIDENT_RESERVE_GB", "2")
            ),
            environment=preview.environment,
        )
        extreme_disabled = args.extreme is False
        args.extreme = decision.activate and not extreme_disabled
        if args.profile == "auto" and not args.extreme:
            args.profile = resolve_capacity_profile(
                preview,
                decision.mode,
            ).profile
        print(
            "[cccp-benchmark] 自动容量："
            f"结论={decision.mode}；profile={args.profile}；"
            f"专家={decision.expert_bytes / 2**30:.2f}GiB；"
            f"需转入GPU={decision.spill_bytes / 2**30:.2f}GiB；"
            f"GPU专家余量={decision.gpu_expert_capacity / 2**30:.2f}GiB",
            flush=True,
        )
    elif args.extreme is None:
        args.extreme = False
    if args.extreme:
        if args.device != "cuda" or args.tp not in (None, 1):
            raise SystemExit("--extreme 需要 --device cuda --tp 1")
        if args.profile not in ("auto", "ram"):
            raise SystemExit("--extreme 不能搭配 resident/parallel profile")
        if args.cache_gb is not None or args.vram_gb is not None:
            raise SystemExit(
                "--extreme 自动规划容量，不能同时指定 --cache-gb/--vram-gb"
            )
        from .extreme import configure_extreme_environment

        configure_extreme_environment()
        os.environ["CCCP_EXTREME_PLACEMENT"] = args.extreme_placement
        if args.extreme_score_file:
            os.environ["CCCP_EXTREME_SCORE_FILE"] = args.extreme_score_file
        if args.extreme_load_workspace_gb is not None:
            if args.extreme_load_workspace_gb < 0.25:
                raise SystemExit(
                    "--extreme-load-workspace-gb 不能小于 0.25"
                )
            os.environ["CCCP_EXTREME_LOAD_WORKSPACE_GB"] = str(
                args.extreme_load_workspace_gb
            )
        args.tp = 1
        args.profile = "ram"
        args.dense_residency = "gpu"
    try:
        cpu_thread_sweep = _parse_cpu_thread_sweep(
            args.cpu_thread_sweep
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.warmup < 1 or args.steps < 1 or args.repeat < 1:
        raise SystemExit("--warmup、--steps、--repeat 必须大于 0")
    if args.prompt_repeat < 1:
        raise SystemExit("--prompt-repeat 必须大于 0")
    if (
        args.prefill_block_tokens is not None
        and args.prefill_block_tokens < 1
    ):
        raise SystemExit("--prefill-block-tokens must be positive")
    if (
        args.prefill_moe_batch is not None
        and not 1 <= args.prefill_moe_batch <= 8192
    ):
        raise SystemExit("--prefill-moe-batch must be in 1..8192")
    if args.prefill_block_tokens is not None:
        os.environ["CCCP_PREFILL_BLOCK_TOKENS"] = str(
            args.prefill_block_tokens
        )
    if args.prefill_moe_batch is not None:
        os.environ["CCCP_PREFILL_MOE_BATCH"] = str(
            args.prefill_moe_batch
        )
    if args.prompt_repeat > 1:
        args.prompt = args.prompt_separator.join(
            [args.prompt] * args.prompt_repeat
        )
    if args.cpu_contaminated_retries < 0:
        raise SystemExit("--cpu-contaminated-retries 不能小于 0")
    if args.window < 1:
        raise SystemExit("--window 必须大于 0")
    if args.dense_residency in {"gpu", "ram"} and args.device != "cuda":
        raise SystemExit("--dense-residency gpu/ram 需要 --device cuda")
    if (
        args.cpu_compile is not None
        and args.device != "cpu"
        and args.dense_residency != "ram"
    ):
        raise SystemExit(
            "--cpu-compile 只适用于 --device cpu 或 --dense-residency ram"
        )
    if args.vram_limit_gb is not None:
        if args.vram_limit_gb <= 0:
            raise SystemExit("--vram-limit-gb 必须大于 0")
        os.environ["CCCP_VRAM_LIMIT_GB"] = str(args.vram_limit_gb)
    if args.pin_gb is not None:
        if args.pin_gb < 0:
            raise SystemExit("--pin-gb 不能小于 0")
        value = str(args.pin_gb)
        os.environ["CCCP_PIN_GB"] = value
        os.environ["CCCP_HOST_PIN_GB"] = value
    if cpu_thread_sweep and args.device != "cpu":
        raise SystemExit("--cpu-thread-sweep 只适用于 --device cpu")
    if args.cpu_wait_idle_percent is not None and args.device != "cpu":
        raise SystemExit("--cpu-wait-idle-percent 只适用于 --device cpu")
    try:
        _wait_for_cpu_idle(
            None,
            args.cpu_wait_idle_samples,
            args.cpu_wait_timeout,
        )
        if args.cpu_wait_idle_percent is not None:
            if not 0.0 < args.cpu_wait_idle_percent <= 100.0:
                raise ValueError(
                    "--cpu-wait-idle-percent 必须在 (0,100] 内"
                )
            if args.cpu_wait_idle_samples < 1:
                raise ValueError(
                    "--cpu-wait-idle-samples 必须大于 0"
                )
            if args.cpu_wait_timeout < 1:
                raise ValueError("--cpu-wait-timeout 必须大于 0")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    preset = resolve_preset(args.model, profile=args.profile, tp=args.tp)
    _apply_preset_environment(args, preset)
    if args.save_route_scores:
        os.environ["CCCP_ROUTE_COUNTS"] = "1"
    if args.device == "cpu" or args.dense_residency == "ram":
        from .runtime_defaults import configure_cpu_operator_defaults

        if args.cpu_compile is not None:
            # 显式 CLI 必须优先于模型 preset 和继承的 shell 环境。
            os.environ["CCCP_CPU_COMPILE"] = args.cpu_compile
        configure_cpu_operator_defaults(
            cpu_compile=args.cpu_compile or "auto"
        )

    # CUDA_VISIBLE_DEVICES 必须在首次导入 torch/Engine 前设置。
    import torch

    from .engine import Engine

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用")

    required = (
        args.warmup
        + args.steps * args.repeat
        + int(args.probe_stages)
    )
    load_started = time.perf_counter()
    engine = Engine(
        str(preset.model_dir),
        cache_gb=args.cache_gb,
        max_ctx=args.max_ctx,
        device=args.device,
        vram_cache_gb=args.vram_gb,
        tp_size=preset.tp,
        dense_residency=args.dense_residency,
        extreme_mode=args.extreme,
    )
    load_seconds = time.perf_counter() - load_started
    actual_device = torch.device(
        getattr(engine.model, "device", args.device)
    ).type
    hybrid_device = getattr(engine.model, "accelerator_device", None)
    if actual_device != args.device and not (
        args.device == "cuda"
        and args.dense_residency == "ram"
        and hybrid_device is not None
        and torch.device(hybrid_device).type == "cuda"
    ):
        raise SystemExit(
            f"请求 device={args.device}，实际回退到 {actual_device}；"
            "拒绝生成会误标硬件的基准结果"
        )
    effective_tp = int(getattr(engine.model, "effective_tp_size", 1))
    if preset.tp > 1 and effective_tp != preset.tp:
        raise SystemExit(
            f"请求 tp={preset.tp}，实际 effective_tp={effective_tp}；"
            "拒绝把 RAM 回退标成多卡性能"
        )
    from .chat_adapters import (
        ChatMessage,
        ChatOptions,
        adapter_for_arch,
    )

    options = ChatOptions(
        thinking_mode="chat",
        reasoning_effort=None,
        temperature=0.0,
        top_p=1.0,
        max_new=required,
    )
    prompt_plan = adapter_for_arch(preset.architecture).prepare(
        engine,
        [ChatMessage(role="user", content=args.prompt)],
        options,
        None,
    )
    prompt_ids = prompt_plan.input_ids
    if not prompt_ids:
        raise SystemExit("prompt 编码为空")
    if len(prompt_ids) + required + 1 > args.max_ctx:
        raise SystemExit(
            f"prompt({len(prompt_ids)}) + warmup/测量({required}) "
            f"超过 max_ctx={args.max_ctx}"
        )

    if args.speculative_ab > 0:
        if preset.architecture != "kimi_k3" or args.device != "cpu":
            raise SystemExit(
                "--speculative-ab currently requires Kimi K3 CPU TP1"
            )
        if preset.tp != 1:
            raise SystemExit("--speculative-ab requires --tp 1")
        # Build every lazy resident/native directory before either measured
        # side.  Otherwise the first (greedy) run pays model construction and
        # the second (candidate) run receives an artificial speedup.
        warmup_tokens = engine.generate(
            prompt_ids,
            max_new=args.warmup,
            temp=0.0,
        )
        engine.reset()
        _wait_for_cpu_idle(
            args.cpu_wait_idle_percent,
            args.cpu_wait_idle_samples,
            args.cpu_wait_timeout,
        )
        started = time.perf_counter()
        baseline = engine.generate(
            prompt_ids,
            max_new=args.steps,
            temp=0.0,
        )
        baseline_total_seconds = time.perf_counter() - started
        baseline_prefill_seconds = (
            float(engine.last_kv_stats.prefill_ms) / 1000.0
        )
        baseline_seconds = max(
            0.0,
            baseline_total_seconds - baseline_prefill_seconds,
        )
        _wait_for_cpu_idle(
            args.cpu_wait_idle_percent,
            args.cpu_wait_idle_samples,
            args.cpu_wait_timeout,
        )
        engine.reset()
        started = time.perf_counter()
        candidate = engine.generate_speculative(
            prompt_ids,
            max_new=args.steps,
            k=args.speculative_ab,
        )
        candidate_total_seconds = time.perf_counter() - started
        candidate_prefill_seconds = (
            float(engine.last_kv_stats.prefill_ms) / 1000.0
        )
        candidate_seconds = max(
            0.0,
            candidate_total_seconds - candidate_prefill_seconds,
        )
        exact = candidate == baseline
        result = {
            "mode": "kimi_cpu_speculative_ab",
            "model": str(preset.model_dir),
            "load_seconds": load_seconds,
            "prompt": "<redacted>" if args.redact_prompt else args.prompt,
            "prompt_sha256": (
                hashlib.sha256(args.prompt.encode("utf-8")).hexdigest()
                if args.redact_prompt else None
            ),
            "prompt_tokens": len(prompt_ids),
            "requested_tokens": args.steps,
            "draft_tokens": int(args.speculative_ab),
            "warmup_tokens": warmup_tokens,
            "threads": int(torch.get_num_threads()),
            "baseline_tokens": baseline,
            "candidate_tokens": candidate,
            "tokens_exact": exact,
            "baseline_seconds": baseline_seconds,
            "candidate_seconds": candidate_seconds,
            "baseline_total_seconds": baseline_total_seconds,
            "candidate_total_seconds": candidate_total_seconds,
            "baseline_prefill_seconds": baseline_prefill_seconds,
            "candidate_prefill_seconds": candidate_prefill_seconds,
            "baseline_tok_s": len(baseline) / max(baseline_seconds, 1e-9),
            "candidate_tok_s": len(candidate) / max(candidate_seconds, 1e-9),
            "speedup": (
                baseline_seconds / candidate_seconds
                if exact and candidate_seconds > 0
                else 0.0
            ),
            "spec_stats": dict(getattr(engine, "spec_stats", {})),
            "expanded_index_bytes": int(
                getattr(
                    getattr(engine.model, "pool", None),
                    "expanded_index_bytes",
                    0,
                )
            ),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        if args.json:
            output = Path(args.json)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        if not exact:
            raise SystemExit("speculative candidate token sequence diverged")
        return

    if args.device == "cpu":
        _wait_for_cpu_idle(
            args.cpu_wait_idle_percent,
            args.cpu_wait_idle_samples,
            args.cpu_wait_timeout,
        )
    engine.reset()
    cuda_profiler_active = (
        args.device == "cuda"
        and os.environ.get("CCCP_CUDA_PROFILER_RANGE", "0") == "1"
    )
    prefill_stage_probe = None
    if args.probe_prefill:
        start_profile = getattr(engine.model, "start_profile", None)
        if not callable(start_profile):
            raise RuntimeError(
                f"{preset.architecture} 当前没有 Prefill 分阶段探针"
            )
        start_profile()
    prefill_started = time.perf_counter()
    logits = _model_prefill(engine.model, prompt_ids)
    if args.device == "cuda":
        torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - prefill_started) * 1000.0
    route_score_summary = (
        _save_route_scores(
            getattr(engine.model, "pool", None),
            args.save_route_scores,
        )
        if args.save_route_scores
        else None
    )
    print(
        "[cccp-benchmark-prefill] "
        f"tokens={len(prompt_ids)} elapsed={prefill_ms / 1000.0:.6f}s "
        f"throughput={len(prompt_ids) / max(prefill_ms / 1000.0, 1e-9):.2f}tok/s",
        flush=True,
    )
    if args.probe_prefill:
        finish_profile = getattr(engine.model, "finish_profile", None)
        if not callable(finish_profile):
            raise RuntimeError("Prefill 分阶段探针无法结束")
        prefill_stage_probe = finish_profile()

    logits, warmup_tokens = _steps(
        preset.architecture,
        engine.model,
        logits,
        args.warmup,
        args.window,
    )
    if args.device == "cuda":
        torch.cuda.synchronize()
    # Capture only the steady decode interval.  Starting before prefill also
    # records construction of every CUDA Graph bucket and model-state warmup,
    # which can multiply a per-token kernel by the number of buckets and hide
    # the actual runtime schedule in Nsight reports.
    if cuda_profiler_active:
        torch.cuda.cudart().cudaProfilerStart()

    runs: list[dict[str, Any]] = []
    discarded_cpu_runs: list[dict[str, Any]] = []
    all_tokens: list[int] = []
    while len(runs) < args.repeat:
        if args.device == "cpu":
            _wait_for_cpu_idle(
                args.cpu_wait_idle_percent,
                args.cpu_wait_idle_samples,
                args.cpu_wait_timeout,
            )
        measured_position = int(getattr(engine.model, "pos", 0))
        cuda_events = None
        if args.device == "cuda":
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            cuda_events = (begin, end)
        cpu_activity_before = _cpu_activity_snapshot(
            args.device == "cpu" and args.cpu_wait_idle_percent is not None
        )
        wall_started = time.perf_counter()
        logits, tokens = _steps(
            preset.architecture,
            engine.model,
            logits,
            args.steps,
            args.window,
        )
        if cuda_events is not None:
            cuda_events[1].record()
            cuda_events[1].synchronize()
            cuda_ms = float(cuda_events[0].elapsed_time(cuda_events[1]))
        else:
            cuda_ms = None
        wall_ms = (time.perf_counter() - wall_started) * 1000.0
        cpu_activity_after = _cpu_activity_snapshot(
            args.device == "cpu" and args.cpu_wait_idle_percent is not None
        )
        external_busy_percent = _external_cpu_busy_percent(
            cpu_activity_before,
            cpu_activity_after,
        )
        run = {
            "repeat": len(runs) + 1,
            "measured_position": measured_position,
            "steps": args.steps,
            "wall_ms": wall_ms,
            "cuda_ms": cuda_ms,
            "external_cpu_busy_percent": external_busy_percent,
            "throughput_tok_s": args.steps / (wall_ms / 1000.0),
            "allocated_vram_gib": (
                torch.cuda.memory_allocated() / 2**30
                if args.device == "cuda"
                else None
            ),
            "tokens": tokens,
        }
        if _cpu_run_is_contaminated(
            external_busy_percent,
            args.cpu_wait_idle_percent,
        ):
            run["discard_reason"] = "external_cpu_busy"
            discarded_cpu_runs.append(run)
            print(
                "discard cpu run "
                f"position={measured_position} "
                f"external_busy={external_busy_percent:.2f}% "
                f"limit={100.0 - args.cpu_wait_idle_percent:.2f}%",
                flush=True,
            )
            if len(discarded_cpu_runs) > args.cpu_contaminated_retries:
                raise RuntimeError(
                    "CPU 正式计时连续受到外部负载污染；"
                    "已拒绝生成误导性跑分"
                )
            # A rejected autoregressive interval has already mutated KDA/KV
            # state. Rebuild the exact accepted prefix before retrying so the
            # next result keeps the same position and routing workload.
            engine.reset()
            logits = _model_prefill(
                engine.model,
                prompt_ids + warmup_tokens + all_tokens
            )
            print(
                "replayed accepted prefix after contaminated CPU run "
                f"position={getattr(engine.model, 'pos', 0)}",
                flush=True,
            )
            continue
        all_tokens.extend(tokens)
        runs.append(run)
        print(
            f"repeat={len(runs)} position={measured_position} "
            f"throughput={run['throughput_tok_s']:.3f} token/s",
            flush=True,
        )

    if cuda_profiler_active:
        torch.cuda.cudart().cudaProfilerStop()

    throughputs = [float(run["throughput_tok_s"]) for run in runs]
    stage_probe = None
    discarded_cpu_stage_probes: list[dict[str, Any]] = []
    if args.probe_stages:
        try:
            (
                logits,
                stage_probe,
                discarded_cpu_stage_probes,
            ) = _measure_stage_probe(
                torch=torch,
                engine=engine,
                architecture=preset.architecture,
                logits=logits,
                window=args.window,
                device=args.device,
                wait_idle_percent=args.cpu_wait_idle_percent,
                wait_idle_samples=args.cpu_wait_idle_samples,
                wait_idle_timeout=args.cpu_wait_timeout,
                contaminated_retries=args.cpu_contaminated_retries,
                replay_ids=prompt_ids + warmup_tokens + all_tokens,
            )
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
    primary_cpu_threads = (
        int(torch.get_num_threads()) if args.device == "cpu" else None
    )
    thread_sweep_results: list[dict[str, Any]] = []
    if cpu_thread_sweep:
        for thread_count in cpu_thread_sweep:
            if thread_count == primary_cpu_threads:
                thread_sweep_results.append(
                    {
                        "requested_threads": thread_count,
                        "effective_threads": primary_cpu_threads,
                        "prefill_ms": prefill_ms,
                        "warmup_tokens": warmup_tokens,
                        "throughput_tok_s_median": statistics.median(
                            throughputs
                        ),
                        "throughput_tok_s_min": min(throughputs),
                        "throughput_tok_s_max": max(throughputs),
                        "decoded_measured_text": engine.decode(all_tokens),
                        "runs": runs,
                        "reused_primary_measurement": True,
                    }
                )
                continue
            _wait_for_cpu_idle(
                args.cpu_wait_idle_percent,
                args.cpu_wait_idle_samples,
                args.cpu_wait_timeout,
            )
            thread_sweep_results.append(
                _measure_cpu_thread_count(
                    torch=torch,
                    engine=engine,
                    architecture=preset.architecture,
                    prompt_ids=prompt_ids,
                    threads=thread_count,
                    warmup=args.warmup,
                    steps=args.steps,
                    repeat=args.repeat,
                    window=args.window,
                    wait_idle_percent=args.cpu_wait_idle_percent,
                    wait_idle_samples=args.cpu_wait_idle_samples,
                    wait_idle_timeout=args.cpu_wait_timeout,
                    contaminated_retries=args.cpu_contaminated_retries,
                )
            )
        torch.set_num_threads(int(primary_cpu_threads))
    hardware: dict[str, Any]
    if args.device == "cuda":
        props = torch.cuda.get_device_properties(0)
        hardware = {
            "name": props.name,
            "compute_capability": (
                f"{props.major}.{props.minor}"
            ),
            "sm": f"sm_{props.major}{props.minor}",
            "vram_gib": props.total_memory / 2**30,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device_count": torch.cuda.device_count(),
        }
    else:
        hardware = _cpu_hardware(torch)
    result = {
        "schema": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": _source_state(),
        "model": str(Path(args.model).resolve()),
        "architecture": preset.architecture,
        "profile": preset.profile,
        "tp": preset.tp,
        "effective_tp": effective_tp,
        "ep_layout": preset.ep_layout,
        "device": args.device,
        "dense_residency": dict(engine.dense_residency),
        "hardware": hardware,
        "process_memory": _process_memory(),
        "environment": {
            key: os.environ[key]
            for key in (
                "CCCP_COMPUTE_DTYPE",
                "CCCP_DENSE_BF16",
                "CCCP_FUSED",
                "CCCP_PROJECTION_FUSED",
                "CCCP_PROJECTION_WARPS",
                "CCCP_PROJECTION_TILE_VIEW",
                "CCCP_PROJECTION_TILE_FUSED",
                "CCCP_PROJECTION_DOWN_REDUCE",
                "CCCP_STATIC_DECODE_GRAPHS",
                "CCCP_STATIC_FFN_GRAPH",
                "CCCP_P10_SHARED",
                "CCCP_PAGED_KV_FUSED",
                "CCCP_LATENT_KV",
                "CCCP_RAM_MIRROR",
                "CCCP_RESIDENT_RESERVE_GB",
                "CCCP_RAM_RESERVE_GB",
                "CCCP_VRAM_RESERVE_GB",
                "CCCP_VRAM_LIMIT_GB",
                "CCCP_HOST_PIN_GB",
                "CCCP_EP_LAYOUT",
                "CCCP_CPU_THREADS",
                "CCCP_CPU_NUMA",
                "CCCP_CPU_FUSED",
                "CCCP_CPU_COMPILE",
                "CCCP_CPU_PACKED_LAYOUT",
                "CCCP_CPU_BLOCK_MAJOR",
                "CCCP_CPU_PACKED_SINGLE_TEAM",
                "CCCP_CPU_PACKED_ROWS16",
                "CCCP_CPU_PACKED_NUMA_ROWS",
                "CCCP_CPU_PACKED_BF16",
                "CCCP_CPU_PACKED_PROFILE",
                "CCCP_CPU_L2_BYTES",
                "CCCP_CPU_LLC_BYTES",
                "CCCP_CPU_L2_TASK_TILES",
                "CCCP_FULL_RESIDENT",
                "CCCP_PREFETCH",
                "CCCP_EXTREME_MODE",
                "CCCP_CPU_ATTN_MANY",
                "CCCP_CPU_QKV_POST",
                "CCCP_CPU_DN_BLOCK",
                "CCCP_CPU_VQ_INT8",
                "CCCP_SINGLE_GPU_LAYER_GRAPH",
                "CCCP_H2D_BATCH",
                "CCCP_PACKED_MOE_GRAPH",
                "CCCP_LOAD_WORKERS",
                "CCCP_TOKEN_GRAPH",
                "CCCP_TP_LAYER_GRAPH",
                "CCCP_ROUTED_WARPS",
                "CCCP_PREFILL_BLOCK_TOKENS",
                "CCCP_PREFILL_MOE_BATCH",
                "OMP_PROC_BIND",
                "OMP_PLACES",
                "GOMP_CPU_AFFINITY",
            )
            if key in os.environ
        },
        "load_seconds": load_seconds,
        "prompt": "<redacted>" if args.redact_prompt else args.prompt,
        "prompt_sha256": (
            hashlib.sha256(args.prompt.encode("utf-8")).hexdigest()
            if args.redact_prompt else None
        ),
        "prompt_mode": "production_chat_adapter",
        "prompt_tokens": len(prompt_ids),
        "prefill_ms": prefill_ms,
        "warmup_steps": args.warmup,
        "warmup_tokens": warmup_tokens,
        "steps_per_repeat": args.steps,
        "repeat": args.repeat,
        "throughput_tok_s_median": statistics.median(throughputs),
        "throughput_tok_s_min": min(throughputs),
        "throughput_tok_s_max": max(throughputs),
        "decoded_measured_text": engine.decode(all_tokens),
        "runs": runs,
    }
    if prefill_stage_probe is not None:
        result["prefill_stage_probe"] = prefill_stage_probe
    if discarded_cpu_runs:
        result["discarded_cpu_runs"] = discarded_cpu_runs
    packed_operator_name = getattr(
        engine.model, "packed_operator_name", None
    )
    if packed_operator_name:
        result["packed_operator"] = packed_operator_name
    result["tp_dataflow"] = getattr(
        engine.model,
        "tp_dataflow",
        "model-default",
    )
    collectives = getattr(
        engine.model,
        "tp_collectives_per_layer",
        None,
    )
    if collectives is not None:
        result["tp_collectives_per_layer"] = int(collectives)
    token_graph = getattr(engine.model, "tp_token_graph_info", None)
    if token_graph:
        result["token_graph"] = dict(token_graph)
    if stage_probe is not None:
        result["stage_probe"] = stage_probe
    if discarded_cpu_stage_probes:
        result["discarded_cpu_stage_probes"] = (
            discarded_cpu_stage_probes
        )
    if thread_sweep_results:
        result["cpu_thread_sweep"] = thread_sweep_results
    pool = getattr(engine.model, "pool", None)
    if pool is not None:
        result["expert_cache"] = {
            "full_resident": bool(
                getattr(pool, "full_resident", False)
            ),
            "host_expert_gib": (
                getattr(pool, "host_expert_bytes", 0) / 2**30
            ),
            "compact_resident_entries": int(
                getattr(pool, "compact_resident_entries", 0)
            ),
            "compact_full_resident": bool(
                getattr(pool, "compact_full_resident", False)
            ),
            "expanded_index_bytes": int(
                getattr(pool, "expanded_index_bytes", 0)
            ),
            "cpu_compile_mode": str(
                getattr(pool, "cpu_compile_mode", "off")
            ),
            "cpu_relayout_entries": int(
                getattr(pool, "block_major_entries", 0)
            ),
            "cpu_relayout_gib": (
                getattr(pool, "block_major_bytes", 0) / 2**30
            ),
            "compiled_source_gib": (
                getattr(pool, "compiled_source_bytes", 0) / 2**30
            ),
            "compiled_index_gib": (
                getattr(pool, "compiled_index_bytes", 0) / 2**30
            ),
            "native_packed_layers": sum(
                value is not False
                for value in getattr(pool, "_native_layers", {}).values()
            ),
            "native_packed_hits": int(
                getattr(pool, "native_hits", 0)
            ),
            "native_packed_fallbacks": int(
                getattr(pool, "native_fallbacks", 0)
            ),
            "extreme_mode": bool(
                getattr(engine, "extreme_mode", False)
                or getattr(pool, "fixed_extreme_residency", False)
            ),
            "extreme_strategy": str(
                getattr(engine, "extreme_strategy", "disabled")
            ),
            "extreme_ram_layers": list(
                getattr(pool, "extreme_ram_layers", ())
            ),
            "extreme_gpu_layers": list(
                getattr(pool, "extreme_gpu_layers", ())
            ),
            "extreme_mixed_layers": list(
                getattr(pool, "extreme_mixed_layers", ())
            ),
            "extreme_placement_mode": str(
                getattr(pool, "extreme_placement_mode", "layer")
            ),
            "extreme_score_source": str(
                getattr(pool, "extreme_score_source", "none")
            ),
            "extreme_gpu_expert_count": int(
                getattr(pool, "extreme_gpu_expert_count", 0)
            ),
            "extreme_storage_ratio": float(
                getattr(pool, "extreme_storage_ratio", 0.0)
            ),
            "gpu_storage_gib": (
                getattr(pool, "gpu_storage_bytes", 0) / 2**30
            ),
            "gpu_storage_gib_by_rank": [
                value / 2**30
                for value in getattr(
                    pool,
                    "gpu_storage_bytes_by_rank",
                    (),
                )
            ],
            "hits": int(getattr(pool, "hits", 0)),
            "misses": int(getattr(pool, "miss", 0)),
            "prefetch_hits": int(getattr(pool, "prefetch_hits", 0)),
            "device_route_lookups": int(
                getattr(pool, "device_route_lookups", 0)
            ),
            "device_route_full_hits": int(
                getattr(pool, "device_route_full_hits", 0)
            ),
            "device_route_fallbacks": int(
                getattr(pool, "device_route_fallbacks", 0)
            ),
            "device_cache_telemetry": (
                pool.device_cache_telemetry()
                if callable(getattr(pool, "device_cache_telemetry", None))
                else {}
            ),
            "decode_executor": str(
                getattr(pool, "decode_executor_name", "unavailable")
            ),
            "uploaded_gib": (
                getattr(pool, "uploaded_bytes", 0) / 2**30
            ),
            "transfer_seconds": float(
                getattr(pool, "transfer_seconds", 0.0)
            ),
            "host_shard_seconds": float(
                getattr(pool, "shard_seconds", 0.0)
            ),
            "h2d_batch_submissions": int(
                getattr(getattr(pool, "_stage", None), "batch_submissions", 0)
            ),
            "h2d_batch_copies": int(
                getattr(getattr(pool, "_stage", None), "batch_copies", 0)
            ),
            "h2d_batch_fallbacks": int(
                getattr(getattr(pool, "_stage", None), "batch_fallbacks", 0)
            ),
            "prefill_batch_rows": int(
                getattr(pool, "prefill_batch_rows", 0)
            ),
            "prefill_batch_submissions": int(
                getattr(pool, "prefill_batch_submissions", 0)
            ),
            "prefill_executor": str(
                getattr(pool, "prefill_executor", "unavailable")
            ),
            "prefill_batch_max": int(
                getattr(pool, "prefill_batch_max", 0)
            ),
            "prefill_expert_chunk_capacity": int(
                getattr(pool, "prefill_expert_chunk_capacity", 0)
            ),
            "prefill_expert_chunk_submissions": int(
                getattr(pool, "prefill_expert_chunk_submissions", 0)
            ),
            "prefill_layer_unique_max": int(
                getattr(pool, "prefill_layer_unique_max", 0)
            ),
            "adaptive_decode_arena": bool(
                getattr(pool, "_adaptive_decode_arena", False)
            ),
            "adaptive_decode_repartitions": int(
                getattr(pool, "adaptive_decode_repartitions", 0)
            ),
            "mapped_slots_per_layer": int(
                getattr(pool, "_mapped_slots_per_layer", 0)
            ),
            "mapped_total_slots": int(
                getattr(pool, "_mapped_total_slots", 0)
            ),
            "mapped_cache_hits": int(
                getattr(pool, "mapped_cache_hits", 0)
            ),
            "mapped_cache_misses": int(
                getattr(pool, "mapped_cache_misses", 0)
            ),
            "mapped_elastic_cache": bool(
                getattr(pool, "_mapped_elastic_enabled", False)
            ),
            "mapped_peak_slots": int(
                getattr(pool, "_mapped_peak_slots", 0)
            ),
            "mapped_cache_evictions": int(
                getattr(pool, "mapped_cache_evictions", 0)
            ),
            "mapped_cache_hits_by_class": {
                str(size): int(count)
                for size, count in getattr(
                    pool, "mapped_cache_hits_by_class", {}
                ).items()
            },
            "mapped_cache_misses_by_class": {
                str(size): int(count)
                for size, count in getattr(
                    pool, "mapped_cache_misses_by_class", {}
                ).items()
            },
            "mapped_cache_uploaded_gib": (
                getattr(pool, "mapped_cache_uploaded_bytes", 0) / 2**30
            ),
            "mapped_cache_refresh_seconds": float(
                getattr(pool, "mapped_cache_refresh_seconds", 0.0)
            ),
        }
        route_tier_profile = getattr(pool, "route_tier_profile", None)
        if route_tier_profile is not None:
            result["expert_cache"]["route_tiers"] = route_tier_profile()
        if route_score_summary is not None:
            result["route_scores"] = route_score_summary
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if args.json:
        output = Path(args.json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
