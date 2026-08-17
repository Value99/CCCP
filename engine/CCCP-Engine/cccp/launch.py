"""统一启动入口：自动识别模型、加载专属预设，再进入聊天或 API 服务。"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from .presets import (
    apply_preset_environment,
    model_context_limit,
    resolve_capacity_profile,
    resolve_preset,
)
from .runtime_defaults import configure_cpu_operator_defaults
from .speculative import (
    provider_attachment_available_in_manifest,
    provider_for_architecture,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CCCP 通用启动器（自动识别 GLM / DeepSeek-V4 / Kimi K3）",
    )
    parser.add_argument(
        "action",
        nargs="?",
        choices=("chat", "serve"),
        default="chat",
        help="chat=CLI 对话，serve=OpenAI 兼容 API",
    )
    parser.add_argument("--model", required=True, help="CCCP 模型目录")
    parser.add_argument(
        "--profile",
        choices=("auto", "ram", "resident", "mapped", "parallel"),
        default="auto",
        help=(
            "auto 根据模型、--gpus 卡数和实时容量选择；"
            "resident 为单卡 packed 全显存，"
            "ram 为单卡专家卸载，parallel 为模型配置声明的多卡路径"
        ),
    )
    parser.add_argument(
        "--tp",
        type=int,
        help="并行卡数；通常省略并由 --gpus 数量自动推导",
    )
    parser.add_argument(
        "--gpus",
        help="设置 CUDA_VISIBLE_DEVICES，例如 0 或 0,1,2",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        help="覆盖模型专属配置中的计算设备",
    )
    parser.add_argument("--max-ctx", type=int)
    parser.add_argument("--max-new", type=int)
    parser.add_argument(
        "--prefill-block-tokens",
        type=int,
        help="resident packed prefill token block (default 8192)",
    )
    parser.add_argument(
        "--prefill-moe-batch",
        type=int,
        help="public packed-MoE prefill micro-batch in 1..8192",
    )
    parser.add_argument("--cache-gb", type=float, help="主机专家缓存预算")
    parser.add_argument("--vram-gb", type=float, help="主卡专家显存缓存预算")
    parser.add_argument(
        "--vram-limit-gb",
        type=float,
        help="整进程 CUDA allocator 硬上限（包含算子、状态和专家 arena）",
    )
    parser.add_argument(
        "--extreme",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "默认自动检测；--extreme 强制单卡 RAM+VRAM 极限常驻，"
            "--no-extreme 禁止自动切换"
        ),
    )
    parser.add_argument(
        "--extreme-placement",
        choices=("auto", "layer", "precision"),
        default="auto",
        help=(
            "极限模式专家放置；异构码本默认按 cccp.json 量化精度预算"
            "选择 GPU 热专家"
        ),
    )
    parser.add_argument(
        "--extreme-score-file",
        help=(
            "可选专家常驻分数 JSON；支持 CCCP expert-preference 审计"
        ),
    )
    parser.add_argument(
        "--extreme-load-workspace-gb",
        type=float,
        help="极限模式加载峰值 RAM 余量；至少 0.25GiB",
    )
    parser.add_argument(
        "--dense-residency",
        choices=("auto", "gpu", "ram"),
        default="auto",
        help=(
            "auto=显存足够时 Dense 仅驻 GPU，否则回退 CPU；"
            "gpu=必须仅驻 GPU，容量不足立即失败"
        ),
    )
    parser.add_argument(
        "--dense-bf16",
        help=(
            "Dense 常驻精度组：none、all，或以逗号分隔的 attention/"
            "compressor/embed/head/hyper/indexer/norm/shared；覆盖模型预设"
        ),
    )
    parser.add_argument(
        "--ram-reserve-gb",
        type=float,
        help="至少留给系统/运行时的 RAM；同时控制镜像与全量常驻判定",
    )
    parser.add_argument(
        "--vram-reserve-gb",
        type=float,
        help="专家 arena、上下文与临时工作区共用的唯一显存总预留",
    )
    parser.add_argument(
        "--pin-gb",
        type=float,
        help=(
            "RAM 模式锁页预算；同时适用于普通热专家缓存和原地 packed 专家"
        ),
    )
    parser.add_argument(
        "--single-gpu-layer-graph",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="单卡固定地址整层 Graph（Kimi RAM profile 默认开启）",
    )
    parser.add_argument(
        "--h2d-batch",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="将独立 packed 专家段合并为 CUDA 批量 H2D 提交",
    )
    parser.add_argument(
        "--spec",
        type=int,
        help="投机草稿数；Kimi CPU --spec 8 启用无损prompt-lookup块验证",
    )
    parser.add_argument(
        "--cpu-compile",
        choices=("off", "auto", "u16", "q4"),
        default=None,
        help=(
            "CPU专家执行镜像：off保持紧凑；auto容量足够时在线编译；"
            "u16强制精确uint16 row-tile内存像，空间不足立即报错"
        ),
    )
    parser.add_argument("--temp", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--think", action="store_true", help="CLI 开启 Think")
    parser.add_argument("--prompt", help="CLI 单轮提示词")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--served-model-name")
    parser.add_argument(
        "--reasoning",
        choices=("chat", "low", "medium", "high", "max"),
        help="CLI/API 推理级别；Kimi CLI 支持 low/medium/high/max",
    )
    parser.add_argument("--max-queue", type=int)
    parser.add_argument("--api-key")
    parser.add_argument("--metrics-jsonl")
    parser.add_argument("--cors-allow-origin", action="append", default=[])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只输出识别结果和最终配置，不加载模型",
    )
    return parser


def _value(args: argparse.Namespace, preset: Any, name: str) -> Any:
    value = getattr(args, name)
    return preset.defaults.get(name) if value is None else value


def _context_limit(args: argparse.Namespace, preset: Any) -> int:
    """Use an explicit debug override or the model's own logical limit."""
    declared = model_context_limit(preset.manifest)
    if args.max_ctx is None:
        return declared
    requested = int(args.max_ctx)
    if requested < 64 or requested > declared:
        raise ValueError(
            f"--max-ctx={requested} 超出模型声明范围 64..{declared}"
        )
    return requested


def _spec_value(args: argparse.Namespace, preset: Any) -> int:
    """Resolve an architecture-published draft count for this backend.

    CPU and accelerator kernels have different break-even points.  Keep the
    policy in the public architecture config and preserve an explicit
    ``--spec`` as the highest-priority reproducibility override.
    """
    if args.spec is not None:
        return int(args.spec)
    backend = str(
        os.environ.get("CCCP_RUNTIME_BACKEND")
        or _value(args, preset, "device")
    ).lower()
    configured = preset.defaults.get("spec_by_device") or {}
    if isinstance(configured, dict) and backend in configured:
        drafts = int(configured[backend])
    else:
        drafts = int(preset.defaults.get("spec") or 0)
    if drafts <= 0:
        return 0
    try:
        provider = provider_for_architecture(preset.architecture)
    except ValueError:
        return 0
    if not provider_attachment_available_in_manifest(
        provider,
        preset.manifest,
        preset.model_dir,
    ):
        return 0
    return min(drafts, int(provider.policy.max_draft))


def _normalize_launch_request(args: argparse.Namespace) -> tuple[str, ...]:
    """规范化显卡列表，并让卡数成为 auto profile 的 TP 数。

    用户只需写 ``--gpus 4,5,6,7``；不再要求同时重复 ``--tp 4`` 和
    ``--profile parallel``。显式 profile/TP 仍保留给调试和回归使用。
    """

    if not args.gpus:
        if args.device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return ()
    devices = tuple(
        part.strip() for part in str(args.gpus).split(",") if part.strip()
    )
    if not devices:
        raise ValueError("--gpus 不能为空")
    if len(set(devices)) != len(devices):
        raise ValueError("--gpus 不能包含重复设备")
    if args.device == "cpu":
        raise ValueError("--device cpu 不能同时指定 --gpus")
    args.gpus = ",".join(devices)

    if args.tp is None and args.profile in {"auto", "parallel"}:
        args.tp = len(devices)
        args._tp_source = "由--gpus自动推导"
    elif args.tp is None:
        if len(devices) != 1:
            raise ValueError(
                f"--profile {args.profile} 是单卡模式；--gpus 只能给一张卡"
            )
        args.tp = 1
        args._tp_source = "单卡"
    elif int(args.tp) != len(devices):
        raise ValueError(
            f"--gpus 给出 {len(devices)} 张卡，但 --tp={args.tp}；"
            "两者必须一致"
        )
    else:
        args._tp_source = "显式"
    return devices


def _apply_automatic_runtime_defaults(
    args: argparse.Namespace,
    preset: Any,
) -> None:
    """把已验证的公共运行参数收敛到 CLI 自动配置。

    外部环境变量和显式 CLI 参数始终优先；这里不按模型名选择算子，只按
    CPU/CUDA 设备能力启用公共注册层。
    """

    selected_device = _value(args, preset, "device")
    if selected_device != "cpu":
        # Both GPU-Dense and RAM-Dense hybrid modes execute routed experts
        # through the public accelerator operators.
        os.environ.setdefault("CCCP_REQUIRE_FUSED", "1")
    if (
        selected_device != "cpu"
        and args.dense_residency != "ram"
    ):
        # A resident profile is selected only after the capacity planner has
        # accounted for compact experts, Dense and the runtime reserve.  Keep
        # Dense on the GPU explicitly so the simplified CLI cannot silently
        # fall back to another placement policy.
        if (
            preset.profile in {"resident", "mapped"}
            and args.dense_residency == "auto"
        ):
            args.dense_residency = "gpu"
        args._cpu_compile_source = "不适用"
        return
    configure_cpu_operator_defaults(
        cpu_compile=args.cpu_compile or "auto",
    )
    if args.cpu_compile is None:
        os.environ.setdefault("CCCP_CPU_COMPILE", "auto")
        args._cpu_compile_source = "自动"
    else:
        args._cpu_compile_source = "显式"


def _configured(
    args: argparse.Namespace,
    preset: Any,
    argument: str,
    config_key: str,
) -> Any:
    value = getattr(args, argument)
    return preset.defaults.get(config_key) if value is None else value


def _apply_environment(
    args: argparse.Namespace,
    preset: Any,
) -> None:
    if args.gpus:
        devices = [part.strip() for part in args.gpus.split(",") if part.strip()]
        if len(devices) != preset.tp:
            raise ValueError(
                f"--gpus 给出 {len(devices)} 张卡，但 tp={preset.tp}"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)

    if getattr(args, "extreme", False):
        from .extreme import configure_extreme_environment

        configure_extreme_environment()
        os.environ["CCCP_EXTREME_PLACEMENT"] = args.extreme_placement
        if args.extreme_score_file:
            os.environ["CCCP_EXTREME_SCORE_FILE"] = args.extreme_score_file
        if args.extreme_load_workspace_gb is not None:
            if args.extreme_load_workspace_gb < 0.25:
                raise ValueError(
                    "--extreme-load-workspace-gb 不能小于 0.25"
                )
            os.environ["CCCP_EXTREME_LOAD_WORKSPACE_GB"] = str(
                args.extreme_load_workspace_gb
            )
        args.dense_residency = "gpu"

    # 极限模式先写入 0.25GiB 公共安全默认值，再让架构预设通过
    # setdefault 补齐其他算子环境，避免普通 profile 的 3GiB 余量覆盖极限规划。
    apply_preset_environment(preset)

    # GUI/ordinary CLI launches are automatic.  Do not inherit stale tuning
    # variables from a parent terminal or an older launcher process: those
    # values previously kept the obsolete 3-GiB reserve or a test-only VRAM
    # hard cap alive after users copied a new engine over an old package.
    # Explicit current CLI flags below remain the only supported override.
    if not getattr(args, "extreme", False) and args.vram_reserve_gb is None:
        os.environ["CCCP_VRAM_RESERVE_GB"] = str(
            preset.environment.get("CCCP_VRAM_RESERVE_GB", "1")
        )
    if getattr(args, "vram_limit_gb", None) is None:
        os.environ.pop("CCCP_VRAM_LIMIT_GB", None)
    if not getattr(args, "extreme", False):
        os.environ.pop("CCCP_EXTREME_VRAM_CAP_GB", None)

    dense_bf16 = getattr(args, "dense_bf16", None)
    if dense_bf16 is not None:
        os.environ["CCCP_DENSE_BF16"] = dense_bf16

    if args.ram_reserve_gb is not None:
        value = str(args.ram_reserve_gb)
        os.environ["CCCP_RAM_RESERVE_GB"] = value
        os.environ["CCCP_RESIDENT_RESERVE_GB"] = value
    if args.vram_reserve_gb is not None:
        os.environ["CCCP_VRAM_RESERVE_GB"] = str(
            args.vram_reserve_gb
        )
    prefill_block_tokens = getattr(args, "prefill_block_tokens", None)
    if prefill_block_tokens is not None:
        if prefill_block_tokens < 1:
            raise ValueError("--prefill-block-tokens must be positive")
        os.environ["CCCP_PREFILL_BLOCK_TOKENS"] = str(
            prefill_block_tokens
        )
    prefill_moe_batch = getattr(args, "prefill_moe_batch", None)
    if prefill_moe_batch is not None:
        if not 1 <= prefill_moe_batch <= 8192:
            raise ValueError("--prefill-moe-batch must be in 1..8192")
        os.environ["CCCP_PREFILL_MOE_BATCH"] = str(
            prefill_moe_batch
        )
    vram_limit_gb = getattr(args, "vram_limit_gb", None)
    if vram_limit_gb is not None:
        if vram_limit_gb <= 0:
            raise ValueError("--vram-limit-gb 必须大于 0")
        os.environ["CCCP_VRAM_LIMIT_GB"] = str(vram_limit_gb)
    if args.pin_gb is not None:
        value = str(args.pin_gb)
        # One public CLI flag covers both storage backends.  GLM's staged
        # cache consumes CCCP_PIN_GB; compact p8/p10/p12/p14/p16 archives use
        # in-place cudaHostRegister through CCCP_HOST_PIN_GB.  The model config
        # selects the backend, not a model-specific command-line branch.
        os.environ["CCCP_PIN_GB"] = value
        os.environ["CCCP_HOST_PIN_GB"] = value
    single_gpu_layer_graph = getattr(
        args,
        "single_gpu_layer_graph",
        None,
    )
    if single_gpu_layer_graph is not None:
        os.environ["CCCP_SINGLE_GPU_LAYER_GRAPH"] = (
            "1" if single_gpu_layer_graph else "0"
        )
    h2d_batch = getattr(args, "h2d_batch", None)
    if h2d_batch is not None:
        os.environ["CCCP_H2D_BATCH"] = "1" if h2d_batch else "0"
    cpu_compile = getattr(args, "cpu_compile", None)
    if cpu_compile is not None:
        os.environ["CCCP_CPU_COMPILE"] = cpu_compile


def _summary(args: argparse.Namespace, preset: Any) -> None:
    device = _value(args, preset, "device")
    max_ctx = _context_limit(args, preset)
    spec = _spec_value(args, preset)
    layout = preset.ep_layout or "-"
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "系统默认")
    print(
        "[cccp-launch] "
        f"模型={preset.model_dir.name}；架构={preset.architecture}；"
        f"profile={preset.profile}({getattr(args, '_profile_source', '显式')})；"
        f"device={device}；tp={preset.tp}"
        f"({getattr(args, '_tp_source', '配置')})；"
        f"layout={layout}；context=动态扩展(模型上限{max_ctx})；spec={spec}；"
        f"dense={args.dense_residency}；"
        f"dense_bf16={os.environ.get('CCCP_DENSE_BF16', 'none')}；"
        f"extreme={getattr(args, '_extreme_source', '否')}；"
        f"extreme_placement={getattr(args, 'extreme_placement', 'auto')}；"
        f"extreme_scores={'是' if args.extreme_score_file else '否'}；"
        f"single_graph={os.environ.get('CCCP_SINGLE_GPU_LAYER_GRAPH', '0')}；"
        f"cpu_compile={os.environ.get('CCCP_CPU_COMPILE', 'off')}"
        f"({getattr(args, '_cpu_compile_source', '配置')})；"
        f"CUDA_VISIBLE_DEVICES={visible}",
        flush=True,
    )
    reserve_parts = [
        f"RAM={os.environ.get('CCCP_RESIDENT_RESERVE_GB', 'auto')}GB"
    ]
    if device != "cpu":
        reserve_parts.append(
            f"VRAM总预留={os.environ.get('CCCP_VRAM_RESERVE_GB', 'auto')}GB"
        )
    reserve_parts.append(
        f"锁页={os.environ.get('CCCP_HOST_PIN_GB', os.environ.get('CCCP_PIN_GB', 'auto'))}GB"
    )
    print("[cccp-launch] 内存预留：" + "；".join(reserve_parts), flush=True)
    decision = getattr(args, "_auto_extreme_decision", None)
    if decision is not None:
        print(
            "[cccp-launch] 自动容量："
            f"结论={decision.mode}；专家={decision.expert_bytes / 2**30:.2f}GiB；"
            f"RAM安全容量={decision.normal_ram_capacity / 2**30:.2f}GiB；"
            f"需转入GPU={decision.spill_bytes / 2**30:.2f}GiB；"
            f"GPU专家余量={decision.gpu_expert_capacity / 2**30:.2f}GiB",
            flush=True,
        )


def _chat_argv(args: argparse.Namespace, preset: Any) -> list[str]:
    result = [
        "--model",
        str(preset.model_dir),
        "--device",
        str(_value(args, preset, "device")),
        "--tp",
        str(preset.tp),
        "--max-ctx",
        str(_context_limit(args, preset)),
        "--spec",
        str(_spec_value(args, preset)),
        "--temp",
        str(_configured(args, preset, "temp", "temperature")),
        "--top-p",
        str(_value(args, preset, "top_p")),
        "--dense-residency",
        args.dense_residency,
    ]
    if getattr(args, "extreme", False):
        result.append("--extreme")
    else:
        result.append("--no-extreme")
    max_new = _value(args, preset, "max_new")
    if max_new is None or int(max_new) <= 0:
        result.append("--no-max-new")
    else:
        result.extend(("--max-new", str(max_new)))
    if args.cache_gb is not None:
        result.extend(("--cache-gb", str(args.cache_gb)))
    if args.vram_gb is not None:
        result.extend(("--vram-gb", str(args.vram_gb)))
    if args.think:
        result.append("--think")
    if args.reasoning is not None:
        result.extend(("--reasoning", args.reasoning))
    if args.prompt is not None:
        result.extend(("--prompt", args.prompt))
    return result


def _serve_argv(args: argparse.Namespace, preset: Any) -> list[str]:
    result = [
        "--model",
        str(preset.model_dir),
        "--device",
        str(_value(args, preset, "device")),
        "--tp",
        str(preset.tp),
        "--max-ctx",
        str(_context_limit(args, preset)),
        "--spec",
        str(_spec_value(args, preset)),
        "--host",
        str(_value(args, preset, "host")),
        "--port",
        str(_value(args, preset, "port")),
        "--default-reasoning",
        str(_value(args, preset, "reasoning")),
        "--max-queue",
        str(_value(args, preset, "max_queue")),
        "--dense-residency",
        args.dense_residency,
    ]
    if getattr(args, "extreme", False):
        result.append("--extreme")
    else:
        result.append("--no-extreme")
    if args.cache_gb is not None:
        result.extend(("--cache-gb", str(args.cache_gb)))
    if args.vram_gb is not None:
        result.extend(("--vram-gb", str(args.vram_gb)))
    if args.served_model_name:
        result.extend(("--served-model-name", args.served_model_name))
    if args.api_key:
        result.extend(("--api-key", args.api_key))
    if args.metrics_jsonl:
        result.extend(("--metrics-jsonl", args.metrics_jsonl))
    for origin in args.cors_allow_origin:
        result.extend(("--cors-allow-origin", origin))
    return result


def _validate_extreme_archive(manifest: dict[str, Any]) -> None:
    """确认归档能在极限模式下保持专家索引紧凑。

    兼容旧式 projection_layouts 与公共异构专家 VQ 描述。这里按格式能力
    判断，不按模型名称分支；异构归档还必须让每个精度档位都有 packing。
    """
    quant = manifest.get("quant") or manifest.get("quantization") or {}
    packing = quant.get("index_packing")
    if not isinstance(packing, dict) or not packing:
        raise ValueError(
            "--extreme 要求完整 index_packing，当前归档可能展开索引"
        )

    if quant.get("projection_layouts"):
        return

    tiering = quant.get("heterogeneous_expert_tiering")
    if not isinstance(tiering, dict):
        raise ValueError(
            "--extreme 要求紧凑三投影或异构专家 VQ 归档"
        )
    if tiering.get("format") != "cccp-heterogeneous-expert-vq-v1":
        raise ValueError(
            "--extreme 不支持该 heterogeneous_expert_tiering 格式"
        )
    levels = tiering.get("precision_levels")
    assignments = tiering.get("layer_expert_levels")
    if not isinstance(levels, dict) or not levels:
        raise ValueError("异构专家归档缺少 precision_levels")
    if not isinstance(assignments, dict) or not assignments:
        raise ValueError("异构专家归档缺少 layer_expert_levels")

    referenced_layouts = {
        layout
        for projections in levels.values()
        if isinstance(projections, dict)
        for layout in projections.values()
        if isinstance(layout, str)
    }
    missing = sorted(referenced_layouts.difference(packing))
    if missing:
        preview = ", ".join(missing[:8])
        suffix = "..." if len(missing) > 8 else ""
        raise ValueError(
            "异构专家 precision_levels 引用了未定义 packing 的布局："
            f"{preview}{suffix}"
        )


def _maybe_select_auto_extreme(
    args: argparse.Namespace,
    preset: Any,
) -> bool:
    """按公共归档容量自动启用极限模式；返回是否自动激活。"""

    if args.extreme is True:
        args._extreme_source = "强制"
        return True
    extreme_disabled = args.extreme is False
    if extreme_disabled and args.profile != "auto":
        args._extreme_source = "禁用"
        return False
    args.extreme = False
    args._extreme_source = "否"
    if (
        args.profile not in {"auto", "ram"}
        or preset.tp != 1
        or _value(args, preset, "device") != "cuda"
        or getattr(args, "dense_residency", "auto") == "ram"
        or any(
            value is not None
            for value in (args.cache_gb, args.vram_gb, args.ram_reserve_gb)
        )
    ):
        return False

    from .extreme import detect_auto_extreme

    normal_reserve = float(
        preset.environment.get("CCCP_RESIDENT_RESERVE_GB", "2")
    )
    auto_environment = dict(preset.environment)
    extreme_load_workspace_gb = getattr(
        args,
        "extreme_load_workspace_gb",
        None,
    )
    if extreme_load_workspace_gb is not None:
        auto_environment["CCCP_EXTREME_LOAD_WORKSPACE_GB"] = str(
            extreme_load_workspace_gb
        )
    decision = detect_auto_extreme(
        str(preset.model_dir),
        # Capacity admission uses one normal Prefill block. The logical KV
        # ceiling is model-driven and pages grow only when requests need them.
        max_ctx=min(4096, _context_limit(args, preset)),
        device="cuda",
        tp_size=1,
        normal_ram_reserve_gib=normal_reserve,
        environment=auto_environment,
    )
    args._auto_extreme_decision = decision
    if decision.activate and not extreme_disabled:
        args.extreme = True
        args._extreme_source = "自动"
        return True
    args._extreme_source = (
        f"禁用({decision.mode})"
        if extreme_disabled
        else f"否({decision.mode})"
    )
    return False

def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        _normalize_launch_request(args)
    except ValueError as exc:
        print(f"[cccp-launch] 配置错误：{exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    requested_extreme = args.extreme
    if args.gpus:
        # 必须在 detect_auto_extreme 首次导入 torch 前限定物理卡。
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    if requested_extreme:
        if args.tp not in (None, 1):
            raise SystemExit("[cccp-launch] --extreme 当前只支持单卡 tp=1")
        if args.device not in (None, "cuda"):
            raise SystemExit("[cccp-launch] --extreme 需要 --device cuda")
        if args.profile not in ("auto", "ram"):
            raise SystemExit(
                "[cccp-launch] --extreme 使用 RAM+VRAM，不能搭配 resident/parallel"
            )
        if any(
            value is not None
            for value in (
                args.cache_gb,
                args.vram_gb,
                args.ram_reserve_gb,
            )
        ):
            raise SystemExit(
                "[cccp-launch] --extreme 自动规划容量，不能同时手工指定 "
                "--cache-gb/--vram-gb/--ram-reserve-gb"
            )
    if args.think and args.reasoning == "chat":
        raise SystemExit(
            "[cccp-launch] --think 不能与 --reasoning chat 同时使用"
        )
    if (
        args.action == "serve"
        and args.reasoning in {"low", "medium"}
    ):
        raise SystemExit(
            "[cccp-launch] API 当前支持 reasoning=chat/high/max；"
            "low/medium 仅用于 Kimi CLI"
        )
    try:
        auto_profile_requested = args.profile == "auto"
        preset = resolve_preset(
            args.model,
            profile="ram" if requested_extreme else args.profile,
            tp=1 if requested_extreme else args.tp,
        )
        args._profile_source = (
            "自动" if auto_profile_requested else "显式"
        )
        _maybe_select_auto_extreme(args, preset)
        if args.extreme and preset.profile != "ram":
            preset = resolve_preset(args.model, profile="ram", tp=1)
            args._profile_source = "自动极限"
        elif (
            auto_profile_requested
            and not args.extreme
            and getattr(args, "_auto_extreme_decision", None) is not None
        ):
            capacity_mode = args._auto_extreme_decision.mode
            selected = resolve_capacity_profile(preset, capacity_mode)
            if selected.profile != preset.profile:
                preset = selected
            if preset.profile == "resident":
                args._profile_source = "自动全显存"
            elif preset.profile == "mapped":
                args._profile_source = "自动映射整图"
            else:
                args._profile_source = "自动RAM"
        if args.extreme:
            from .presets import load_manifest

            _root, manifest = load_manifest(args.model)
            _validate_extreme_archive(manifest)
        elif requested_extreme is False:
            os.environ["CCCP_AUTO_EXTREME"] = "0"
        _apply_automatic_runtime_defaults(args, preset)
        _apply_environment(args, preset)
    except (OSError, ValueError) as exc:
        print(f"[cccp-launch] 配置错误：{exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    if (
        preset.architecture == "dsv4"
        and args.reasoning in {"low", "medium"}
    ):
        raise SystemExit(
            "[cccp-launch] DeepSeek-V4 官方模板只支持 "
            "reasoning=chat/high/max；low/medium 是 Kimi 专用档位"
        )

    if _value(args, preset, "device") == "cpu" and preset.tp > 1:
        raise SystemExit("[cccp-launch] CPU 模式不能使用 tp > 1")
    if (
        args.dense_residency in {"gpu", "ram"}
        and _value(args, preset, "device") != "cuda"
    ):
        raise SystemExit(
            "[cccp-launch] --dense-residency gpu/ram 需要 --device cuda"
        )

    _summary(args, preset)
    if args.dry_run:
        return

    if args.action == "serve":
        from .serve import main as serve_main

        serve_main(_serve_argv(args, preset))
    else:
        from .chat import main as chat_main

        chat_main(_chat_argv(args, preset))


if __name__ == "__main__":
    main()
