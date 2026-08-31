"""Serve one CCCP CCCP model through the OpenAI-compatible API."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from .chat_adapters import adapter_for_arch


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve one CCCP CCCP model through an OpenAI API",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--cache-gb", type=float)
    parser.add_argument("--vram-gb", type=float)
    parser.add_argument("--vram-limit-gb", type=float)
    parser.add_argument(
        "--extreme",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="默认自动检测；可用 --extreme 强制或 --no-extreme 禁用",
    )
    parser.add_argument(
        "--dense-residency",
        choices=("auto", "gpu", "ram"),
        default="auto",
        help="auto 自动尝试 Dense GPU-only；gpu 容量不足即失败",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="GPU parallel size (GLM expert parallel or Kimi tensor parallel)",
    )
    parser.add_argument("--max-ctx", type=int, default=32768)
    parser.add_argument(
        "--default-reasoning", choices=("chat", "high", "max"), default="chat"
    )
    parser.add_argument("--spec", type=int, default=0)
    parser.add_argument("--max-queue", type=int, default=16)
    parser.add_argument("--api-key", default=os.environ.get("CCCP_API_KEY"))
    parser.add_argument(
        "--cors-allow-origin",
        action="append",
        default=[],
    )
    parser.add_argument("--metrics-jsonl")
    parser.add_argument(
        "--preload-vision",
        action="store_true",
        help="显式预载 GLM-5.3 视觉塔；默认关闭，不影响纯文本启动",
    )
    parser.add_argument(
        "--warmup-vision",
        action="store_true",
        help="启动后运行一次合成图片热身；隐含 --preload-vision",
    )
    args = parser.parse_args(argv)
    if args.max_queue <= 0:
        parser.error("--max-queue must be positive")
    if args.tp <= 0:
        parser.error("--tp must be positive")
    if args.tp > 1 and args.device != "cuda":
        parser.error("--tp > 1 requires --device cuda")
    if args.dense_residency in {"gpu", "ram"} and args.device != "cuda":
        parser.error("--dense-residency gpu/ram requires --device cuda")
    if args.vram_limit_gb is not None:
        if args.vram_limit_gb <= 0:
            parser.error("--vram-limit-gb must be positive")
        os.environ["CCCP_VRAM_LIMIT_GB"] = str(args.vram_limit_gb)
    if args.device == "cpu" or args.dense_residency == "ram":
        from .runtime_defaults import configure_cpu_operator_defaults

        configure_cpu_operator_defaults(cpu_compile="auto")
    if args.extreme:
        if args.device != "cuda" or args.tp != 1:
            parser.error("--extreme requires --device cuda --tp 1")
        if args.cache_gb is not None or args.vram_gb is not None:
            parser.error("--extreme cannot be combined with --cache-gb/--vram-gb")
        from .extreme import configure_extreme_environment

        configure_extreme_environment()
        args.dense_residency = "gpu"
    elif args.extreme is False:
        os.environ["CCCP_AUTO_EXTREME"] = "0"
    if args.served_model_name is None:
        args.served_model_name = Path(args.model).resolve().name
    return args


def build_service(args: argparse.Namespace) -> tuple[Any, Any]:
    try:
        from .chat_service import ChatService
        from .openai_api import create_app
    except ModuleNotFoundError as exc:
        missing = (exc.name or "").split(".", 1)[0]
        if missing not in {"fastapi", "pydantic", "starlette"}:
            raise
        raise RuntimeError(
            "OpenAI API 依赖未安装；请执行 pip install -e '.[api]'"
        ) from exc
    from .engine import Engine

    if args.preload_vision or args.warmup_vision:
        os.environ["CCCP_PRELOAD_VISION"] = "1"
    engine = Engine(
        args.model,
        cache_gb=args.cache_gb,
        max_ctx=args.max_ctx,
        device=args.device,
        vram_cache_gb=args.vram_gb,
        tp_size=args.tp,
        dense_residency=args.dense_residency,
        extreme_mode=args.extreme,
    )
    if args.warmup_vision:
        engine.warmup_multimodal()
    adapter = adapter_for_arch(engine.arch)
    service = ChatService(
        engine,
        adapter=adapter,
        served_model_name=args.served_model_name,
        default_reasoning=args.default_reasoning,
        spec=args.spec,
        max_queue=args.max_queue,
        metrics_jsonl=args.metrics_jsonl,
    )
    app = create_app(
        service,
        served_model_name=args.served_model_name,
        api_key=args.api_key,
        cors_allow_origins=tuple(args.cors_allow_origin),
        context_length=args.max_ctx,
    )
    return service, app


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        import uvicorn
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "OpenAI API 依赖未安装；请执行 pip install -e '.[api]'"
        ) from exc
    try:
        _service, app = build_service(args)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
