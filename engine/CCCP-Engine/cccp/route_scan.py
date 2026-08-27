"""Measured expert routing over role-preserving, teacher-forced conversations."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="用完整聊天模板执行纯 prefill，导出逐层专家命中计数",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True, help="JSONL，每行包含 messages")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--profile", choices=("ram", "mapped"), default="ram")
    parser.add_argument("--token-budget", type=int, required=True)
    parser.add_argument("--prefill-block-tokens", type=int, default=4096)
    return parser


class _TokenizerEngine:
    def __init__(self, model_dir: Path, architecture: str):
        if architecture == "kimi_k3":
            from .kimi_tokenizer import KimiTokenizer

            self.tok = KimiTokenizer(str(model_dir))
        else:
            from tokenizers import Tokenizer

            self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))

    def encode(self, text: str) -> list[int]:
        return list(self.tok.encode(text).ids)


def _messages(raw: object) -> list[dict[str, str]]:
    if not isinstance(raw, dict) or not isinstance(raw.get("messages"), list):
        raise ValueError("扫描输入每行必须是含 messages 数组的 JSON 对象")
    result: list[dict[str, str]] = []
    for item in raw["messages"]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        content = item.get("content")
        if role not in {"system", "developer", "user", "assistant"}:
            raise ValueError(f"扫描语料含不支持的角色：{role!r}")
        if isinstance(content, str) and content:
            result.append({"role": role, "content": content})
    if not result:
        raise ValueError("扫描记录没有有效消息")
    return result


def _encode_conversation(
    architecture: str,
    tokenizer_engine: _TokenizerEngine,
    raw_messages: list[dict[str, str]],
) -> list[int]:
    """Render the complete transcript, including existing assistant answers."""
    from .chat_adapters import ChatMessage, ChatOptions

    messages = tuple(
        ChatMessage(role=message["role"], content=message["content"])
        for message in raw_messages
    )
    options = ChatOptions(
        thinking_mode="chat",
        reasoning_effort=None,
        temperature=0.0,
        top_p=1.0,
        max_new=0,
    )
    if architecture == "dsv4":
        from .chat_adapters.dsv4_encoding import encode_messages

        rendered = encode_messages(
            list(raw_messages),
            thinking_mode="chat",
            drop_thinking=True,
        )
        return tokenizer_engine.encode(rendered)
    if architecture in {"glm", "glm5_next"}:
        from .chat_adapters.glm import _render_prompt

        return tokenizer_engine.encode(_render_prompt(messages, options))
    if architecture == "kimi_k3":
        from .chat_adapters.kimi_k3 import _encode_segments, _render_messages

        return _encode_segments(
            tokenizer_engine,
            _render_messages(messages, thinking=False),
        )
    raise ValueError(f"不支持的模型架构：{architecture}")


def _load_token_documents(
    input_path: Path,
    architecture: str,
    tokenizer_engine: _TokenizerEngine,
    token_budget: int,
    model_max_context: int,
) -> tuple[list[list[int]], int]:
    documents: list[list[int]] = []
    total = 0
    truncated = 0
    with input_path.open("r", encoding="utf-8", errors="strict") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                token_ids = _encode_conversation(
                    architecture, tokenizer_engine, _messages(raw)
                )
            except Exception as exc:
                raise ValueError(f"扫描输入第 {line_number} 行无效：{exc}") from exc
            if not token_ids:
                continue
            if len(token_ids) > model_max_context:
                token_ids = token_ids[:model_max_context]
                truncated += 1
            remaining = token_budget - total
            if remaining <= 0:
                break
            if len(token_ids) > remaining:
                token_ids = token_ids[:remaining]
            documents.append(token_ids)
            total += len(token_ids)
            if total >= token_budget:
                break
    return documents, truncated


def _event(processed: int, budget: int, stage: str) -> None:
    print(
        "CCCP_ROUTE_SCAN_PROGRESS "
        + json.dumps(
            {
                "processed_tokens": int(processed),
                "token_budget": int(budget),
                "stage": stage,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        flush=True,
    )


def _expert_pool(engine: object) -> object:
    pool = getattr(getattr(engine, "model", None), "pool", None)
    if pool is None:
        raise RuntimeError("当前 CCCP 模型没有公开专家池")
    return pool


def _atomic_write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> None:
    total_started = time.perf_counter()
    args = _parser().parse_args(argv)
    if args.token_budget <= 0:
        raise SystemExit("--token-budget 必须大于 0")
    if args.prefill_block_tokens != 4096:
        raise SystemExit("训练路由扫描固定使用 4096 token prefill 块")
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    report_path = Path(args.report).resolve()
    if not input_path.is_file():
        raise SystemExit(f"扫描输入不存在：{input_path}")

    from .presets import apply_preset_environment, resolve_preset

    # CPU 扫描统一使用模型发布的 RAM 算子栈，但不把完整专家强制塞进内存。
    # auto 让 LRU 尽量使用可用 RAM；mapped 把缓存压到 0.5 GiB，形成明确
    # 的强制磁盘路径。二者都能路由到全部专家，不裁减专家集合。
    preset = resolve_preset(args.model, profile="ram", tp=1)
    apply_preset_environment(preset)
    os.environ["CCCP_ROUTE_COUNTS"] = "1"
    os.environ["CCCP_PREFILL_BLOCK_TOKENS"] = "4096"
    os.environ["CCCP_PREFILL_MOE_BATCH"] = "4096"
    os.environ["CCCP_CPU_GROUPED_DEQUANT_MIN_ROWS"] = "2"
    os.environ["CCCP_ROUTE_SCAN_LAYER_LOCAL"] = "1"
    os.environ["CCCP_FULL_RESIDENT"] = "0"
    os.environ["CCCP_PREFETCH"] = "1"
    from .runtime_defaults import configure_cpu_operator_defaults

    # Kimi's very wide multi-row Dense projections currently use the exact
    # block-FP8 grouped path.  Expanding its runtime-only Q4 decode image for a
    # prefill GEMM can exceed the native operator's safe temporary geometry and
    # used to terminate the complete training process with SIGSEGV.  Routed
    # experts remain exact packed VQ for every architecture.
    cpu_compile_mode = "off" if preset.architecture == "kimi_k3" else "q4"
    os.environ["CCCP_CPU_COMPILE"] = cpu_compile_mode
    configure_cpu_operator_defaults(cpu_compile=cpu_compile_mode)
    print(
        "[route-scan] Prefill 扫描：4096 token 整块逐层计算；"
        "换层时释放上一层动态专家",
        flush=True,
    )
    manifest_context = int(
        preset.manifest.get("config", {}).get("max_position_embeddings") or 32768
    )
    tokenize_started = time.perf_counter()
    tokenizer_engine = _TokenizerEngine(preset.model_dir, preset.architecture)
    documents, truncated = _load_token_documents(
        input_path,
        preset.architecture,
        tokenizer_engine,
        args.token_budget,
        manifest_context,
    )
    tokenize_seconds = time.perf_counter() - tokenize_started
    processed_target = sum(len(document) for document in documents)
    if not documents:
        raise SystemExit("没有可扫描的 token")
    if processed_target < args.token_budget:
        raise SystemExit(
            f"语料 token 不足：需要 {args.token_budget:,}，"
            f"聊天模板编码后只有 {processed_target:,}；请增加语料后重试"
        )
    max_context = max(len(document) for document in documents)
    _event(0, args.token_budget, "加载模型")

    from .engine import Engine

    cache_gb = None
    if args.profile == "mapped":
        cache_gb = 0.5
    elif preset.architecture in {
        "dsv4", "glm", "glm5_next", "kimi_k3",
    }:
        import psutil
        available_gib = psutil.virtual_memory().available / 2**30
        context_gib = 0.0
        if preset.architecture == "dsv4":
            from .capacity import dsv4_context_runtime_bytes

            context_gib = (
                dsv4_context_runtime_bytes(
                    preset.manifest["config"], max_context
                ).total_bytes
                / 2**30
            )
        declared_layers = preset.manifest.get("expert_files") or {}
        if not declared_layers:
            declared_layers = (
                (preset.manifest.get("routed_experts") or {}).get("layer_files")
                or {}
            )
        layer_files = []
        for item in declared_layers.values():
            filename = item.get("path") if isinstance(item, dict) else item
            if filename:
                layer_files.append(preset.model_dir / str(filename))
        largest_layer_gib = max(
            (
                path.stat().st_size / 2**30
                for path in layer_files
                if path.is_file()
            ),
            default=0.5,
        )
        # 扫描一次只需要当前层专家。文件体积外留 0.75 GiB 给码本和
        # 索引对象，禁止按启动时全部可用 RAM 扩大成跨层缓存。
        cache_gb = max(0.5, largest_layer_gib + 0.75)
        context_text = (
            f" / {context_gib:.2f} GiB" if context_gib else ""
        )
        print(
            f"[route-scan] RAM 自动规划：可用 {available_gib:.2f} GiB，"
            f"最长上下文 {max_context:,} token{context_text}，"
            f"单层专家上限 {largest_layer_gib:.2f} GiB，"
            f"层级缓存 {cache_gb:.2f} GiB",
            flush=True,
        )
    model_load_started = time.perf_counter()
    engine = Engine(
        str(preset.model_dir),
        cache_gb=cache_gb,
        max_ctx=max_context,
        device="cpu",
        tp_size=1,
        quiet=False,
    )
    model_load_seconds = time.perf_counter() - model_load_started
    _event(0, args.token_budget, "模型已加载 · 准备 prefill")
    _pool = _expert_pool(engine)
    store = getattr(_pool, "store", None)
    if store is None:
        raise RuntimeError("当前 CCCP 专家池没有模型专家清单")
    layers = sorted(int(layer) for layer in store.man.expert_files)
    expert_count = int(store.cfg["n_experts"])

    def route_output(processed: int, *, complete: bool) -> dict[str, Any]:
        counts = getattr(_pool, "route_counts", None)
        if counts is None:
            counts = {}
        return {
            "format": "cccp-expert-residency-scores-v1",
            "scores": {
                f"{layer}:{expert}": int(counts.get((layer, expert), 0))
                for layer in layers
                for expert in range(expert_count)
                if complete or counts.get((layer, expert), 0)
            },
            "observations": int(sum(counts.values())),
            "source": "cccp route-scan teacher-forced prefill",
            "processed_tokens": int(processed),
            "token_budget": int(args.token_budget),
            "complete": bool(complete),
        }

    last_snapshot = -1
    last_live_snapshot = 0.0

    def report_completed_block(processed: int, stage: str) -> None:
        nonlocal last_snapshot
        if processed > last_snapshot:
            _atomic_write_json(
                output_path,
                route_output(processed, complete=False),
            )
            last_snapshot = processed
        _event(processed, args.token_budget, stage)

    completed = 0
    import torch
    import psutil

    scan_process = psutil.Process()

    prefill_started = time.perf_counter()
    for index, token_ids in enumerate(documents, start=1):
        engine.reset()
        if hasattr(engine.model, "prefill_chunked"):
            base = completed
            tensor = torch.tensor([token_ids], device=engine.model.device)
            document_blocks = max(1, (len(token_ids) + 4095) // 4096)

            def report_layer(
                start: int,
                _end: int,
                layer: int,
                layer_count: int,
                *,
                base: int = base,
                document_index: int = index,
                document_count: int = len(documents),
                document_tokens: int = len(token_ids),
            ) -> None:
                nonlocal last_live_snapshot
                # Layer callbacks expose liveness inside a 4096-token block.
                # Only completed outer blocks count as processed tokens.
                # One callback per completed layer keeps the GUI truthful.
                # Printing 43 short JSON lines per block is negligible next
                # to layer compute and avoids appearing frozen for four
                # complete layers at a time.
                cadence = 1
                if layer != layer_count and layer % cadence:
                    return
                block_index = start // 4096 + 1
                now = time.monotonic()
                if now - last_live_snapshot >= 2.0 or layer == layer_count:
                    # A 4096-token layer can take long enough that waiting for
                    # the whole outer block leaves the GUI heatmap empty. The
                    # snapshot is atomic and contains only routes observed so
                    # far; processed_tokens remains at the last complete block.
                    _atomic_write_json(
                        output_path,
                        route_output(base, complete=False),
                    )
                    last_live_snapshot = now
                if preset.architecture in {"glm", "glm5_next"}:
                    resident_gib = scan_process.memory_info().rss / 2**30
                    available_gib = psutil.virtual_memory().available / 2**30
                    _event(
                        base,
                        args.token_budget,
                        (
                            f"分层 prefill {document_index}/{document_count} · "
                            f"层 {layer}/{layer_count} · "
                            f"块 {block_index}/{document_blocks} · "
                            f"token {document_tokens} · "
                            f"RAM {resident_gib:.2f} GiB · "
                            f"可用 {available_gib:.2f} GiB"
                        ),
                    )
                    return
                _event(
                    base + start,
                    args.token_budget,
                    (
                        f"prefill {document_index}/{document_count} · "
                        f"块 {block_index}/{document_blocks} · "
                        f"层 {layer}/{layer_count}"
                    ),
                )

            engine.model.prefill_chunked(
                tensor,
                chunk_size=4096,
                progress_callback=lambda current, base=base: report_completed_block(
                    base + current, f"prefill {index}/{len(documents)}"
                ),
                layer_progress_callback=report_layer,
            )
        else:
            engine.model.forward(token_ids)
        completed += len(token_ids)
        _event(completed, args.token_budget, f"prefill {index}/{len(documents)}")
    prefill_seconds = time.perf_counter() - prefill_started

    counts = getattr(_pool, "route_counts", None)
    if counts is None:
        raise RuntimeError("当前 CCCP 专家池在 prefill 后仍没有公开 route_counts")
    output = route_output(completed, complete=True)
    from .cpuext import extension_status

    operator_status = extension_status()
    cached_weights = {
        id(weight): weight
        for weight in getattr(engine.model, "_w", {}).values()
        if hasattr(weight, "layout")
    }
    dense_q4_images = sum(
        str(getattr(weight, "layout", "")) == "q4_0"
        for weight in cached_weights.values()
    )
    cpu_audit = {
        "native_extension_available": bool(operator_status.get("available")),
        "native_extension_source": str(operator_status.get("source") or "unknown"),
        "native_extension_name": str(operator_status.get("name") or ""),
        "threads": int(operator_status.get("threads") or 0),
        "packed_operator": str(
            getattr(engine.model, "packed_operator_name", "") or ""
        ),
        "requested_compile_mode": str(os.environ.get("CCCP_CPU_COMPILE", "off")),
        "expert_compile_mode": str(getattr(_pool, "cpu_compile_mode", "off")),
        "dense_q4_execution_images": int(dense_q4_images),
        "dense_cached_weight_objects": int(len(cached_weights)),
        "packed_layout": str(os.environ.get("CCCP_CPU_PACKED_LAYOUT", "")),
        "l2_bytes": int(os.environ.get("CCCP_CPU_L2_BYTES", "0") or 0),
        "llc_bytes": int(os.environ.get("CCCP_CPU_LLC_BYTES", "0") or 0),
        "l2_task_tiles": int(
            os.environ.get("CCCP_CPU_L2_TASK_TILES", "0") or 0
        ),
        "grouped_prefill_enabled": bool(
            getattr(_pool, "prefill_rows_supported", False)
        ),
        "grouped_prefill_calls": int(
            getattr(_pool, "prefill_rows_calls", 0)
        ),
        "grouped_prefill_tokens": int(
            getattr(_pool, "prefill_rows_tokens", 0)
        ),
        "grouped_prefill_micro_batches": int(
            getattr(_pool, "prefill_rows_micro_batches", 0)
        ),
        "grouped_prefill_fallbacks": int(
            getattr(_pool, "prefill_rows_fallbacks", 0)
        ),
        "grouped_prefill_batch_size": int(
            os.environ.get("CCCP_PREFILL_MOE_BATCH", "4096")
        ),
        "grouped_prefill_algorithm": "expert-grouped-dequant-gemm",
        "layer_local_expert_cache": bool(
            getattr(_pool, "layer_local_cache", False)
        ),
        "layer_cache_resets": int(
            getattr(_pool, "layer_cache_resets", 0)
        ),
        "expert_cache_gib": round(float(cache_gb or 0.0), 3),
        "expert_cache_hits": int(getattr(_pool, "hits", 0)),
        "expert_cache_misses": int(getattr(_pool, "miss", 0)),
        "expert_prefetch_calls": int(getattr(_pool, "prefetch_calls", 0)),
        "expert_prefetch_keys": int(getattr(_pool, "prefetch_keys", 0)),
        "codebook_object_cache_entries": int(
            len(getattr(store, "_cb_cache", {}))
        ),
    }
    report: dict[str, Any] = {
        "schema": "cccp-token-route-scan-report-v1",
        "architecture": preset.architecture,
        "token_budget": args.token_budget,
        "processed_tokens": completed,
        "documents": len(documents),
        "max_context_tokens": max_context,
        "truncated_documents": truncated,
        "prefill_block_tokens": 4096,
        "generation_tokens": 0,
        "tokenize_seconds": round(tokenize_seconds, 3),
        "model_load_seconds": round(model_load_seconds, 3),
        "prefill_seconds": round(prefill_seconds, 3),
        "prefill_tokens_per_second": round(
            completed / prefill_seconds if prefill_seconds > 0 else 0.0, 4
        ),
        "elapsed_seconds": round(time.perf_counter() - total_started, 3),
        "route_observations": output["observations"],
        "cpu_operator_audit": cpu_audit,
    }
    _atomic_write_json(output_path, output)
    _atomic_write_json(report_path, report)
    _event(completed, args.token_budget, "完成")
    if completed < args.token_budget:
        print(
            f"[route-scan] 语料不足：{completed:,}/{args.token_budget:,} token",
            flush=True,
        )


if __name__ == "__main__":
    main()
