"""Synchronous, transport-independent ownership of one CCCP chat engine."""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from .chat_adapters import (
    AssistantOutput,
    ChatAdapter,
    ChatMessage,
    ChatOptions,
    PromptPlan,
    StreamDelta,
)


class ChatQueueFull(RuntimeError):
    """Raised when all configured waiting slots are occupied."""


@dataclass(frozen=True)
class GenerationMetrics:
    """Content-free measurements for one completed generation."""

    request_id: str
    prompt_tokens: int
    processed_tokens: int
    completion_tokens: int
    queue_delay_ms: float
    kv_mode: str
    kv_reason: str
    prefill_ms: float | None
    ttft_ms: float | None
    generation_ms: float
    finish_reason: str
    tokens_per_second: float
    output_token_sha256: str
    periodic_tail_detected: bool
    cancelled: bool

    @property
    def token_rate(self) -> float:
        """Concise alias used by transports and diagnostics."""
        return self.tokens_per_second

    @property
    def generation_duration_ms(self) -> float:
        return self.generation_ms


@dataclass(frozen=True)
class HotConversation:
    """The adapter ledger corresponding to the engine's canonical history."""

    model: str
    adapter_name: str
    ledger: object


@dataclass(frozen=True)
class GenerationReady:
    """Content-free identity available immediately before generation."""

    request_id: str
    created: int
    model: str


@dataclass
class ChatResult:
    request_id: str
    created: int
    model: str
    output: AssistantOutput
    output_ids: list[int]
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    metrics: GenerationMetrics


@dataclass(frozen=True)
class _Admission:
    acquired: bool
    queued: bool
    admitted_at: float


def _token_ids_sha256(token_ids: list[int]) -> str:
    payload = json.dumps(token_ids, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _periodic_tail_detected(
    token_ids: list[int],
    *,
    repeats: int = 4,
    min_period: int = 1,
    max_period: int = 64,
) -> bool:
    """Detect an exact repeated token block without retaining its contents."""
    maximum = min(max_period, len(token_ids) // repeats)
    for period in range(min_period, maximum + 1):
        pattern = token_ids[-period:]
        if all(
            token_ids[
                len(token_ids) - (repeat + 1) * period : len(token_ids)
                - repeat * period
            ]
            == pattern
            for repeat in range(1, repeats)
        ):
            return True
    return False


def _decode_token_rate(
    completion_tokens: int,
    *,
    generation_started: float,
    first_token_at: float | None,
    generation_finished: float,
) -> float:
    """Measure steady decode after TTFT instead of charging prefill to tok/s."""
    count = max(0, int(completion_tokens))
    if count <= 0:
        return 0.0
    if first_token_at is not None and count > 1:
        seconds = max(0.0, generation_finished - first_token_at)
        return (count - 1) / seconds if seconds > 0 else 0.0
    seconds = max(0.0, generation_finished - generation_started)
    return count / seconds if seconds > 0 else 0.0


def _expert_cache_snapshot(engine: object) -> dict[str, float]:
    model = getattr(engine, "model", None)
    pool = getattr(model, "pool", None)
    if pool is None:
        return {}
    stage = getattr(pool, "_stage", None)
    watcher = getattr(engine, "_vwatch", None)
    snapshot = {
        "hits": float(getattr(pool, "hits", 0)),
        "miss": float(getattr(pool, "miss", 0)),
        "prefetch_hits": float(getattr(pool, "prefetch_hits", 0)),
        "uploaded_bytes": float(getattr(pool, "uploaded_bytes", 0)),
        "transfer_seconds": float(getattr(pool, "transfer_seconds", 0.0)),
        "arena_bytes": float(getattr(pool, "gpu_arena_bytes", 0)),
        "pinned_bytes": float(getattr(pool, "_host_pinned_bytes", 0)),
        "host_expert_bytes": float(getattr(pool, "host_expert_bytes", 0)),
        "host_staging_seconds": float(
            getattr(stage, "host_staging_seconds", 0.0)
        ),
        "direct_upload_bytes": float(
            getattr(stage, "direct_upload_bytes", 0)
        ),
        "staged_upload_bytes": float(
            getattr(stage, "staged_upload_bytes", 0)
        ),
        "upload_submissions": float(
            getattr(stage, "upload_submissions", 0)
        ),
        "upload_copies": float(getattr(stage, "upload_copies", 0)),
        "batch_submissions": float(
            getattr(stage, "batch_submissions", 0)
        ),
        "batch_copies": float(getattr(stage, "batch_copies", 0)),
        "batch_fallbacks": float(getattr(stage, "batch_fallbacks", 0)),
        "watcher_trims": float(getattr(watcher, "trims", 0)),
        "watcher_grows": float(getattr(watcher, "grows", 0)),
        "arena_slots": float(sum(getattr(pool, "arena_slots", {}).values())),
        "profile_hot_slots": float(getattr(pool, "profile_hot_slots", 0)),
        "profile_hot_enabled": float(
            bool(getattr(pool, "profile_hot_cache_enabled", False))
        ),
        "initial_free_slots": float(getattr(pool, "initial_free_slots", 0)),
        "decode_fused_submissions": float(
            getattr(pool, "decode_fused_submissions", 0)
        ),
        "decode_graph_submissions": float(
            getattr(pool, "decode_graph_submissions", 0)
        ),
        "decode_reference_submissions": float(
            getattr(pool, "decode_reference_submissions", 0)
        ),
        "process_limit_bytes": float(
            getattr(engine, "_vram_limit_bytes", 0)
        ),
        "runtime_headroom_gb": float(
            getattr(engine, "_vram_runtime_reserve_gb", 0.0)
        ),
    }
    device = getattr(model, "device", None)
    if getattr(device, "type", None) == "cuda":
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            snapshot.update({
                "cuda_allocated_bytes": float(
                    torch.cuda.memory_allocated(device)
                ),
                "cuda_reserved_bytes": float(
                    torch.cuda.memory_reserved(device)
                ),
                "driver_free_bytes": float(free_bytes),
                "driver_total_bytes": float(total_bytes),
            })
        except (RuntimeError, TypeError, ValueError):
            pass
    return snapshot


def _decode_executor_name(
    engine: object,
    delta: dict[str, float],
) -> str:
    """Report the executor actually evidenced by this request.

    The former literal ``cuda.packed_moe_topk_fused`` was printed even for
    HIP and Dense VQ models, and could therefore hide the exact AMD failure
    mode where all experts were resident but the TP1 parent graph was not
    replayed.  Graph counters are authoritative for routed DSV4; architecture
    adapters supply their own public operator name for non-routed models.
    """
    model = getattr(engine, "model", None)
    device = getattr(model, "device", None)
    if getattr(device, "type", None) == "cuda":
        backend = "hip" if torch.version.hip is not None else "cuda"
    else:
        backend = "cpu"
    if int(delta.get("decode_graph_submissions", 0)) > 0:
        return f"{backend}.tp1-token-graph"
    operator = str(getattr(model, "packed_operator_name", "") or "")
    if operator:
        if backend == "hip" and operator.startswith("cuda."):
            operator = "hip." + operator[len("cuda."):]
        return operator
    return f"{backend}.eager-or-unreported"


def _metric_value(stats: object | None, name: str, default: Any) -> Any:
    if stats is None:
        return default
    if isinstance(stats, dict):
        return stats.get(name, default)
    return getattr(stats, name, default)


class _StopTextFilter:
    """Hide stop strings without delaying text beyond an unstable suffix."""

    def __init__(self, stops: tuple[str, ...]) -> None:
        self._stops = tuple(stop for stop in stops if stop)
        self._buffer = ""
        self._offset = 0
        self.stopped = False
        self.stop_at: int | None = None

    def feed(self, text: str) -> str:
        if self.stopped or not text:
            return ""
        self._buffer += text
        matches = [
            self._buffer.find(stop)
            for stop in self._stops
            if self._buffer.find(stop) >= 0
        ]
        if matches:
            stop_at = min(matches)
            visible = self._buffer[:stop_at]
            self.stop_at = self._offset + stop_at
            self._buffer = ""
            self.stopped = True
            return visible
        retain = max(
            (
                length
                for stop in self._stops
                for length in range(
                    1,
                    min(len(stop), len(self._buffer) + 1),
                )
                if self._buffer.endswith(stop[:length])
            ),
            default=0,
        )
        split = len(self._buffer) - retain
        visible, self._buffer = self._buffer[:split], self._buffer[split:]
        self._offset += len(visible)
        return visible

    def finish(self) -> str:
        if self.stopped:
            return ""
        visible, self._buffer = self._buffer, ""
        self._offset += len(visible)
        return visible


class ChatService:
    """Serialize requests through one engine and retain one exact-token ledger."""

    def __init__(
        self,
        engine: object,
        *,
        adapter: ChatAdapter,
        served_model_name: str,
        default_reasoning: bool | None = None,
        spec: int = 0,
        max_queue: int = 16,
        metrics_jsonl: str | Path | None = None,
    ) -> None:
        if isinstance(max_queue, bool) or not isinstance(max_queue, int):
            raise TypeError("max_queue must be an integer")
        if max_queue < 0:
            raise ValueError("max_queue must be non-negative")
        if isinstance(spec, bool) or not isinstance(spec, int):
            raise TypeError("spec must be an integer")
        if spec < 0:
            raise ValueError("spec must be non-negative")
        self.engine = engine
        self.adapter = adapter
        self.served_model_name = served_model_name
        self.default_reasoning = default_reasoning
        self.spec = spec
        self.max_queue = max_queue
        self.metrics_jsonl = Path(metrics_jsonl) if metrics_jsonl is not None else None

        self._engine_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._metrics_lock = threading.Lock()
        self._waiting_count = 0
        self._hot_conversation: HotConversation | None = None

    @property
    def busy(self) -> bool:
        return self._engine_lock.locked()

    @property
    def waiting_count(self) -> int:
        with self._state_lock:
            return self._waiting_count

    @property
    def hot_conversation(self) -> HotConversation | None:
        with self._state_lock:
            return self._hot_conversation

    def _admit(self) -> _Admission:
        admitted_at = time.perf_counter()
        with self._state_lock:
            if self._engine_lock.acquire(blocking=False):
                return _Admission(
                    acquired=True,
                    queued=False,
                    admitted_at=admitted_at,
                )
            if self._waiting_count >= self.max_queue:
                raise ChatQueueFull(f"chat queue is full ({self.max_queue} waiting)")
            self._waiting_count += 1
            return _Admission(
                acquired=False,
                queued=True,
                admitted_at=admitted_at,
            )

    def _wait_for_owner(
        self,
        admission: _Admission,
        cancel_event: threading.Event,
    ) -> bool:
        if admission.acquired:
            return True
        cancelled_after_acquire = False
        try:
            while not cancel_event.is_set():
                if self._engine_lock.acquire(timeout=0.05):
                    if cancel_event.is_set():
                        cancelled_after_acquire = True
                        return False
                    return True
            return False
        finally:
            with self._state_lock:
                self._waiting_count -= 1
                if self._waiting_count < 0:
                    self._waiting_count = 0
                    raise RuntimeError("chat waiting count underflow")
                if cancelled_after_acquire:
                    self._engine_lock.release()

    def _release_owner(self) -> None:
        # Admission and release share the state lock, so a request cannot
        # observe a transiently free engine while the waiting count changes.
        with self._state_lock:
            self._engine_lock.release()

    def _generate(
        self,
        plan: PromptPlan,
        options: ChatOptions,
        callback: object,
        should_stop: object,
    ) -> list[int]:
        state = plan.adapter_state if isinstance(plan.adapter_state, dict) else {}
        common = {
            "max_new": options.max_new,
            "callback": callback,
            "should_stop": should_stop,
            "kv_baseline_len": plan.kv_baseline_len,
            "media_digest": state.get("media_digest"),
            "media_slots": state.get("media_slots", ()),
            "media_state": state.get("media_state"),
        }
        # 当前投机路径是严格贪心验收；启用任一惩罚时改走标准采样，
        # 避免请求参数在 spec>0 的服务上被静默忽略。
        if (
            self.spec > 0
            and options.repetition_penalty == 1.0
            and options.presence_penalty == 0.0
            and options.no_repeat_ngram_size == 0
        ):
            return self.engine.generate_speculative(
                plan.input_ids,
                k=self.spec,
                **common,
            )
        return self.engine.generate(
            plan.input_ids,
            temp=options.temperature,
            top_p=options.top_p,
            rep_penalty=options.repetition_penalty,
            presence_penalty=options.presence_penalty,
            no_repeat_ngram=options.no_repeat_ngram_size,
            **common,
        )

    @staticmethod
    def _finish_reason(
        *,
        output: AssistantOutput,
        output_count: int,
        prompt_count: int,
        options: ChatOptions,
        max_context: int | None,
        stopped: bool,
    ) -> str:
        if stopped:
            return "stop"
        if options.max_new is not None and output_count >= options.max_new:
            return "length"
        if max_context is not None and prompt_count + output_count >= max_context:
            return "length"
        if output.tool_calls:
            return "tool_calls"
        return "stop"

    def _write_metrics(self, metrics: GenerationMetrics) -> None:
        if self.metrics_jsonl is None:
            return
        line = json.dumps(
            asdict(metrics),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._metrics_lock:
            with self.metrics_jsonl.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def complete(
        self,
        messages: list[ChatMessage],
        options: ChatOptions,
        *,
        request_id: str | None = None,
        cancel_event: threading.Event | None = None,
        on_ready: Callable[[GenerationReady], None] | None = None,
        on_stream_delta: Callable[[StreamDelta], None] | None = None,
    ) -> ChatResult:
        """Generate one complete response while exclusively owning the engine."""
        if not isinstance(messages, list):
            raise TypeError("messages must be a list")
        if not isinstance(options, ChatOptions):
            raise TypeError("options must be ChatOptions")
        request_id = request_id or f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        cancellation = cancel_event or threading.Event()
        admission = self._admit()
        owns_engine = self._wait_for_owner(admission, cancellation)
        if not owns_engine:
            now = time.perf_counter()
            empty_output = AssistantOutput(
                reasoning_content=None,
                content="",
                tool_calls=[],
            )
            metrics = GenerationMetrics(
                request_id=request_id,
                prompt_tokens=0,
                processed_tokens=0,
                completion_tokens=0,
                queue_delay_ms=(now - admission.admitted_at) * 1000,
                kv_mode="not-started",
                kv_reason="cancelled-in-queue",
                prefill_ms=None,
                ttft_ms=None,
                generation_ms=0.0,
                finish_reason="stop",
                tokens_per_second=0.0,
                output_token_sha256=_token_ids_sha256([]),
                periodic_tail_detected=False,
                cancelled=True,
            )
            self._write_metrics(metrics)
            return ChatResult(
                request_id=request_id,
                created=created,
                model=self.served_model_name,
                output=empty_output,
                output_ids=[],
                finish_reason="stop",
                prompt_tokens=0,
                completion_tokens=0,
                metrics=metrics,
            )

        queue_acquired_at = time.perf_counter()
        generation_started = queue_acquired_at
        try:
            hot = self.hot_conversation
            hot_ledger = (
                hot.ledger
                if hot is not None
                and hot.model == self.served_model_name
                and hot.adapter_name
                == getattr(self.adapter, "name", type(self.adapter).__name__)
                else None
            )
            plan = self.adapter.prepare(
                self.engine,
                messages,
                options,
                hot_ledger,
            )
            if on_ready is not None:
                on_ready(
                    GenerationReady(
                        request_id=request_id,
                        created=created,
                        model=self.served_model_name,
                    )
                )
            decode_stream = self.engine.new_decode_stream(skip_special_tokens=False)
            parser = self.adapter.new_stream_parser(self.engine, options)
            stop_filter = _StopTextFilter(options.stop)
            callback_ids: list[int] = []
            decoded_length = 0
            text_ranges: list[tuple[int, int]] = []
            first_token_at: float | None = None

            def on_token(token_id: int, _ignored_piece: str) -> None:
                nonlocal decoded_length, first_token_at
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                callback_ids.append(token_id)
                chunk = decode_stream.step(self.engine.tok, token_id) or ""
                start = decoded_length
                decoded_length += len(chunk)
                text_ranges.append((start, decoded_length))
                visible = stop_filter.feed(chunk)
                if visible:
                    for delta in parser.feed(visible):
                        if on_stream_delta is not None:
                            on_stream_delta(delta)

            def should_stop() -> bool:
                return cancellation.is_set() or stop_filter.stopped

            generation_started = time.perf_counter()
            cache_before = _expert_cache_snapshot(self.engine)
            generated_ids = list(
                self._generate(
                    plan,
                    options,
                    on_token,
                    should_stop,
                )
            )
            generation_finished = time.perf_counter()
            kv_stats = getattr(self.engine, "last_kv_stats", None)

            if callback_ids != generated_ids:
                raise RuntimeError(
                    "engine callback token IDs must exactly match generated output"
                )

            if not stop_filter.stopped:
                visible_tail = stop_filter.finish()
                if visible_tail:
                    for delta in parser.feed(visible_tail):
                        if on_stream_delta is not None:
                            on_stream_delta(delta)
            parsed, final_deltas = parser.finish()
            if on_stream_delta is not None:
                for delta in final_deltas:
                    on_stream_delta(delta)

            visible_ids = generated_ids
            hidden_stop = stop_filter.stopped
            if stop_filter.stop_at is not None:
                stop_at = stop_filter.stop_at
                visible_count = next(
                    (
                        index
                        for index, (_start, end) in enumerate(text_ranges)
                        if end > stop_at
                    ),
                    len(generated_ids),
                )
                visible_ids = generated_ids[:visible_count]

            max_context = getattr(
                getattr(self.engine, "model", None),
                "max_ctx",
                None,
            )
            finish_reason = self._finish_reason(
                output=parsed,
                output_count=len(visible_ids),
                prompt_count=len(plan.input_ids),
                options=options,
                max_context=max_context,
                stopped=(cancellation.is_set() or hidden_stop),
            )

            context_exhausted_without_generation = (
                max_context is not None
                and len(plan.input_ids) >= max_context
                and not generated_ids
            )
            if hidden_stop or context_exhausted_without_generation:
                # Hidden stop tokens / context exhaustion produce states that
                # must not be inherited by the next request.
                self.engine.reset()
            else:
                # Normal completion OR user cancellation: commit the KV state
                # so the next turn can reuse it as an exact prefix.  When the
                # user hits stop, visible_ids holds the truncated output; the
                # live KV has exactly prompt+those tokens, which is a valid
                # prefix for the next round.
                ledger = self.adapter.commit(
                    self.engine,
                    plan,
                    visible_ids,
                    parsed,
                )
                promote_history = getattr(
                    self.engine,
                    "commit_canonical_history",
                    None,
                )
                canonical_ids = getattr(ledger, "completed_ids", None)
                if callable(promote_history) and canonical_ids is not None:
                    promote_history(list(canonical_ids))
                committed_hot = HotConversation(
                    model=self.served_model_name,
                    adapter_name=getattr(
                        self.adapter,
                        "name",
                        type(self.adapter).__name__,
                    ),
                    ledger=ledger,
                )
                with self._state_lock:
                    self._hot_conversation = committed_hot

            generation_seconds = max(
                0.0,
                generation_finished - generation_started,
            )
            completion_count = len(visible_ids)
            token_rate = _decode_token_rate(
                completion_count,
                generation_started=generation_started,
                first_token_at=first_token_at,
                generation_finished=generation_finished,
            )
            metrics = GenerationMetrics(
                request_id=request_id,
                prompt_tokens=len(plan.input_ids),
                processed_tokens=_metric_value(
                    kv_stats,
                    "processed_tokens",
                    len(plan.input_ids),
                ),
                completion_tokens=completion_count,
                queue_delay_ms=(queue_acquired_at - admission.admitted_at) * 1000,
                kv_mode=str(_metric_value(kv_stats, "mode", "unknown")),
                kv_reason=str(_metric_value(kv_stats, "reason", "unknown")),
                prefill_ms=_metric_value(kv_stats, "prefill_ms", None),
                ttft_ms=(
                    None
                    if first_token_at is None
                    else (first_token_at - generation_started) * 1000
                ),
                generation_ms=generation_seconds * 1000,
                finish_reason=finish_reason,
                tokens_per_second=token_rate,
                output_token_sha256=_token_ids_sha256(visible_ids),
                periodic_tail_detected=_periodic_tail_detected(visible_ids),
                cancelled=cancellation.is_set(),
            )
            prefill_ms = _metric_value(kv_stats, "prefill_ms", None)
            ttft_ms = (
                None
                if first_token_at is None
                else (first_token_at - generation_started) * 1000
            )
            print(
                "[cccp-generation] "
                f"request={request_id} prompt={len(plan.input_ids)} "
                f"completion={completion_count} "
                f"prefill_ms={prefill_ms if prefill_ms is not None else 'reuse'} "
                f"ttft_ms={ttft_ms if ttft_ms is not None else 'none'} "
                f"decode_tok_s={token_rate:.3f}",
                flush=True,
            )
            cache_after = _expert_cache_snapshot(self.engine)
            if cache_after:
                delta = {
                    key: cache_after.get(key, 0.0) - cache_before.get(key, 0.0)
                    for key in (
                        "hits", "miss", "prefetch_hits", "uploaded_bytes",
                        "transfer_seconds", "host_staging_seconds",
                        "direct_upload_bytes", "staged_upload_bytes",
                        "upload_submissions", "upload_copies",
                        "batch_submissions", "batch_copies", "batch_fallbacks",
                        "decode_fused_submissions", "decode_graph_submissions",
                        "decode_reference_submissions",
                        "watcher_trims", "watcher_grows",
                    )
                }
                print(
                    "[cccp-cache] "
                    f"hit={int(delta['hits'])} miss={int(delta['miss'])} "
                    f"prefetch_hit={int(delta['prefetch_hits'])} "
                    f"uploaded={delta['uploaded_bytes'] / 2**30:.3f}GiB "
                    f"dma={delta['transfer_seconds']:.3f}s "
                    f"host_stage={delta['host_staging_seconds']:.3f}s "
                    f"direct={delta['direct_upload_bytes'] / 2**30:.3f}GiB "
                    f"staged={delta['staged_upload_bytes'] / 2**30:.3f}GiB "
                    f"submissions={int(delta['upload_submissions'])} "
                    f"copies={int(delta['upload_copies'])} "
                    f"compiled_batches={int(delta['batch_submissions'])}/"
                    f"{int(delta['batch_copies'])} "
                    f"batch_fallbacks={int(delta['batch_fallbacks'])} "
                    f"decode_executor={_decode_executor_name(self.engine, delta)} "
                    f"decode_fused={int(delta['decode_fused_submissions'])} "
                    f"decode_graph={int(delta['decode_graph_submissions'])} "
                    f"decode_reference={int(delta['decode_reference_submissions'])} "
                    f"arena={cache_after['arena_bytes'] / 2**30:.2f}GiB "
                    f"arena_slots={int(cache_after['arena_slots'])} "
                    f"policy=strict-lru "
                    f"warmup_hot_slots={int(cache_after['profile_hot_slots'])} "
                    f"initial_free_slots={int(cache_after['initial_free_slots'])} "
                    "permanent_protection=0 prefetch=off "
                    f"pinned={cache_after['pinned_bytes'] / 2**30:.2f}GiB "
                    f"ram_experts={cache_after['host_expert_bytes'] / 2**30:.2f}GiB "
                    f"cuda_alloc={cache_after.get('cuda_allocated_bytes', 0.0) / 2**30:.2f}GiB "
                    f"cuda_reserved={cache_after.get('cuda_reserved_bytes', 0.0) / 2**30:.2f}GiB "
                    f"driver_free={cache_after.get('driver_free_bytes', 0.0) / 2**30:.2f}/"
                    f"{cache_after.get('driver_total_bytes', 0.0) / 2**30:.2f}GiB "
                    f"process_limit={cache_after['process_limit_bytes'] / 2**30:.2f}GiB "
                    f"headroom={cache_after['runtime_headroom_gb']:.2f}GiB "
                    f"watcher_trims={int(delta['watcher_trims'])} "
                    f"watcher_grows={int(delta['watcher_grows'])}",
                    flush=True,
                )
            self._write_metrics(metrics)
            return ChatResult(
                request_id=request_id,
                created=created,
                model=self.served_model_name,
                output=parsed,
                output_ids=list(visible_ids),
                finish_reason=finish_reason,
                prompt_tokens=len(plan.input_ids),
                completion_tokens=completion_count,
                metrics=metrics,
            )
        except BaseException:
            # A failed engine call can leave KV partially advanced. Reset it,
            # but retain the previous canonical ledger so an exact extension
            # can be safely rebuilt from token IDs on a later request.
            self.engine.reset()
            raise
        finally:
            self._release_owner()


__all__ = [
    "ChatQueueFull",
    "ChatResult",
    "ChatService",
    "GenerationReady",
    "GenerationMetrics",
    "HotConversation",
]
