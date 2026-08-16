"""CCCP 聊天命令行：统一适配 GLM、DeepSeek-V4 与 Kimi K3。

对话模板（与模型自带 chat_template 对齐的最小实现，arch 由引擎自动判定）：
    glm:  [gMASK]<sop>[<|system|>Reasoning Effort: Max]<|user|>{问题}\n<|assistant|><think></think>
    dsv4: <｜begin▁of▁sentence｜><｜User｜>{问题}<｜Assistant｜>{<think>|</think>}（on 补 <think>，off 补 </think>）
DSV4 多轮历史保存主模型实际 token IDs；think 模式按 token 切掉推理段，
不会通过回答文字 decode→encode 重建 prompt。
命令：/think [off|low|medium|high|max] 调整思维链；/clear 清空上下文与 KV cache；/stats 专家缓存命中；
/kv 查看主模型 KV 状态；/exit 退出。
也可 --prompt "..." 单轮非交互运行（冒烟测试用）。
采样默认 temperature=1.0、top_p=1.0（DeepSeek-V4 官方推荐）；--temp 0 可切回贪心。
生成实时流式打印，统计行 token 数为实际生成 token 数 len(out)。
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from .chat_adapters import ChatMessage, ChatOptions, adapter_for_arch
from .chat_adapters.dsv4 import DSV4TokenLedger as DSV4TokenLedger
from .engine import Engine
from .dsv4cache import ContextCapacityError

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    # Windows GBK 控制台遇到 H₂O 下标等字符会 UnicodeEncodeError，强制 UTF-8 容错
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
# stdin 容错：管道/重定向里的非法字节按 U+FFFD 替换，而不是 surrogateescape 成
# 孤代理——孤代理 str 会让 tokenizers Rust 层抛 TypeError: TextInputSequence must be str
try:
    sys.stdin.reconfigure(errors="replace")
except Exception:
    pass
# store.py 的 torch.frombuffer 零拷贝视图会在首个 decode 打一条 UserWarning，
# 恰好插在 "DSV4: " 与正文之间，碍眼且无实际风险（只读使用），压掉。
import warnings as _warnings
_warnings.filterwarnings("ignore", message="The given buffer is not writable")

def _terminal_options(
    *,
    think: bool,
    reasoning_effort: str | None,
    max_new: int | None,
    temp: float,
    top_p: float,
    rep_penalty: float = 1.0,
    no_repeat_ngram: int = 0,
) -> ChatOptions:
    return ChatOptions(
        thinking_mode="thinking" if think else "chat",
        reasoning_effort=reasoning_effort if think else None,
        temperature=temp,
        top_p=top_p,
        max_new=max_new,
        repetition_penalty=rep_penalty,
        no_repeat_ngram_size=no_repeat_ngram,
    )


class _TerminalStream:
    """将 tokenizer 原始流交给架构适配器，绝不显示协议控制标签。"""

    def __init__(
        self,
        eng: Engine,
        adapter,
        options: ChatOptions,
        *,
        write=None,
    ) -> None:
        self._eng = eng
        self._parser = adapter.new_stream_parser(eng, options)
        self._decoder = eng.new_decode_stream(skip_special_tokens=False)
        self._thinking = options.thinking_mode == "thinking"
        self._last_kind: str | None = None
        self._write = write or (
            lambda text: print(text, end="", flush=True)
        )

    def _emit(self, delta) -> None:
        if not delta.text:
            return
        if self._thinking and delta.kind != self._last_kind:
            if delta.kind == "reasoning":
                self._write("[思考]\n")
            elif delta.kind == "content":
                self._write(
                    "\n[回答]\n"
                    if self._last_kind == "reasoning"
                    else "[回答]\n"
                )
        self._write(delta.text)
        self._last_kind = delta.kind

    def on_token(self, token_id: int, _ignored_piece: str) -> None:
        chunk = self._decoder.step(self._eng.tok, token_id) or ""
        for delta in self._parser.feed(chunk):
            self._emit(delta)

    def finish(self):
        parsed, final_deltas = self._parser.finish()
        for delta in final_deltas:
            self._emit(delta)
        return parsed


def _int_range(values: list[int]) -> str:
    lo, hi = min(values), max(values)
    return str(lo) if lo == hi else f"{lo}-{hi}"


def format_kv_stats(eng: Engine) -> str:
    """Format live KV metadata without reading tensor values or syncing CUDA."""
    model = eng.model
    pos = int(getattr(model, "pos", 0))
    max_ctx = getattr(model, "max_ctx", None)
    head = f"[KV] raw={pos}"
    if max_ctx is not None:
        head += f" / max_ctx={int(max_ctx)}"

    states = getattr(model, "states", None)
    ratios = list(getattr(model, "ratios", ()) or ())
    if getattr(eng, "arch", "glm") != "dsv4" or not states:
        return head

    groups: dict[int, dict[str, list[int]]] = {}
    for ratio, state in zip(ratios, states):
        ratio = int(ratio)
        compressed = state.get("compressed") if ratio > 0 else None
        if compressed is None:
            continue
        group = groups.setdefault(
            ratio,
            {
                "lengths": [],
                "page_items": [],
                "indexer_lengths": [],
            },
        )
        group["lengths"].append(int(compressed.length))
        group["page_items"].append(int(compressed.page_items))
        indexer = state.get("indexer")
        if indexer is not None:
            group["indexer_lengths"].append(int(indexer.keys.length))

    lines = [head]
    for ratio in sorted(groups):
        group = groups[ratio]
        lengths = group["lengths"]
        page_items_values = group["page_items"]
        pages = [
            (length + page_items - 1) // page_items
            for length, page_items in zip(lengths, page_items_values)
        ]
        line = (
            f"     ratio={ratio}: layers={len(lengths)}, "
            f"compressed={_int_range(lengths)}, "
            f"pages={_int_range(pages)}, "
            f"page_items={_int_range(page_items_values)}"
        )
        indexer_lengths = group["indexer_lengths"]
        if indexer_lengths:
            line += f", indexer={_int_range(indexer_lengths)}"
            if max(indexer_lengths) <= 512:
                line += ", topk=OFF"
            elif min(indexer_lengths) > 512:
                line += (
                    f", topk=ON ({_int_range(indexer_lengths)} -> 512)"
                )
            else:
                line += ", topk=MIXED"
        lines.append(line)
    return "\n".join(lines)


def chat_loop(
    eng: Engine,
    max_new: int | None,
    temp: float,
    top_p: float,
    think: bool,
    reasoning_effort: str | None = None,
    rep_penalty: float = 1.0,
    no_repeat_ngram: int = 0,
    spec: int = 0,
    should_stop=None,
) -> None:
    messages: list[ChatMessage] = []
    arch = getattr(eng, "arch", "glm")
    adapter = adapter_for_arch(arch)
    hot_ledger: object | None = None
    name = (
        "DSV4"
        if arch == "dsv4"
        else ("Kimi K3" if arch == "kimi_k3" else "GLM")
    )
    effort = reasoning_effort or ("max" if think else None)
    print("CCCP 对话已就绪（/stop 停止生成, "
          "/think [off|low|medium|high|max], "
          "/clear 清空, /stats 专家缓存, /kv KV状态, /exit 退出）"
          f"  当前 think={effort or 'OFF'}", flush=True)
    while True:
        try:
            q = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q == "/exit":
            break
        if q == "/clear":
            messages.clear()
            clear_ledger = getattr(hot_ledger, "clear", None)
            if callable(clear_ledger):
                clear_ledger()
            hot_ledger = None
            eng.reset()
            print("[已清空上下文]")
            continue
        if q.startswith("/think"):
            arg = q[6:].strip().lower()
            supported = {"high", "max"}
            if arch == "kimi_k3":
                supported.update({"low", "medium"})
            if not arg:
                effort = None if effort is not None else "max"
            elif arg in {"off", "chat", "0", "false", "关"}:
                effort = None
            elif arg in {"on", "1", "true", "开"}:
                effort = "max"
            elif arg in supported:
                effort = arg
            else:
                print(
                    "[用法: /think off|"
                    + "|".join(sorted(supported))
                    + "]"
                )
                continue
            think = effort is not None
            print(f"[think {effort or 'OFF'}]")
            continue
        if q == "/stats":
            p = eng.model.pool
            print(f"[专家缓存: 命中 {p.hits} / 未命中 {p.miss} "
                  f"({p.hits / max(p.hits + p.miss, 1):.1%}), "
                  f"驻留 {p.bytes / 2**30:.1f}GB]")
            continue
        if q == "/kv":
            print(format_kv_stats(eng))
            continue
        options = _terminal_options(
            think=think,
            reasoning_effort=effort,
            max_new=max_new,
            temp=temp,
            top_p=top_p,
            rep_penalty=rep_penalty,
            no_repeat_ngram=no_repeat_ngram,
        )
        plan = adapter.prepare(
            eng,
            [*messages, ChatMessage(role="user", content=q)],
            options,
            hot_ledger,
        )
        ids = plan.input_ids
        kv_baseline_len = plan.kv_baseline_len
        print(f"{name}: ", end="", flush=True)
        t0 = time.time()
        stream = _TerminalStream(eng, adapter, options)

        try:
            if spec > 0:
                out = eng.generate_speculative(
                    ids,
                    max_new=max_new,
                    k=spec,
                    callback=stream.on_token,
                    should_stop=should_stop,
                    kv_baseline_len=kv_baseline_len,
                )
            else:
                out = eng.generate(ids, max_new=max_new, temp=temp, top_p=top_p,
                                   rep_penalty=rep_penalty,
                                   no_repeat_ngram=no_repeat_ngram,
                                   callback=stream.on_token, should_stop=should_stop,
                                   kv_baseline_len=kv_baseline_len)
        except ContextCapacityError as exc:
            stream.finish()
            print(
                f"\n[KV cache 扩容失败，已输出 {exc.committed} token；"
                f"position={exc.position}: {exc.cause}]",
                flush=True,
            )
            if arch == "dsv4":
                eng.reset()
            continue
        parsed = stream.finish()
        dt = time.time() - t0
        print(f"\n[{len(out)} token, {dt:.1f}s, {len(out) / max(dt, 1e-6):.2f} tok/s]")
        hot_ledger = adapter.commit(eng, plan, out, parsed)
        committed_messages = getattr(hot_ledger, "committed_messages", None)
        if committed_messages is None:
            messages = [
                *plan.normalized_messages,
                ChatMessage(
                    role="assistant",
                    content=parsed.content,
                    reasoning_content=parsed.reasoning_content,
                    tool_calls=tuple(parsed.tool_calls),
                ),
            ]
        else:
            messages = list(committed_messages)


def main(argv=None, should_stop=None) -> None:
    ap = argparse.ArgumentParser(description="CCCP CCCP 量化模型推理聊天")
    ap.add_argument(
        "--model",
        required=True,
        help="GLM/DeepSeek-V4/Kimi K3 CCCP 模型目录",
    )
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="cuda=GPU 加速推理（dense 常驻显存，专家流式上卡）")
    ap.add_argument("--cache-gb", type=float, default=None,
                    help="专家缓存预算（缺省自动：可用RAM − 固定开销）")
    ap.add_argument("--vram-gb", type=float, default=None,
                    help="专家显存缓存预算（缺省自动：空闲显存 − dense常驻 − KV）")
    ap.add_argument(
        "--vram-limit-gb",
        type=float,
        help="整进程 CUDA allocator 硬上限（GiB）",
    )
    ap.add_argument(
        "--extreme",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="默认自动检测；可用 --extreme 强制或 --no-extreme 禁用",
    )
    ap.add_argument(
        "--dense-residency",
        choices=("auto", "gpu", "ram"),
        default="auto",
        help=(
            "auto=CUDA 容量足够时 Dense 仅驻 GPU，否则回退 CPU；"
            "gpu=强制 Dense 仅驻 GPU"
        ),
    )
    ap.add_argument(
        "--tp",
        type=int,
        default=1,
        help="GLM 专家并行或 Kimi 张量并行卡数",
    )
    ap.add_argument("--max-ctx", type=int, default=None,
                    help="最大上下文；缺省按架构选择，长程部署应显式设置")
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--no-max-new", action="store_true",
                    help="不设置人为输出 token 上限；仍受 EOS 和模型上下文上限约束")
    ap.add_argument("--temp", type=float, default=1.0, help="0=贪心，默认 1.0")
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--rep-penalty", type=float, default=1.0,
                    help="重复惩罚（>1 压制复读，建议 1.1–1.2；knee 档自由文本用）")
    ap.add_argument("--no-repeat-ngram", type=int, default=0,
                    help="禁止重复 n-gram（如 3）")
    ap.add_argument("--spec", type=int, default=0,
                    help="投机解码草稿数（0=关闭；GLM-MTP 建议2；Kimi CPU无损prompt-lookup建议8；仅贪心有效）")
    ap.add_argument("--think", action="store_true", help="开启思维链推理")
    ap.add_argument(
        "--reasoning",
        choices=("chat", "low", "medium", "high", "max"),
        help="CLI Think 级别；chat 关闭，Kimi 支持 low/medium/high/max",
    )
    ap.add_argument("--prompt", default=None, help="单轮非交互运行")
    a = ap.parse_args(argv)
    if a.tp <= 0:
        ap.error("--tp must be positive")
    if a.tp > 1 and a.device != "cuda":
        ap.error("--tp > 1 requires --device cuda")
    if a.dense_residency in {"gpu", "ram"} and a.device != "cuda":
        ap.error("--dense-residency gpu/ram requires --device cuda")
    if a.vram_limit_gb is not None:
        if a.vram_limit_gb <= 0:
            ap.error("--vram-limit-gb must be positive")
        os.environ["CCCP_VRAM_LIMIT_GB"] = str(a.vram_limit_gb)
    if a.device == "cpu" or a.dense_residency == "ram":
        from .runtime_defaults import configure_cpu_operator_defaults

        configure_cpu_operator_defaults(cpu_compile="auto")
    if a.extreme:
        if a.device != "cuda" or a.tp != 1:
            ap.error("--extreme requires --device cuda --tp 1")
        if a.cache_gb is not None or a.vram_gb is not None:
            ap.error("--extreme cannot be combined with --cache-gb/--vram-gb")
        from .extreme import configure_extreme_environment

        configure_extreme_environment()
        a.dense_residency = "gpu"
    elif a.extreme is False:
        os.environ["CCCP_AUTO_EXTREME"] = "0"
    if a.think and a.reasoning == "chat":
        ap.error("--think cannot be combined with --reasoning chat")
    think = a.think or (
        a.reasoning is not None and a.reasoning != "chat"
    )
    reasoning_effort = (
        a.reasoning
        if a.reasoning not in {None, "chat"}
        else ("max" if a.think else None)
    )
    max_new = None if a.no_max_new else a.max_new

    # GLM 默认使用 MLA latent KV，约 0.09MB/token；缺省仍保持 1024，
    # 长文运行按需显式提高。DSV4 使用环形窗+压缩槽。
    max_ctx = a.max_ctx
    if max_ctx is None:
        import json as _json
        import os as _os
        arch_hint = "glm"
        try:
            with open(_os.path.join(a.model, "cccp.json"), encoding="utf-8") as _f:
                _manifest = _json.load(_f)
                _config = _manifest["config"]
                if (
                    _manifest.get("model_family") == "kimi_k3"
                    or "kda_layers" in _config
                ):
                    arch_hint = "kimi_k3"
                elif "hc_mult" in _config:
                    arch_hint = "dsv4"
                elif (
                    _manifest.get("tensor_vq")
                    and not _manifest.get("expert_files")
                    and str(
                        _config.get("text_model_type")
                        or _config.get("outer_model_type")
                        or ""
                    ).startswith("qwen3_5")
                ):
                    arch_hint = "qwen3_5_dense"
        except Exception:
            pass
        max_ctx = (
            4096
            if arch_hint in {"dsv4", "kimi_k3", "qwen3_5_dense"}
            else 1024
        )
        print(f"[chat] max_ctx 按架构缺省: {arch_hint} → {max_ctx}", flush=True)

    if reasoning_effort in {"low", "medium"}:
        try:
            from .presets import detect_architecture, load_manifest

            _root, _manifest = load_manifest(a.model)
            if detect_architecture(_manifest) == "dsv4":
                ap.error(
                    "DeepSeek-V4 官方模板只支持 "
                    "--reasoning chat/high/max；low/medium 是 Kimi 专用档位"
                )
        except (OSError, ValueError):
            pass

    eng = Engine(a.model, cache_gb=a.cache_gb, max_ctx=max_ctx,
                 device=a.device, vram_cache_gb=a.vram_gb,
                 tp_size=a.tp, dense_residency=a.dense_residency,
                 extreme_mode=a.extreme)
    if a.prompt is not None:
        prompt_options = _terminal_options(
            think=think,
            reasoning_effort=reasoning_effort,
            max_new=max_new,
            temp=a.temp,
            top_p=a.top_p,
            rep_penalty=a.rep_penalty,
            no_repeat_ngram=a.no_repeat_ngram,
        )
        prompt_adapter = adapter_for_arch(
            getattr(eng, "arch", "glm")
        )
        prompt_plan = prompt_adapter.prepare(
            eng,
            [ChatMessage(role="user", content=a.prompt)],
            prompt_options,
            None,
        )
        ids = prompt_plan.input_ids
        kv_baseline_len = prompt_plan.kv_baseline_len
        t0 = time.time()
        stream = _TerminalStream(
            eng,
            prompt_adapter,
            prompt_options,
        )

        try:
            if a.spec > 0:
                out = eng.generate_speculative(
                    ids,
                    max_new=max_new,
                    k=a.spec,
                    callback=stream.on_token,
                    should_stop=should_stop,
                    kv_baseline_len=kv_baseline_len,
                )
            else:
                out = eng.generate(ids, max_new=max_new, temp=a.temp, top_p=a.top_p,
                                   rep_penalty=a.rep_penalty,
                                   no_repeat_ngram=a.no_repeat_ngram,
                                   callback=stream.on_token, should_stop=should_stop,
                                   kv_baseline_len=kv_baseline_len)
        except ContextCapacityError as exc:
            stream.finish()
            print(
                f"\n[KV cache 扩容失败，已输出 {exc.committed} token；"
                f"position={exc.position}: {exc.cause}]",
                flush=True,
            )
            return
        stream.finish()
        dt = time.time() - t0
        print(f"\n[{len(out)} token, {dt:.1f}s, {len(out) / max(dt, 1e-6):.2f} tok/s]")
        return
    chat_loop(
        eng,
        max_new,
        a.temp,
        a.top_p,
        think,
        reasoning_effort=reasoning_effort,
        rep_penalty=a.rep_penalty,
        no_repeat_ngram=a.no_repeat_ngram,
        spec=a.spec,
        should_stop=should_stop,
    )


if __name__ == "__main__":
    main()
