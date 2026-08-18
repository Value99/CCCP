"""Component microbenchmark for Qwen3.5 Dense VQ decode (CUDA events, whole-loop).

Loads the real int4/fp8-compiled modules, then times one representative
decoder layer's projection groups and the lm_head in isolation with CUDA
events around complete loops (no cross-stream pairs).  Multiplies by the
real per-model group counts to attribute ms/token.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cccp.qwen35_model import Qwen35DenseVQModel


def time_loop(fn, repeats: int = 200, warmup: int = 20) -> float:
    for _ in range(warmup):
        fn()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats  # ms/call


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()

    model = Qwen35DenseVQModel(args.model, device="cuda", max_ctx=512)
    model.preload()
    torch.cuda.synchronize()

    net = model.network.model
    layers = net.layers
    n_layers = len(layers)
    hidden_size = int(getattr(net.config, "hidden_size", 4096))

    # 收集每层 int4 GEMV 对象(decoder 层属性里是 DenseVQLinear/Group)
    from cccp.dense_vq import DenseVQLinear, DenseVQLinearGroup

    singles, groups, seen = [], [], set()
    for module in net.modules():
        for name, child in module.named_children():
            if id(child) in seen:
                continue
            if isinstance(child, DenseVQLinearGroup):
                seen.add(id(child)); groups.append(child)
            elif isinstance(child, DenseVQLinear):
                seen.add(id(child)); singles.append(child)

    x_single = torch.randn(1, hidden_size, device="cuda", dtype=torch.bfloat16)
    per_call = {}
    def input_for(module):
        cols = int(getattr(module, "in_features", 0) or 0)
        if cols <= 0:
            weight = getattr(module, "weight", None)
            cols = int(weight.shape[1]) if weight is not None else hidden_size
        return torch.randn(1, cols, device="cuda", dtype=torch.bfloat16)

    if singles:
        per_call["linear_avg_ms"] = sum(
            time_loop(lambda m=m: m(input_for(m))) for m in singles[:40]
        ) / min(len(singles), 40)
    # DenseVQLinearGroup 是视图容器(无 forward);其成员已按 singles 计入。

    # lm_head
    lm = model.network.lm_head
    per_call["lm_head_ms"] = time_loop(lambda: lm(x_single))

    out = {
        "layers": n_layers,
        "hidden_size": hidden_size,
        "single_linears": len(singles),
        "groups": len(groups),
        "group_rows_total": sum(
            int(getattr(g, "out_features", 0) or 0) for g in groups
        ),
        **{k: round(v, 6) for k, v in per_call.items()},
        "cuda_memory_gib": round(torch.cuda.memory_reserved() / 2**30, 2),
    }
    # 每 token 投影估算:全部 singles + groups(每层一次)
    proj = per_call.get("linear_avg_ms", 0) * len(singles) + per_call.get(
        "group_avg_ms", 0) * len(groups)
    out["projected_projection_ms_per_token"] = round(proj, 3)
    out["projected_lm_head_ms"] = round(per_call.get("lm_head_ms", 0), 3)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
