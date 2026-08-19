#!/usr/bin/env bash
# DSV4 H20 满速配置(Round 34-36 三层启用链,零代码改动):
# 打包池全显存常驻 + TP 隐藏输入 + 层/共享/静态图 + dense 张量核 FP8。
# 实测(2026-08-20): 打印 25.87 tok/s(256 token 含 prefill);
# 稳态 decode ~31.3 tok/s((9.90s-1.7s)/256)。
export CCCP_PACKED_FULL_GPU=1        # PackedDevicePool:11008 专家全显存常驻,H2D=0
export CCCP_TP_HIDDEN=1              # hidden_mode:bind_hidden_inputs 必需
export CCCP_SINGLE_GPU_LAYER_GRAPH=1 # TP 共享专家图(43 层)+层图
export CCCP_STATIC_DECODE_GRAPHS=1
export CCCP_STATIC_FFN_GRAPH=1
export CCCP_DENSE_BF16=all           # dense/共享专家 BF16 展开
export CCCP_GPU_FP8_EXECUTION=on     # dense BlockFP8→tensor-fp8 scaled_mm(+12%)
export CCCP_COMPUTE_DTYPE=bf16
export CCCP_ROUTED_WARPS=16          # 扫描 8/16/32,16 最优
export CCCP_VRAM_LIMIT_GB=130
export CCCP_VRAM_RESERVE_GB=3
export CCCP_VRAM_RUNTIME_GB=3.0
# 用法: bash run_dsv4_fast.sh <model_dir> [max_new] [warmup]
exec python scripts/bench_any_model.py "${1:-/media/tyh20/disk22/dsv4-cccp-s-noblack-v2}" --max-new "${2:-256}" --warmup "${3:-128}"
