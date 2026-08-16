"""DeepSeek-V4（deepseek_v4，DeepSeek-V4-Flash-DSpark）纯 PyTorch 前向模块。

用途：CCCP 量化流水线的路由摘选（profile）、KL 评估与后续推理复用。
逐行对照官方参考实现 inference/model.py（+ kernel.py），全 f32 计算
（RMSNorm 内部亦 fp32），只依赖 torch（+ CCCP.fp4io），不依赖 tilelang 等官方包。

文件结构：
  1) SafeFile / DSV4Checkpoint —— safetensors 纯文件 I/O 读取（8 字节头长 + JSON 头
     + 原始字节，无 mmap，模式参考 cccp/store.py），按层/按专家懒加载；
     FP8（e4m3 值 × 2^(scale-127)，128×128 块，scale 直接乘不取倒数）与
     FP4（fp4io.dequant_fp4，低半字节在前、e2m1 LUT、32 块 ue8m0）反量化。
  2) 前向数学 —— Hyper-Connections（4 通道残差流 [B,T,4,D]：hc_pre(sigmoid 混合)
     → 子层 → hc_post(post + sinkhorn(20 轮) comb)）、低秩 Q（wq_a→q_norm→wq_b→逐头
     无权重 RMS）+ MQA（64 头共享同一 512 维 kv，key=value）+ 分组 LoRA O（o_groups=8）、
     相邻复数对 RoPE（仅最后 64 维；ratio>0 层 YaRN theta=160000/factor=16/orig=65536/
     beta_fast=32/beta_slow=1，无 mscale；ratio=0 层 theta=10000）+ 注意力输出反旋转
     （value 也被 rope 过）、注意力集合 = 最近 128 原始 token（滑窗环形缓存）+ 全部
     压缩 token（T≤2048 时 Indexer top-512 恒等于全选，规范 §4，故跳过 Indexer），
     softmax 分母含 attn_sink（不进分子），scale=512^-0.5。
     KV Compressor（ratio=4 时 coff=2 overlap：8 窗口 4 步长；ratio=128 时 coff=1）：
     wkv/wgate → ape 位置偏置 → 分组 softmax 池化 → RMSNorm → 相位=窗口首 token 的
     RoPE（最后 64 维）→ 写入压缩槽位；decode 增量：每步算 1 token 的 wkv/wgate 入
     state，每 ratio 步产出 1 个压缩 token。
     sqrtsoftplus 路由（sqrt(softplus(logits))，fp32）：层 0-2 用 tid2eid 静态表选择
     + learned 权重（无 bias）；层≥3 noaux_tc top-6（bias 只用于选择不进权重）→
     权重归一 ×1.5。专家 SwiGLU：up=clamp(±10)、gate=clamp(max=10)、silu(gate)*up→w2。
     embed 复制到 4 个 hc 通道（官方 model.py generate 流程）；结尾 hc_head（4 通道
     加权求和，无 sinkhorn）→ final RMSNorm → head。MTP/DSpark 层（43-45）整块跳过。
  3) DSV4Model —— prefill 批式 + decode 单步（KV 环形缓存 + 压缩槽位 + 增量 Compressor）。
  4) main() 自包含自检（python -m CCCP.dsv4）：微小合成模型对照逐元素朴素实现。

与官方实现的偏离（语义不变或精度更高，均已在自检中覆盖）：
  - 省略全部 QAT quant-dequant 模拟（kv 前 448 维 fp8 块 64、indexer Hadamard+fp4）：
    官方亦注明 "kv could also use fp8, current implementation uses bf16"，此处按 f32 精确算。
  - Indexer 整块省略：T≤2048 时其 top-512 恒等于全选（规范 §4），数学上精确。
  - 专家中间结果不转 bf16（官方 w2 前 .to(bf16)），全程 f32，精度更高。
  - HF 检查点里 wo_a 实为 FP8+scale（官方 convert.py 之后才转 BF16），加载器两者皆可。
"""

from __future__ import annotations

import glob
import json
import math
import os
import struct
import sys

import torch
import torch.nn.functional as F

from .cconfig import DSV4Config
from .fp4io import dequant_fp4

FP8_BLOCK = 128  # 块级 FP8 的块边长

def _lin(x, w):
    """线性层统一入口：默认 torch F.linear；CCCP 推理侧以 Int4Weight/VQWeight
    分派实现替换（见 cccp/dsv4model.py 的 _cccp_lin），量化感知代码无需改动。"""
    return torch.nn.functional.linear(x, w)


# safetensors dtype → torch dtype；F8_* 一律按 uint8 原始字节读入，反量化时再解释
_ST_DTYPES = {
    "U8": torch.uint8, "I8": torch.int8, "I16": torch.int16, "I32": torch.int32,
    "I64": torch.int64, "F16": torch.float16, "F32": torch.float32,
    "F64": torch.float64, "BF16": torch.bfloat16,
    "F8_E4M3": torch.uint8, "F8_E8M0": torch.uint8,
}


class SafeFile:
    """极简 safetensors 读取器（纯文件 I/O，无 mmap；模式参考 cccp/store.py）。"""

    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n).decode("utf-8"))
        self.meta = {k: v for k, v in header.items() if k != "__metadata__"}
        self.data_start = 8 + n

    def keys(self):
        return self.meta.keys()

    def info(self, name: str) -> dict:
        return self.meta[name]

    def get_tensor(self, name: str) -> torch.Tensor:
        info = self.meta[name]
        buf = bytearray(info["data_offsets"][1] - info["data_offsets"][0])
        with open(self.path, "rb") as f:
            f.seek(self.data_start + info["data_offsets"][0])
            f.readinto(buf)
        return torch.frombuffer(buf, dtype=_ST_DTYPES[info["dtype"]]).reshape(info["shape"])


def dequant_fp8(w: torch.Tensor, scale: torch.Tensor, block: int = FP8_BLOCK) -> torch.Tensor:
    """块级 FP8 反量化到 f32：W = e4m3值(w) × 2^(scale-127)，块 128×128。

    w: [R, C] uint8 原始字节（或 float8_e4m3fn）；scale: [ceil(R/128), ceil(C/128)]
    ue8m0 字节（2 的幂次指数，HF ckpt 的 weight_scale_inv 存的就是正向乘子，直接乘）。
    """
    wf = w.view(torch.float8_e4m3fn).float() if w.dtype == torch.uint8 else w.float()
    s = torch.pow(2.0, scale.view(torch.uint8).float() - 127.0)
    s = s.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)
    return wf * s[: w.shape[0], : w.shape[1]]


class DSV4Checkpoint:
    """DeepSeek-V4 HF 检查点（官方 convert 规则命名的分片）流式读取器。

    张量命名（已从本机分片头核实）：
      layers.{i}.attn.{wq_a,q_norm,wq_b,wkv,kv_norm,attn_sink,wo_a,wo_b}[.weight/.scale]
      layers.{i}.attn.compressor.{wkv,wgate}.weight / .ape / .norm.weight   （ratio>0 层）
      layers.{i}.ffn.gate.{weight,bias(层≥3),tid2eid(层0-2,I64)}
      layers.{i}.ffn.experts.{e}.w{1,2,3}.weight(I8 打包 FP4)/.scale(F8_E8M0)
      layers.{i}.ffn.shared_experts.w{1,2,3}.weight/.scale                  （FP8）
      layers.{i}.{attn_norm,ffn_norm}.weight、layers.{i}.hc_{attn,ffn}_{fn,base,scale}
      顶层 embed.weight / norm.weight / head.weight / hc_head_{fn,base,scale}
    FP8 张量 = .weight(F8_E4M3) + .scale(F8_E8M0)。注意 wo_a 在 HF ckpt 中也是 FP8。
    """

    def __init__(self, root: str, device: str | torch.device = "cpu",
                 cache_layers: int = 2):
        self.root = root
        self.device = device
        self.loc: dict[str, SafeFile] = {}
        for p in sorted(glob.glob(os.path.join(root, "model-*.safetensors"))):
            sf = SafeFile(p)
            for k in sf.keys():
                self.loc[k] = sf
        if not self.loc:
            raise FileNotFoundError(f"{root} 下未找到 model-*.safetensors 分片")
        self._layer_cache: dict[int, dict] = {}
        # 层权重驻留上限：prefill 顺序过层，容量 2 足够；旧版无界缓存 43 层 f32
        # ≈17GB，会把显存顶进共享显存换页（实测 +7.1GB）。
        self._cache_layers = max(1, cache_layers)

    def has(self, name: str) -> bool:
        return name in self.loc

    def get_raw(self, name: str) -> torch.Tensor:
        """按存储 dtype 读取（F8_* 为 uint8 原始字节）。"""
        return self.loc[name].get_tensor(name)

    def get_f32(self, name: str) -> torch.Tensor:
        """读取并反量化到 f32（带伴生 .scale 的走块级 FP8 反量化；BF16/F32 直转）。

        伴生 scale 命名：X.weight ↔ X.scale（官方 convert 把 weight_scale_inv 改名为 scale）。
        """
        sname = name[:-len("weight")] + "scale" if name.endswith("weight") else name + ".scale"
        if self.has(sname):
            w = dequant_fp8(self.get_raw(name), self.get_raw(sname))
        else:
            w = self.get_raw(name).float()
        return w.to(self.device)

    # ---- 按层加载 ----
    def layer(self, i: int) -> dict:
        """一层全部非专家权重（f32；tid2eid 转 long）。"""
        if i in self._layer_cache:
            return self._layer_cache[i]
        p = f"layers.{i}"
        w = {
            "wq_a": self.get_f32(f"{p}.attn.wq_a.weight"),
            "q_norm": self.get_f32(f"{p}.attn.q_norm.weight"),
            "wq_b": self.get_f32(f"{p}.attn.wq_b.weight"),
            "wkv": self.get_f32(f"{p}.attn.wkv.weight"),
            "kv_norm": self.get_f32(f"{p}.attn.kv_norm.weight"),
            "attn_sink": self.get_f32(f"{p}.attn.attn_sink"),
            "wo_a": self.get_f32(f"{p}.attn.wo_a.weight"),
            "wo_b": self.get_f32(f"{p}.attn.wo_b.weight"),
            "attn_norm": self.get_f32(f"{p}.attn_norm.weight"),
            "ffn_norm": self.get_f32(f"{p}.ffn_norm.weight"),
            "gate": self.get_f32(f"{p}.ffn.gate.weight"),
            "sh_w1": self.get_f32(f"{p}.ffn.shared_experts.w1.weight"),
            "sh_w3": self.get_f32(f"{p}.ffn.shared_experts.w3.weight"),
            "sh_w2": self.get_f32(f"{p}.ffn.shared_experts.w2.weight"),
            "hc_attn_fn": self.get_f32(f"{p}.hc_attn_fn"),
            "hc_attn_base": self.get_f32(f"{p}.hc_attn_base"),
            "hc_attn_scale": self.get_f32(f"{p}.hc_attn_scale"),
            "hc_ffn_fn": self.get_f32(f"{p}.hc_ffn_fn"),
            "hc_ffn_base": self.get_f32(f"{p}.hc_ffn_base"),
            "hc_ffn_scale": self.get_f32(f"{p}.hc_ffn_scale"),
        }
        if self.has(f"{p}.attn.compressor.wkv.weight"):
            w["cmp"] = {
                "wkv": self.get_f32(f"{p}.attn.compressor.wkv.weight"),
                "wgate": self.get_f32(f"{p}.attn.compressor.wgate.weight"),
                "ape": self.get_f32(f"{p}.attn.compressor.ape"),
                "norm": self.get_f32(f"{p}.attn.compressor.norm.weight"),
            }
        if self.has(f"{p}.ffn.gate.bias"):
            w["gate_bias"] = self.get_f32(f"{p}.ffn.gate.bias")
        if self.has(f"{p}.ffn.gate.tid2eid"):
            w["tid2eid"] = self.get_raw(f"{p}.ffn.gate.tid2eid").long().to(self.device)
        self._layer_cache[i] = w
        while len(self._layer_cache) > self._cache_layers:
            for k in list(self._layer_cache):
                if k != i:  # 驱逐最老的非当前层（插入序），f32 张量归还分配器
                    del self._layer_cache[k]
                    break
        return w

    def expert(self, layer: int, eid: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """单个 routed 专家 (w1, w3, w2)，FP4 反量化到 f32（低半字节在前、e2m1、32 块）。"""
        p = f"layers.{layer}.ffn.experts.{eid}"
        out = []
        for k in ("w1", "w3", "w2"):
            q = self.get_raw(f"{p}.{k}.weight")      # [R, C/2] I8
            s = self.get_raw(f"{p}.{k}.scale")       # [R, C/32] ue8m0
            r, c2 = q.shape
            out.append(dequant_fp4(q, s, r, c2 * 2, device=self.device))
        return tuple(out)

    # ---- 顶层 ----
    def embed(self) -> torch.Tensor:
        return self.get_f32("embed.weight")

    def head(self) -> torch.Tensor:
        return self.get_f32("head.weight")

    def final_norm(self) -> torch.Tensor:
        return self.get_f32("norm.weight")

    def hc_head(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (self.get_f32("hc_head_fn"), self.get_f32("hc_head_scale"),
                self.get_f32("hc_head_base"))


# =====================================================================
# 前向数学（全 f32）
# =====================================================================

def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm：f32 计算方差，输出保持输入 dtype（无 bias）。"""
    v = x.float().square().mean(-1, keepdim=True)
    out = w.float() * (x.float() * torch.rsqrt(v + eps))
    return out.to(x.dtype)


class RopeCache:
    """RoPE 频率预计算（相邻复数对 (x0,x1),(x2,x3),...；只作用最后 rope_dim 维）。

    ratio=0 层：theta=rope_theta(10000)，YaRN 关闭；ratio>0 层：theta=
    compress_rope_theta(160000) + YaRN（factor/orig/beta_fast/beta_slow 来自
    rope_scaling，无 mscale 增益）。输出 cos/sin [max_seq, rope_dim//2]。
    """

    def __init__(self, rope_dim: int, max_seq: int, theta: float, yarn: dict | None = None):
        dim = rope_dim
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        if yarn:
            orig = float(yarn.get("original_max_position_embeddings", 0))
            if orig > 0:
                factor = float(yarn.get("factor", 1.0))
                beta_fast = float(yarn.get("beta_fast", 32))
                beta_slow = float(yarn.get("beta_slow", 1))

                def fcd(num_rot: float) -> float:
                    return dim * math.log(orig / (num_rot * 2 * math.pi)) / (2 * math.log(theta))

                low = max(math.floor(fcd(beta_fast)), 0)
                high = min(math.ceil(fcd(beta_slow)), dim - 1)
                if low == high:
                    high += 0.001
                ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1)
                smooth = 1.0 - ramp
                freqs = freqs / factor * (1 - smooth) + freqs * smooth
        t = torch.arange(max_seq, dtype=torch.float32)
        f = torch.outer(t, freqs)                       # [T, dim/2]
        self.cos = f.cos()
        self.sin = f.sin()


def rope_apply(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
               inverse: bool = False) -> torch.Tensor:
    """相邻对旋转：最后一维（rope_dim）按 (x0,x1),(x2,x3),... 配对旋转。

    cos/sin 需已可广播到 x（最后一维 = rope_dim//2）。inverse=True 取共轭
    （注意力输出反旋转：value 也被 rope 过，输出最后 64 维要反向转回）。
    """
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    if inverse:
        sin = -sin
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.stack([y1, y2], dim=-1).flatten(-2)


# ---- Hyper-Connections ----

def hc_split(mixes: torch.Tensor, scale: torch.Tensor, base: torch.Tensor,
             hc: int, iters: int, eps: float):
    """24=(2+hc)×hc 维混合系数 → (pre, post, comb)。

    pre[j]=sigmoid(m[j]*scale[0]+base[j])+eps；post[j]=2*sigmoid(m[hc+j]*scale[1]+base[hc+j])；
    comb[j,k]=m[2hc+4j+k]*scale[2]+base[...]，然后 sinkhorn：softmax(-1)+eps → 列归一 →
    19×(行归一+列归一)（共 20 轮，注意首轮顺序：softmax 后先列归一）。
    """
    pre = torch.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + eps
    post = 2.0 * torch.sigmoid(mixes[..., hc:2 * hc] * scale[1] + base[hc:2 * hc])
    comb = mixes[..., 2 * hc:].unflatten(-1, (hc, hc)) * scale[2] + base[2 * hc:].view(hc, hc)
    comb = comb.softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def hc_pre(x: torch.Tensor, fn: torch.Tensor, scale: torch.Tensor, base: torch.Tensor,
           cfg: DSV4Config):
    """hc_pre：[B,T,hc,D] → (y [B,T,D], post, comb)；rsqrt 用整 4D 维均方 + rms_eps。"""
    shape = x.shape
    xf = x.flatten(2).float()
    r = torch.rsqrt(xf.square().mean(-1, keepdim=True) + cfg.rms_eps)
    # CPU 上 CCCP 可能让该矩阵保持 Int4Weight 打包态；已安装的 _lin
    # 分派器能直接处理，强制调用 .float() 会在首层之前报错。
    mixes = _lin(xf, fn.float() if isinstance(fn, torch.Tensor) else fn) * r
    pre, post, comb = hc_split(mixes, scale.float(), base.float(),
                               cfg.hc_mult, cfg.hc_sinkhorn_iters, cfg.hc_eps)
    y = (pre.unsqueeze(-1) * xf.view(shape)).sum(dim=2)
    dtype = x.dtype
    return y.to(dtype), post.to(dtype), comb.to(dtype)


def hc_post(out: torch.Tensor, residual: torch.Tensor, post: torch.Tensor,
            comb: torch.Tensor) -> torch.Tensor:
    """hc_post（按官方 model.py）：y[:,:,k,:] = post[k]*out + Σ_j comb[j,k]*residual[:,:,j,:]。

    注意：spec 伪代码写成 y[j]=Σ_k comb[j,k]*res[k]（comb 的转置），以官方代码为准——
    comb.unsqueeze(-1)*residual.unsqueeze(-2) 按 dim=2 求和即 y[k]=Σ_j comb[j,k]*res[j]。
    """
    dtype = residual.dtype
    out = out.to(dtype)
    post = post.to(dtype)
    comb = comb.to(dtype)
    return (
        post.unsqueeze(-1) * out.unsqueeze(-2)
        + (comb.unsqueeze(-1) * residual.unsqueeze(-2)).sum(dim=2)
    )


def hc_head(x: torch.Tensor, fn: torch.Tensor, scale: torch.Tensor, base: torch.Tensor,
            cfg: DSV4Config) -> torch.Tensor:
    """末尾 hc 头：pre=sigmoid(mixes*scale+base)+eps（无 sinkhorn），4 通道加权求和 → [B,T,D]。"""
    shape = x.shape
    xf = x.flatten(2).float()
    r = torch.rsqrt(xf.square().mean(-1, keepdim=True) + cfg.rms_eps)
    mixes = _lin(xf, fn.float() if isinstance(fn, torch.Tensor) else fn) * r
    pre = torch.sigmoid(mixes * scale[0] + base) + cfg.hc_eps
    return (pre.unsqueeze(-1) * xf.view(shape)).sum(dim=2).to(x.dtype)


# ---- KV Compressor ----

def _overlap_transform(t: torch.Tensor, value: float, ratio: int, d: int) -> torch.Tensor:
    """[B,N,r,2d] → [B,N,2r,d]：槽位 r..2r = 当前组后半通道(d..2d)，
    槽位 0..r（组 g≥1）= 上一组前半通道(0..d)（即窗口 8 token、步长 4 的重叠池化）。"""
    B, N = t.shape[0], t.shape[1]
    out = t.new_full((B, N, 2 * ratio, d), value)
    out[:, :, ratio:] = t[:, :, :, d:]
    out[:, 1:, :ratio] = t[:, :-1, :, :d]
    return out


def compressor_prefill(x: torch.Tensor, w: dict, ratio: int, d: int, rd: int,
                       cos: torch.Tensor, sin: torch.Tensor, eps: float,
                       st: dict) -> torch.Tensor | None:
    """KV Compressor 批式前向。x [B,T,D] f32；cos/sin: 压缩相位（窗口首 token，[1,N,rd/2]）。

    末尾 T%ratio 个 token 写入 st['ckv']/st['cscore']（score 已加 ape）；
    overlap 时另把最后完整窗口存入 state 前 r 槽（供 decode 的重叠池化）。
    返回压缩 kv [B, T//ratio, d]（无完整窗口时 None）。
    """
    B, T, _ = x.shape
    coff = w["wkv"].shape[0] // d
    overlap = coff == 2
    kv = _lin(x, w["wkv"])                        # [B,T,coff*d]
    score = _lin(x, w["wgate"])
    rem = T % ratio
    cutoff = T - rem
    pooled = None
    if overlap and cutoff >= ratio:
        st["ckv"][:, :ratio] = kv[:, cutoff - ratio:cutoff]
        st["cscore"][:, :ratio] = score[:, cutoff - ratio:cutoff] + w["ape"]
    if cutoff > 0:
        kvg = kv[:, :cutoff].unflatten(1, (-1, ratio))            # [B,N,r,coff*d]
        scg = score[:, :cutoff].unflatten(1, (-1, ratio)) + w["ape"]
        if overlap:
            kvg = _overlap_transform(kvg, 0.0, ratio, d)
            scg = _overlap_transform(scg, float("-inf"), ratio, d)
        probs = scg.float().softmax(dim=2)
        pooled = (kvg.float() * probs).sum(dim=2)                 # [B,N,d]
        pooled = rmsnorm(pooled, w["norm"], eps)
        pooled[..., d - rd:] = rope_apply(pooled[..., d - rd:], cos, sin)
    if rem > 0:
        off = ratio if overlap else 0
        st["ckv"][:, off:off + rem] = kv[:, cutoff:]
        st["cscore"][:, off:off + rem] = score[:, cutoff:] + w["ape"][:rem]
    return pooled


def compressor_decode(x: torch.Tensor, w: dict, ratio: int, d: int, rd: int,
                      cos: torch.Tensor, sin: torch.Tensor, eps: float,
                      st: dict, pos: int) -> torch.Tensor | None:
    """KV Compressor 增量单步。x [B,1,D]；cos/sin: 窗口首 token 相位（位置 pos+1-ratio）。

    每步把当前 token 的 wkv/wgate(+ape[pos%r]) 写入 state；仅当 (pos+1)%ratio==0
    时才池化产出一个压缩 kv [B,1,d]，否则返回 None。
    overlap：池化用「上一完整窗口（state 前 r 槽，前半通道）+ 当前窗口（后 r 槽，
    后半通道）」，随后窗口滑动（state[:r]=state[r:]）。
    """
    coff = w["wkv"].shape[0] // d
    overlap = coff == 2
    kv = _lin(x, w["wkv"])                        # [B,1,coff*d]
    score = _lin(x, w["wgate"]) + w["ape"][pos % ratio]
    should = (pos + 1) % ratio == 0
    if overlap:
        st["ckv"][:, ratio + pos % ratio] = kv[:, 0]
        st["cscore"][:, ratio + pos % ratio] = score[:, 0]
        if not should:
            return None
        kvs = torch.cat([st["ckv"][:, :ratio, :d], st["ckv"][:, ratio:, d:]], dim=1)
        scs = torch.cat([st["cscore"][:, :ratio, :d], st["cscore"][:, ratio:, d:]], dim=1)
        probs = scs.float().softmax(dim=1)
        pooled = (kvs.float() * probs).sum(dim=1, keepdim=True)
        st["ckv"][:, :ratio] = st["ckv"][:, ratio:].clone()
        st["cscore"][:, :ratio] = st["cscore"][:, ratio:].clone()
    else:
        st["ckv"][:, pos % ratio] = kv[:, 0]
        st["cscore"][:, pos % ratio] = score[:, 0]
        if not should:
            return None
        probs = st["cscore"].float().softmax(dim=1)
        pooled = (st["ckv"].float() * probs).sum(dim=1, keepdim=True)
    pooled = rmsnorm(pooled, w["norm"], eps)
    pooled[..., d - rd:] = rope_apply(pooled[..., d - rd:], cos, sin)
    return pooled


# ---- 注意力 ----

def _qkv(x: torch.Tensor, w: dict, cfg: DSV4Config, cache: RopeCache, pos0: int):
    """低秩 Q + MQA kv。x [B,T,D] → q [B,T,H,hd]（逐头无权重 RMS + 末 64 维 RoPE）、
    kv [B,T,hd]（kv_norm + 末 64 维 RoPE；64 个 q 头共享同一份 kv，key=value）。"""
    B, T, _ = x.shape
    H, hd, rd = cfg.n_heads, cfg.head_dim, cfg.qk_rope_head_dim
    # QKV/attention 暂保留 FP32 局部状态以命中现有 decode 融合核；层边界
    # hidden 与 dense GEMM 仍为 BF16。SM120 sparse kernel 将移除此转换。
    qr = rmsnorm(_lin(x, w["wq_a"]).float(), w["q_norm"], cfg.rms_eps)
    q = _lin(qr, w["wq_b"]).view(B, T, H, hd).float()
    qf = q.float()
    q = (
        qf * torch.rsqrt(qf.square().mean(-1, keepdim=True) + cfg.rms_eps)
    ).to(q.dtype)   # 逐头无权重 RMS，易漏！
    cos = cache.cos[pos0:pos0 + T]
    sin = cache.sin[pos0:pos0 + T]
    q[..., hd - rd:] = rope_apply(q[..., hd - rd:], cos.view(1, T, 1, -1), sin.view(1, T, 1, -1))
    kv = rmsnorm(_lin(x, w["wkv"]).float(), w["kv_norm"], cfg.rms_eps)
    kv[..., hd - rd:] = rope_apply(kv[..., hd - rd:], cos.view(1, T, -1), sin.view(1, T, -1))
    return qr, q, kv


def _o_proj(o: torch.Tensor, w: dict, cfg: DSV4Config) -> torch.Tensor:
    """分组 LoRA O：o [B,T,H*hd] → view [B,T,G,H*hd/G] → 每组独立降维到 o_lora_rank → wo_b。"""
    B, T = o.shape[0], o.shape[1]
    G = cfg.o_groups
    o = o.reshape(B, T, G, -1)
    wo_a = w["wo_a"].view(G, cfg.o_lora_rank, -1)
    o = torch.einsum("btgd,grd->btgr", o, wo_a)
    return _lin(o.flatten(2), w["wo_b"])


# O 投影钩子：CCCP 推理侧以 Int4Weight 分组反量化实现替换（见 cccp/dsv4model.py）
_o_proj_hook = _o_proj
# 单 token attention core 钩子：CCCP CUDA 扩展可融合 score/softmax/value/RoPE。
_attn_decode_core_hook = None


def _compressed_write_many(st: dict, win: int, start: int, values: torch.Tensor) -> None:
    paged = st.get("compressed")
    if paged is not None:
        paged.write_many(start, values)
    else:
        st["kv"][:, win + start:win + start + values.shape[1]] = values


def _compressed_write(st: dict, win: int, item: int, value: torch.Tensor) -> None:
    paged = st.get("compressed")
    if paged is not None:
        paged.write(item, value)
    else:
        st["kv"][:, win + item] = value


def _compressed_prefix(st: dict, win: int, length: int) -> torch.Tensor:
    paged = st.get("compressed")
    if paged is not None:
        return paged.contiguous_prefix(length)
    return st["kv"][:, win:win + length]


def attn_prefill(x: torch.Tensor, w: dict, st: dict, cfg: DSV4Config,
                 cache: RopeCache, ratio: int) -> torch.Tensor:
    """注意力批式前向（start_pos=0）。x [B,T,D]（已过 attn_norm）→ [B,T,D]。

    集合 = 最近 win 个因果原始 token（带状掩码）+ 压缩 token（query i 可见 j<(i+1)/ratio）；
    softmax 分母含 attn_sink（exp(sink)/denom，不进分子）；scale=hd^-0.5；输出末 64 维反旋转。
    同时更新环形窗口缓存（槽位=位置%win）与压缩槽位。
    """
    B, T, _ = x.shape
    H, hd, rd = cfg.n_heads, cfg.head_dim, cfg.qk_rope_head_dim
    win = cfg.sliding_window
    _qr, q, kv = _qkv(x, w, cfg, cache, 0)
    scale = hd ** -0.5
    n = min(T, win)
    slots = torch.arange(T - n, T, device=x.device) % win
    st["kv"][:, slots] = kv[:, T - n:]
    st["win_pos"][:, slots] = torch.arange(T - n, T, device=x.device)
    scores = torch.einsum("bthd,bsd->bhts", q * scale, kv)
    ii = torch.arange(T, device=x.device)
    allow = (ii[None, :] <= ii[:, None]) & (ii[None, :] > ii[:, None] - win)
    scores = scores.masked_fill(~allow, float("-inf"))
    values = kv
    if ratio:
        n_full = T // ratio
        if n_full > 0:
            cos = cache.cos[0:n_full * ratio:ratio].view(1, n_full, -1)
            sin = cache.sin[0:n_full * ratio:ratio].view(1, n_full, -1)
        else:
            # 短序列（T<ratio）无完整窗口：压缩相位不会被使用（pooled=None），
            # 占位即可；compressor_prefill 仍会把尾部 token 写入 state 供 decode 续接
            cos = cache.cos[:1].view(1, 1, -1)
            sin = cache.sin[:1].view(1, 1, -1)
        ck = compressor_prefill(x, w["cmp"], ratio, hd, rd, cos, sin, cfg.rms_eps, st)
        if ck is not None:
            nc = ck.shape[1]
            _compressed_write_many(st, win, 0, ck)
            comp = _compressed_prefix(st, win, nc).to(q.dtype)
            cs = torch.einsum("bthd,bnd->bhtn", q * scale, comp)
            jj = torch.arange(nc, device=x.device)
            cmallow = jj[None, :] < ((ii[:, None] + 1) // ratio)
            cs = cs.masked_fill(~cmallow, float("-inf"))
            scores = torch.cat([scores, cs], dim=-1)
            values = torch.cat([values, comp], dim=1)
    m = scores.amax(dim=-1, keepdim=True)                 # max 不含 sink（与 kernel 一致）
    e = (scores - m).exp()
    denom = e.sum(dim=-1) + (w["attn_sink"].view(1, -1, 1) - m.squeeze(-1)).exp()
    o = torch.einsum("bhts,bsd->bthd", e, values) / denom.transpose(1, 2).unsqueeze(-1)
    cos = cache.cos[:T].view(1, T, 1, -1)
    sin = cache.sin[:T].view(1, T, 1, -1)
    o[..., hd - rd:] = rope_apply(o[..., hd - rd:], cos, sin, inverse=True)   # 输出反旋转，易漏！
    return _o_proj_hook(o.flatten(2), w, cfg)


def attn_decode(x: torch.Tensor, w: dict, st: dict, cfg: DSV4Config,
                cache: RopeCache, ratio: int, pos: int) -> torch.Tensor:
    """注意力增量单步。x [B,1,D] → [B,1,D]；pos = 当前 token 绝对位置（>0）。"""
    B = x.shape[0]
    H, hd, rd = cfg.n_heads, cfg.head_dim, cfg.qk_rope_head_dim
    win = cfg.sliding_window
    _qr, q, kv = _qkv(x, w, cfg, cache, pos)              # T=1
    scale = hd ** -0.5
    st["kv"][:, pos % win] = kv[:, 0]
    st["win_pos"][:, pos % win] = pos
    nc = 0
    if ratio:
        cos = cache.cos[pos + 1 - ratio].view(1, 1, -1)   # 窗口首 token 相位
        sin = cache.sin[pos + 1 - ratio].view(1, 1, -1)
        ck = compressor_decode(x, w["cmp"], ratio, hd, rd, cos, sin, cfg.rms_eps, st, pos)
        if ck is not None:
            _compressed_write(st, win, pos // ratio, ck[:, 0])
        nc = (pos + 1) // ratio
    out_cos = cache.cos[pos].view(1, 1, 1, -1)
    out_sin = cache.sin[pos].view(1, 1, 1, -1)
    comp = _compressed_prefix(st, win, nc).to(q.dtype) if nc > 0 else None
    if _attn_decode_core_hook is not None:
        fused = _attn_decode_core_hook(
            q[:, 0], st["kv"][:, :win], st["win_pos"],
            comp if comp is not None else st["kv"][:, :0],
            w["attn_sink"], out_cos, out_sin, scale,
        )
        if fused is not None:
            return _o_proj_hook(fused.unsqueeze(1).flatten(2), w, cfg)
    scores = torch.einsum("bhd,bwd->bhw", q[:, 0] * scale, st["kv"][:, :win])
    scores = scores.masked_fill((st["win_pos"] < 0)[:, None, :], float("-inf"))
    values = st["kv"][:, :win]
    if nc > 0:
        cs = torch.einsum("bhd,bnd->bhn", q[:, 0] * scale, comp)
        scores = torch.cat([scores, cs], dim=-1)
        values = torch.cat([values, comp], dim=1)
    m = scores.amax(dim=-1)
    e = (scores - m.unsqueeze(-1)).exp()
    denom = e.sum(dim=-1) + (w["attn_sink"] - m).exp()
    o = torch.einsum("bhn,bnd->bhd", e, values) / denom.unsqueeze(-1)
    o = o.unsqueeze(1)                                    # [B,1,H,hd]
    o[..., hd - rd:] = rope_apply(o[..., hd - rd:], out_cos, out_sin, inverse=True)
    return _o_proj_hook(o.flatten(2), w, cfg)


# ---- MoE ----

def gate_route(x: torch.Tensor, w: dict, cfg: DSV4Config, ids: torch.Tensor):
    """sqrtsoftplus 路由。x [N,D] f32，ids [N] long → (weights [N,K], indices [N,K])。

    scores = sqrt(softplus(x@W^T))（fp32）；hash 层：indices = tid2eid[ids]（静态选择，
    无 bias）；其余层 noaux_tc：scores+bias 选 top-k（bias 只影响选择不进权重）。
    权重 = gather(原始 scores) → 归一化（norm_topk_prob）→ ×routed_scaling(1.5)。
    """
    scores = F.softplus(_lin(x, w["gate"].float())).sqrt()
    tid2eid = w.get("tid2eid")
    if tid2eid is not None:
        indices = tid2eid[ids]
    else:
        indices = (scores + w["gate_bias"].float()).topk(cfg.top_k, dim=-1).indices
    weights = scores.gather(1, indices)
    if cfg.norm_topk_prob:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights * cfg.routed_scaling, indices


def expert_mlp(x: torch.Tensor, w1: torch.Tensor, w3: torch.Tensor, w2: torch.Tensor,
               limit: float, weight: torch.Tensor | None = None) -> torch.Tensor:
    """专家 SwiGLU（f32）：up=clamp(±limit)、gate=clamp(max=limit)（gate 只截上界）、
    silu(gate)*up →（可选乘路由权重）→ w2。"""
    gate = _lin(x, w1)
    up = _lin(x, w3)
    if limit > 0:
        up = up.clamp(min=-limit, max=limit)
        gate = gate.clamp(max=limit)
    h = F.silu(gate) * up
    if weight is not None:
        h = weight * h
    return _lin(h, w2)


def moe_forward(x: torch.Tensor, w: dict, cfg: DSV4Config, ids: torch.Tensor,
                get_expert, layer: int) -> torch.Tensor:
    """MoE：top-k routed 专家（按专家循环 gather token，同官方简单路径）+ 1 共享专家。"""
    B, T, D = x.shape
    xf = x.reshape(B * T, D).float()
    weights, indices = gate_route(xf, w, cfg, ids.reshape(-1))
    y = torch.zeros_like(xf)
    for e in range(cfg.n_experts):
        sel = (indices == e).nonzero(as_tuple=True)
        if sel[0].numel() == 0:
            continue
        w1, w3, w2 = get_expert(layer, e)
        y[sel[0]] += expert_mlp(xf[sel[0]], w1, w3, w2, cfg.swiglu_limit,
                                weights[sel[0], sel[1], None])
    y += expert_mlp(xf, w["sh_w1"], w["sh_w3"], w["sh_w2"], cfg.swiglu_limit)
    return y.view(B, T, D)


# ---- Block / Model ----

def block_forward(h: torch.Tensor, w: dict, st: dict, cfg: DSV4Config,
                  cache: RopeCache, ratio: int, ids: torch.Tensor, pos0: int,
                  get_expert, layer: int) -> torch.Tensor:
    """一个 Block：hc_pre→attn_norm→attn→hc_post；hc_pre→ffn_norm→ffn→hc_post。"""
    residual = h
    y, post, comb = hc_pre(h, w["hc_attn_fn"], w["hc_attn_scale"], w["hc_attn_base"], cfg)
    y = rmsnorm(y, w["attn_norm"], cfg.rms_eps)
    if pos0 == 0:
        a = attn_prefill(y, w, st, cfg, cache, ratio)
    else:
        a = attn_decode(y, w, st, cfg, cache, ratio, pos0)
    h = hc_post(a, residual, post, comb)
    residual = h
    y, post, comb = hc_pre(h, w["hc_ffn_fn"], w["hc_ffn_scale"], w["hc_ffn_base"], cfg)
    y = rmsnorm(y, w["ffn_norm"], cfg.rms_eps)
    f = moe_forward(y, w, cfg, ids, get_expert, layer)
    return hc_post(f, residual, post, comb)


class DSV4Model:
    """DeepSeek-V4 主模型前向（prefill 批式 + decode 单步；MTP/DSpark 层 43-45 不涉及）。

    provider 协议：layer(i)->dict、expert(i,e)->(w1,w3,w2)、embed()、head()、
    final_norm()、hc_head()->(fn,scale,base)。DSV4Checkpoint 即该协议的实现。
    """

    def __init__(self, cfg: DSV4Config, provider, max_seq: int = 2048, device="cpu"):
        self.cfg = cfg
        self.p = provider
        self.max_seq = max_seq
        self.device = device
        ratios = list(cfg.compress_ratios) if cfg.compress_ratios else []
        self.ratios = (ratios + [0] * cfg.n_layers)[: cfg.n_layers]
        rd = cfg.qk_rope_head_dim
        self.rope_base = RopeCache(rd, max_seq, cfg.rope_theta, None)
        self.rope_cmp = RopeCache(rd, max_seq, cfg.compress_rope_theta, cfg.rope_scaling or None)
        for rc in (self.rope_base, self.rope_cmp):  # RopeCache 默认 CPU，与权重设备对齐
            rc.cos = rc.cos.to(device)
            rc.sin = rc.sin.to(device)
        self._embed = provider.embed().to(device)
        self._head = provider.head().to(device)
        self._norm = provider.final_norm().to(device)
        self._hc_head = tuple(t.to(device) for t in provider.hc_head())
        self._layers: dict[int, dict] = {}
        self.states: list[dict] | None = None

    def _layer(self, i: int) -> dict:
        w = self._layers.get(i)
        if w is None:
            w = self.p.layer(i)
            self._layers[i] = w
            # 模型级层缓存同样有界（容量 2，顺序过层）：旧版无界持有 43 层 f32
            # ≈17GB，会把显存顶进共享显存换页（实测 +7.1GB）。
            while len(self._layers) > 2:
                for k in list(self._layers):
                    if k != i:
                        del self._layers[k]
                        break
        return w

    def _rope(self, i: int) -> RopeCache:
        return self.rope_cmp if self.ratios[i] else self.rope_base

    def _alloc(self, B: int) -> None:
        """按 batch 分配每层 KV 缓存（环形窗口 + 压缩槽位）与 Compressor state。"""
        cfg = self.cfg
        win, hd = cfg.sliding_window, cfg.head_dim
        self.states = []
        for i in range(cfg.n_layers):
            ratio = self.ratios[i]
            st = {
                "kv": torch.zeros(B, win + (self.max_seq // ratio + 1 if ratio else 0), hd,
                                  device=self.device),
                "win_pos": torch.full((B, win), -1, dtype=torch.long, device=self.device),
            }
            if ratio:
                coff = 2 if ratio == 4 else 1       # 规范：ratio=4 → coff=2，ratio=128 → coff=1
                st["ckv"] = torch.zeros(B, coff * ratio, coff * hd, device=self.device)
                st["cscore"] = torch.full((B, coff * ratio, coff * hd), float("-inf"),
                                          device=self.device)
            self.states.append(st)

    def reset(self) -> None:
        self.states = None

    @torch.no_grad()
    def prefill(self, ids: torch.Tensor, full_logits: bool = True) -> torch.Tensor:
        """批式前向（start_pos=0）。ids [B,T] long → logits [B,T,V]（full_logits=False 时 [B,V] 末位）。"""
        ids = ids.to(self.device).long()
        B, T = ids.shape
        self._alloc(B)
        cfg = self.cfg
        # 嵌入复制到 hc_mult 个通道（官方 model.py：unsqueeze(2).repeat(...)）
        h = self._embed[ids].unsqueeze(2).repeat(1, 1, cfg.hc_mult, 1)
        for i in range(cfg.n_layers):
            h = block_forward(h, self._layer(i), self.states[i], cfg, self._rope(i),
                              self.ratios[i], ids, 0, self.p.expert, i)
        y = hc_head(h, *self._hc_head, cfg)
        y = rmsnorm(y, self._norm, cfg.rms_eps)
        logits = _lin(y, self._head)
        return logits if full_logits else logits[:, -1]

    @torch.no_grad()
    def decode(self, ids: torch.Tensor, pos: int) -> torch.Tensor:
        """增量单步。ids [B] long（位置 pos 的输入 token，pos>0）→ logits [B,V]。"""
        assert self.states is not None and pos > 0, "decode 前须先 prefill"
        ids = ids.to(self.device).long()
        cfg = self.cfg
        h = self._embed[ids].unsqueeze(1).unsqueeze(2).repeat(1, 1, cfg.hc_mult, 1)
        for i in range(cfg.n_layers):
            h = block_forward(h, self._layer(i), self.states[i], cfg, self._rope(i),
                              self.ratios[i], ids.view(-1, 1), pos, self.p.expert, i)
        y = hc_head(h, *self._hc_head, cfg)
        y = rmsnorm(y, self._norm, cfg.rms_eps)
        return _lin(y[:, 0], self._head)


# =====================================================================
# 自检：微小合成模型对照逐元素朴素实现（python -m CCCP.dsv4）
# =====================================================================

def _naive_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    v = (x * x).sum(-1, keepdim=True) / x.shape[-1]
    return w * x / torch.sqrt(v + eps)


def _naive_rope_complex(x: torch.Tensor, fc: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """官方复数乘法形式的 RoPE（view_as_complex 相邻对），fc 为复数 freqs_cis（已塑形可广播）。"""
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        fc = fc.conj()
    return torch.view_as_real(xc * fc).flatten(-2).to(x.dtype)


def _naive_hc_pre(x, fn, scale, base, cfg):
    B, T, hc, D = x.shape
    xf = x.reshape(B, T, hc * D)
    y = torch.zeros(B, T, D)
    posts, combs = [], []
    for b in range(B):
        for t in range(T):
            v = xf[b, t]
            r = 1.0 / math.sqrt(float((v * v).mean()) + cfg.rms_eps)
            m = fn @ v * r
            pre = torch.sigmoid(m[:hc] * scale[0] + base[:hc]) + cfg.hc_eps
            post = 2 * torch.sigmoid(m[hc:2 * hc] * scale[1] + base[hc:2 * hc])
            comb = m[2 * hc:].view(hc, hc) * scale[2] + base[2 * hc:].view(hc, hc)
            comb = comb.softmax(-1) + cfg.hc_eps
            comb = comb / (comb.sum(0, keepdim=True) + cfg.hc_eps)
            for _ in range(cfg.hc_sinkhorn_iters - 1):
                comb = comb / (comb.sum(1, keepdim=True) + cfg.hc_eps)
                comb = comb / (comb.sum(0, keepdim=True) + cfg.hc_eps)
            y[b, t] = (pre[:, None] * v.view(hc, D)).sum(0)
            posts.append(post)
            combs.append(comb)
    return y, torch.stack(posts).view(B, T, hc), torch.stack(combs).view(B, T, hc, hc)


def _naive_hc_post(out, residual, post, comb):
    # 与官方一致：y[j] = post[j]*out + Σ_k comb[k,j]*res[k]（comb 第一维是残差通道）
    B, T, hc, D = residual.shape
    y = torch.zeros_like(residual)
    for j in range(hc):
        acc = post[..., j, None] * out
        for k in range(hc):
            acc = acc + comb[..., k, j, None] * residual[:, :, k]
        y[:, :, j] = acc
    return y


def _naive_hc_head(x, fn, scale, base, cfg):
    B, T, hc, D = x.shape
    xf = x.reshape(B, T, hc * D)
    y = torch.zeros(B, T, D)
    for b in range(B):
        for t in range(T):
            v = xf[b, t]
            r = 1.0 / math.sqrt(float((v * v).mean()) + cfg.rms_eps)
            m = fn @ v * r
            pre = torch.sigmoid(m * scale[0] + base) + cfg.hc_eps
            y[b, t] = (pre[:, None] * v.view(hc, D)).sum(0)
    return y


def _naive_compressor_prefill(x, w, ratio, d, rd, cache: RopeCache, eps):
    """分组 softmax 池化的逐组朴素实现（overlap 时 8 窗口 4 步长）。"""
    B, T, _ = x.shape
    coff = w["wkv"].shape[0] // d
    kv = x @ w["wkv"].t()
    score = x @ w["wgate"].t()
    N = T // ratio
    out = x.new_zeros(B, N, d)
    for g in range(N):
        for b in range(B):
            ks, ss = [], []
            if coff == 2:
                if g > 0:                       # 上一组前半通道（g=0 时这些槽位为 -inf，跳过）
                    for s in range(ratio):
                        ks.append(kv[b, (g - 1) * ratio + s, :d])
                        ss.append(score[b, (g - 1) * ratio + s, :d] + w["ape"][s, :d])
                for s in range(ratio):          # 当前组后半通道
                    ks.append(kv[b, g * ratio + s, d:])
                    ss.append(score[b, g * ratio + s, d:] + w["ape"][s, d:])
            else:
                for s in range(ratio):
                    ks.append(kv[b, g * ratio + s])
                    ss.append(score[b, g * ratio + s] + w["ape"][s])
            K = torch.stack(ks)
            S = torch.stack(ss)
            out[b, g] = (K * S.softmax(0)).sum(0)
    out = _naive_rmsnorm(out, w["norm"], eps)
    fc = torch.complex(cache.cos, cache.sin)    # 相位 = 窗口首 token
    fcs = fc[0:N * ratio:ratio].view(1, N, -1)
    out[..., d - rd:] = _naive_rope_complex(out[..., d - rd:], fcs)
    return out


def _naive_attn(x, w, cfg, cache: RopeCache, ratio: int = 0, ckv: torch.Tensor | None = None):
    """逐 (query, head) 循环的朴素注意力（滑窗因果 + 压缩 token + attn_sink + 输出反旋转）。"""
    B, T, _ = x.shape
    H, hd, rd = cfg.n_heads, cfg.head_dim, cfg.qk_rope_head_dim
    win = cfg.sliding_window
    qr = _naive_rmsnorm(x @ w["wq_a"].t(), w["q_norm"], cfg.rms_eps)
    q = (qr @ w["wq_b"].t()).view(B, T, H, hd)
    q = q / torch.sqrt(q.square().mean(-1, keepdim=True) + cfg.rms_eps)
    fc = torch.complex(cache.cos, cache.sin)
    q[..., hd - rd:] = _naive_rope_complex(q[..., hd - rd:], fc[:T].view(1, T, 1, -1))
    kv = _naive_rmsnorm(x @ w["wkv"].t(), w["kv_norm"], cfg.rms_eps)
    kv[..., hd - rd:] = _naive_rope_complex(kv[..., hd - rd:], fc[:T].view(1, T, -1))
    scale = hd ** -0.5
    o = x.new_zeros(B, T, H, hd)
    for b in range(B):
        for i in range(T):
            lo = max(0, i - win + 1)
            keys = [kv[b, j] for j in range(lo, i + 1)]
            if ckv is not None:
                keys += [ckv[b, j] for j in range((i + 1) // ratio)]
            K = torch.stack(keys)
            for h in range(H):
                s = K @ (q[b, i, h] * scale)
                m = s.max()
                e = (s - m).exp()
                denom = e.sum() + (w["attn_sink"][h] - m).exp()
                o[b, i, h] = (e.unsqueeze(-1) * K).sum(0) / denom
    o[..., hd - rd:] = _naive_rope_complex(o[..., hd - rd:], fc[:T].view(1, T, 1, -1), inverse=True)
    G = cfg.o_groups
    og = o.reshape(B, T, G, -1)
    wo_a = w["wo_a"].view(G, cfg.o_lora_rank, -1)
    r = torch.einsum("btgd,grd->btgr", og, wo_a)
    return r.flatten(2) @ w["wo_b"].t()


def _naive_moe(x, w, cfg, ids, get_expert, layer):
    B, T, D = x.shape
    y = torch.zeros(B, T, D)
    for b in range(B):
        for t in range(T):
            xt = x[b, t]
            scores = torch.log1p(torch.exp(w["gate"] @ xt)).sqrt()
            if "tid2eid" in w:
                idx = w["tid2eid"][ids[b, t]]
            else:
                idx = (scores + w["gate_bias"]).topk(cfg.top_k).indices
            ww = scores[idx]
            ww = ww / ww.sum() * cfg.routed_scaling
            acc = torch.zeros(D)
            for k, e in enumerate(idx.tolist()):
                w1, w3, w2 = get_expert(layer, e)
                gate = (w1 @ xt).clamp(max=cfg.swiglu_limit)
                up = (w3 @ xt).clamp(-cfg.swiglu_limit, cfg.swiglu_limit)
                acc = acc + ww[k] * (w2 @ (F.silu(gate) * up))
            g = (w["sh_w1"] @ xt).clamp(max=cfg.swiglu_limit)
            u = (w["sh_w3"] @ xt).clamp(-cfg.swiglu_limit, cfg.swiglu_limit)
            acc = acc + w["sh_w2"] @ (F.silu(g) * u)
            y[b, t] = acc
    return y


def _naive_block(h, w, cfg, cache, ratio, ids, layer, get_expert):
    y, post, comb = _naive_hc_pre(h, w["hc_attn_fn"], w["hc_attn_scale"], w["hc_attn_base"], cfg)
    y = _naive_rmsnorm(y, w["attn_norm"], cfg.rms_eps)
    ckv = None
    if ratio:
        ckv = _naive_compressor_prefill(y, w["cmp"], ratio, cfg.head_dim,
                                        cfg.qk_rope_head_dim, cache, cfg.rms_eps)
    a = _naive_attn(y, w, cfg, cache, ratio, ckv)
    h = _naive_hc_post(a, h, post, comb)
    y, post, comb = _naive_hc_pre(h, w["hc_ffn_fn"], w["hc_ffn_scale"], w["hc_ffn_base"], cfg)
    y = _naive_rmsnorm(y, w["ffn_norm"], cfg.rms_eps)
    f = _naive_moe(y, w, cfg, ids, get_expert, layer)
    return _naive_hc_post(f, h, post, comb)


def _naive_net(ids, cfg, prov, ropes):
    h = prov.embed()[ids].unsqueeze(2).repeat(1, 1, cfg.hc_mult, 1)
    for i in range(cfg.n_layers):
        h = _naive_block(h, prov.layer(i), cfg, ropes[i], cfg.compress_ratios[i],
                         ids, i, prov.expert)
    fn, scale, base = prov.hc_head()
    y = _naive_hc_head(h, fn, scale, base, cfg)
    y = _naive_rmsnorm(y, prov.final_norm(), cfg.rms_eps)
    return y @ prov.head().t()


def _tiny_cfg() -> DSV4Config:
    return DSV4Config(
        n_layers=3, hidden=64, n_experts=4, top_k=2, moe_inter=32, n_shared=1,
        n_heads=4, head_dim=16, q_lora_rank=16, o_lora_rank=16, o_groups=2,
        kv_dim=16, qk_rope_head_dim=8, n_kv_heads=1, vocab=128, rms_eps=1e-6,
        scoring_func="sqrtsoftplus", norm_topk_prob=True, routed_scaling=1.5,
        swiglu_limit=10.0, n_hash_layers=1, sliding_window=8, index_topk=512,
        rope_theta=10000.0,
        rope_scaling={"type": "yarn", "factor": 16,
                      "original_max_position_embeddings": 65536,
                      "beta_fast": 32, "beta_slow": 1},
        eos_token_id=[1], compress_ratios=[0, 4, 0],
        compress_rope_theta=160000.0, hc_mult=4,
    )


class _TinyProvider:
    """自备用微型合成权重（DSV4Model 的 provider 协议；层0=tid2eid，层1=ratio4 压缩）。"""

    def __init__(self, cfg: DSV4Config, seed: int = 0):
        g = torch.Generator().manual_seed(seed)

        def r(*shape, s=0.05):
            return torch.randn(*shape, generator=g) * s

        D, hd = cfg.hidden, cfg.head_dim
        mix = (2 + cfg.hc_mult) * cfg.hc_mult
        self._layers = []
        for i in range(cfg.n_layers):
            w = {
                "wq_a": r(cfg.q_lora_rank, D), "q_norm": 1 + r(cfg.q_lora_rank, s=0.02),
                "wq_b": r(cfg.n_heads * hd, cfg.q_lora_rank),
                "wkv": r(hd, D), "kv_norm": 1 + r(hd, s=0.02),
                "attn_sink": r(cfg.n_heads, s=0.1),
                "wo_a": r(cfg.o_groups * cfg.o_lora_rank, cfg.n_heads * hd // cfg.o_groups),
                "wo_b": r(D, cfg.o_groups * cfg.o_lora_rank),
                "attn_norm": 1 + r(D, s=0.02), "ffn_norm": 1 + r(D, s=0.02),
                "gate": r(cfg.n_experts, D),
                "sh_w1": r(cfg.moe_inter, D), "sh_w3": r(cfg.moe_inter, D),
                "sh_w2": r(D, cfg.moe_inter),
                "hc_attn_fn": r(mix, cfg.hc_mult * D, s=0.02), "hc_attn_base": r(mix, s=0.1),
                "hc_attn_scale": 1 + r(3, s=0.02),
                "hc_ffn_fn": r(mix, cfg.hc_mult * D, s=0.02), "hc_ffn_base": r(mix, s=0.1),
                "hc_ffn_scale": 1 + r(3, s=0.02),
            }
            ratio = cfg.compress_ratios[i]
            if ratio:
                coff = 2 if ratio == 4 else 1
                w["cmp"] = {
                    "wkv": r(coff * hd, D), "wgate": r(coff * hd, D),
                    "ape": r(ratio, coff * hd, s=0.1), "norm": 1 + r(hd, s=0.02),
                }
            if i < cfg.n_hash_layers:
                # 真实 tid2eid 每行 6 个专家互不相同，此处同样生成不重复索引
                perm = torch.argsort(torch.rand(cfg.vocab, cfg.n_experts, generator=g), dim=1)
                w["tid2eid"] = perm[:, :cfg.top_k]
            else:
                w["gate_bias"] = r(cfg.n_experts, s=0.2)
            self._layers.append(w)
        self._experts = {(i, e): (r(cfg.moe_inter, D), r(cfg.moe_inter, D), r(D, cfg.moe_inter))
                         for i in range(cfg.n_layers) for e in range(cfg.n_experts)}
        self._embed = r(cfg.vocab, D)
        self._head = r(cfg.vocab, D)
        self._norm = 1 + r(D, s=0.02)
        self._hc_head = (r(cfg.hc_mult, cfg.hc_mult * D, s=0.02),
                         1 + r(1, s=0.02), r(cfg.hc_mult, s=0.1))

    def layer(self, i):
        return self._layers[i]

    def expert(self, i, e):
        return self._experts[(i, e)]

    def embed(self):
        return self._embed

    def head(self):
        return self._head

    def final_norm(self):
        return self._norm

    def hc_head(self):
        return self._hc_head


def _write_safetensors(path: str, tensors: dict) -> None:
    """极简 safetensors 写入（自检用）：tensors = {name: (dtype_str, shape, raw_bytes)}。"""
    header, off, blobs = {}, 0, []
    for name, (dt, shape, raw) in tensors.items():
        header[name] = {"dtype": dt, "shape": list(shape),
                        "data_offsets": [off, off + len(raw)]}
        off += len(raw)
        blobs.append(raw)
    hj = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for raw in blobs:
            f.write(raw)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    torch.manual_seed(0)
    fails = []

    def check(name, ok, detail=""):
        print(f"[{'OK' if ok else 'FAIL'}] {name}  {detail}")
        if not ok:
            fails.append(name)

    def diff(a, b):
        return float((a - b).abs().max())

    cfg = _tiny_cfg()
    prov = _TinyProvider(cfg, seed=0)
    rope_base = RopeCache(cfg.qk_rope_head_dim, 32, cfg.rope_theta, None)
    rope_cmp = RopeCache(cfg.qk_rope_head_dim, 32, cfg.compress_rope_theta, cfg.rope_scaling)

    # 1) RoPE 相邻对旋转 vs 官方复数乘法 + 反旋转往返
    x = torch.randn(2, 5, 3, 8)
    cosv = rope_base.cos[:5].view(1, 5, 1, -1)
    sinv = rope_base.sin[:5].view(1, 5, 1, -1)
    fc = torch.complex(rope_base.cos, rope_base.sin)
    mine = rope_apply(x, cosv, sinv)
    ref = _naive_rope_complex(x, fc[:5].view(1, 5, 1, -1))
    d1 = diff(mine, ref)
    back = rope_apply(mine, cosv, sinv, inverse=True)
    d2 = diff(back, x)
    check("RoPE 旋转 vs 复数乘法 / 反旋转往返", d1 < 1e-6 and d2 < 1e-6,
          f"diff={d1:.2e}/{d2:.2e}")

    # 2) YaRN 频率 vs 手工逐元素公式
    yarn = cfg.rope_scaling
    dim = cfg.qk_rope_head_dim
    theta = cfg.compress_rope_theta
    inv = [theta ** (-2.0 * i / dim) for i in range(dim // 2)]
    orig = float(yarn["original_max_position_embeddings"])

    def fcd(r_):
        return dim * math.log(orig / (r_ * 2 * math.pi)) / (2 * math.log(theta))

    low = max(math.floor(fcd(yarn["beta_fast"])), 0)
    high = min(math.ceil(fcd(yarn["beta_slow"])), dim - 1)
    fr = []
    for i in range(dim // 2):
        ramp = min(max((i - low) / (high - low), 0.0), 1.0)
        fr.append(inv[i] / yarn["factor"] * ramp + inv[i] * (1 - ramp))
    terr = 0.0
    for p in range(16):
        for i in range(dim // 2):
            terr = max(terr, abs(float(rope_cmp.cos[p, i]) - math.cos(p * fr[i])))
            terr = max(terr, abs(float(rope_cmp.sin[p, i]) - math.sin(p * fr[i])))
    check("YaRN 频率（theta=160000/factor=16/orig=65536/beta 32,1）", terr < 1e-6,
          f"err={terr:.2e}")

    # 3) q 逐头无权重 RMS norm
    q = torch.randn(2, 3, 4, 16)
    mine = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + cfg.rms_eps)
    ref = q.clone()
    for b in range(2):
        for t in range(3):
            for h in range(4):
                v = float((q[b, t, h] ** 2).mean())
                ref[b, t, h] = q[b, t, h] / math.sqrt(v + cfg.rms_eps)
    d = diff(mine, ref)
    check("q 逐头无权重 RMS norm", d < 1e-6, f"diff={d:.2e}")

    # 4) sinkhorn 双随机性 + hc_pre/hc_post vs 朴素逐位置实现
    mixes = torch.randn(2, 3, 24) * 2
    pre, post, comb = hc_split(mixes, prov.layer(0)["hc_attn_scale"],
                               prov.layer(0)["hc_attn_base"], cfg.hc_mult,
                               cfg.hc_sinkhorn_iters, cfg.hc_eps)
    row_err = float((comb.sum(-1) - 1).abs().max())
    col_err = float((comb.sum(-2) - 1).abs().max())
    x4 = torch.randn(1, 5, cfg.hc_mult, cfg.hidden)
    y1, p1, c1 = hc_pre(x4, prov.layer(0)["hc_attn_fn"], prov.layer(0)["hc_attn_scale"],
                        prov.layer(0)["hc_attn_base"], cfg)
    y2, p2, c2 = _naive_hc_pre(x4, prov.layer(0)["hc_attn_fn"], prov.layer(0)["hc_attn_scale"],
                               prov.layer(0)["hc_attn_base"], cfg)
    d_pre = max(diff(y1, y2), diff(p1, p2), diff(c1, c2))
    out = torch.randn(1, 5, cfg.hidden)
    h1 = hc_post(out, x4, p1, c1)
    h2 = _naive_hc_post(out, x4, p1, c1)
    d_post = diff(h1, h2)
    ok = row_err < 1e-3 and col_err < 1e-3 and d_pre < 1e-6 and d_post < 1e-6
    check("sinkhorn 双随机性 + hc_pre/hc_post vs 朴素", ok,
          f"row={row_err:.2e} col={col_err:.2e} pre={d_pre:.2e} post={d_post:.2e}")

    # 5) Compressor（ratio=4 overlap）批式池化 vs 手工分组 softmax
    w1_ = prov.layer(1)
    xc = torch.randn(1, 11, cfg.hidden)
    st = {"ckv": torch.zeros(1, 8, 32), "cscore": torch.full((1, 8, 32), float("-inf"))}
    cos = rope_cmp.cos[0:8:4].view(1, 2, -1)
    sin = rope_cmp.sin[0:8:4].view(1, 2, -1)
    mine = compressor_prefill(xc, w1_["cmp"], 4, cfg.head_dim, cfg.qk_rope_head_dim,
                              cos, sin, cfg.rms_eps, st)
    ref = _naive_compressor_prefill(xc, w1_["cmp"], 4, cfg.head_dim, cfg.qk_rope_head_dim,
                                    rope_cmp, cfg.rms_eps)
    d = diff(mine, ref)
    check("Compressor 分组 softmax 池化（overlap 8窗4步）vs 朴素", d < 1e-6, f"diff={d:.2e}")

    # 6) ratio=0 层 T≤win：与朴素因果全注意力严格一致
    w0 = prov.layer(0)
    xa = torch.randn(1, 6, cfg.hidden)
    sta = {"kv": torch.zeros(1, 8, cfg.head_dim),
           "win_pos": torch.full((1, 8), -1, dtype=torch.long)}
    mine = attn_prefill(xa, w0, sta, cfg, rope_base, 0)
    ref = _naive_attn(xa, w0, cfg, rope_base)
    d = diff(mine, ref)
    check("ratio=0 注意力（滑窗+sink+反旋转）vs 朴素全注意力", d < 1e-6, f"diff={d:.2e}")

    # 7) gate：sqrtsoftplus / noaux_tc（bias 只选择不进权重）/ tid2eid
    xg = torch.randn(5, cfg.hidden)
    ids_g = torch.randint(0, cfg.vocab, (5,))
    w2_ = prov.layer(2)
    ww, ii = gate_route(xg, w2_, cfg, ids_g)
    scores_ref = torch.log1p(torch.exp(xg @ w2_["gate"].t())).sqrt()
    ii_ref = (scores_ref + w2_["gate_bias"]).topk(cfg.top_k, dim=-1).indices
    ww_ref = scores_ref.gather(1, ii_ref)
    ww_ref = ww_ref / ww_ref.sum(-1, keepdim=True) * cfg.routed_scaling
    d_i = 0 if torch.equal(ii, ii_ref) else 1
    d_w = diff(ww, ww_ref)
    # bias 翻转选择：把低分专家 bias 拉到 +100，应被选中但权重仍取未加 bias 的原始分数
    w_mod = dict(w2_)
    bias_mod = w2_["gate_bias"].clone()
    low_e = int(scores_ref[0].argmin())
    bias_mod[low_e] += 100.0
    w_mod["gate_bias"] = bias_mod
    ww2, ii2 = gate_route(xg, w_mod, cfg, ids_g)
    sel_flip = low_e in ii2[0].tolist()
    exp_w = scores_ref[0].gather(0, ii2[0])
    exp_w = exp_w / exp_w.sum() * cfg.routed_scaling
    w_small = diff(ww2[0], exp_w) < 1e-6
    # tid2eid 层：选择完全由 token id 静态决定
    ww3, ii3 = gate_route(xg, w0, cfg, ids_g)
    tid_ok = torch.equal(ii3, w0["tid2eid"][ids_g])
    ww3_ref = scores_ref0 = None
    sc0 = torch.log1p(torch.exp(xg @ w0["gate"].t())).sqrt()
    ww3_ref = sc0.gather(1, ii3)
    ww3_ref = ww3_ref / ww3_ref.sum(-1, keepdim=True) * cfg.routed_scaling
    d3 = diff(ww3, ww3_ref)
    ok = d_i == 0 and d_w < 1e-6 and sel_flip and w_small and tid_ok and d3 < 1e-6
    check("gate sqrtsoftplus/noaux_tc(bias仅选择)/tid2eid", ok,
          f"idx={d_i} w={d_w:.2e} flip={sel_flip} tid={tid_ok} w3={d3:.2e}")

    # 8) 专家 clamp：up 截 ±10、gate 只截上界
    D8, I8 = 8, 16
    w1c = torch.zeros(I8, D8); w3c = torch.zeros(I8, D8)
    w1c[0, 0] = 1.0; w1c[1, 0] = -1.0
    w3c[0, 0] = 1.0; w3c[1, 0] = -1.0
    w2c = torch.zeros(D8, I8); w2c[0, 0] = 1.0; w2c[1, 1] = 1.0
    xin = torch.tensor([[30.0] + [0.0] * 7])
    out = expert_mlp(xin, w1c, w3c, w2c, 10.0)
    exp0 = float(F.silu(torch.tensor(10.0)) * 10.0)     # gate=30→截10, up=30→截10
    exp1 = float(F.silu(torch.tensor(-30.0)) * -10.0)   # gate=-30 不截下界, up=-30→截-10
    d = max(abs(float(out[0, 0]) - exp0), abs(float(out[0, 1]) - exp1))
    check("专家 SwiGLU clamp（up±10 / gate 仅上界）", d < 1e-6, f"diff={d:.2e}")

    # 9) 整网 prefill vs 朴素参考
    ids = torch.randint(0, cfg.vocab, (1, 12))
    model = DSV4Model(cfg, prov, max_seq=32)
    logits = model.prefill(ids)
    ref = _naive_net(ids, cfg, prov, [rope_base, rope_cmp, rope_base])
    d = diff(logits, ref)
    check("整网前向（3层含 ratio=4 压缩层/tid2eid 层）vs 朴素", d < 1e-5, f"max_diff={d:.2e}")

    # 10) prefill 批式 vs decode 增量逐步（环形缓存 + 增量 Compressor）
    model2 = DSV4Model(cfg, _TinyProvider(cfg, seed=0), max_seq=32)
    model2.prefill(ids[:, :8])
    cols = []
    for p in range(8, 12):
        cols.append(model2.decode(ids[:, p], p))
    logits_d = torch.stack(cols, dim=1)                 # [1,4,V]
    d = diff(logits_d, logits[:, 8:12])
    check("decode 增量（Compressor/环形窗）与 prefill 一致", d < 1e-5, f"max_diff={d:.2e}")

    # 11) FP4 反量化往返（低半字节在前、e2m1 LUT、2^(b-127)、32 块）
    from .fp4io import dequant_fp4_check
    lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.])
    w = torch.randn(8, 64) * 2
    wb = w.view(8, 2, 32)
    amax = wb.abs().amax(-1)
    exp = torch.where(amax > 0, torch.ceil(torch.log2(amax / 6.0)), torch.zeros_like(amax))
    s = 2.0 ** exp
    qv = (wb / s[..., None]).clamp(-6, 6)
    idx = (qv[..., None] - lut).abs().argmin(-1).view(8, 64)
    packed = (idx[:, 0::2] | (idx[:, 1::2] << 4)).to(torch.uint8)
    sbyte = (exp + 127).to(torch.uint8)
    dq = dequant_fp4(packed, sbyte, 8, 64)
    ref = lut[idx] * s.repeat_interleave(32, dim=1)
    d = diff(dq, ref)
    rmin, rmax = dequant_fp4_check(packed, sbyte, 8, 64)
    check("FP4 反量化往返（nibble 顺序/LUT/ue8m0）", d == 0.0 and 3.0 <= rmin and rmax <= 6.0,
          f"diff={d:.2e} amax/scale=[{rmin:.2f},{rmax:.2f}]")

    # 12) FP8 反量化往返（e4m3 × 2^(scale-127)，128×128 块，非整除形状）
    w = torch.randn(130, 300) * 3
    e = torch.randint(-3, 4, (2, 3))
    sb = (2.0 ** e.float()).repeat_interleave(128, 0).repeat_interleave(128, 1)[:130, :300]
    q8 = (w / sb).clamp(-448, 448).to(torch.float8_e4m3fn)
    s8 = (e + 127).to(torch.uint8)
    dq = dequant_fp8(q8.view(torch.uint8), s8)
    ref = q8.float() * sb
    d = diff(dq, ref)
    check("FP8 反量化（e4m3×2^(b-127)，128×128 块）", d == 0.0, f"diff={d:.2e}")

    # 13) SafeFile / DSV4Checkpoint 读取（手写 safetensors 文件往返）
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        t_bf = torch.randn(3, 4).bfloat16()
        t_i64 = torch.randint(0, 5, (7, 2), dtype=torch.int64)
        tensors = {
            "w.weight": ("F8_E4M3", list(q8.shape), q8.view(torch.uint8).numpy().tobytes()),
            "w.scale": ("F8_E8M0", list(s8.shape), s8.numpy().tobytes()),
            "t": ("BF16", [3, 4], t_bf.view(torch.int16).numpy().tobytes()),
            "ids": ("I64", [7, 2], t_i64.numpy().tobytes()),
        }
        fp = os.path.join(td, "model-00001-of-00001.safetensors")
        _write_safetensors(fp, tensors)
        sf = SafeFile(fp)
        d1 = 0 if torch.equal(sf.get_tensor("t"), t_bf) else 1
        d2 = 0 if torch.equal(sf.get_tensor("ids"), t_i64) else 1
        ck = DSV4Checkpoint(td)
        d3 = diff(ck.get_f32("w.weight"), ref)
        ok = d1 == 0 and d2 == 0 and d3 == 0.0
        check("SafeFile/Checkpoint 读取（BF16/I64/FP8 对）", ok,
              f"bf16={d1} i64={d2} fp8={d3:.2e}")

    print()
    if fails:
        print(f"自检未通过：{len(fails)} 项 —— {fails}")
        return 1
    print("全部自检通过。")
    print("文件结构：SafeFile/DSV4Checkpoint/dequant_fp8（加载） | "
          "rmsnorm/RopeCache/rope_apply/hc_*/compressor_*/attn_*/gate_route/expert_mlp/"
          "moe_forward/block_forward（前向） | DSV4Model（prefill+decode） | main（自检）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
