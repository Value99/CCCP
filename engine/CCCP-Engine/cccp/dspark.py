"""CCCP DSpark 块并行投机解码头（DeepSeek-V4-Flash-DSpark 自带的 mtp.* 三层）。

结构（已核对原始分片 model-00046/47/48-of-00048 的张量名与形状）：
  mtp.0/1/2 = 三个 DSpark 层（compress_ratio=0、无 Compressor，其余与主层同构：
  低秩 Q 注意力 + Hyper-Connections + 256 专家 MoE）；
  stage 0 另有 main_norm/main_proj [4096, 3·4096]——输入为主模型层 40/41/42 的
  hc 均值隐态拼接 [., 3·4096]；stage 2 另有 norm、hc_head_fn/base/scale、
  markov_head.markov_w1/w2 [129280, 256]（低秩 logits 偏置头）与
  confidence_head.proj（v1 未接）。embed/head 与主模型共享（官方 convert 跳过
  mtp.*emb* 与 mtp.*head.weight）。

草稿前向（官方 inference/model.py 的 forward_spec / DSparkAttention 纯 torch 复刻）：
  main_x = main_norm(main_proj(main_hidden))            # [1,1,D]
  草稿输入 = [t1, noise_token×4] 的 embedding 复制到 4 个 hc 通道 [1,5,4,D]
  每层 DSparkAttention：main_kv = wkv(main_x)（相位 start_pos）写入环形槽
  start_pos%128；5 个草稿位（相位 start_pos+1..+5）对「环内全部活槽 + 自身 5 位」
  做注意力（块并行：草稿位之间无因果掩码，noise 位不含真实未来信息，与官方一致；
  softmax 分母含 attn_sink、输出末 64 维反旋转，同主模型）；
  MoE 为 sqrtsoftplus top-6 + gate.bias 选择（与主模型层≥3 相同）；
  末层 hc_head → norm → 共享 lm_head 得 5 位 logits，再沿块顺序加 markov 偏置
  （logits[j] += markov_w2 @ markov_w1[prev_token]）贪心生出 5 个草稿。

KV 同步（草稿质量的关键，正确性由主模型贪心验证兜底）：
  DSpark 环只保存「已验证接受位置」的 main_kv。prefill 后由 prefill_kv 用
  main_hidden 全量建立（仅留最后 128 个，环形摆放）；每轮验证后由 update_kv
  写入接受前缀（最末接受位由下一次 draft 调用内部写入，幂等）。被拒草稿
  从不进入 DSpark 环，无需回滚。

权重来源：CCCP 产物目录的 dspark.safetensors（由 `python -m CCCP dspark-export`
自原始检查点导出，张量名/dtype 原样保留；产物自包含，不依赖原始模型目录）。
FP8 反量化值在 bf16 中精确可表，大矩阵按 bf16 驻留。routed 专家为 FP4 e2m1
打包 + ue8m0 缩放：FP4Weight 结构与 Int4Weight 同构（256 字节 LUT 一次取双
半字节，分块在线反量化 matmul），打包态经 LRU 驻内存（默认 1.5GB，
CCCP_DSPARK_GB 可调），命中失败读产物文件。
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict

import torch
import torch.nn.functional as F

from .dsv4 import SafeFile, dequant_fp8, rmsnorm, rope_apply, hc_pre, hc_post, hc_head, \
    gate_route, expert_mlp

from .kernels import VQWeight

# e2m1 全 16 值表（bit3 为符号位；与 CCCP/fp4io.py 一致）
_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

_FP4_GROUP = 32  # ue8m0 缩放粒度（每行每 32 元素一个指数字节）


def _make_fp4_lut() -> torch.Tensor:
    """256 字节 → (低半字节值, 高半字节值)（低半字节在前 = 偶数列先）。"""
    tab = torch.tensor(_E2M1, dtype=torch.float32)
    t = torch.arange(256)
    return torch.stack([tab[t & 15], tab[t >> 4]], 1)


_FP4_LUT = _make_fp4_lut()
_LUTS: dict = {"cpu": _FP4_LUT}


def _lut_on(device) -> torch.Tensor:
    key = str(device)
    lut = _LUTS.get(key)
    if lut is None:
        lut = _FP4_LUT.to(device)
        _LUTS[key] = lut
    return lut


class FP4Weight:
    """FP4 e2m1 打包权重（I8 [R, C//2] 低半字节在前 + ue8m0 [R, C//32]）。
    matmul 按行块在线反量化到 f32 后 torch.mm（与 Int4Weight 同构的驻留方案）。"""

    __slots__ = ("q", "s", "cols")

    def __init__(self, q: torch.Tensor, s: torch.Tensor, cols: int):
        self.q = q          # u8 [R, C//2]
        self.s = s          # u8 [R, C//32]（ue8m0 指数字节）
        self.cols = cols

    @property
    def shape(self) -> torch.Size:
        return torch.Size([self.q.shape[0], self.cols])

    @property
    def nbytes(self) -> int:
        return self.q.numel() + self.s.numel()

    def dequant_rows(self, r0: int, r1: int, device) -> torch.Tensor:
        q = self.q[r0:r1].to(device)
        s = self.s[r0:r1].to(device)
        w = _lut_on(device)[q.long()].view(r1 - r0, self.cols)
        sp = torch.pow(2.0, s.float() - 127.0)
        w.view(r1 - r0, self.cols // _FP4_GROUP, _FP4_GROUP).mul_(sp.unsqueeze(-1))
        return w

    def matmul_T(self, x: torch.Tensor, chunk: int | None = None) -> torch.Tensor:
        """y = x @ W.T。x: [T, C] f32 → [T, R] f32，逐行块反量化（块 ≤64MB，自适应）。"""
        R = self.q.shape[0]
        if chunk is None:
            chunk = max(512, min(R, (64 * 2**20) // max(self.cols * 4, 1)))
        out = torch.empty(x.shape[0], R, dtype=torch.float32, device=x.device)
        for r0 in range(0, R, chunk):
            r1 = min(r0 + chunk, R)
            out[:, r0:r1] = x @ self.dequant_rows(r0, r1, x.device).t()
        return out


class DSparkStore:
    """产物目录 dspark.safetensors 的读取（产物自包含：只认 cccp.json 的
    dspark_file 指引，不依赖原始模型目录）。张量名与原始检查点一致（mtp.*），
    FP8 伴生 .scale 由导出器原样保留。产物由 `python -m CCCP dspark-export` 生成。"""

    def __init__(self, model_dir: str):
        man_path = os.path.join(model_dir, "cccp.json")
        with open(man_path, "r", encoding="utf-8") as f:
            man = json.load(f)
        fn = man.get("dspark_file")
        if not fn:
            raise FileNotFoundError(
                f"{model_dir}/cccp.json 缺少 dspark_file：请先运行 "
                f"python -m CCCP dspark-export --src <原始模型目录> --out {model_dir}")
        self.man = man
        self.sf = SafeFile(os.path.join(model_dir, fn))
        self.keys = set(self.sf.keys())

    def hyper(self) -> dict:
        """DSpark 超参（block_size/noise_id/targets），manifest 缺省取官方发布值。"""
        d = self.man.get("dspark", {})
        return {"block_size": int(d.get("block_size", 5)),
                "noise_id": int(d.get("noise_id", 128799)),
                "targets": tuple(d.get("targets", (40, 41, 42)))}

    def has_scale(self, name: str) -> bool:
        # 伴生 scale 命名：X.weight ↔ X.scale（官方 convert 只改名不取倒数）
        sname = name[:-len("weight")] + "scale" if name.endswith("weight") else name + ".scale"
        return sname in self.keys

    def get_raw(self, name: str) -> torch.Tensor:
        return self.sf.get_tensor(name)

    def get_f32(self, name: str) -> torch.Tensor:
        """带伴生 .scale 的走块级 FP8 反量化；其余按存储 dtype 转 f32。"""
        sname = name[:-len("weight")] + "scale" if name.endswith("weight") else name + ".scale"
        if sname in self.keys:
            return dequant_fp8(self.get_raw(name), self.get_raw(sname))
        return self.get_raw(name).float()

    # ---- VQ 产物（dspark-vq.safetensors）----
    def is_vq(self) -> bool:
        """产物是否为 VQ 量化版（含 stage 码本键）。"""
        return "s0.cb.gu" in self.keys

    def cb(self, stage: int) -> tuple[torch.Tensor, torch.Tensor]:
        """stage 共享码本 (cb_gu, cb_dn) f32 [K, dim]。"""
        return (self.get_raw(f"s{stage}.cb.gu").float(),
                self.get_raw(f"s{stage}.cb.dn").float())


def _linb(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """bf16 大矩阵的线性层（e4m3×2^k 在 bf16 中精确；激活仿官方按 bf16 进 GEMM，
    输出回 f32 保持后续归一化/softmax 精度）。"""
    return (x.bfloat16() @ w.t()).float()


class DSparkHead:
    """DeepSeek-V4-Flash-DSpark 的 DSpark 三层草稿头（provider 复用 DSV4CCCPModel：
    embed/head/rope/设备来自主模型；mtp.* 权重自原始分片加载）。"""

    N_STAGES = 3
    WIN = 128

    def __init__(self, model, src: str | None = None):
        self.m = model
        self.device = model.device
        self.cfg = model._cfg_obj()
        self.store = DSparkStore(model.store.root)   # 只从产物目录加载（dspark_file）
        hp = self.store.hyper()
        self.block_size = hp["block_size"]
        self.noise_id = hp["noise_id"]
        self.targets = hp["targets"]
        self.rope = model.rope_base          # ratio=0 → theta=10000、无 YaRN
        self._stages: dict[int, dict] = {}
        self._experts: OrderedDict[tuple[int, int], tuple[FP4Weight, FP4Weight]] = OrderedDict()
        self._ebytes = 0
        self._ebudget = int(float(os.environ.get("CCCP_DSPARK_GB", "1.5")) * 2**30)
        self.ehits = 0
        self.emiss = 0
        self.rings: list[torch.Tensor] | None = None

    # ---- 权重 ----
    def stage_w(self, s: int) -> dict:
        w = self._stages.get(s)
        if w is not None:
            return w
        p = f"mtp.{s}"
        st = self.store

        def bf16(name: str) -> torch.Tensor:
            # FP8(e4m3×2^k) 反量化值在 bf16 中精确可表；BF16 原样；F32 转 f32 另存
            t = st.get_f32(name)
            return t.bfloat16().to(self.device)

        def f32(name: str) -> torch.Tensor:
            return st.get_f32(name).to(self.device)

        w = {
            "wq_a": bf16(f"{p}.attn.wq_a.weight"),
            "q_norm": f32(f"{p}.attn.q_norm.weight"),
            "wq_b": bf16(f"{p}.attn.wq_b.weight"),
            "wkv": bf16(f"{p}.attn.wkv.weight"),
            "kv_norm": f32(f"{p}.attn.kv_norm.weight"),
            "attn_sink": f32(f"{p}.attn.attn_sink"),
            "wo_a": bf16(f"{p}.attn.wo_a.weight"),
            "wo_b": bf16(f"{p}.attn.wo_b.weight"),
            "attn_norm": f32(f"{p}.attn_norm.weight"),
            "ffn_norm": f32(f"{p}.ffn_norm.weight"),
            "gate": f32(f"{p}.ffn.gate.weight"),
            "gate_bias": f32(f"{p}.ffn.gate.bias"),
            "sh_w1": f32(f"{p}.ffn.shared_experts.w1.weight"),
            "sh_w3": f32(f"{p}.ffn.shared_experts.w3.weight"),
            "sh_w2": f32(f"{p}.ffn.shared_experts.w2.weight"),
            "hc_attn_fn": f32(f"{p}.hc_attn_fn"),
            "hc_attn_base": f32(f"{p}.hc_attn_base"),
            "hc_attn_scale": f32(f"{p}.hc_attn_scale"),
            "hc_ffn_fn": f32(f"{p}.hc_ffn_fn"),
            "hc_ffn_base": f32(f"{p}.hc_ffn_base"),
            "hc_ffn_scale": f32(f"{p}.hc_ffn_scale"),
        }
        if s == 0:
            w["main_proj"] = bf16(f"{p}.main_proj.weight")
            w["main_norm"] = f32(f"{p}.main_norm.weight")
        if s == self.N_STAGES - 1:
            w["norm"] = f32(f"{p}.norm.weight")
            w["hc_head_fn"] = f32(f"{p}.hc_head_fn")
            w["hc_head_base"] = f32(f"{p}.hc_head_base")
            w["hc_head_scale"] = f32(f"{p}.hc_head_scale")
            w["markov_w1"] = st.get_raw(f"{p}.markov_head.markov_w1.weight").to(self.device)
            w["markov_w2"] = st.get_raw(f"{p}.markov_head.markov_w2.weight").to(self.device)
        self._stages[s] = w
        return w

    def _cbs(self, stage: int) -> tuple[torch.Tensor, torch.Tensor]:
        """stage 共享码本的设备副本（懒加载缓存）。"""
        cache = getattr(self, "_cb_cache", None)
        if cache is None:
            cache = self._cb_cache = {}
        cb = cache.get(stage)
        if cb is None:
            g, d = self.store.cb(stage)
            cb = (g.to(self.device), d.to(self.device))
            cache[stage] = cb
        return cb

    def _expert(self, stage: int, eid: int) -> tuple:
        """routed 专家 (gu, dn)。VQ 产物：VQWeight（u8 索引 + stage 共享码本，
        LUT 免还原矩阵乘，GPU 驻留 LRU）；FP4 产物：FP4Weight（打包态驻内存 LRU）。"""
        key = (stage, eid)
        ent = self._experts.get(key)
        if ent is not None:
            self.ehits += 1
            self._experts.move_to_end(key)
            return ent
        self.emiss += 1
        st = self.store
        if st.is_vq():
            cb_gu, cb_dn = self._cbs(stage)
            gu = VQWeight(st.get_raw(f"s{stage}.e{eid}.gu").to(self.device),
                          cb_gu, self.cfg.hidden)
            dn = VQWeight(st.get_raw(f"s{stage}.e{eid}.dn").to(self.device),
                          cb_dn, self.cfg.moe_inter)
        else:
            p = f"mtp.{stage}.ffn.experts.{eid}"
            q1 = st.get_raw(f"{p}.w1.weight").view(torch.uint8)
            s1 = st.get_raw(f"{p}.w1.scale").view(torch.uint8)
            q3 = st.get_raw(f"{p}.w3.weight").view(torch.uint8)
            s3 = st.get_raw(f"{p}.w3.scale").view(torch.uint8)
            q2 = st.get_raw(f"{p}.w2.weight").view(torch.uint8)
            s2 = st.get_raw(f"{p}.w2.scale").view(torch.uint8)
            mi = self.cfg.moe_inter
            gu = FP4Weight(torch.cat([q1, q3], 0), torch.cat([s1, s3], 0), self.cfg.hidden)
            dn = FP4Weight(q2, s2, mi)
        ent = (gu, dn)
        nb = gu.nbytes + dn.nbytes
        while self._ebytes + nb > self._ebudget and self._experts:
            _, (g, d) = self._experts.popitem(last=False)
            self._ebytes -= g.nbytes + d.nbytes
        self._experts[key] = ent
        self._ebytes += nb
        return ent

    # ---- KV 环（只存已接受位置的 main_kv） ----
    def reset(self) -> None:
        self.rings = None

    def _alloc(self) -> None:
        hd = self.cfg.head_dim
        self.rings = [torch.zeros(1, self.WIN, hd, device=self.device)
                      for _ in range(self.N_STAGES)]

    def _main_x(self, mh: torch.Tensor) -> torch.Tensor:
        """main_hidden [., 3D] → main_norm(main_proj(mh)) [., D]（3 层共用）。"""
        w0 = self.stage_w(0)
        return rmsnorm(_linb(mh, w0["main_proj"]), w0["main_norm"], self.cfg.rms_eps)

    def _kv_write(self, main_x: torch.Tensor, pos0: int) -> None:
        """把 positions pos0..pos0+T-1 的 main_kv 写入各层环（槽 = pos % 128）。"""
        cfg = self.cfg
        hd, rd = cfg.head_dim, cfg.qk_rope_head_dim
        T = main_x.shape[0]
        cos = self.rope.cos[pos0:pos0 + T]
        sin = self.rope.sin[pos0:pos0 + T]
        slots = (torch.arange(pos0, pos0 + T, device=self.device) % self.WIN)
        for s in range(self.N_STAGES):
            w = self.stage_w(s)
            kv = rmsnorm(_linb(main_x, w["wkv"]), w["kv_norm"], cfg.rms_eps)  # [T, hd]
            kv[:, hd - rd:] = rope_apply(kv[:, hd - rd:], cos.view(T, -1), sin.view(T, -1))
            self.rings[s][:, slots] = kv

    @torch.no_grad()
    def prefill_kv(self, mh: torch.Tensor) -> None:
        """用 prompt 全部位置的 main_hidden [T, 3D] 建环（只留最后 128 个，环形摆放）。"""
        if self.rings is None:
            self._alloc()
        T = mh.shape[0]
        n = min(T, self.WIN)
        main_x = self._main_x(mh[T - n:])  # 只有最后 128 个位置会留在环内，少算proj
        self._kv_write(main_x, T - n)

    @torch.no_grad()
    def update_kv(self, mh_rows: torch.Tensor, pos0: int) -> None:
        """写入一段已接受位置的 main_kv。mh_rows [n, 3D] ↔ positions pos0..pos0+n-1。"""
        if mh_rows.shape[0] == 0:
            return
        self._kv_write(self._main_x(mh_rows), pos0)

    # ---- 草稿前向 ----
    def _qkv(self, x: torch.Tensor, w: dict, pos0: int, T: int):
        """低秩 Q + MQA kv（与 CCCP.dsv4._qkv 同数学，大矩阵走 bf16 GEMM）。"""
        cfg = self.cfg
        H, hd, rd = cfg.n_heads, cfg.head_dim, cfg.qk_rope_head_dim
        qr = rmsnorm(_linb(x, w["wq_a"]), w["q_norm"], cfg.rms_eps)
        q = _linb(qr, w["wq_b"]).view(1, T, H, hd)
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + cfg.rms_eps)  # 逐头无权重 RMS
        cos = self.rope.cos[pos0:pos0 + T]
        sin = self.rope.sin[pos0:pos0 + T]
        q[..., hd - rd:] = rope_apply(q[..., hd - rd:], cos.view(1, T, 1, -1),
                                      sin.view(1, T, 1, -1))
        kv = rmsnorm(_linb(x, w["wkv"]), w["kv_norm"], cfg.rms_eps)
        kv[..., hd - rd:] = rope_apply(kv[..., hd - rd:], cos.view(1, T, -1),
                                       sin.view(1, T, -1))
        return q, kv

    def _o_proj(self, o: torch.Tensor, w: dict) -> torch.Tensor:
        """分组 LoRA O（bf16 GEMM；o [1,T,H*hd] → [1,T,D]）。"""
        cfg = self.cfg
        G = cfg.o_groups
        o = o.reshape(1, -1, G, cfg.n_heads * cfg.head_dim // G).bfloat16()
        wo_a = w["wo_a"].view(G, cfg.o_lora_rank, -1)
        o = torch.einsum("btgd,grd->btgr", o, wo_a)
        return _linb(o.flatten(2), w["wo_b"])

    def _attn(self, x: torch.Tensor, w: dict, ring: torch.Tensor,
              main_x: torch.Tensor, start_pos: int) -> torch.Tensor:
        """DSparkAttention decode：main_kv 入环（start_pos 槽），5 个草稿位对
        「环全部活槽 + 自身 5 位」注意力（无草稿间掩码；sink 在分母）。"""
        cfg = self.cfg
        H, hd, rd = cfg.n_heads, cfg.head_dim, cfg.qk_rope_head_dim
        T = self.block_size
        # main_kv（相位 start_pos）写入环槽 start_pos % win（幂等：同隐态同值）
        mkv = rmsnorm(_linb(main_x, w["wkv"]), w["kv_norm"], cfg.rms_eps)  # [1, 1, hd]
        cos1 = self.rope.cos[start_pos:start_pos + 1]
        sin1 = self.rope.sin[start_pos:start_pos + 1]
        mkv[..., hd - rd:] = rope_apply(mkv[..., hd - rd:], cos1.view(1, 1, -1),
                                        sin1.view(1, 1, -1))
        ring[:, start_pos % self.WIN] = mkv[0, 0]
        # 草稿 q/kv（相位 start_pos+1..+T）
        q, dkv = self._qkv(x, w, start_pos + 1, T)
        n = min(self.WIN, start_pos + 1)   # 活槽数（prefill 后槽位即位置；满环后全活）
        keys = torch.cat([ring[:, :n], dkv], dim=1)                       # [1, n+T, hd]
        scores = torch.einsum("bthd,bsd->bhts", q * (hd ** -0.5), keys)
        m = scores.amax(dim=-1)                                   # max 不含 sink
        e = (scores - m.unsqueeze(-1)).exp()
        denom = e.sum(dim=-1) + (w["attn_sink"].view(1, -1, 1) - m).exp()
        o = torch.einsum("bhts,bsd->bthd", e, keys) / denom.transpose(1, 2).unsqueeze(-1)
        cosT = self.rope.cos[start_pos + 1:start_pos + 1 + T].view(1, T, 1, -1)
        sinT = self.rope.sin[start_pos + 1:start_pos + 1 + T].view(1, T, 1, -1)
        o[..., hd - rd:] = rope_apply(o[..., hd - rd:], cosT, sinT, inverse=True)  # 输出反旋转
        return self._o_proj(o.flatten(2), w)

    def _moe(self, x: torch.Tensor, w: dict, stage: int, ids: torch.Tensor) -> torch.Tensor:
        """sqrtsoftplus top-6 路由 + FP4 专家 + 共享专家（数学同主模型 _moe；
        argsort+searchsorted 分派，避免逐专家 nonzero 的隐式同步）。"""
        cfg = self.cfg
        B, T, D = x.shape
        xf = x.reshape(B * T, D).float()
        gw = {"gate": w["gate"], "gate_bias": w["gate_bias"]}
        weights, indices = gate_route(xf, gw, cfg, ids.reshape(-1))
        y = torch.zeros_like(xf)
        limit = cfg.swiglu_limit
        mi = cfg.moe_inter
        K = indices.shape[1]
        flat = indices.reshape(-1)
        order = torch.argsort(flat)
        bounds = torch.searchsorted(flat[order],
                                    torch.arange(cfg.n_experts + 1, device=flat.device))
        bl = bounds.tolist()
        rows_all = torch.div(order, K, rounding_mode="floor")
        cols_all = order % K
        for e in range(cfg.n_experts):
            if bl[e + 1] == bl[e]:
                continue
            sl = slice(bl[e], bl[e + 1])
            rows, cols = rows_all[sl], cols_all[sl]
            gu, dn = self._expert(stage, e)
            h = gu.matmul_T(xf[rows])
            g, u = h[:, :mi], h[:, mi:]
            if limit:
                u = u.clamp(-limit, limit)
                g = g.clamp(max=limit)
            y[rows] += dn.matmul_T(F.silu(g) * u) \
                * weights[rows, cols, None]
        y += expert_mlp(xf, w["sh_w1"], w["sh_w3"], w["sh_w2"], limit)
        return y.view(B, T, D)

    def _block(self, h: torch.Tensor, stage: int, draft_ids: torch.Tensor,
               main_x: torch.Tensor, start_pos: int) -> torch.Tensor:
        cfg = self.cfg
        w = self.stage_w(stage)
        residual = h
        y, post, comb = hc_pre(h, w["hc_attn_fn"], w["hc_attn_scale"], w["hc_attn_base"], cfg)
        y = rmsnorm(y, w["attn_norm"], cfg.rms_eps)
        a = self._attn(y, w, self.rings[stage], main_x, start_pos)
        h = hc_post(a, residual, post, comb)
        residual = h
        y, post, comb = hc_pre(h, w["hc_ffn_fn"], w["hc_ffn_scale"], w["hc_ffn_base"], cfg)
        y = rmsnorm(y, w["ffn_norm"], cfg.rms_eps)
        f = self._moe(y, w, stage, draft_ids)
        return hc_post(f, residual, post, comb)

    @torch.no_grad()
    def draft(self, t1: int, mh_last: torch.Tensor, start_pos: int) -> list[int]:
        """一次 DSpark 前向产出 block_size 个草稿 token。

        t1: 主模型在当前位置的贪心 token（position start_pos+1 的输入）；
        mh_last: 主模型在 position start_pos 的 main_hidden [3D] 或 [1, 3D]；
        start_pos: 最末已接受位置（>0，prefill 后 = prompt_len-1）。
        """
        cfg = self.cfg
        T = self.block_size
        main_x = self._main_x(mh_last.view(1, -1)).unsqueeze(0)        # [1, 1, D]
        ids = torch.full((1, T), self.noise_id, dtype=torch.long, device=self.device)
        ids[0, 0] = t1
        h = self.m._embed(ids).unsqueeze(2).repeat(1, 1, cfg.hc_mult, 1)
        for s in range(self.N_STAGES):
            h = self._block(h, s, ids, main_x, start_pos)
        wl = self.stage_w(self.N_STAGES - 1)
        x = hc_head(h, wl["hc_head_fn"], wl["hc_head_scale"], wl["hc_head_base"], cfg)
        x = rmsnorm(x, wl["norm"], cfg.rms_eps)
        logits = self.m.logits_of(x[0])                # [T, V] f32（共享 lm_head）
        mw1, mw2 = wl["markov_w1"], wl["markov_w2"]    # [V, 256] bf16
        prev = torch.tensor([t1], dtype=torch.long, device=self.device)
        outs = []
        for j in range(T):
            emb = mw1[prev]                            # [1, 256]
            logits[j] += (emb @ mw2.t()).float()[0]    # markov 低秩偏置
            prev = logits[j].argmax().view(1)
            outs.append(prev)
        return [int(t) for t in outs]
