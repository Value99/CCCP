"""CCCP MTP 投机解码：GLM-5.2 自带 MTP 层（layer 78）的前向与草稿生成。

MTP 前向规格（DeepSeek-V3 式，张量形状已在 FP8 检查点核实）：
    x = eh_proj(cat([hnorm(h_main), enorm(embed(t_next))], -1))   # [., 6144]
    h78 = decoder_layer_78(x)          # 完整 MLA 注意力 + MoE（256 专家，全 v 档）
    logits = lm_head(shared_head.norm(h78))
草稿链式：把上一步 h78 与草稿 token 的嵌入再喂回同一模块。
MTP 层有独立 KV cache（第 78 层自己的 K/V），随主模型 reset 同步清空。

投机解码（贪心验收，输出与纯贪心**逐 token 一致**，零质量风险）：
  1. 主模型一次前向得 next token t1（真值）与主 hidden；
  2. MTP 链式起草 k 个草稿 d1..dk；
  3. 主模型一次前向验证 [t1, d1..dk]：逐位 argmax 比对，接受最长连续匹配前缀，
     首个不匹配位置的 argmax 作为"奖励 token"；
  4. 每轮流式成本≈1 次主前向，产出 1+接受数 个 token。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .kernels import Int4Weight, VQWeight, rmsnorm


def _lin(x: torch.Tensor, w) -> torch.Tensor:
    if isinstance(w, Int4Weight):
        return w.matmul_T(x)
    return x.float() @ w.t()


class MTPHead:
    """GLM-5.2 的 MTP 层（layer 78）推理头。"""

    LAYER = 78

    def __init__(self, model):
        self.m = model
        self.store = model.store
        assert self.store.has_mtp(), "模型目录缺 MTP 附件（mtp.safetensors / experts.L78.safetensors）"
        self._w: dict[str, object] = {}
        self.kv: tuple[torch.Tensor, torch.Tensor] | None = None

    def reset(self) -> None:
        self.kv = None

    def w(self, name: str):
        wt = self._w.get(name)
        if wt is None:
            wt = self.store.get_mtp(name)
            dev = self.m.device
            if dev.type != "cpu":
                if isinstance(wt, Int4Weight):
                    wt = Int4Weight(wt.q.to(dev), wt.s.to(dev), wt.cols, wt.gs)
                else:
                    wt = wt.to(dev)
            self._w[name] = wt
        return wt

    # ---- 第 78 层前向（数学与 model.py 的注意力/MoE 一致） ----
    def _attention(self, x: torch.Tensor, pos0: int) -> torch.Tensor:
        c = self.m.cfg
        H = c["n_heads"]
        T = x.shape[0]
        q_resid = rmsnorm(_lin(x, self.w("attn.q_a")), self.w("attn.q_a_norm"), 1e-6)
        q = _lin(q_resid, self.w("attn.q_b")).view(T, H, c["qk_head_dim"]).transpose(0, 1)
        q_nope, q_rot = q.split([c["qk_nope_head_dim"], c["qk_rope_head_dim"]], dim=-1)
        kv = _lin(x, self.w("attn.kv_a"))
        k_pass, k_rot = kv.split([c["kv_lora_rank"], c["qk_rope_head_dim"]], dim=-1)
        k_pass = rmsnorm(k_pass, self.w("attn.kv_a_norm"), 1e-6)
        k_pass = _lin(k_pass, self.w("attn.kv_b"))
        k_pass = k_pass.view(T, H, c["qk_nope_head_dim"] + c["v_head_dim"]).transpose(0, 1)
        k_nope, v = k_pass.split([c["qk_nope_head_dim"], c["v_head_dim"]], dim=-1)
        q_rot, k_rot = self.m.rope.apply(q_rot, k_rot.view(1, T, c["qk_rope_head_dim"]), pos0)
        k_rot = k_rot.expand(H, T, c["qk_rope_head_dim"])
        q_f = torch.cat([q_nope, q_rot], dim=-1)
        k_f = torch.cat([k_nope, k_rot], dim=-1)
        if self.kv is not None:
            k_f = torch.cat([self.kv[0].float(), k_f], dim=1)
            v = torch.cat([self.kv[1].float(), v], dim=1)
        self.kv = (k_f.half(), v.half())
        scores = (q_f.float() @ k_f.float().transpose(1, 2)) / math.sqrt(c["qk_head_dim"])
        S = scores.shape[-1]
        if T > 1:
            kpos = torch.arange(S, device=x.device)
            qpos = torch.arange(pos0, pos0 + T, device=x.device)
            scores = scores.masked_fill((kpos[None, :] > qpos[:, None])[None], float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ v.float()).transpose(0, 1).reshape(T, H * c["v_head_dim"])
        return _lin(out, self.w("attn.o"))

    def _moe(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 0:
            # 空批(续算路径):0 行 MoE 恒等返回(同 model.py._moe 守卫)。
            return x
        c = self.m.cfg
        logits = _lin(x, self.w("router")).float()
        prob = logits.sigmoid()
        choice = prob + self.w("router_bias").float()
        mask = self.m._mask(self.LAYER)
        choice = choice.masked_fill(~mask, float("-inf"))
        idx = choice.topk(c["top_k"], dim=-1).indices
        w = prob.gather(1, idx)
        w = w / (w.sum(-1, keepdim=True) + 1e-20) * c["routed_scaling"]
        activation = self.m.operator_config.expert_activation
        activation_beta = float(c.get("situ_beta", 4.0))
        activation_linear_beta = c.get("situ_linear_beta")
        limit = float(c.get("swiglu_limit", 0.0))

        if (
            x.shape[0] > 1
            and getattr(self.m.pool, "prefill_rows_supported", False)
            and callable(getattr(self.m.pool, "run_rows", None))
        ):
            self.m.pool.prefetch([
                (self.LAYER, int(expert))
                for expert in torch.unique(idx).detach().cpu().tolist()
            ])
            routed = self.m.pool.run_rows(
                self.LAYER,
                x,
                idx,
                w,
                activation=activation,
                activation_beta=activation_beta,
                activation_linear_beta=activation_linear_beta,
                limit=limit,
            )
        elif x.shape[0] == 1:
            expert_ids = [int(expert) for expert in idx[0].tolist()]
            selected = self.m.pool.get_many([
                (self.LAYER, expert) for expert in expert_ids
            ])
            experts = [
                selected[(self.LAYER, expert)] for expert in expert_ids
            ]
            if not all(
                isinstance(gu, VQWeight) and isinstance(dn, VQWeight)
                for gu, dn in experts
            ):
                # 长生成后池内该专家可能是紧凑/展开形态(非 VQWeight
                # 包装):复用 run_rows 单行路径而非硬失败——同进程二次
                # generate 续算路径的实测触发点(第二十九轮)。
                run_rows = getattr(self.m.pool, "run_rows", None)
                if not callable(run_rows):
                    raise RuntimeError(
                        "MTP fused packed top-k decode requires VQ experts; "
                        "the legacy single-token expert projection was "
                        "deleted"
                    )
                routed = run_rows(
                    self.LAYER,
                    x,
                    idx,
                    w,
                    activation=activation,
                    activation_beta=activation_beta,
                    activation_linear_beta=activation_linear_beta,
                    limit=limit,
                )
                shared = _lin(
                    F.silu(_lin(x, self.w("shared_gate")))
                    * _lin(x, self.w("shared_up")),
                    self.w("shared_down"),
                )
                return routed.to(shared.dtype) + shared
            from .grouped import moe_mlp_grouped_mixed

            routed = moe_mlp_grouped_mixed(
                x,
                experts,
                w[0],
                activation=activation,
                situ_beta=activation_beta,
                situ_linear_beta=activation_linear_beta,
                limit=limit,
            ).reshape_as(x)
        else:
            raise RuntimeError(
                "MTP grouped Prefill operator unavailable; the legacy "
                "per-token/per-expert projection implementation was deleted"
            )
        shared = _lin(F.silu(_lin(x, self.w("shared_gate")))
                      * _lin(x, self.w("shared_up")), self.w("shared_down"))
        return routed.to(shared.dtype) + shared

    def _layer78(self, x: torch.Tensor, pos0: int) -> torch.Tensor:
        eps = self.m.cfg["rms_eps"]
        h = self._attention(rmsnorm(x, self.w("input_norm"), eps), pos0)
        x = x + h
        return x + self._moe(rmsnorm(x, self.w("post_norm"), eps))

    # ---- MTP 接口 ----
    def _combine(self, h_main: torch.Tensor, tok_ids: list[int]) -> torch.Tensor:
        """eh_proj(cat[enorm(embed(tok)), hnorm(h_main)])——拼接顺序为 嵌入在前、隐藏在后
        （隔离测试判定：[emb,h] 预测分布与主模型一致，[h,emb] 为乱码）。"""
        emb = self.m.embed(tok_ids)
        hn = rmsnorm(h_main, self.w("hnorm"), 1e-5)
        en = rmsnorm(emb, self.w("enorm"), 1e-5)
        return _lin(torch.cat([en, hn], dim=-1), self.w("eh_proj"))

    def prefill(self, h_main: torch.Tensor, ids: list[int]) -> torch.Tensor:
        """主模型 hidden [T, hidden] 与 token 序列 → 最后位置 h78 [1, hidden]。

        位置 j 的 MTP 输入 = (h_main[j], embed(ids[j+1]))，预测 ids[j+2]。
        """
        from .prefill import end_prefill_block

        T = len(ids)
        x = self._combine(h_main[: T - 1], ids[1:])
        h78 = self._layer78(x, 1)  # MTP 输入在 RoPE 位置 1..T-1（与 token 对齐）
        # 本块 run_rows 在共享主池上保留的专家展开工作区不能带进解码阶段。
        # 主池的 arena 相位由 engine 管理，这里只释放工作区。
        end_prefill_block(self.m.pool, restore_decode=False)
        return h78[-1:]

    def step(self, h78_prev: torch.Tensor, tok_id: int, pos: int) -> tuple[torch.Tensor, torch.Tensor]:
        """单步：上一步 h78 + 草稿 token → (新 h78, logits[vocab])。"""
        x = self._combine(h78_prev, [tok_id])
        h78 = self._layer78(x, pos)
        logits = _lin(rmsnorm(h78, self.w("shared_head_norm"), 1e-5),
                      self.m.w("lm_head.weight")).squeeze(0)
        return h78, logits
