# WINUI-EXE ⇄ TPQ-Final 接口契约

> 版本:v1.0(2026-08-08)| 状态:**I‑0 已落地;I‑1~I‑5 待 TPQ 开发**
> 约束重申:WINUI-EXE 仓库**不修改 TPQ-Final 的任何文件**。本文档是给
> TPQ-Final 后续开发的接口需求单与现有集成说明。编号以 `I‑x` 引用,
> WINUI-EXE 代码中的 `TODO(TPQ-DEV I‑x)` 即对应等待点。
> 与 WEBUI 仓库的同名契约保持一致(同一批接口需求),TPQ 侧只需实现一次。

---

## 0. 角色与数据流

```
WINUI-EXE(启动器)                         TPQ-Final(推理引擎)
───────────────                           ───────────────────
1. 组装发动计划 ──(子进程+CLI+env)──▶  加载 cccp 模型
2. 生成偏好文件 ──(score/counts json)▶ 专家驻留/热专家固定
3. 转发聊天请求 ──(HTTP /v1)────────▶ OpenAI 兼容 API
4. 语料扫描任务 ──(HTTP /v1 或子进程)▶ CPU/硬盘全量推理
5. 消费统计输出 ◀──(I‑3 router-stats)─ 路由命中计数
6. 校准体积/规格 ◀──(I‑1/I‑2)───────── 专家字节表/层×专家数
```

WINUI-EXE 产物文件统一写到 `WINUI-EXE/data/`,**不会写入 TPQ-Final 目录**;
TPQ 侧如需落文件,约定写到模型目录或 `%TEMP%/tpq-winui/` 并在文档中固定。

---

## 1. 现有集成(I‑0,已核实 TPQ-Final v1.2.0 支持,无需改动)

| # | 方式 | 内容 | WINUI-EXE 使用处 |
|---|---|---|---|
| I‑0a | 子进程 | `python -m tpq launch serve --model <dir> --host H --port P --served-model-name N [--profile auto\|ram\|resident\|mapped\|parallel] [--device cuda\|cpu] [--cache-gb X] [--vram-gb Y] [--dense-residency auto\|gpu\|ram] [--cpu-compile auto\|u16\|q4] [--extreme --extreme-placement auto --extreme-score-file F]` | `tpq_adapter.build_command` |
| I‑0b | 预检 | 同一命令追加 `--dry-run`,要求 `returncode==0` 且 stderr 无致命错误 | `tpq_adapter.dry_run` |
| I‑0c | 环境变量 | `TPQ_PROFILE_JSON=<path>`:热专家计数档案(schema 见 §2.2);`TPQ_API_KEY`(可选) | `tpq_adapter._env` |
| I‑0d | HTTP | `GET /health`(字段 `ready/busy/model/architecture`),`GET /v1/models`,`POST /v1/chat/completions`(`stream:true` SSE 或 false 一次性) | `tpq_adapter.health`、`chat.py` |
| I‑0e | 文件 | `--extreme-score-file`:schema `tpq-expert-residency-scores-v1`,`{scores:{"layer:expert": float}}`,**当前要求恰好覆盖存档全部专家**(有限、非负) | `tpq_adapter._score_file`、`training.export_scores` |

行为约定(WINUI-EXE 已按此实现,TPQ 请勿破坏):
- `/health.ready==true` 前 WINUI-EXE 不转发聊天;`busy` 仅用于 UI 展示。
- SSE 以 `data: …\n\n` 分块、`data: [DONE]` 结束;WINUI-EXE 逐 chunk 透传。
- 鉴权:设置了 `TPQ_API_KEY` 时 WINUI-EXE 以 `Authorization: Bearer` 透传。
- drop 语义:模型归档中被标记 drop 的专家由 TPQ 路由前的掩码自动排除
  (`masked_fill(~available, -inf)`);WINUI-EXE 的 profile drop 占位专家
  **只在启动器内解析**为组合中最相关的已加载专家并写入发动计划 meta,
  不要求 TPQ 做任何配合。

---

## 2. WINUI-EXE 产出、TPQ 消费的文件 Schema

### 2.1 `tpq-expert-residency-scores-v1`(发动/训练均产出)

```json
{
  "schema": "tpq-expert-residency-scores-v1",
  "scores": { "12:33": 1.0, "12:45": 0.42, "...": 0.0 },
  "meta": { "generator": "tpq-winui-launcher", "profiles": ["contract", "python-code"],
            "calibrated": false, "target_gb": 128, "selected": 5140,
            "drop_resolution": {"contract": "18:77"} }
}
```
- 发动时:组合内专家 `1.0`;训练产物:命中计数归一化到 `[0,1]`。
- meta 块 TPQ 可忽略;`calibrated=false` 表示体积/热度为估算。
- `drop_resolution` 仅为启动器的解析记录,TPQ 可忽略(见 §1 drop 语义)。

### 2.2 `counts` profile(TPQ_PROFILE_JSON)

```json
{ "counts": { "12": { "33": 17, "45": 3 }, "13": { "2": 9 } } }
```
发动时 WINUI-EXE 对组合内专家置 `1`;训练产物为真实(或估算)命中计数。
与 TPQ 现有 `<model>/profile.json` schema 完全一致。

---

## 3. 待开发接口需求(按优先级)

### I‑1 专家字节表(校准体积)— 优先级 P0

**需求**:输出每个打包专家实例的字节数,使 profile 体积从"估算"变为"校准"。

建议其一(任选):
- CLI:`python -m tpq check --model <dir> --expert-bytes-json <out>` →
  `{"schema":"tpq-expert-bytes-v1","layers":L,"experts_per_layer":E,"bytes":{"layer:expert":int}}`
- 或启动后 HTTP:`GET /v1/expert-bytes` 同 schema。

WINUI-EXE 侧接入点:`profiles.py` recipe 的 `size_mb` 替换;profile `meta.calibrated=true`。
未提供前 WINUI-EXE 标注"估算"并继续工作(降级安全)。

### I‑2 模型规格发现 — P0

**需求**:机器可读地返回 `layers`、`experts_per_layer`、每个 layer 的实际专家数(异构时)。
当前 WINUI-EXE 用 `settings.model_layers/model_experts_per_layer`(默认 60×256)手工校准,
并为 score 文件**补零全覆盖**(I‑0e 的全覆盖要求)。
建议合入 I‑1 的同一输出(已含 L/E 字段)。

> 若 TPQ 放宽"全覆盖"校验(允许只给非零项,其余按 0),WINUI-EXE 可省掉补零;
> 请在此文档回复确认后,WINUI-EXE 端删除自动补零逻辑(改动点:`training.export_scores`)。

### I‑3 路由激活统计输出(训练选项卡真数据源)— P1

**需求**:一次推理会话(或语料批)后,导出每层路由命中计数,使"训练"页从估算切换为实测。

建议:
- HTTP:`GET /v1/expert-stats`(自服务启动累计)→
  `{"schema":"tpq-router-stats-v1","counts":{"layer:expert":int},"tokens_seen":int,"wall_time_s":float}`
  搭配 `POST /v1/expert-stats/reset` 清零;**或**
- 文件:env `TPQ_ROUTER_STATS_JSONL=<path>`,每 N tokens append
  `{"counts":{"layer:expert":int}}`(WINUI-EXE 尾部聚合)。

WINUI-EXE 接入点:`training.tpq_router_stats_available()`(探测)→ `TrainingEngine._run` 的
分支 A(`data_source=tpq-router-stats, calibrated=true`)。语料扫描流程(不动 TPQ):
WINUI-EXE 以 `--device cpu`(或 disk 模式,见 I‑4)启动一个**扫描实例**,把语料分片走
`/v1/chat/completions` 喂入,轮询 I‑3 输出取增量。

### I‑4 硬盘(disk)全量推理档位 — P2

**需求**:确认/暴露"权重 mostly 在磁盘、逐层 mmap 流入"的纯离线档位(比 `--device cpu`
更省 RAM),用于"当前硬件推理不起来的模型"的语料扫描。若现有 `--cpu-compile`/mmap
行为已可满足,请文档化推荐 flag 组合,WINUI-EXE 把它固化为 `mode=disk` 的命令模板。

### I‑5 发动验收回执 — P3

**需求**(可选):serve 就绪后 `GET /health` 增加
`{"experts_resident_vram":n,"experts_resident_ram":m,"plan_file_ok":true}`,
使 WINUI-EXE 能在 UI 显示"偏好文件是否被采纳"。无此字段时 WINUI-EXE 忽略(降级安全)。

---

## 4. 兼容与版本

- 两个 json schema 均带 `schema` 字符串 + meta,TPQ 忽略未知键;新增键不视为破坏。
- WINUI-EXE 对 I‑1~I‑5 全部**降级安全**:缺失时保持估算/启发式,UI 显著标注
  (`calibrated:false`、能力徽标)。
- TPQ 实现任一接口后,请更新本文档状态表并 bump 版本:

| 接口 | 状态 | TPQ 版本 | 落地 PR/commit |
|---|---|---|---|
| I‑0 | ✅ 已集成 | 1.2.0 | — |
| I‑1 专家字节表 | ⏳ 待开发 | — | — |
| I‑2 规格发现 | ⏳ 待开发 | — | — |
| I‑3 路由统计 | ⏳ 待开发 | — | — |
| I‑4 disk 档位 | ⏳ 待确认 | — | — |
| I‑5 验收回执 | ⏳ 可选 | — | — |

---

## 5. WINUI-EXE 侧承诺(给 TPQ 开发者的定心丸)

1. 永不写 `TPQ-Final/` 目录;任何文件集成只在 WINUI-EXE `data/` 或约定临时目录。
2. 仅通过 `python -m tpq …` 与 OpenAI HTTP 交互;不 import tpq 内部模块、不 patch。
3. 所有新接口在 WINUI-EXE 端有保底路径,TPQ 未实现不阻塞 WINUI-EXE 使用。
4. 接口探测失败即回退并在 UI 明示,不静默伪造"校准"数据。
