#!/usr/bin/env bash
# 端到端冒烟(不加载真模型):后端 → profiles 组合去重 → 训练任务 → 导出 → API 信息
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python}
PORT=${SMOKE_PORT:-8795}
BASE="http://127.0.0.1:${PORT}"

echo "== 启动后端(端口 ${PORT})…"
"$PY" -m launcher.app --no-shell --port "${PORT}" > data/smoke-server.log 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

for i in $(seq 1 60); do
  curl -sf "${BASE}/api/health" > /dev/null 2>&1 && break
  sleep 0.5
done
curl -sf "${BASE}/api/health" > /dev/null || { echo "FAIL: 后端未就绪"; cat data/smoke-server.log; exit 1; }
echo "OK: /api/health"

echo "== profiles 列表(应含 3 个内置)…"
"$PY" - "$BASE" <<'PYEOF'
import json, sys, urllib.request
base = sys.argv[1]
d = json.load(urllib.request.urlopen(base + "/api/profiles"))
ids = {p["id"] for p in d["profiles"]}
assert {"roleplay", "python-code", "contract"} <= ids, ids
print("OK: builtins", sorted(ids))

req = urllib.request.Request(
    base + "/api/profiles/combine",
    data=json.dumps({"ids": ["contract", "python-code"]}).encode(),
    headers={"Content-Type": "application/json"},
)
c = json.load(urllib.request.urlopen(req))
assert c["overlap_mb"] > 0, "重叠应 > 0"
assert set(c["drop_resolution"]) == {"contract", "python-code"}
per_sum = sum(p["memory_gb"] for p in c["per_profile"])
assert c["memory_gb"] < per_sum, "并集应小于逐项之和(去重叠)"
print(f"OK: combine union={c['memory_gb']:.1f}GB overlap={c['overlap_gb']:.1f}GB "
      f"(逐项之和={per_sum:.1f}GB) drop={c['drop_resolution']}")

# 语料上传 -> 训练任务 -> 完成 -> 导出
corpus = '{"prompt": "写 python 代码"}\n{"prompt": "审阅合同条款"}\n'
boundary = "----smoke"
body = (
    f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
    f"filename=\"smoke.jsonl\"\r\nContent-Type: application/octet-stream\r\n\r\n"
    f"{corpus}\r\n--{boundary}--\r\n"
).encode()
req = urllib.request.Request(
    base + "/api/training/corpus", data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
)
up = json.load(urllib.request.urlopen(req))
assert up["ok"]

req = urllib.request.Request(
    base + "/api/training/jobs",
    data=json.dumps({"corpus_files": ["smoke.jsonl"], "target_gb": 30,
                     "related_profiles": ["python-code", "contract"]}).encode(),
    headers={"Content-Type": "application/json"},
)
job = json.load(urllib.request.urlopen(req))["job"]
import time
for _ in range(100):
    j = json.load(urllib.request.urlopen(base + f"/api/training/jobs/{job['id']}"))
    if j["status"] in ("done", "failed"):
        break
    time.sleep(0.3)
assert j["status"] == "done", j["message"]
assert j["plan_bytes_mb"] / 1024 <= 30.0
print(f"OK: training job done, plan={j['plan_bytes_mb']/1024:.1f}GiB "
      f"(source={j['data_source']}, calibrated={j['calibrated']})")

exp = json.load(urllib.request.urlopen(base + f"/api/training/jobs/{job['id']}/export?kind=scores"))
assert exp["schema"] == "tpq-expert-residency-scores-v1"
n_scores = len(exp["scores"])
assert n_scores == j["layers"] * j["experts_per_layer"], "scores 必须全覆盖"
print(f"OK: export scores 全覆盖 {n_scores} 项")

info = json.load(urllib.request.urlopen(base + "/api/service/info"))
assert info["openai_endpoint"].endswith("/v1/chat/completions")
print("OK: /api/service/info base=", info["base_url"])

# v0.3: 社区 + 下载 API(不触网,仅结构校验)
comm = json.load(urllib.request.urlopen(base + "/api/community/config"))
assert "discord_url" in comm and "index_url" in comm
dj = json.load(urllib.request.urlopen(base + "/api/models/download/jobs"))
assert "jobs" in dj and "default_dir" in dj
bad = urllib.request.Request(
    base + "/api/models/download",
    data=json.dumps({"repo": ""}).encode(),
    headers={"Content-Type": "application/json"},
)
try:
    urllib.request.urlopen(bad)
    raise AssertionError("空 repo 应返回 400")
except urllib.error.HTTPError as e:
    assert e.code == 400
print("OK: community + downloads API")
PYEOF

echo "== SMOKE 全部通过 ✅"
