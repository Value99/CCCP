/* WINUI-EXE · TPQ Launcher 前端逻辑(原生 JS,无构建依赖) */
"use strict";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];
const state = {
  profiles: [],
  selected: new Set(),
  instance: null,
  ready: false,
  messages: [],           // {role, content}
  sessionId: null,
  abort: null,            // 生成中断
  currentJob: null,
  settings: null,
};

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  const text = await r.text();
  let data;
  try { data = JSON.parse(text); } catch { data = { raw: text }; }
  if (!r.ok || data.error) {
    const msg = data.error?.message || `HTTP ${r.status}`;
    throw new Error(msg);
  }
  return data;
}
const fmtGB = (gb) => (gb >= 100 ? gb.toFixed(0) : gb.toFixed(1));
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

/* ---------- 选项卡 ---------- */
$$(".tab").forEach((b) =>
  b.addEventListener("click", () => {
    $$(".tab").forEach((x) => x.classList.toggle("active", x === b));
    $$(".page").forEach((p) => p.classList.toggle("active", p.id === `page-${b.dataset.tab}`));
    if (b.dataset.tab === "training") refreshTraining();
    if (b.dataset.tab === "api") refreshApiInfo();
  })
);

/* ---------- 状态轮询 ---------- */
async function pollStatus() {
  try {
    const h = await api("/api/health");
    $("#ver").textContent = `v${h.version}`;
    state.ready = !!h.tpq?.ready;
    const dot = $("#statusDot");
    dot.className = "dot " + (state.ready ? "ready" : h.tpq?.running ? "loading" : "");
    $("#statusText").textContent = state.ready ? "模型就绪" : h.tpq?.running ? "启动中…" : "未启动";
    $("#stopBtn").disabled = !h.tpq?.running;
    if (state.ready) $("#launchHint").textContent = "";
  } catch { $("#statusText").textContent = "后端断连"; $("#statusDot").className = "dot"; }
}
setInterval(pollStatus, 5000);

/* ---------- 配置页 ---------- */
async function loadProfiles() {
  const d = await api("/api/profiles");
  state.profiles = d.profiles;
  if (d.selected?.length) state.selected = new Set(d.selected);
  renderCards();
  renderRelatedSelect();
  updateSummary();
}

function renderCards() {
  const root = $("#profileCards");
  root.innerHTML = "";
  for (const p of state.profiles) {
    const el = document.createElement("div");
    el.className = "card" + (state.selected.has(p.id) ? " selected" : "");
    el.innerHTML = `
      <div class="badges">
        <span class="badge ${p.source}">${{ builtin: "内置", imported: "导入", trained: "训练" }[p.source] || p.source}</span>
        ${p.drop?.enabled ? '<span class="badge drop">drop</span>' : ""}
      </div>
      <h3>${esc(p.name)}</h3>
      <div class="desc">${esc(p.description || "")}</div>
      <div class="stats">
        <span><b>${p.expert_count}</b> 专家</span>
        <span><b>${fmtGB(p.memory_gb)}</b> GB</span>
      </div>
      <div class="calib">${p.calibrated ? "体积已校准" : "体积为估算"}</div>
      ${p.source !== "builtin" || !p.builtin ? "" : ""}
      <div class="check">${state.selected.has(p.id) ? "✓" : ""}</div>`;
    if (p.source === "imported" || p.source === "trained") {
      const del = document.createElement("button");
      del.className = "icon-btn del";
      del.textContent = "✕";
      del.title = "删除(内置被覆盖时还原内置)";
      del.addEventListener("click", async (e) => {
        e.stopPropagation();
        if (!confirm(`删除 profile ${p.name}?`)) return;
        await api(`/api/profiles/${p.id}`, { method: "DELETE" });
        state.selected.delete(p.id);
        await persistSelection();
        loadProfiles();
      });
      el.appendChild(del);
    }
    el.addEventListener("click", () => toggleProfile(p.id));
    root.appendChild(el);
  }
}

async function toggleProfile(id) {
  state.selected.has(id) ? state.selected.delete(id) : state.selected.add(id);
  await persistSelection();
  renderCards();
  updateSummary();
}

async function persistSelection() {
  await api("/api/profiles/select", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ids: [...state.selected] }),
  });
}

async function updateSummary() {
  const ids = [...state.selected];
  $("#sumCount").textContent = ids.length;
  $("#chatProfiles").innerHTML = ids.map((i) => `<span class="chip acc">${esc(i)}</span>`).join("");
  if (!ids.length) {
    $("#sumExperts").textContent = "0"; $("#sumMem").textContent = "0";
    $("#sumOverlap").textContent = "0"; $("#sumDrop").innerHTML = "";
    $("#perProfile").innerHTML = "";
    return;
  }
  const c = await api("/api/profiles/combine", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ids }),
  });
  $("#sumExperts").textContent = c.expert_count;
  $("#sumMem").textContent = fmtGB(c.memory_gb);
  $("#sumOverlap").textContent = fmtGB(c.overlap_gb);
  $("#sumDrop").innerHTML = Object.entries(c.drop_resolution || {})
    .map(([pid, key]) => `<span class="chip acc">${esc(pid)} → ${key ?? "?"}</span>`).join("");
  $("#perProfile").innerHTML = (c.per_profile || [])
    .map((p) => `<span class="chip">${esc(p.id)}: ${fmtGB(p.memory_gb)} GB</span>`).join("");
}

/* 模型 / 启动 */
async function loadModels() {
  try {
    const { models } = await api("/api/models");
    const sel = $("#modelSelect");
    sel.innerHTML = models.length
      ? models.map((m) => `<option value="${esc(m.path)}">${esc(m.name)} (${esc(m.architecture)})</option>`).join("")
      : `<option value="">— 未发现模型(在下方设置 model_roots)—</option>`;
  } catch { /* 静默 */ }
}

function showLog(text) {
  const el = $("#launchLog");
  el.hidden = false; el.textContent = text;
}

async function launch(dry) {
  const ids = [...state.selected];
  if (!ids.length) return alert("请先选择至少一个 profile");
  const model = $("#modelSelect").value;
  if (!model) return alert("请先选择模型目录");
  const body = {
    profile_ids: ids, model_path: model,
    profile_mode: $("#modeSelect").value, device: $("#deviceSelect").value,
    port: +$("#portInput").value, dry_run_only: dry,
  };
  $("#launchBtn").disabled = true;
  $("#launchHint").textContent = dry ? "预检中…" : "启动中…(大模型加载可能需要几分钟)";
  try {
    const r = await api(dry ? "/api/launch/dry-run" : "/api/launch", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    if (dry) {
      showLog(`预检 ${r.ok ? "通过" : "失败"}\n命令: ${(r.cmd || []).join(" ")}\n${r.stdout}\n${r.stderr}`);
      $("#launchHint").textContent = r.ok ? "预检通过,可启动" : "预检失败";
    } else {
      $("#launchHint").textContent = "已发起,等待就绪…";
      pollStatus();
    }
  } catch (e) { alert(e.message); $("#launchHint").textContent = ""; }
  $("#launchBtn").disabled = false;
}
$("#dryRunBtn").addEventListener("click", () => launch(true));
$("#launchBtn").addEventListener("click", () => launch(false));
$("#stopBtn").addEventListener("click", async () => {
  await api("/api/launch/stop", { method: "POST" }); pollStatus();
});

/* 导入 */
$("#importBtn").addEventListener("click", () => $("#importFile").click());
$("#importFile").addEventListener("change", async (ev) => {
  const f = ev.target.files[0];
  if (!f) return;
  const fd = new FormData();
  fd.append("file", f);
  try {
    const r = await api("/api/profiles/import", { method: "POST", body: fd });
    alert(`已导入: ${r.profile.name}(${r.profile.expert_count} 专家 / ${fmtGB(r.profile.memory_gb)} GB)`);
    loadProfiles();
  } catch (e) { alert(`导入失败: ${e.message}`); }
  ev.target.value = "";
});

/* ---------- 设置面板(配置页) ---------- */
async function loadSettings() {
  const s = await api("/api/settings");
  state.settings = s;
  $("#setTpqPath").value = s.tpq_path || "";
  $("#setModelRoots").value = (s.model_roots || []).join("\n");
}
$("#saveSettingsBtn").addEventListener("click", async () => {
  const body = {
    tpq_path: $("#setTpqPath").value.trim(),
    model_roots: $("#setModelRoots").value.split("\n").map((x) => x.trim()).filter(Boolean),
  };
  try {
    await api("/api/settings", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    $("#settingsHint").textContent = "已保存,重新扫描模型…";
    await loadModels();
    setTimeout(() => ($("#settingsHint").textContent = ""), 3000);
  } catch (e) { alert(e.message); }
});

/* ---------- 聊天页 ---------- */
function newSessionId() { return `s-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`; }

function addMsg(role, content) {
  const box = $("#messages");
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.innerHTML = `<div class="role">${role === "user" ? "你" : "模型"}</div><div class="body"></div>`;
  el.querySelector(".body").textContent = content;
  box.appendChild(el);
  box.scrollTop = box.scrollHeight;
  return el;
}

function renderMessages() {
  $("#messages").innerHTML = "";
  for (const m of state.messages) addMsg(m.role, m.content);
}

async function saveSession() {
  if (!state.messages.length) return;
  const firstUser = state.messages.find((m) => m.role === "user");
  const title = (firstUser?.content || "会话").slice(0, 24);
  await api(`/api/chat/sessions/${state.sessionId}`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title, messages: state.messages }),
  });
  refreshSessions();
}

async function refreshSessions() {
  try {
    const { sessions } = await api("/api/chat/sessions");
    $("#sessionList").innerHTML = sessions.map((s) => `
      <div class="session-item ${s.id === state.sessionId ? "active" : ""}" data-sid="${esc(s.id)}">
        <span class="t">${esc(s.title)}</span>
        <button class="icon-btn" data-del="${esc(s.id)}" title="删除">✕</button>
      </div>`).join("") || `<div class="hint">暂无历史会话</div>`;
    $$(".session-item").forEach((el) =>
      el.addEventListener("click", () => loadSession(el.dataset.sid)));
    $$("[data-del]").forEach((b) =>
      b.addEventListener("click", async (e) => {
        e.stopPropagation();
        await api(`/api/chat/sessions/${b.dataset.del}`, { method: "DELETE" });
        if (b.dataset.del === state.sessionId) startNewSession();
        refreshSessions();
      }));
  } catch { /* 静默 */ }
}

async function loadSession(sid) {
  if (sid === state.sessionId) return;
  try {
    const s = await api(`/api/chat/sessions/${sid}`);
    state.sessionId = sid;
    state.messages = s.messages || [];
    renderMessages();
    refreshSessions();
  } catch (e) { alert(e.message); }
}

function startNewSession() {
  state.sessionId = newSessionId();
  state.messages = [];
  renderMessages();
  refreshSessions();
}

async function send() {
  const input = $("#chatInput");
  const text = input.value.trim();
  if (!text) return;
  if (!state.ready) return alert("模型未就绪:请到「配置」页启动组合");
  input.value = "";
  state.messages.push({ role: "user", content: text });
  addMsg("user", text);
  const bubble = addMsg("assistant", "");
  bubble.classList.add("streaming");
  $("#stopGenBtn").hidden = false;
  state.abort = new AbortController();

  const payload = {
    model: state.instance?.model || "winui-model",
    messages: state.messages,
    temperature: +$("#tempInput").value,
    max_tokens: +$("#maxTokInput").value,
    stream: true,
    profile_ids: [...state.selected],
  };
  let answer = "";
  try {
    const r = await fetch("/api/chat/completions", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload), signal: state.abort.signal,
    });
    if (!r.ok || !r.body) throw new Error(`HTTP ${r.status}`);
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop() || "";
      for (const ln of lines) {
        const s = ln.trim();
        if (!s.startsWith("data:")) continue;
        const d = s.slice(5).trim();
        if (d === "[DONE]") continue;
        try {
          const j = JSON.parse(d);
          const delta = j.choices?.[0]?.delta?.content || j.choices?.[0]?.message?.content || j.error || "";
          if (delta) { answer += delta; bubble.querySelector(".body").textContent = answer; }
        } catch { /* 半包忽略 */ }
      }
      $("#messages").scrollTop = $("#messages").scrollHeight;
    }
    state.messages.push({ role: "assistant", content: answer });
    saveSession();
  } catch (e) {
    if (e.name !== "AbortError") bubble.querySelector(".body").textContent = `⚠ ${e.message}`;
  }
  bubble.classList.remove("streaming");
  $("#stopGenBtn").hidden = true;
  state.abort = null;
}
$("#sendBtn").addEventListener("click", send);
$("#chatInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
});
$("#stopGenBtn").addEventListener("click", () => state.abort?.abort());
$("#clearChatBtn").addEventListener("click", () => {
  state.messages = []; $("#messages").innerHTML = "";
  api(`/api/chat/sessions/${state.sessionId}`, { method: "DELETE" }).catch(() => {});
  refreshSessions();
});
$("#newSessionBtn").addEventListener("click", startNewSession);

/* 聊天侧栏实例信息 */
setInterval(async () => {
  try {
    const s = await api("/api/launch/status");
    state.instance = s.instance;
    $("#chatInstance").textContent = s.instance
      ? `${s.instance.model}\nport ${s.instance.port} · profiles: ${(s.instance.profiles || []).join(", ")}`
      : "未启动模型";
  } catch { /* 静默 */ }
}, 8000);

/* ---------- 训练页 ---------- */
async function refreshTraining() {
  const { files } = await api("/api/training/corpus");
  $("#corpusList").innerHTML = files.map((f) => `
    <div class="file-item">
      <span>${esc(f.name)}</span>
      <span class="dim">${(f.bytes / 1024).toFixed(1)} KB
        <button class="icon-btn" data-cdel="${esc(f.name)}" title="删除">✕</button></span>
    </div>`).join("") || `<div class="hint">暂无语料</div>`;
  $$("[data-cdel]").forEach((b) =>
    b.addEventListener("click", async () => {
      await api(`/api/training/corpus/${encodeURIComponent(b.dataset.cdel)}`, { method: "DELETE" });
      refreshTraining();
    }));
  const { jobs } = await api("/api/training/jobs");
  renderJobs(jobs);
}

function renderJobs(jobs) {
  $("#jobList").innerHTML = jobs.map((j) => `
    <div class="job" data-jid="${j.id}">
      <div class="jhead">
        <span class="mono">${j.id}</span>
        <span class="jstatus ${j.status}">${{ pending: "排队", running: "运行", done: "完成", failed: "失败" }[j.status]}</span>
      </div>
      <div class="progress"><div style="width:${(j.progress * 100).toFixed(0)}%"></div></div>
      <div class="jmsg">${j.mode} · 目标 ${j.target_gb} GB · ${esc(j.message || "")}</div>
    </div>`).join("") || `<div class="hint">暂无任务</div>`;
  $$(".job").forEach((el) => el.addEventListener("click", () => showJob(el.dataset.jid)));
  const running = jobs.some((j) => j.status === "running" || j.status === "pending");
  if (running) setTimeout(refreshTraining, 2000);
}

async function showJob(jid) {
  const j = await api(`/api/training/jobs/${jid}`);
  state.currentJob = j;
  const d = $("#jobDetail");
  d.hidden = false;
  $("#jobDetailTitle").textContent = `任务 ${j.id}`;
  $("#jobDetailMeta").textContent =
    `status=${j.status} · source=${j.data_source} · calibrated=${j.calibrated}\n` +
    `样本=${j.total_samples} · 激活专家=${Object.keys(j.counts || {}).length} · ` +
    `规划=${(j.plan_keys || []).length} 个 / ${(j.plan_bytes_mb / 1024).toFixed(1)} GiB`;
  const counts = j.counts || {};
  const top = Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 12);
  const max = top[0]?.[1] || 1;
  $("#topExperts").innerHTML = top.map(([k, v]) => `
    <div class="bar-row"><span>${k}</span>
      <div class="bar"><div style="width:${(v / max * 100).toFixed(1)}%"></div></div>
      <span>${v}</span></div>`).join("");
}

$("#corpusBtn").addEventListener("click", () => $("#corpusFile").click());
$("#corpusFile").addEventListener("change", async (ev) => {
  for (const f of ev.target.files) {
    const fd = new FormData(); fd.append("file", f);
    try { await api("/api/training/corpus", { method: "POST", body: fd }); }
    catch (e) { alert(`上传失败 ${f.name}: ${e.message}`); }
  }
  ev.target.value = "";
  refreshTraining();
});

$("#targetGbRange").addEventListener("input", (e) => ($("#targetGb").value = e.target.value));
$("#targetGb").addEventListener("input", (e) => ($("#targetGbRange").value = e.target.value));

function renderRelatedSelect() {
  const sel = $("#relatedProfiles");
  sel.innerHTML = state.profiles.map((p) => `<option value="${p.id}">${esc(p.name)}</option>`).join("");
}

$("#trainStartBtn").addEventListener("click", async () => {
  const { files } = await api("/api/training/corpus");
  if (!files.length) return alert("先上传语料");
  const body = {
    corpus_files: files.map((f) => f.name),
    mode: document.querySelector('input[name="tmode"]:checked').value,
    target_gb: +$("#targetGb").value,
    sample_limit: +$("#sampleLimit").value,
    related_profiles: [...$("#relatedProfiles").selectedOptions].map((o) => o.value),
  };
  try {
    const r = await api("/api/training/jobs", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    refreshTraining();
    showJob(r.job.id);
  } catch (e) { alert(e.message); }
});

$$("[data-export]").forEach((b) =>
  b.addEventListener("click", async () => {
    if (!state.currentJob) return;
    const kind = b.dataset.export;
    const data = await api(`/api/training/jobs/${state.currentJob.id}/export?kind=${kind}`);
    const blob = new Blob([kind === "profile" ? toYamlFallback(data) : JSON.stringify(data, null, 2)],
                          { type: "application/octet-stream" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${kind}-${state.currentJob.id}.${kind === "profile" ? "yaml" : "json"}`;
    a.click();
  })
);
function toYamlFallback(data) { return JSON.stringify(data, null, 2); }  // profile 导出本身是 dict;注册走服务端
$("#registerBtn").addEventListener("click", async () => {
  if (!state.currentJob) return;
  const name = prompt("profile 显示名:", `训练产物 ${state.currentJob.id}`);
  if (name === null) return;
  try {
    const r = await api(`/api/training/jobs/${state.currentJob.id}/register`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name }),
    });
    alert(`已注册: ${r.profile.name}`);
    loadProfiles();
  } catch (e) { alert(e.message); }
});
$("#jobDeleteBtn").addEventListener("click", async () => {
  if (!state.currentJob) return;
  await api(`/api/training/jobs/${state.currentJob.id}`, { method: "DELETE" });
  $("#jobDetail").hidden = true;
  state.currentJob = null;
  refreshTraining();
});

/* ---------- API 服务页 ---------- */
async function refreshApiInfo() {
  try {
    const d = await api("/api/service/info");
    $("#apiBase").textContent = d.base_url;
    $("#apiEndpoint").textContent = d.openai_endpoint;
    $("#apiCurl").textContent = d.curl_example;
    $("#apiModel").textContent = d.served_model || "未启动";
    const pill = $("#apiStatePill");
    pill.textContent = d.tpq_ready ? "TPQ 就绪" : "TPQ 未就绪";
    pill.className = "state-pill " + (d.tpq_ready ? "on" : "off");
    const auth = $("#apiAuthPill");
    auth.textContent = d.auth ? "已启用" : "未启用";
    auth.className = "state-pill " + (d.auth ? "on" : "off");
    const ul = $("#apiEndpoints");
    ul.innerHTML = "";
    for (const [group, items] of Object.entries(d.endpoints || {})) {
      for (const it of items) {
        const li = document.createElement("li");
        li.innerHTML = `${esc(it)} <span>${esc(group)}</span>`;
        ul.appendChild(li);
      }
    }
  } catch (e) { $("#apiCurl").textContent = `加载失败: ${e.message}`; }
}
$("#apiRefreshBtn").addEventListener("click", refreshApiInfo);

/* ---------- 启动 ---------- */
(async function init() {
  state.sessionId = newSessionId();
  await loadProfiles();
  await loadModels();
  await loadSettings().catch(() => {});
  refreshSessions();
  pollStatus();
  const s = await api("/api/launch/status").catch(() => null);
  if (s?.instance) state.instance = s.instance;
})();
