/* CCCP 启动器 前端逻辑(原生 JS,无构建依赖) — 浅色应用外壳 */
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
/* ---------- 主题:system/light/dark ---------- */
function applyTheme(mode) {
  document.documentElement.setAttribute("data-theme", mode || "system");
  localStorage.setItem("theme", mode || "system");
}
applyTheme(localStorage.getItem("theme") || "system");

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

/* ---------- 左侧导航 ---------- */
$$(".nav-item").forEach((b) =>
  b.addEventListener("click", () => {
    $$(".nav-item").forEach((x) => x.classList.toggle("active", x === b));
    $$(".page").forEach((p) => p.classList.toggle("active", p.id === `page-${b.dataset.tab}`));
    if (b.dataset.tab === "training") refreshTraining();
    if (b.dataset.tab === "api") refreshApiInfo();
    if (b.dataset.tab === "settings") { loadSettings().catch(() => {}); loadSystem().catch(() => {}); }
    if (b.dataset.tab === "models") refreshModelsPage();
    if (b.dataset.tab === "home") refreshHome();
    if (b.dataset.tab === "terminal") startTerminal();
    if (b.dataset.tab !== "terminal") stopTerminal();
  })
);

/* ---------- 状态轮询(侧栏状态卡) ---------- */
async function pollStatus() {
  try {
    const h = await api("/api/health");
    $("#ver").textContent = `v${h.version}`;
    state.ready = !!h.tpq?.ready;
    const dot = $("#statusDot");
    dot.className = "dot " + (state.ready ? "ready" : h.tpq?.running ? "loading" : "");
    $("#statusText").textContent = state.ready ? "模型就绪" : h.tpq?.running ? "启动中…" : "未启动";
    $("#stopBtn").disabled = !h.tpq?.running;
    $("#homeStopBtn").disabled = !h.tpq?.running;
    const pill = $("#homeStatusPill");
    pill.textContent = state.ready ? "模型就绪" : h.tpq?.running ? "启动中…" : "未启动";
    pill.className = "state-pill hero-pill " + (state.ready ? "on" : "off");
    if (state.ready) { $("#launchHint").textContent = ""; $("#homeLaunchHint").textContent = ""; }
  } catch { $("#statusText").textContent = "后端断连"; $("#statusDot").className = "dot"; }
}
setInterval(pollStatus, 5000);

/* ---------- 配置页 ---------- */
async function loadProfiles() {
  const d = await api("/api/profiles");
  state.profiles = d.profiles;
  if (d.selected?.length) state.selected = new Set(d.selected);
  renderCards();
  renderHomeChips();
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
  renderHomeChips();
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
    updateHomeSummary(null);
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
  updateHomeSummary(c);
}

/* 模型 / 启动 */
async function loadModels() {
  try {
    const { models } = await api("/api/models");
    state.models = models;
    const html = models.length
      ? models.map((m) => `<option value="${esc(m.path)}">${esc(m.name)} (${esc(m.architecture)})</option>`).join("")
      : `<option value="">— 未发现模型(「模型库」下载或「设置」添加目录)—</option>`;
    $("#modelSelect").innerHTML = html;
    $("#homeModelSelect").innerHTML = html;
    renderLocalModels(models);
  } catch { /* 静默 */ }
}

function showLog(text) {
  const el = $("#launchLog");
  el.hidden = false; el.textContent = text;
}

async function launch(dry, src = "profiles") {
  const ids = [...state.selected];
  if (!ids.length) return alert("请先在「首页」或「配置库」勾选至少一个配置");
  const sel = src === "home" ? $("#homeModelSelect") : $("#modelSelect");
  const portEl = src === "home" ? $("#homePortInput") : $("#portInput");
  const hint = src === "home" ? $("#homeLaunchHint") : $("#launchHint");
  const btn = src === "home" ? $("#homeLaunchBtn") : $("#launchBtn");
  const logEl = src === "home" ? $("#homeLaunchLog") : $("#launchLog");
  const model = sel.value;
  if (!model) return alert("请先选择模型(「模型库」页可下载;「设置」可添加本地目录)");
  const body = {
    profile_ids: ids, model_path: model,
    profile_mode: $("#modeSelect").value,
    device: (src === "home" ? $("#homeDeviceSelect") : $("#deviceSelect")).value,
    port: +portEl.value, dry_run_only: dry,
  };
  btn.disabled = true;
  hint.textContent = dry ? "预检中…" : "启动中…(大模型加载可能需要几分钟)";
  try {
    const r = await api(dry ? "/api/launch/dry-run" : "/api/launch", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    if (dry) {
      logEl.hidden = false;
      logEl.textContent = `预检 ${r.ok ? "通过" : "失败"}\n命令: ${(r.cmd || []).join(" ")}\n${r.stdout}\n${r.stderr}`;
      hint.textContent = r.ok ? "预检通过,可启动" : "预检失败";
    } else {
      hint.textContent = "已发起,等待就绪…";
      pollStatus();
    }
  } catch (e) { alert(e.message); hint.textContent = ""; }
  btn.disabled = false;
}
$("#dryRunBtn").addEventListener("click", () => launch(true));
$("#launchBtn").addEventListener("click", () => launch(false));
$("#homeDryBtn").addEventListener("click", () => launch(true, "home"));
$("#homeLaunchBtn").addEventListener("click", () => launch(false, "home"));
const stopTpq = async () => { await api("/api/launch/stop", { method: "POST" }); pollStatus(); };
$("#stopBtn").addEventListener("click", stopTpq);
$("#homeStopBtn").addEventListener("click", stopTpq);

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

/* ---------- 首页(快速选择 / 一键启动 / 上次启动 / 社区入口)---------- */
function renderHomeChips() {
  const root = $("#homeChips");
  if (!root) return;
  root.innerHTML = state.profiles.map((p) => `
    <span class="qchip ${state.selected.has(p.id) ? "on" : ""}" data-pid="${esc(p.id)}" title="${esc(p.description || "")}">${esc(p.name)} <small>${fmtGB(p.memory_gb)}G</small></span>`).join("");
  $$("#homeChips .qchip").forEach((el) => el.addEventListener("click", () => toggleProfile(el.dataset.pid)));
}

function updateHomeSummary(c) {
  $("#homeSummary").textContent = c
    ? `合计 ${c.expert_count} 专家 · 去重后 ${fmtGB(c.memory_gb)} GB(重叠省 ${fmtGB(c.overlap_gb)} GB)`
    : "勾选配置后显示合计体积";
  const ms = $("#homeModelSelect");
  $("#homeSelChip").textContent = `${ms?.selectedOptions[0]?.textContent || "未选择模型"} · ${state.selected.size} 个配置`;
}

async function refreshHome() {
  renderHomeChips();
  try {
    const s = await api("/api/launch/status");
    const ll = s.last_launch;
    const box = $("#lastLaunchBox");
    if (ll) {
      box.textContent = `${ll.model}\nprofiles: ${(ll.profiles || []).join(", ")}\nport ${ll.port} · ${new Date((ll.at || 0) * 1000).toLocaleString()}`;
      const btn = $("#relaunchBtn");
      btn.disabled = false;
      btn.onclick = async () => {
        state.selected = new Set(ll.profiles || []);
        await persistSelection();
        renderCards(); renderHomeChips(); updateSummary();
        const sel = $("#homeModelSelect");
        if ([...sel.options].some((o) => o.value === ll.model)) sel.value = ll.model;
        if (ll.port) $("#homePortInput").value = ll.port;
        launch(false, "home");
      };
    } else { box.textContent = "暂无记录"; $("#relaunchBtn").disabled = true; }
  } catch { /* 静默 */ }
  try {
    const cfg = await api("/api/community/config");
    $("#communityCfgHint").textContent = cfg.discord_url ? "" : "Discord 链接未配置(到「设置 · 社区与下载」填写)";
  } catch { /* 静默 */ }
}

$("#discordBtn").addEventListener("click", async () => {
  try {
    const cfg = await api("/api/community/config");
    if (cfg.discord_url) window.open(cfg.discord_url, "_blank");
    else alert("社区 Discord 链接未配置:请到「设置 · 社区与下载」填写");
  } catch (e) { alert(e.message); }
});
$("#goModelsBtn").addEventListener("click", () =>
  $$(".nav-item").find((b) => b.dataset.tab === "models")?.click());

/* ---------- 社区配置下载 ---------- */
$("#communityInstallBtn").addEventListener("click", async () => {
  const url = $("#communityUrlInput").value.trim();
  if (!url) return alert("请粘贴 profile 文件 URL(yaml/json)");
  installCommunity(url);
});
async function installCommunity(url) {
  try {
    const r = await api("/api/community/profiles/install", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ url }) });
    alert(`已安装: ${r.profile.name}(${r.profile.expert_count} 专家 / ${fmtGB(r.profile.memory_gb)} GB)`);
    $("#communityUrlInput").value = "";
    loadProfiles();
  } catch (e) { alert(`安装失败: ${e.message}`); }
}
$("#communityRefreshBtn").addEventListener("click", async () => {
  const box = $("#communityList");
  box.innerHTML = `<div class="hint">拉取社区索引中…</div>`;
  try {
    const d = await api("/api/community/profiles");
    box.innerHTML = d.profiles.length
      ? d.profiles.map((e) => `
        <div class="community-item">
          <div class="ci-t"><b>${esc(e.name || e.id || "未命名")}</b><span class="dim">${esc(e.description || "")}</span></div>
          <button class="primary sm" data-curl="${esc(e.url)}">安装</button>
        </div>`).join("")
      : `<div class="hint">${esc(d.note || "社区索引为空")}</div>`;
    $$("#communityList [data-curl]").forEach((b) =>
      b.addEventListener("click", () => installCommunity(b.dataset.curl)));
  } catch (e) { box.innerHTML = `<div class="hint">拉取失败: ${esc(e.message)}</div>`; }
});

/* ---------- 模型库页 ---------- */
function renderLocalModels(models) {
  const box = $("#localModels");
  if (!box) return;
  box.innerHTML = models.length ? models.map((m) => `
    <div class="model-item">
      <div class="mi-t"><b>${esc(m.name)}</b><span class="badge builtin">${esc(m.architecture)}</span></div>
      <div class="mi-p mono dim">${esc(m.path)}</div>
      <button class="ghost sm" data-use="${esc(m.path)}">设为启动模型</button>
    </div>`).join("")
    : `<div class="hint">未发现含 cccp.json 的模型目录;在「设置」添加 model_roots,或下载后完成 CCCP 归档。</div>`;
  $$("#localModels [data-use]").forEach((b) =>
    b.addEventListener("click", () => {
      if ([...$("#homeModelSelect").options].some((o) => o.value === b.dataset.use))
        $("#homeModelSelect").value = b.dataset.use;
      $("#modelSelect").value = b.dataset.use;
      $$(".nav-item").find((x) => x.dataset.tab === "home")?.click();
    }));
}

async function refreshDlJobs() {
  try {
    const d = await api("/api/models/download/jobs");
    if (d.default_dir) $("#dlTarget").placeholder = `留空使用默认目录: ${d.default_dir}`;
    $("#dlJobs").innerHTML = d.jobs.map((j) => `
      <div class="job">
        <div class="jhead">
          <span class="mono">${esc(j.repo)} <span class="dim">[${esc(j.source)}]</span></span>
          <span class="jstatus ${j.status}">${{ running: "下载中", done: "完成", failed: "失败" }[j.status]}</span>
        </div>
        <div class="jmsg">${esc(j.message)}${j.error ? " · " + esc(j.error) : ""}</div>
        <div class="jmsg mono">${esc(j.result_path || j.target_dir)}</div>
      </div>`).join("") || `<div class="hint">暂无下载任务</div>`;
    if (d.jobs.some((j) => j.status === "running")) setTimeout(refreshDlJobs, 3000);
  } catch { /* 静默 */ }
}

function refreshModelsPage() { loadModels(); refreshDlJobs(); }
$("#modelsRefreshBtn").addEventListener("click", refreshModelsPage);
$("#dlStartBtn").addEventListener("click", async () => {
  const repo = $("#dlRepo").value.trim();
  if (!repo) return alert("请填写仓库 ID(如 Qwen/Qwen2.5-7B-Instruct)");
  try {
    await api("/api/models/download", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ repo, source: $("#dlSource").value,
        revision: $("#dlRevision").value.trim(), target_dir: $("#dlTarget").value.trim() }) });
    refreshDlJobs();
  } catch (e) { alert(e.message); }
});

/* ---------- 设置页 ---------- */
async function loadSettings() {
  const s = await api("/api/settings");
  state.settings = s;
  $("#setTpqPath").value = s.tpq_path || "";
  $("#setModelRoots").value = (s.model_roots || []).join("\n");
  $("#setDiscord").value = s.discord_url || "";
  $("#setIndexUrl").value = s.community_index_url || "";
  $("#setHfEndpoint").value = s.hf_endpoint || "";
  $("#setDlDir").value = s.model_download_dir || "";
  $("#setDefaultDevice").value = s.default_device || "cuda";
  const th = s.theme_mode || "system";
  $("#setTheme").value = th; applyTheme(th);
  $("#deviceSelect").value = s.default_device || "cuda";
  $("#homeDeviceSelect").value = s.default_device || "cuda";
  $("#aboutTpq").textContent = s.tpq_path || "未探测到";
}
$("#setTheme").addEventListener("change", (e) => applyTheme(e.target.value));
$("#saveSettingsBtn").addEventListener("click", async () => {
  const body = {
    tpq_path: $("#setTpqPath").value.trim(),
    model_roots: $("#setModelRoots").value.split("\n").map((x) => x.trim()).filter(Boolean),
    discord_url: $("#setDiscord").value.trim(),
    community_index_url: $("#setIndexUrl").value.trim(),
    hf_endpoint: $("#setHfEndpoint").value.trim(),
    model_download_dir: $("#setDlDir").value.trim(),
    default_device: $("#setDefaultDevice").value,
    theme_mode: $("#setTheme").value,
  };
  try {
    await api("/api/settings", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    $("#settingsHint").textContent = "已保存,重新扫描模型…";
    $("#aboutTpq").textContent = body.tpq_path || "自动探测";
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
  if (!state.ready) return alert("模型未就绪:请到「首页」勾选配置后一键启动");
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
    if (e.name !== "AbortError") bubble.querySelector(".body").textContent = `[错误] ${e.message}`;
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
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/octet-stream" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${kind}-${state.currentJob.id}.${kind === "profile" ? "yaml" : "json"}`;
    a.click();
  })
);
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
  refreshSessions();
  pollStatus();
  const h = await api("/api/health").catch(() => null);
  if (h?.version) $("#aboutVer").textContent = `v${h.version}`;
  loadSettings().catch(() => {});
  refreshHome();
  refreshDlJobs();
  const s = await api("/api/launch/status").catch(() => null);
  if (s?.instance) state.instance = s.instance;
})();

/* ---------- 无边框自绘标题栏(pywebview 原生窗口;浏览器模式静默降级)---------- */
(function titlebarWiring() {
  const native = () => window.pywebview?.api;
  $("#tbMin")?.addEventListener("click", (e) => { e.stopPropagation(); native()?.win_minimize?.(); });
  $("#tbMax")?.addEventListener("click", (e) => { e.stopPropagation(); native()?.win_toggle_max?.(); });
  $("#tbClose")?.addEventListener("click", (e) => {
    e.stopPropagation();
    if (native()?.win_close) native().win_close(); else window.close();
  });
  const tb = $("#titlebar");
  if (!tb) return;
  tb.addEventListener("dblclick", (e) => {
    if (e.target.closest(".tb-btn")) return;
    native()?.win_toggle_max?.();
  });
  tb.addEventListener("mousedown", (e) => {
    if (e.button !== 0 || e.target.closest(".tb-btn")) return;
    const api = native();
    if (!api?.win_move) return;
    const ox = e.screenX - window.screenX, oy = e.screenY - window.screenY;
    const onmove = (ev) => api.win_move(ev.screenX - ox, ev.screenY - oy);
    const onup = () => {
      document.removeEventListener("mousemove", onmove);
      document.removeEventListener("mouseup", onup);
    };
    document.addEventListener("mousemove", onmove);
    document.addEventListener("mouseup", onup);
  });
})();

/* ---------- 终端页(可见时轮询)---------- */
let termTimer = null;
function renderTerm(el, lines, empty) {
  el.textContent = lines && lines.length ? lines.join("\n") : empty;
  el.scrollTop = el.scrollHeight;
}
async function refreshTerminal() {
  try {
    const [t, a] = await Promise.all([api("/api/terminal/tpq"), api("/api/terminal/app")]);
    renderTerm($("#termTpq"), t.lines, "TPQ 未在运行 / 暂无输出");
    renderTerm($("#termApp"), a.lines, "暂无启动器日志");
  } catch (e) { $("#termApp").textContent = `读取失败: ${e.message}`; }
}
function startTerminal() { stopTerminal(); refreshTerminal(); termTimer = setInterval(refreshTerminal, 3000); }
function stopTerminal() { if (termTimer) { clearInterval(termTimer); termTimer = null; } }
$("#termRefreshBtn").addEventListener("click", refreshTerminal);

/* ---------- 运行环境(CUDA / CPU)---------- */
async function loadSystem() {
  const d = await api("/api/system");
  $("#sysInfo").textContent = d.cuda_available
    ? `CUDA 可用 · GPU: ${d.gpus.join(" | ")} · CPU ${d.cpu_count} 线程 · ${d.platform} · Python ${d.python}`
    : `未检测到 CUDA(nvidia-smi 不可用)· 将以 CPU 推理 · CPU ${d.cpu_count} 线程 · ${d.platform} · Python ${d.python}`;
}
