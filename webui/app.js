/* CCCP 启动器 前端逻辑(原生 JS,无构建依赖) — 浅色应用外壳 */
"use strict";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];
const state = {
  profiles: [],
  models: [],
  selected: new Set(),
  fullModelPath: "",
  instance: null,
  ready: false,
  messages: [],           // {role, content}
  sessionId: null,
  abort: null,            // 生成中断
  currentJob: null,
  corpusSelected: new Set(),
  settings: null,
  update: null,
  editingProfile: null,
  completedDownloadScans: new Set(),
};

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  const text = await r.text();
  let data;
  try { data = JSON.parse(text); } catch { data = { raw: text }; }
  if (!r.ok || data.error) {
    const msg = data.error?.message || data.detail || `HTTP ${r.status}`;
    throw new Error(msg);
  }
  return data;
}
const fmtGB = (gb) => (gb >= 100 ? gb.toFixed(0) : gb.toFixed(1));
const profileResidentGiB = (profile) => Number(
  profile?.meta?.configuration_resident_gib ??
  profile?.meta?.configuration_budget_gib ??
  profile?.memory_gb ?? 0
);
/* ---------- 主题:system/light/dark ---------- */
function applyTheme(mode) {
  document.documentElement.setAttribute("data-theme", mode || "system");
  localStorage.setItem("theme", mode || "system");
}
applyTheme(localStorage.getItem("theme") || "system");

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

/* ---------- 左侧导航 ---------- */
function activateTab(tab) {
  const button = $$(".nav-item").find((item) => item.dataset.tab === tab);
  if (!button) return;
  $$(".nav-item").forEach((item) => item.classList.toggle("active", item === button));
  $$(".page").forEach((page) => page.classList.toggle("active", page.id === `page-${tab}`));
  if (tab === "training") refreshTraining();
  if (tab === "api") refreshApiInfo();
  if (tab === "settings") { loadSettings().catch(() => {}); loadSystem().catch(() => {}); }
  if (tab === "models") refreshModelsPage();
  if (tab === "home") refreshHome();
  if (tab === "terminal") startTerminal();
  else stopTerminal();
}
$$(".nav-item").forEach((button) =>
  button.addEventListener("click", () => activateTab(button.dataset.tab))
);

/* ---------- 状态轮询(侧栏状态卡) ---------- */
async function pollStatus() {
  try {
    const h = await api("/api/health");
    $("#ver").textContent = `v${h.version}`;
    const wasReady = state.ready;
    state.ready = !!h.cccp?.ready;
    const dot = $("#statusDot");
    dot.className = "dot " + (state.ready ? "ready" : h.cccp?.running ? "loading" : "");
    $("#statusText").textContent = state.ready ? "模型就绪" : h.cccp?.running ? "启动中…" : "未启动";
    $("#stopBtn").disabled = !h.cccp?.running;
    $("#homeStopBtn").disabled = !h.cccp?.running;
    const pill = $("#homeStatusPill");
    pill.textContent = state.ready ? "模型就绪" : h.cccp?.running ? "启动中…" : "未启动";
    pill.className = "state-pill hero-pill " + (state.ready ? "on" : "off");
    if (state.ready) { $("#launchHint").textContent = ""; $("#homeLaunchHint").textContent = ""; }
    if (state.ready && (!wasReady || !state.thinkingCapabilityLoaded)) {
      refreshThinkingCapabilities().catch(() => {});
    }
  } catch { $("#statusText").textContent = "后端断连"; $("#statusDot").className = "dot"; }
}
setInterval(pollStatus, 5000);

/* ---------- 配置页 ---------- */
async function loadProfiles() {
  const d = await api("/api/profiles");
  state.profiles = d.profiles;
  const activeModel = $("#modelSelect")?.value || $("#homeModelSelect")?.value || "";
  state.selected = new Set((d.selected || []).filter((id) =>
    state.profiles.some((profile) =>
      profile.id === id && profile.model_available && profile.matched_model_path === activeModel)));
  renderCards();
  renderHomeChips();
  updateSummary();
}

function renderCards() {
  const root = $("#profileCards");
  root.innerHTML = "";
  const modelPath = $("#modelSelect")?.value || "";
  if (!modelPath) {
    root.innerHTML = `<div class="empty-placeholder"><b>请先选择模型</b><span>选择后显示该模型的专家配置。</span></div>`;
    return;
  }
  const profiles = profilesForModel(modelPath);
  if (!profiles.length) {
    const model = state.models.find((item) => item.path === modelPath);
    root.innerHTML = model?.has_dynamic_experts === false
      ? `<div class="empty-placeholder"><b>Dense 模型不使用专家配置</b><span>权重由模型清单完整加载，无需语料扫描或专家路由预设。</span></div>`
      : `<div class="empty-placeholder"><b>该模型暂无专家配置</b><span>可到“训练”生成配置、导入匹配配置，或选择下方全量专家模式。</span></div>`;
    if (model) root.appendChild(fullModelCard(model));
    return;
  }
  for (const p of profiles) {
    const el = document.createElement("div");
    el.className = "card" + (state.selected.has(p.id) ? " selected" : "");
    el.innerHTML = `
      <h3 title="${esc(p.name)}">${esc(p.name)}</h3>
      <div class="desc">${esc(p.description || "")}</div>
      <div class="stats">
        <span><b>${p.expert_count}</b> 专家</span>
        <span><b>${fmtGB(profileResidentGiB(p))}</b> GiB 配置总驻留</span>
      </div>
      <div class="card-actions"><span class="card-action-spacer"></span></div>`;
    const actions = el.querySelector(".card-actions");
    const del = document.createElement("button");
    del.className = "icon-btn del";
    del.innerHTML = '<svg class="ic"><use href="#i-trash"/></svg>';
    del.setAttribute("aria-label", `删除配置 ${p.name}`);
    del.title = "删除配置文件";
    del.addEventListener("click", async (e) => {
      e.stopPropagation();
      if (!confirm(`确定删除「${p.name}」的配置文件？\n模型权重不会被删除。`)) return;
      await api(`/api/profiles/${encodeURIComponent(p.id)}`, { method: "DELETE" });
      state.selected.delete(p.id);
      await persistSelection();
      await loadProfiles();
    });
    actions.appendChild(del);
    const edit = document.createElement("button");
    edit.className = "icon-btn edit";
    edit.innerHTML = '<svg class="ic"><use href="#i-edit"/></svg>';
    edit.setAttribute("aria-label", `编辑配置 ${p.name}`);
    edit.title = "编辑配置名称与说明";
    edit.addEventListener("click", (e) => {
      e.stopPropagation();
      openProfileEdit(p);
    });
    actions.appendChild(edit);
    const exp = document.createElement("a");
    exp.className = "icon-btn export";
    exp.innerHTML = '<svg class="ic"><use href="#i-download"/></svg>';
    exp.setAttribute("aria-label", `导出配置 ${p.name}`);
    exp.title = "导出完整配置（可分享、可再次导入）";
    exp.href = `/api/profiles/${encodeURIComponent(p.id)}/export`;
    exp.download = `${p.id}.json`;
    exp.addEventListener("click", (e) => e.stopPropagation());
    actions.appendChild(exp);
    const check = document.createElement("span");
    check.className = "check";
    check.innerHTML = state.selected.has(p.id) ? '<svg class="ic"><use href="#i-check"/></svg>' : "";
    check.setAttribute("aria-hidden", "true");
    actions.appendChild(check);
    if (!p.model_available) el.classList.add("unavailable");
    el.addEventListener("click", () => {
      if (!p.model_available) return alert(p.model_status || "找不到配置对应的模型");
      toggleProfile(p.id);
    });
    root.appendChild(el);
  }
}

function profilesForModel(path) {
  if (!path) return [];
  return state.profiles.filter((profile) =>
    profile.model_available && profile.matched_model_path === path);
}

function fullModelSummary(path) {
  const model = state.models.find((item) => item.path === path);
  if (!model) return null;
  const expertLayerCount = Number(model.expert_layer_count || model.expert_layers?.length || model.layers || 0);
  return {
    expert_count: expertLayerCount * Number(model.experts_per_layer || 0),
    memory_gb: Number(model.expert_gb || 0),
    configuration_resident_gib: Number(model.dense_gb || 0) + Number(model.expert_gb || 0),
    total_deduplicated_gib: 0,
    full_model: true,
  };
}

function fullModelCard(model) {
  const selected = state.fullModelPath === model.path;
  const dense = model.has_dynamic_experts === false;
  const expertLayerCount = Number(model.expert_layer_count || model.expert_layers?.length || model.layers || 0);
  const count = expertLayerCount * Number(model.experts_per_layer || 0);
  const el = document.createElement("div");
  el.className = "card full-model-card" + (selected ? " selected" : "");
  el.innerHTML = `
    <div class="badges"><span class="badge model">${dense ? "Dense VQ" : "无配置模式"}</span><span class="badge trained">${dense ? "完整权重" : "全部专家可路由"}</span></div>
    <h3>${dense ? "完整 Dense 模型" : "全量专家加载"}</h3>
    <div class="desc">${dense ? "模型没有动态专家，直接按 cccp.json 加载全部 Dense VQ 投影；无需训练或专家配置。" : "不应用专家配置或路由白名单；先尝试完整常驻，容量不足时保留全部专家并自动降级到 LRU 与磁盘映射。"}</div>
    <div class="stats">${dense ? `<span><b>${fmtGB(Number(model.dense_gb || 0))}</b> GiB Dense VQ 权重</span><span><b>${Number(model.layers || 0)}</b> 层</span>` : `<span><b>${count}</b> 专家</span><span><b>${fmtGB(Number(model.expert_gb || 0))}</b> GiB 专家权重</span>`}</div>
    <div class="calib">完整模型约 ${fmtGB(Number(model.total_gb || 0))} GiB · ${esc(model.model_version || model.name)}</div>
    <div class="card-actions"><span class="card-action-spacer"></span><span class="check">${selected ? '<svg class="ic"><use href="#i-check"/></svg>' : ""}</span></div>`;
  el.addEventListener("click", () => toggleFullModel(model.path));
  return el;
}

async function toggleFullModel(path) {
  state.fullModelPath = state.fullModelPath === path ? "" : path;
  state.selected.clear();
  await persistSelection();
  renderCards(); renderHomeChips(); updateSummary();
}

async function toggleProfile(id) {
  state.fullModelPath = "";
  state.selected.has(id) ? state.selected.delete(id) : state.selected.add(id);
  const profile = state.profiles.find((item) => item.id === id);
  if (state.selected.has(id) && profile?.matched_model_path) selectMatchedModel(profile.matched_model_path);
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

function activeChatProfileIds() {
  if (state.instance?.full_model) return [];
  const running = state.instance?.profiles;
  return Array.isArray(running) && running.length
    ? [...running]
    : [...state.selected];
}

function renderChatProfiles() {
  const full = !!state.instance?.full_model || !!state.fullModelPath;
  const modelPath = state.instance?.model || state.fullModelPath || $("#homeModelSelect")?.value || "";
  const dense = state.models.find((item) => item.path === modelPath)?.has_dynamic_experts === false;
  const ids = activeChatProfileIds();
  $("#chatProfiles").innerHTML = full
    ? `<span class="chip acc">${dense ? "完整 Dense 模型" : "全量专家（无配置路由限制）"}</span>`
    : ids.length
    ? ids.map((id) => `<span class="chip acc">${esc(id)}</span>`).join("")
    : `<span class="empty-inline">未选择配置</span>`;
}

async function updateSummary() {
  const ids = [...state.selected];
  const full = state.fullModelPath ? fullModelSummary(state.fullModelPath) : null;
  $("#sumCount").textContent = full ? "全量" : ids.length;
  renderChatProfiles();
  if (full) {
    const dense = state.models.find((item) => item.path === state.fullModelPath)?.has_dynamic_experts === false;
    $("#sumExperts").textContent = dense ? "不适用" : full.expert_count;
    $("#sumMem").textContent = fmtGB(full.configuration_resident_gib);
    $("#sumOverlap").textContent = "0";
    updateHomeSummary(full);
    return;
  }
  if (!ids.length) {
    $("#sumExperts").textContent = "0"; $("#sumMem").textContent = "0";
    $("#sumOverlap").textContent = "0";
    updateHomeSummary(null);
    return;
  }
  const c = await api("/api/profiles/combine", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ids }),
  });
  $("#sumExperts").textContent = c.expert_count;
  $("#sumMem").textContent = fmtGB(c.configuration_resident_gib || c.memory_gb);
  $("#sumOverlap").textContent = fmtGB(c.total_deduplicated_gib ?? c.overlap_gb);
  updateHomeSummary(c);
}

function selectMatchedModel(path) {
  for (const id of ["modelSelect", "homeModelSelect"]) {
    const select = $("#" + id);
    if (select && [...select.options].some((option) => option.value === path)) select.value = path;
  }
  updateHomeSelectionChip();
}

/* 模型 / 启动 */
async function loadModels() {
  try {
    const { models } = await api("/api/models");
    state.models = models;
    const ids = ["modelSelect", "homeModelSelect", "trainModelSelect"];
    const previous = Object.fromEntries(ids.map((id) => [id, $("#" + id)?.value || ""]));
    const options = models.map((m) => `<option value="${esc(m.path)}" ${m.complete ? "" : "disabled"}>${esc(m.name)} (${esc(m.architecture)} · ${fmtGB(m.total_gb)} GiB${m.complete ? "" : " · 不完整"})</option>`).join("");
    for (const id of ids) {
      const select = $("#" + id);
      select.innerHTML = models.length
        ? `<option value="">— 请选择模型 —</option>${options}`
        : `<option value="">— 未发现模型（可到模型库下载）—</option>`;
      if (previous[id] && [...select.options].some((option) => option.value === previous[id])) {
        select.value = previous[id];
      }
    }
    updateTrainingModelLimit();
    renderLocalModels(models);
    // /api/profiles 会按最新的模型根目录重新扫描各模型的 profiles/。
    // 首次添加模型时，旧的 state.profiles 仍为空，必须在模型发现后立即刷新，
    // 否则用户选中模型也只能看到“暂无专家配置”，直到重启应用。
    await loadProfiles();
    updateHomeSelectionChip();
  } catch { /* 静默 */ }
}

function updateTrainingModelLimit() {
  const selected = state.models.find((model) => model.path === $("#trainModelSelect")?.value);
  const controls = [
    ...$$('input[name="tmode"]'),
    $("#tokenBudget"),
  ].filter(Boolean);
  if (!selected) {
    if ($("#trainModelLimit")) $("#trainModelLimit").textContent = "选择模型后自动读取聊天模板、层数、专家数与上下文上限";
    controls.forEach((control) => (control.disabled = false));
    $("#trainStartBtn").disabled = false;
    return;
  }
  if (selected.supports_route_training === false || selected.has_dynamic_experts === false) {
    $("#trainModelLimit").textContent =
      `Dense VQ · ${Number(selected.layers || 0)} 层 · 上下文上限 ${Number(selected.max_context || 0).toLocaleString()} token · 没有动态专家，无需且不支持语料路由扫描`;
    controls.forEach((control) => (control.disabled = true));
    $("#trainStartBtn").disabled = true;
    $("#trainStartBtn").title = "Dense 模型没有动态专家，请直接从配置库启动完整模型";
    return;
  }
  controls.forEach((control) => (control.disabled = false));
  $("#trainStartBtn").disabled = false;
  $("#trainStartBtn").title = "";
  const expertLayerCount = Number(selected.expert_layer_count || selected.expert_layers?.length || selected.layers || 0);
  const layerText = expertLayerCount === Number(selected.layers || 0)
    ? `${selected.layers} 层`
    : `${expertLayerCount} 个专家层 / ${selected.layers} 个总层`;
  $("#trainModelLimit").textContent =
    `${layerText} × ${selected.experts_per_layer} 专家 · top-k ${selected.top_k} · 上下文上限 ${Number(selected.max_context || 0).toLocaleString()} token`;
}
$("#trainModelSelect").addEventListener("change", () => {
  updateTrainingModelLimit();
});

function showLog(text) {
  const el = $("#launchLog");
  el.hidden = false; el.textContent = text;
}

function memoryRiskMessage(memory, warnings = []) {
  const reasons = memory.risk_reasons || [];
  const lines = [
    `配置预算/总驻留：${memory.configuration_source_resident_gb} GiB（它不是运行内存上限）`,
    memory.capacity_kind === "vram"
      ? `合计驻留估算（显存+主机内存）：${memory.total_estimate_gb} GiB；物理显存：${memory.device_capacity_gb ?? memory.limit_gb} GiB；当前空闲显存：${memory.available_gb} GiB`
      : `运行估算：${memory.total_estimate_gb} GiB；内存上限：${memory.device_capacity_gb ?? memory.limit_gb} GiB；当前空闲：${memory.available_gb} GiB`,
  ];
  if (memory.capacity_kind === "vram") {
    lines.push(`主机内存：${memory.host_available_gb} / ${memory.host_total_gb} GiB 可用`);
    if (memory.minimum_vram_gb > 0) {
      lines.push(`CUDA 最低工作集：${memory.minimum_vram_gb} GiB；全速建议：${memory.recommended_vram_gb} GiB`);
    }
  }
  reasons.forEach((reason) => lines.push(reason.message));
  if (memory.offload_target === "ram") {
    lines.push("完整专家保留在主机内存，GPU 按可用显存自动分块；不会减少专家，当前不使用磁盘。");
  } else if (memory.offload_target === "cpu") {
    lines.push(memory.hybrid_dense_ram
      ? "当前显存连 RAM Dense 混合模式的 CUDA 基础工作区与最小专家块也容纳不下；建议切换 CPU 推理。"
      : "当前显存连 Dense 与 CUDA 基础工作区也容纳不下；缩小专家块无效，建议切换 CPU 推理。");
  } else if (memory.offload_target === "disk" && memory.capacity_kind === "vram") {
    lines.push("降级顺序：显存不足 → 主机内存不足 → 磁盘映射/系统虚拟内存（速度较慢）。");
  } else if (memory.offload_target === "disk") {
    lines.push("当前内存不足，将使用磁盘映射/系统虚拟内存，速度较慢。");
  }
  if (!reasons.length) lines.push(...warnings);
  return lines.join("\n");
}

function preflightStatusLabel(preflight, commandOk = true) {
  if (preflight?.memory?.gpu_execution_tier === "below_minimum") return "失败 · 显存低于 CUDA 最低工作集";
  if (!commandOk || !preflight?.ok || preflight?.status === "blocked") return "失败";
  if (preflight?.status === "danger") return "完成 · 容量高风险";
  if (preflight?.memory?.gpu_execution_tier === "reduced_expert_arena") return "完成 · 受限显存混合加速";
  if (preflight?.status === "warning") return `完成 · 当前可用${preflight?.memory?.capacity_label || "设备内存"}不足`;
  return "通过";
}

function primeTerminal(preflight = null) {
  const target = preflight?.memory?.offload_target || "none";
  const limitedGpu = preflight?.memory?.gpu_execution_tier === "reduced_expert_arena";
  $("#termStage").textContent = target === "disk" ? "正在启用磁盘卸载" : target === "ram" ? "正在启用内存卸载" : "正在提交启动任务";
  $("#termDetail").textContent = target === "disk"
    ? "显存与主机内存均不足，已切换到磁盘映射与系统虚拟内存；速度会较慢"
    : target === "ram"
    ? (limitedGpu ? "显存低于全速建议值；正在缩小 GPU 专家块，完整专家保留在主机内存" : "完整专家保留在主机内存，GPU 将按剩余显存自动分块；不会使用磁盘")
    : "即将创建 CCCP 推理进程";
  $("#termPercent").textContent = "1%";
  $("#termProgressBar").style.width = "1%";
  $("#termProgress").setAttribute("aria-valuenow", "1");
  $("#termMeta").textContent = target === "disk" ? "磁盘卸载模式 · 请勿关闭窗口" : target === "ram" ? "主机内存卸载模式 · 未使用磁盘" : "正在校验模型、配置与本地加速算子";
  $("#termOutput").textContent = target === "disk"
    ? "[启动器] 已依次检查显存和主机内存，二者均不足；系统将使用磁盘换页或映射缓存，速度明显较慢。"
    : target === "ram"
    ? `[启动器] ${limitedGpu ? "显存低于全速建议值，已启用受限显存混合加速" : "已启用动态专家内存驻留"}；专家数量保持不变，当前不会使用磁盘。`
    : "[启动器] 已通过预检，正在创建 CCCP 进程…";
}

function primeTrainingTerminal(job) {
  $("#termStage").textContent = "正在准备 Token 扫描";
  $("#termDetail").textContent = job.message || "正在整理语料并创建路由扫描任务";
  $("#termPercent").textContent = "2%";
  $("#termProgressBar").style.width = "2%";
  $("#termProgress").setAttribute("aria-valuenow", "2");
  $("#termMeta").textContent = `任务 ${job.id} · 0/${Number(job.token_budget || 0).toLocaleString()} token · 4096 token/块`;
  $("#termOutput").textContent = "[启动器] Token 路由扫描已提交；正在读取模型、语料和本地 CPU 加速算子…";
}

async function launch(dry, src = "profiles") {
  const ids = [...state.selected];
  const sel = src === "home" ? $("#homeModelSelect") : $("#modelSelect");
  const portEl = src === "home" ? $("#homePortInput") : $("#portInput");
  const hint = src === "home" ? $("#homeLaunchHint") : $("#launchHint");
  const btn = src === "home" ? $("#homeLaunchBtn") : $("#launchBtn");
  const logEl = src === "home" ? $("#homeLaunchLog") : $("#launchLog");
  const model = sel.value;
  if (!model) return alert("请先选择模型(「模型库」页可下载;「设置」可添加本地目录)");
  const fullModel = state.fullModelPath === model;
  if (!ids.length && !fullModel) {
    return alert("请选择该模型的专家配置；若该模型没有配置，可选择“全量专家加载”");
  }
  const body = {
    profile_ids: ids, model_path: model,
    full_model: fullModel,
    profile_mode: $("#modeSelect").value,
    device: (src === "home" ? $("#homeDeviceSelect") : $("#deviceSelect")).value,
    port: +portEl.value, dry_run_only: dry,
  };
  btn.disabled = true;
  hint.textContent = dry ? "预检中…" : "启动中…(大模型加载可能需要几分钟)";
  try {
    if (!dry) {
      let pre = await api("/api/launch/preflight", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
      });
      let pm = pre.memory || {};
      if (pm.gpu_execution_tier === "below_minimum") {
        logEl.hidden = false;
        logEl.classList.remove("memory-warning");
        logEl.classList.add("memory-danger");
        logEl.textContent = `显存低于 CUDA 最低工作集\n${memoryRiskMessage(pm, pre.warnings || [])}`;
        const switchToCpu = confirm(
          `当前可用显存 ${pm.available_gb} GiB，低于该模型在当前上下文下的 CUDA 最低工作集 ${pm.minimum_vram_gb} GiB。\n\n` +
          (pm.hybrid_dense_ram
            ? "RAM Dense 混合模式的 CUDA 基础工作区与最小专家块也无法容纳。是否自动切换为 CPU 推理继续预检？"
            : "这部分是 Dense 与 CUDA 工作区，缩小专家块也无法解决。是否自动切换为 CPU 推理继续预检？")
        );
        if (!switchToCpu) {
          hint.textContent = "已取消；可关闭占用显存的程序后重试";
          btn.disabled = false;
          return;
        }
        body.device = "cpu";
        if ($("#homeDeviceSelect")) $("#homeDeviceSelect").value = "cpu";
        if ($("#deviceSelect")) $("#deviceSelect").value = "cpu";
        pre = await api("/api/launch/preflight", {
          method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
        });
        pm = pre.memory || {};
      }
      if (!pre.ok) throw new Error((pre.errors || []).join("；") || "启动前检查失败");
      logEl.classList.remove("memory-danger", "memory-warning");
      logEl.hidden = true;
      logEl.textContent = "";
      if (pm.risk_level === "danger") {
        logEl.hidden = false;
        logEl.classList.add("memory-danger");
        logEl.textContent = `高风险\n${memoryRiskMessage(pm, pre.warnings || [])}`;
        const confirmText = pm.capacity_kind === "vram"
          ? "显存不足后已尝试卸载到主机内存，但当前内存仍不足。若继续才会使用磁盘映射/系统虚拟内存，推理速度会明显变慢。仍要继续吗？"
          : "当前内存不足。若继续，启动器会使用磁盘映射/系统虚拟内存完成加载，但推理速度会明显变慢。仍要继续吗？";
        if (!confirm(confirmText)) {
          hint.textContent = "已取消高风险加载";
          btn.disabled = false;
          return;
        }
      } else if (pm.risk_level === "warning") {
        logEl.hidden = false;
        logEl.classList.add("memory-warning");
        logEl.textContent = `内存提醒\n${memoryRiskMessage(pm, pre.warnings || [])}`;
      }
      activateTab("terminal");
      primeTerminal(pre);
    }
    const r = await api(dry ? "/api/launch/dry-run" : "/api/launch", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    if (dry) {
      logEl.hidden = false;
      const mem = r.preflight?.memory;
      logEl.classList.toggle("memory-danger", mem?.risk_level === "danger");
      logEl.classList.toggle("memory-warning", mem?.risk_level === "warning");
      const notes = [
        ...(r.preflight?.errors || []).map((x) => `错误: ${x}`),
        ...(r.preflight?.warnings || []).map((x) => `警告: ${x}`),
      ].join("\n");
      const statusLabel = preflightStatusLabel(r.preflight, r.ok);
      logEl.textContent = `预检 ${statusLabel}` +
        (mem ? `\n配置总驻留: ${mem.configuration_source_resident_gb} GiB（Dense ${mem.dense_without_shared_source_gb} + 共享专家 ${mem.shared_expert_source_gb} + 动态专家 ${mem.routed_expert_source_gb}）\n${mem.capacity_kind === "vram" ? "显存门槛" : "运行内存估算"}: ${mem.capacity_kind === "vram" ? `最低 ${mem.minimum_vram_gb} / 建议 ${mem.recommended_vram_gb}` : mem.total_estimate_gb} GiB；当前可用 ${mem.available_gb} GiB${mem.offload_target !== "none" ? `\n${memoryRiskMessage(mem, r.preflight?.warnings || [])}` : ""}` : "") +
        `\n${notes}\n命令: ${(r.cmd || []).join(" ")}\n${r.stdout || ""}\n${r.stderr || ""}`;
      hint.textContent = `预检${statusLabel}${r.ok && statusLabel !== "通过" ? "，确认风险后可启动" : ""}`;
    } else {
      hint.textContent = "已发起,等待就绪…";
      state.instance = r.instance;
      refreshTerminal();
      pollStatus();
    }
  } catch (e) { alert(e.message); hint.textContent = ""; }
  btn.disabled = false;
}
$("#dryRunBtn").addEventListener("click", () => launch(true));
$("#launchBtn").addEventListener("click", () => launch(false));
$("#homeDryBtn").addEventListener("click", () => launch(true, "home"));
$("#homeLaunchBtn").addEventListener("click", () => launch(false, "home"));
const stopCccp = async () => { await api("/api/launch/stop", { method: "POST" }); pollStatus(); };
$("#stopBtn").addEventListener("click", stopCccp);
$("#homeStopBtn").addEventListener("click", stopCccp);

/* 导入 */
$("#importBtn").addEventListener("click", () => $("#importFile").click());
$("#importFile").addEventListener("change", async (ev) => {
  const f = ev.target.files[0];
  if (!f) return;
  const fd = new FormData();
  fd.append("file", f);
  try {
    const r = await api("/api/profiles/import", { method: "POST", body: fd });
    alert(`已导入: ${r.profile.name}(${r.profile.expert_count} 专家 / ${fmtGB(profileResidentGiB(r.profile))} GiB 配置总驻留)`);
    loadProfiles();
  } catch (e) { alert(`导入失败: ${e.message}`); }
  ev.target.value = "";
});

/* ---------- 首页(快速选择 / 一键启动 / 上次启动 / 社区入口)---------- */
function renderHomeChips() {
  const root = $("#homeChips");
  if (!root) return;
  const modelPath = $("#homeModelSelect")?.value || "";
  const profiles = profilesForModel(modelPath);
  if (!modelPath) {
    root.innerHTML = `<div class="empty-placeholder compact"><span>请先选择模型</span></div>`;
  } else if (profiles.length) {
    root.innerHTML = profiles.map((p) => `
      <span class="qchip ${state.selected.has(p.id) ? "on" : ""}" data-pid="${esc(p.id)}" title="${esc(p.description || p.model_status || "")}">${esc(p.name)} <small>${fmtGB(profileResidentGiB(p))}G 总驻留</small></span>`).join("");
  } else {
    const active = state.fullModelPath === modelPath;
    const dense = state.models.find((item) => item.path === modelPath)?.has_dynamic_experts === false;
    root.innerHTML = `<span class="qchip ${active ? "on" : ""}" data-full-model="${esc(modelPath)}" title="${dense ? "Dense 模型没有动态专家，按清单加载完整权重" : "该模型没有匹配配置；使用全部专家且不限制路由"}">${dense ? "完整 Dense 模型" : "全量专家加载"} <small>${dense ? "无需配置" : "无配置"}</small></span>`;
  }
  $$("#homeChips .qchip").forEach((el) => el.addEventListener("click", () => {
    if (el.dataset.fullModel) return toggleFullModel(el.dataset.fullModel);
    const profile = state.profiles.find((item) => item.id === el.dataset.pid);
    if (!profile?.model_available) return alert(profile?.model_status || "找不到对应模型");
    toggleProfile(el.dataset.pid);
  }));
}

function updateHomeSummary(c) {
  const selectedPath = state.fullModelPath || $("#homeModelSelect")?.value || "";
  const dense = state.models.find((item) => item.path === selectedPath)?.has_dynamic_experts === false;
  $("#homeSummary").textContent = c
    ? c.full_model
      ? dense
        ? `约 ${fmtGB(c.configuration_resident_gib)} GiB Dense VQ 权重 · 无动态专家 · 直接完整加载`
        : `${c.expert_count} 个专家 · 约 ${fmtGB(c.configuration_resident_gib)} GiB 完整模型 · 无路由限制，容量不足自动磁盘降级`
      : `${c.expert_count} 个专家 · ${fmtGB(c.configuration_resident_gib || c.memory_gb)} GiB 配置总驻留 · 已按模型版本与专家编号去重${(c.total_deduplicated_gib ?? c.overlap_gb) > 0 ? `，复用 ${fmtGB(c.total_deduplicated_gib ?? c.overlap_gb)} GiB` : ""}`
    : "选择配置后显示合计体积";
  updateHomeSelectionChip();
}

function updateHomeSelectionChip() {
  const ms = $("#homeModelSelect");
  $("#homeSelChip").textContent = ms?.selectedOptions[0]?.textContent || "未选择模型";
}
$("#homeModelSelect").addEventListener("change", async () => {
  if ([...$("#modelSelect").options].some((option) => option.value === $("#homeModelSelect").value)) {
    $("#modelSelect").value = $("#homeModelSelect").value;
  }
  state.fullModelPath = "";
  state.selected.clear();
  try {
    await persistSelection();
    await loadProfiles();
  } catch { /* 下次模型列表刷新时会再次扫描 */ }
  updateHomeSelectionChip();
});
$("#modelSelect").addEventListener("change", async () => {
  if ([...$("#homeModelSelect").options].some((option) => option.value === $("#modelSelect").value)) {
    $("#homeModelSelect").value = $("#modelSelect").value;
  }
  state.fullModelPath = "";
  state.selected.clear();
  try {
    await persistSelection();
    await loadProfiles();
  } catch { /* 下次模型列表刷新时会再次扫描 */ }
});

async function refreshHome() {
  renderHomeChips();
  try {
    const s = await api("/api/launch/status");
    const ll = s.last_launch;
    const box = $("#lastLaunchBox");
    if (ll) {
      box.textContent = `${ll.model}\n${ll.full_model ? "全量专家（无配置路由限制）" : `profiles: ${(ll.profiles || []).join(", ")}`}\nport ${ll.port} · ${new Date((ll.at || 0) * 1000).toLocaleString()}`;
      const btn = $("#relaunchBtn");
      btn.disabled = false;
      btn.onclick = async () => {
        state.selected = new Set(ll.profiles || []);
        state.fullModelPath = ll.full_model ? ll.model : "";
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
    const missing = [];
    if (!cfg.discord_url) missing.push("Discord 社区");
    if (!cfg.modelscope_profile_url) missing.push("ModelScope 模型主页");
    $("#communityCfgHint").textContent = missing.length ? `${missing.join("、")}链接未配置` : "";
  } catch { /* 静默 */ }
}

$("#discordBtn").addEventListener("click", async () => {
  try {
    const cfg = await api("/api/community/config");
    if (cfg.discord_url) window.open(cfg.discord_url, "_blank");
    else alert("社区 Discord 链接未配置:请到「设置 · 社区与下载」填写");
  } catch (e) { alert(e.message); }
});
$("#goModelsBtn").addEventListener("click", async () => {
  try {
    const cfg = await api("/api/community/config");
    if (cfg.modelscope_profile_url) window.open(cfg.modelscope_profile_url, "_blank");
    else alert("ModelScope 模型主页未配置：请到「设置 · 社区与下载」填写");
  } catch (e) { alert(e.message); }
});

/* ---------- 模型库页 ---------- */
function renderLocalModels(models) {
  const box = $("#localModels");
  if (!box) return;
  box.innerHTML = models.length ? models.map((m) => `
    <div class="model-item">
      <div class="mi-t"><b>${esc(m.name)}</b><span class="badge model">${esc(m.architecture)} / ${esc(m.model_format || "cccp")}</span></div>
      <div class="mi-p mono dim">${esc(m.path)}</div>
      <div class="mi-p dim">${m.has_dynamic_experts === false
        ? `总计 ${fmtGB(m.total_gb)} GiB · Dense VQ 权重 ${fmtGB(m.dense_gb)} GiB · ${m.layers} 层 · 无动态专家/无需训练`
        : `总计 ${fmtGB(m.total_gb)} GiB · Dense ${fmtGB(m.dense_without_shared_gb ?? m.dense_gb)} GiB · 共享专家 ${fmtGB(m.shared_expert_gb || 0)} GiB · 路由专家 ${fmtGB(m.expert_gb)} GiB · ${m.layers} 层 × ${m.experts_per_layer} 专家 · top-k ${m.top_k}`}</div>
      <div class="mi-p mono dim">模型指纹 ${esc((m.manifest_sha256 || "未提供").slice(0, 16))}…</div>
      ${m.errors?.length ? `<div class="hint">${m.errors.map(esc).join("；")}</div>` : ""}
      <div class="model-actions">
        <button class="ghost sm" data-use="${esc(m.path)}" ${m.complete ? "" : "disabled"}>设为启动模型</button>
        <button class="danger sm" data-model-delete="${esc(m.path)}" data-model-name="${esc(m.name)}"><svg class="ic"><use href="#i-trash"/></svg>删除模型</button>
      </div>
    </div>`).join("")
    : `<div class="empty-placeholder"><b>暂无本地模型</b><span>可在“设置”添加模型目录，或在上方下载模型。</span></div>`;
  $$("#localModels [data-use]").forEach((b) =>
    b.addEventListener("click", () => {
      if ([...$("#homeModelSelect").options].some((o) => o.value === b.dataset.use))
        $("#homeModelSelect").value = b.dataset.use;
      $("#modelSelect").value = b.dataset.use;
      $("#modelSelect").dispatchEvent(new Event("change"));
      $$(".nav-item").find((x) => x.dataset.tab === "home")?.click();
    }));
  $$("#localModels [data-model-delete]").forEach((button) =>
    button.addEventListener("click", async () => {
      const name = button.dataset.modelName || "该模型";
      if (!confirm(`确定永久删除「${name}」的整个模型目录？\n模型权重文件很大，此操作不可撤销。`)) return;
      button.disabled = true;
      try {
        await api("/api/models", {
          method: "DELETE", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ path: button.dataset.modelDelete }),
        });
        await loadModels();
      } catch (error) {
        button.disabled = false;
        alert(error.message);
      }
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
      </div>`).join("") || `<div class="empty-placeholder compact"><span>暂无下载任务</span></div>`;
    const newlyCompleted = d.jobs.filter((job) =>
      job.status === "done" && !state.completedDownloadScans.has(job.id));
    for (const job of newlyCompleted) state.completedDownloadScans.add(job.id);
    if (newlyCompleted.length) await loadModels();
    if (d.jobs.some((j) => j.status === "running")) setTimeout(refreshDlJobs, 3000);
  } catch { /* 静默 */ }
}

function refreshModelsPage() { loadModels(); refreshDlJobs(); }
$("#modelsRefreshBtn").addEventListener("click", refreshModelsPage);
$("#dlStartBtn").addEventListener("click", async () => {
  const repo = $("#dlRepo").value.trim();
  if (!repo) return alert("请填写仓库 ID（如 ValueFX/DeepSeek-V4-Flash-0731-CCCP-L）");
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
  $("#setModelRoots").value = (s.model_roots || []).join("\n");
  $("#dlSource").value = s.default_download_source || "modelscope";
  $("#setDefaultDevice").value = s.default_device || "cpu";
  $("#modeSelect").value = ["auto", "mapped"].includes(s.default_profile_mode) ? s.default_profile_mode : "auto";
  const theme = s.theme_mode || "system";
  $("#setTheme").value = theme;
  applyTheme(theme);
  $("#deviceSelect").value = s.default_device || "cpu";
  $("#homeDeviceSelect").value = s.default_device || "cpu";
  $("#aboutCccp").textContent = s.cccp_engine_path ? "内置 · 已探测" : "未探测到";
  $("#setApiAuthEnabled").checked = !!s.cccp_api_key;
  $("#setApiKey").value = s.cccp_api_key || "";
  syncApiKeyControls();
}

/* ---------- 配置资料编辑 ---------- */
function closeProfileEdit() {
  state.editingProfile = null;
  $("#profileEditOverlay").hidden = true;
}
function openProfileEdit(profile) {
  state.editingProfile = profile;
  $("#profileEditName").value = profile.name || "";
  $("#profileEditDescription").value = profile.description || "";
  $("#profileEditOverlay").hidden = false;
  requestAnimationFrame(() => $("#profileEditName").focus());
}
$("#profileEditCancel").addEventListener("click", closeProfileEdit);
$("#profileEditOverlay").addEventListener("click", (event) => {
  if (event.target === $("#profileEditOverlay")) closeProfileEdit();
});
$("#profileEditForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const profile = state.editingProfile;
  if (!profile) return;
  const name = $("#profileEditName").value.trim();
  const description = $("#profileEditDescription").value.trim();
  if (!name) return $("#profileEditName").focus();
  const save = $("#profileEditSave");
  save.disabled = true;
  try {
    await api(`/api/profiles/${encodeURIComponent(profile.id)}`, {
      method: "PATCH", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, description }),
    });
    closeProfileEdit();
    await loadProfiles();
  } catch (error) {
    alert(error.message);
  } finally {
    save.disabled = false;
  }
});

/* ---------- 非阻塞更新检测 ---------- */
let updatePollTimer = null;
function hideUpdateDialog() { $("#updateOverlay").hidden = true; }
function showUpdateDialog(update) {
  state.update = update;
  $("#updateSource").textContent = `${update.source_label} 检测到更新`;
  $("#updateTitle").textContent = update.title || `发现新版本 ${update.latest_version}`;
  $("#updateSummary").textContent = update.summary || "有新的稳定版本可以下载。";
  $("#updateVersions").textContent = `当前版本 ${update.current_version} · 最新版本 ${update.latest_version}`;
  const notes = $("#updateNotes");
  notes.replaceChildren(...(update.release_notes || []).map((item) => {
    const li = document.createElement("li"); li.textContent = item; return li;
  }));
  notes.hidden = !notes.children.length;
  $("#updateOpenText").textContent = `从${update.source_label}下载`;
  $("#updateOverlay").hidden = false;
}
function describeUpdate(update, manual = false) {
  if (!manual) return;
  const hint = $("#updateCheckHint");
  if (update.status === "current") hint.textContent = `当前已是最新版本（来源：${update.source_label}）。`;
  else if (update.status === "ignored") hint.textContent = `已忽略版本 ${update.latest_version}。`;
  else if (update.status === "unavailable") hint.textContent = "官网和 GitHub 暂时均不可访问，不影响离线使用。";
}
async function pollUpdate(manual = false, attempts = 0) {
  const update = await api("/api/update/status").catch(() => null);
  if (!update) return;
  state.update = update;
  if (update.status === "checking" && attempts < 80) {
    clearTimeout(updatePollTimer);
    updatePollTimer = setTimeout(() => pollUpdate(manual, attempts + 1), 250);
    return;
  }
  if (update.status === "available") showUpdateDialog(update);
  describeUpdate(update, manual);
  if (manual) $("#checkUpdateBtn").disabled = false;
}
async function checkForUpdates(manual = false) {
  if (manual) {
    $("#checkUpdateBtn").disabled = true;
    $("#updateCheckHint").textContent = "正在后台检查官网，失败时自动尝试 GitHub…";
  }
  await api("/api/update/check", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ force: manual }),
  }).catch(() => null);
  pollUpdate(manual);
}
$("#checkUpdateBtn").addEventListener("click", () => checkForUpdates(true));
$("#updateLaterBtn").addEventListener("click", hideUpdateDialog);
$("#updateIgnoreBtn").addEventListener("click", async () => {
  if (!state.update?.latest_version) return hideUpdateDialog();
  await api("/api/update/ignore", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ version: state.update.latest_version }),
  });
  hideUpdateDialog();
  $("#updateCheckHint").textContent = `已选择不升级 ${state.update.latest_version}。`;
});
$("#updateOpenBtn").addEventListener("click", async () => {
  if (!state.update?.source) return;
  await api("/api/update/open", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source: state.update.source }),
  });
  hideUpdateDialog();
});
$("#setTheme").addEventListener("change", async (e) => {
  const theme = e.target.value || "system";
  const previous = state.settings?.theme_mode || "system";
  applyTheme(theme);
  try {
    state.settings = await api("/api/settings", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ theme_mode: theme }),
    });
    $("#settingsHint").textContent = "主题已保存";
    setTimeout(() => {
      if ($("#settingsHint").textContent === "主题已保存") {
        $("#settingsHint").textContent = "";
      }
    }, 2000);
  } catch (error) {
    $("#setTheme").value = previous;
    applyTheme(previous);
    alert(`主题保存失败: ${error.message}`);
  }
});
$("#saveSettingsBtn").addEventListener("click", async () => {
  const body = {
    model_roots: $("#setModelRoots").value.split("\n").map((x) => x.trim()).filter(Boolean),
    default_device: $("#setDefaultDevice").value,
    theme_mode: $("#setTheme").value,
  };
  try {
    state.settings = await api("/api/settings", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    $("#settingsHint").textContent = "已保存,重新扫描模型…";
    applyTheme($("#setTheme").value);
    $("#aboutCccp").textContent = state.settings.cccp_engine_path ? "内置 · 已探测" : "未探测到";
    await loadModels();
    setTimeout(() => ($("#settingsHint").textContent = ""), 3000);
  } catch (e) { alert(e.message); }
});

function syncApiKeyControls() {
  const enabled = $("#setApiAuthEnabled").checked;
  $("#setApiKey").disabled = !enabled;
  $("#toggleApiKeyBtn").disabled = !enabled;
  $("#copyApiKeyBtn").disabled = !enabled || !$("#setApiKey").value;
}

async function copyApiKey(value = $("#setApiKey").value) {
  if (!value) return alert("当前没有 API Key");
  try {
    await navigator.clipboard.writeText(value);
  } catch {
    const input = $("#setApiKey");
    input.disabled = false; input.select(); document.execCommand("copy"); syncApiKeyControls();
  }
  $("#apiKeyHint").textContent = "API Key 已复制";
  setTimeout(() => { if ($("#apiKeyHint")) $("#apiKeyHint").textContent = ""; }, 2000);
}

$("#setApiAuthEnabled").addEventListener("change", syncApiKeyControls);
$("#setApiKey").addEventListener("input", syncApiKeyControls);
$("#toggleApiKeyBtn").addEventListener("click", () => {
  const input = $("#setApiKey");
  input.type = input.type === "password" ? "text" : "password";
  $("#toggleApiKeyBtn").textContent = input.type === "password" ? "显示" : "隐藏";
});
$("#generateApiKeyBtn").addEventListener("click", async () => {
  try {
    const data = await api("/api/settings/api-key/generate", { method: "POST" });
    $("#setApiAuthEnabled").checked = true;
    $("#setApiKey").value = data.api_key;
    $("#setApiKey").type = "text";
    $("#toggleApiKeyBtn").textContent = "隐藏";
    $("#apiKeyHint").textContent = "已生成，请点击“保存鉴权”使其生效";
    syncApiKeyControls();
  } catch (error) { alert(error.message); }
});
$("#copyApiKeyBtn").addEventListener("click", () => copyApiKey());
$("#apiCopyKeyBtn").addEventListener("click", () => copyApiKey(state.settings?.cccp_api_key || ""));
$("#saveApiKeyBtn").addEventListener("click", async () => {
  const enabled = $("#setApiAuthEnabled").checked;
  const key = $("#setApiKey").value.trim();
  if (enabled && !key) return alert("启用鉴权前请生成或填写 API Key");
  try {
    state.settings = await api("/api/settings", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ cccp_api_key: enabled ? key : "" }),
    });
    $("#setApiKey").value = state.settings.cccp_api_key || "";
    $("#apiKeyHint").textContent = enabled ? "鉴权已保存并生效" : "鉴权已关闭";
    syncApiKeyControls();
    refreshApiInfo();
  } catch (error) { alert(error.message); }
});

/* ---------- 聊天页 ---------- */
function newSessionId() { return `s-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`; }

function addMsg(role, content, extra = {}) {
  const box = $("#messages");
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.innerHTML = `<div class="role">${role === "user" ? "你" : "模型"}</div>${role === "assistant" ? `<details class="reasoning" ${extra.reasoning_content ? "" : "hidden"}><summary>思考过程（不会作为正文回传）</summary><div class="reasoning-body"></div></details>` : ""}<div class="body"></div>${role === "assistant" ? `<div class="msg-metrics" ${extra.metrics ? "" : "hidden"}></div>` : ""}`;
  el.querySelector(".body").textContent = content;
  if (extra.reasoning_content && el.querySelector(".reasoning-body")) el.querySelector(".reasoning-body").textContent = extra.reasoning_content;
  if (extra.metrics && el.querySelector(".msg-metrics")) el.querySelector(".msg-metrics").textContent = metricsText(extra.metrics);
  box.appendChild(el);
  box.scrollTop = box.scrollHeight;
  return el;
}

function cleanAssistantText(text) {
  return String(text)
    .replace(/<｜end▁of▁sentence｜>/g, "")
    .replace(/<\|(?:end_of_sentence|eot_id|endoftext)\|>/g, "")
    .trimEnd();
}

function renderMessages() {
  $("#messages").innerHTML = "";
  for (const m of state.messages) addMsg(m.role, m.content, m);
  syncBubbleUndo();
}

function syncBubbleUndo() {
  $$("#messages .bubble-undo").forEach((button) => button.remove());
  if (state.abort || !state.messages.length) return;
  const bubbles = $$("#messages .msg");
  const bubble = bubbles[state.messages.length - 1];
  if (!bubble) return;
  const button = document.createElement("button");
  button.className = "bubble-undo";
  button.textContent = state.messages.at(-1)?.role === "assistant" ? "回退本轮" : "撤回消息";
  button.title = state.messages.at(-1)?.role === "assistant"
    ? "删除最后一轮用户消息和模型回复"
    : "模型尚未回复，删除最后一条用户消息";
  button.addEventListener("click", undoLastTurn);
  bubble.appendChild(button);
}

async function undoLastTurn() {
  if (state.abort) return alert("请先停止当前生成");
  if (!state.messages.length) return;
  // 正常问答删掉 assistant + 它前面的 user；只有 user 时也能单独撤回。
  if (state.messages.at(-1)?.role === "assistant") state.messages.pop();
  if (state.messages.at(-1)?.role === "user") state.messages.pop();
  renderMessages();
  await saveSession();
  $("#chatInput").focus();
}

function apiMessages() {
  return state.messages.map(({ role, content }) => ({ role, content }));
}

function metricsText(metric) {
  if (!metric || !Object.keys(metric).length) return "尚无生成记录";
  const prefill = metric.prefill_ms == null ? "复用/无新增" : `${Number(metric.prefill_ms).toFixed(0)} ms`;
  const ttft = metric.ttft_ms == null ? "—" : `${Number(metric.ttft_ms).toFixed(0)} ms`;
  const kvLabels = {
    hot: "KV 续接", reuse: "KV 复用", prefix: "KV 前缀复用", cold: "KV 冷启动",
    "exact-prefix": "KV 续接", "lcp-replay": "KV 前缀复用", "full-prefill": "KV 冷启动",
  };
  const kv = kvLabels[metric.kv_mode] || `KV ${metric.kv_mode || "未知"}`;
  return `${Number(metric.tokens_per_second || 0).toFixed(2)} tok/s · 首字 ${ttft} · ${kv} · 输入 ${metric.processed_tokens ?? "—"}/${metric.prompt_tokens ?? "—"} tokens · 预填充 ${prefill}`;
}

async function refreshChatMetrics(bubble = null, requestId = "") {
  for (let attempt = 0; attempt < 5; attempt++) {
    try {
      const suffix = requestId ? `?request_id=${encodeURIComponent(requestId)}` : "";
      const { metrics } = await api(`/api/chat/metrics${suffix}`);
      if (metrics && Object.keys(metrics).length) {
        if (bubble) {
          const line = bubble.querySelector(".msg-metrics");
          line.hidden = false;
          line.textContent = metricsText(metrics);
        }
        return metrics;
      }
    } catch { /* 指标不影响聊天正文 */ }
    if (attempt < 4) await new Promise((resolve) => setTimeout(resolve, 120));
  }
  return null;
}

async function saveSession() {
  if (!state.messages.length) {
    await api(`/api/chat/sessions/${state.sessionId}`, { method: "DELETE" }).catch(() => {});
    refreshSessions();
    return;
  }
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
        <button class="icon-btn session-delete" data-del="${esc(s.id)}" title="删除会话" aria-label="删除会话 ${esc(s.title)}"><svg class="ic"><use href="#i-trash"/></svg></button>
      </div>`).join("") || `<div class="empty-placeholder compact"><span>暂无历史会话</span></div>`;
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
  resizeChatInput(true);
  state.messages.push({ role: "user", content: text });
  $$("#messages .bubble-undo").forEach((button) => button.remove());
  addMsg("user", text);
  const bubble = addMsg("assistant", "");
  bubble.classList.add("streaming");
  state.abort = new AbortController();
  syncSendButton();

  const payload = {
    model: state.instance?.served_model_name || "winui-model",
    messages: apiMessages(),
    temperature: +$("#tempInput").value,
    max_tokens: +$("#maxTokInput").value,
    repetition_penalty: +$("#repPenaltyInput").value,
    presence_penalty: +$("#presencePenaltyInput").value,
    stream: true,
    profile_ids: activeChatProfileIds(),
  };
  const thinking = currentThinkingLevel();
  if (thinking === "off") { payload.thinking = false; payload.reasoning_effort = "none"; }
  else if (thinking === "auto" && state.thinkingDefaultEffort) {
    payload.thinking = true;
    payload.reasoning_effort = state.thinkingDefaultEffort;
  } else if (thinking === "on") {
    payload.thinking = true;
  } else if (thinking !== "auto") {
    payload.thinking = true;
    payload.reasoning_effort = thinking;
  }
  let answer = "";
  let reasoning = "";
  let rawAnswer = "";
  let streamError = "";
  let requestId = "";
  let streamDone = false;
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
        if (d === "[DONE]") { streamDone = true; continue; }
        try {
          const j = JSON.parse(d);
          if (j.id) requestId = j.id;
          if (j.error) {
            streamError = j.error?.message || (typeof j.error === "string" ? j.error : JSON.stringify(j.error));
            continue;
          }
          const choice = j.choices?.[0] || {};
          const delta = choice.delta?.content || choice.message?.content || "";
          const thought = choice.delta?.reasoning_content || choice.message?.reasoning_content || "";
          if (thought) {
            reasoning += thought;
            const panel = bubble.querySelector(".reasoning");
            panel.hidden = false;
            panel.querySelector(".reasoning-body").textContent = reasoning;
          }
          if (delta) {
            rawAnswer += delta;
            answer = cleanAssistantText(rawAnswer);
            bubble.querySelector(".body").textContent = answer;
          }
        } catch { /* 半包忽略 */ }
      }
      $("#messages").scrollTop = $("#messages").scrollHeight;
    }
    if (streamError) throw new Error(streamError);
    if (!streamDone) throw new Error("生成连接未正常完成，请查看终端中的详细错误");
    await new Promise((resolve) => setTimeout(resolve, 80));
    const metrics = await refreshChatMetrics(bubble, requestId);
    if (answer || reasoning) {
      state.messages.push({ role: "assistant", content: answer, reasoning_content: reasoning, metrics });
    } else {
      // 后端正常结束但没有产生任何可见内容时，保留用户消息本身，
      // 回退按钮会落在该 user 气泡上，避免出现无法撤回的空回复。
      bubble.remove();
    }
    await saveSession();
  } catch (e) {
    if (e.name === "AbortError") {
      // 引擎会在取消时提交已经生成的 KV；前端也保留同一段部分回复，
      // 这样下一轮恰好沿用相同前缀，而不是因为历史不一致被迫全量重算。
      if (answer || reasoning) {
        await new Promise((resolve) => setTimeout(resolve, 80));
        const metrics = await refreshChatMetrics(bubble, requestId);
        state.messages.push({
          role: "assistant", content: answer,
          reasoning_content: reasoning, metrics, interrupted: true,
        });
        await saveSession();
      } else {
        bubble.remove();
        renderMessages();
      }
    } else {
      bubble.remove();
      renderMessages();
      const last = $("#messages .msg:last-child");
      if (last) {
        const error = document.createElement("span");
        error.className = "msg-error";
        error.textContent = `生成失败：${e.message}`;
        last.insertBefore(error, last.querySelector(".bubble-undo"));
      }
    }
  }
  bubble.classList.remove("streaming");
  state.abort = null;
  syncSendButton();
  syncBubbleUndo();
}
function syncSendButton() {
  const button = $("#sendBtn");
  button.textContent = state.abort ? "停止" : "发送";
  button.classList.toggle("stopping", !!state.abort);
  button.title = state.abort ? "停止当前生成" : "发送消息";
}
$("#sendBtn").addEventListener("click", () => state.abort ? state.abort.abort() : send());
const CHAT_INPUT_MIN = 46;
const CHAT_INPUT_MAX = 180;
const THINKING_LEVELS = [
  ["auto", "自动", "自动识别"], ["off", "关闭", "关闭"],
  ["low", "低", "低强度"], ["medium", "中", "中等强度"],
  ["high", "高", "高强度"], ["max", "最大", "最大强度"],
];
const THINKING_LABELS = Object.fromEntries(THINKING_LEVELS.map((item) => [item[0], item]));
let activeThinkingLevels = [...THINKING_LEVELS];
function currentThinkingLevel() {
  return activeThinkingLevels[Number($("#thinkingRange").value) || 0]?.[0] || "auto";
}
function syncThinkingControl() {
  const [, short, detail] = activeThinkingLevels[Number($("#thinkingRange").value) || 0] || activeThinkingLevels[0];
  $("#thinkingBtn").textContent = `思考 · ${short}`;
  $("#thinkingValue").textContent = currentThinkingLevel() === "auto" && state.thinkingDefaultEffort
    ? `${detail} · 模型默认${THINKING_LABELS[state.thinkingDefaultEffort]?.[1] || state.thinkingDefaultEffort}`
    : detail;
  localStorage.setItem("thinking-level", currentThinkingLevel());
}
function setThinkingLevels(levels, defaultEffort = null) {
  const previous = currentThinkingLevel();
  activeThinkingLevels = levels.length ? levels : [THINKING_LABELS.auto, THINKING_LABELS.off];
  state.thinkingDefaultEffort = defaultEffort && activeThinkingLevels.some(([value]) => value === defaultEffort)
    ? defaultEffort : null;
  const range = $("#thinkingRange");
  range.max = String(activeThinkingLevels.length - 1);
  const restored = activeThinkingLevels.findIndex(([value]) => value === previous);
  range.value = String(restored >= 0 ? restored : 0);
  const scale = $("#thinkingScale");
  scale.replaceChildren(...activeThinkingLevels.map(([, short]) => {
    const span = document.createElement("span");
    span.textContent = short;
    return span;
  }));
  scale.style.gridTemplateColumns = `repeat(${activeThinkingLevels.length},1fr)`;
  syncThinkingControl();
}
async function refreshThinkingCapabilities() {
  const response = await api("/api/chat/models");
  const model = response?.data?.[0] || response?.models?.[0] || null;
  if (!model) return;
  const advertised = model.think_efforts?.valid_efforts || model.support_efforts || [];
  const supported = [...new Set(advertised)].filter((value) => THINKING_LABELS[value]);
  const levels = [THINKING_LABELS.auto, THINKING_LABELS.off];
  if (supported.length) levels.push(...supported.map((value) => THINKING_LABELS[value]));
  else if (model.supports_reasoning || model.reasoning) levels.push(["on", "开启", "模型思考"]);
  const defaultEffort = model.think_efforts?.default_effort || model.default_effort || null;
  setThinkingLevels(levels, defaultEffort);
  state.thinkingCapabilityLoaded = true;
  $("#thinkingHint").textContent = defaultEffort
    ? `自动使用模型声明的默认强度“${THINKING_LABELS[defaultEffort]?.[1] || defaultEffort}”；当前模型仅显示实际支持的档位。`
    : "自动采用当前模型的默认思考能力；当前模型仅显示实际支持的档位。";
}
function closeThinkingPopover() {
  $("#thinkingPopover").hidden = true;
  $("#thinkingBtn").setAttribute("aria-expanded", "false");
}
const savedThinking = localStorage.getItem("thinking-level");
const savedThinkingIndex = activeThinkingLevels.findIndex(([value]) => value === savedThinking);
$("#thinkingRange").value = String(savedThinkingIndex >= 0 ? savedThinkingIndex : 0);
syncThinkingControl();
$("#thinkingBtn").addEventListener("click", (event) => {
  event.stopPropagation();
  const open = $("#thinkingPopover").hidden;
  $("#thinkingPopover").hidden = !open;
  $("#thinkingBtn").setAttribute("aria-expanded", String(open));
});
$("#thinkingPopover").addEventListener("click", (event) => event.stopPropagation());
$("#thinkingRange").addEventListener("input", syncThinkingControl);
document.addEventListener("click", closeThinkingPopover);
document.addEventListener("keydown", (event) => { if (event.key === "Escape") closeThinkingPopover(); });
function resizeChatInput(reset = false) {
  const input = $("#chatInput");
  input.style.height = reset ? `${CHAT_INPUT_MIN}px` : "auto";
  const height = reset ? CHAT_INPUT_MIN : Math.min(CHAT_INPUT_MAX, Math.max(CHAT_INPUT_MIN, input.scrollHeight));
  input.style.height = `${height}px`;
  input.style.overflowY = input.scrollHeight > CHAT_INPUT_MAX ? "auto" : "hidden";
}
$("#chatInput").addEventListener("input", () => resizeChatInput());
$("#chatInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); }
});
$("#clearChatBtn").addEventListener("click", () => {
  state.messages = []; $("#messages").innerHTML = "";
  api(`/api/chat/sessions/${state.sessionId}`, { method: "DELETE" }).catch(() => {});
  refreshSessions();
});
$("#newSessionBtn").addEventListener("click", startNewSession);

/* 聊天侧栏实例信息 */
function renderChatInstance(instance) {
  $("#chatInstance").textContent = instance
    ? `模型 ID: ${instance.served_model_name}\nport ${instance.port} · ${instance.full_model ? "全量专家（无配置路由限制）" : `profiles: ${(instance.profiles || []).join(", ")}`}\n目录: ${instance.model}`
    : "未启动模型";
  renderChatProfiles();
}
setInterval(async () => {
  try {
    const s = await api("/api/launch/status");
    state.instance = s.instance;
    renderChatInstance(s.instance);
  } catch { /* 静默 */ }
}, 8000);

/* ---------- 训练页 ---------- */
async function refreshTraining() {
  const { files } = await api("/api/training/corpus");
  const names = new Set(files.map((file) => file.name));
  state.corpusSelected = new Set([...state.corpusSelected].filter((name) => names.has(name)));
  if (!state.corpusSelected.size) files.forEach((file) => state.corpusSelected.add(file.name));
  $("#corpusCount").textContent = `${files.length} 个文件`;
  $("#corpusList").innerHTML = files.map((f) => `
    <div class="file-item">
      <input class="file-check" type="checkbox" data-corpus-select="${esc(f.name)}" ${state.corpusSelected.has(f.name) ? "checked" : ""} aria-label="本次使用 ${esc(f.name)}">
      <div class="file-main"><span class="file-name" title="${esc(f.name)}">${esc(f.name)}</span>
        <span class="file-meta">${esc(f.format || "未知")} · ${Number(f.samples || 0).toLocaleString()} 条对话 · ${Number(f.messages || 0).toLocaleString()} 条消息 · ${Number(f.characters || 0).toLocaleString()} 字符 · ${formatBytes(f.bytes)}${f.invalid_lines ? ` · ${f.invalid_lines} 行无效` : ""}<br>角色：${esc((f.roles || []).join(" / ") || "user")} · 最长记录 ${Number(f.max_sample_characters || 0).toLocaleString()} 字符 · ${esc(f.stored_path || "data/corpus")}</span></div>
      <button class="icon-btn file-delete" data-cdel="${esc(f.name)}" title="删除语料" aria-label="删除语料 ${esc(f.name)}"><svg class="ic"><use href="#i-trash"/></svg></button>
    </div>`).join("") || `<div class="empty-placeholder"><b>暂无语料</b><span>上传后将保存在应用语料库中。</span></div>`;
  $$('[data-corpus-select]').forEach((box) => box.addEventListener("change", () => {
    box.checked ? state.corpusSelected.add(box.dataset.corpusSelect) : state.corpusSelected.delete(box.dataset.corpusSelect);
  }));
  $$("[data-cdel]").forEach((b) =>
    b.addEventListener("click", async () => {
      if (!confirm(`确定删除语料“${b.dataset.cdel}”？\n将从应用 data/corpus 中永久移除。`)) return;
      await api(`/api/training/corpus/${encodeURIComponent(b.dataset.cdel)}`, { method: "DELETE" });
      state.corpusSelected.delete(b.dataset.cdel);
      refreshTraining();
    }));
  const { jobs } = await api("/api/training/jobs");
  $("#jobCount").textContent = `${jobs.length} 个`;
  renderJobs(jobs);
  if (state.currentJob && jobs.some((job) => job.id === state.currentJob.id)) {
    await showJob(state.currentJob.id);
  }
}

function formatBytes(bytes) {
  const value = Number(bytes || 0);
  if (value >= 1024 ** 3) return `${(value / 1024 ** 3).toFixed(2)} GiB`;
  if (value >= 1024 ** 2) return `${(value / 1024 ** 2).toFixed(1)} MiB`;
  return `${(value / 1024).toFixed(1)} KiB`;
}

function trainingVisualProgress(job) {
  const budget = Math.max(1, Number(job.token_budget || 0));
  const processed = Math.max(0, Number(job.processed_tokens || 0));
  let progressTokens = processed;
  const message = String(job.message || "");
  const layerFirst = message.match(/层\s+(\d+)\/(\d+)\s+·\s+块\s+(\d+)\/(\d+)\s+·\s+token\s+(\d+)/);
  const layer = message.match(/块\s+\d+\/\d+\s+·\s+层\s+(\d+)\/(\d+)/);
  if (layerFirst && processed < budget) {
    const currentLayer = Number(layerFirst[1]), layerCount = Math.max(1, Number(layerFirst[2]));
    const currentBlock = Number(layerFirst[3]), blockCount = Math.max(1, Number(layerFirst[4]));
    const documentTokens = Math.min(Number(layerFirst[5]), budget - processed);
    progressTokens += documentTokens * ((currentLayer - 1) + Math.min(1, currentBlock / blockCount)) / layerCount;
  } else if (layer && processed < budget) {
    const currentLayer = Number(layer[1]), layerCount = Math.max(1, Number(layer[2]));
    progressTokens += Math.min(4096, budget - processed) * Math.min(1, currentLayer / layerCount);
  }
  const calculated = .08 + .82 * progressTokens / budget;
  return Math.max(0, Math.min(1, Math.max(Number(job.progress || 0), calculated)));
}

function trainingLayerStatus(job) {
  const message = String(job.message || "");
  const budget = Math.max(1, Number(job.token_budget || 0));
  const processed = Math.max(0, Number(job.processed_tokens || 0));
  const totalBlocks = Math.max(1, Math.ceil(budget / 4096));
  const layerFirst = message.match(/层\s*(\d+)\/(\d+).*?块\s*(\d+)\/(\d+)/);
  if (layerFirst && /分层 prefill/.test(message)) {
    return `当前计算=第 ${Number(layerFirst[1])}/${Number(layerFirst[2])} 层 · 当前层第 ${Number(layerFirst[3])}/${Number(layerFirst[4])} 块`;
  }
  const parsed = message.match(/块\s*(\d+)\/(\d+).*?层\s*(\d+)\/(\d+)/);
  if (parsed) {
    const block = Number(parsed[1]), blocks = Number(parsed[2]);
    const completedLayer = Number(parsed[3]), layers = Number(parsed[4]);
    const next = completedLayer < layers
      ? `正在计算第 ${completedLayer + 1}/${layers} 层`
      : "本块层计算完成";
    return `当前计算=第 ${block}/${blocks} 块 · 已完成 ${completedLayer}/${layers} 层 · ${next}`;
  }
  if (job.status === "running" && /模型已加载|准备 prefill/.test(message)) {
    const block = Math.min(totalBlocks, Math.floor(processed / 4096) + 1);
    return `当前计算=第 ${block}/${totalBlocks} 块 · 正在计算第 1/${Number(job.layers || 1)} 层`;
  }
  return `当前阶段=${message || "等待任务状态"}`;
}

function jobViewLabel(job, detail) {
  if (state.currentJob?.id === job.id && !detail.hidden) return "收起详情";
  return Number(job.processed_tokens || 0) > 0 || job.status === "done" ? "查看热力图" : "查看进度";
}

function jobCardMarkup(job) {
  return `<div class="job" data-jid="${job.id}">
    <div class="jhead"><span class="jtitle"></span><span class="jstatus"></span></div>
    <div class="progress"><div></div></div>
    <div class="jmsg"></div><div class="jmsg"></div>
    <div class="job-actions">
      <button class="ghost" data-job-view="${job.id}"></button>
      <button class="ghost" data-job-cancel="${job.id}"><svg class="ic"><use href="#i-stop"/></svg>停止</button>
    </div>
  </div>`;
}

function updateJobCard(card, job, detail) {
  const visualProgress = trainingVisualProgress(job);
  const statusText = { pending: "排队", running: "运行", done: "完成", failed: "失败", cancelled: "已停止" }[job.status] || job.status;
  const visibleStatus = job.status === "running" ? `${statusText} ${Math.round(visualProgress * 100)}%` : statusText;
  const active = ["running", "pending"].includes(job.status);
  card.className = `job ${job.status}`;
  card.dataset.jid = job.id;
  const title = card.querySelector(".jtitle");
  title.textContent = job.profile_name || `Token 扫描 · ${job.id}`;
  title.title = job.profile_name || job.id;
  const status = card.querySelector(".jstatus");
  status.className = `jstatus ${job.status}`;
  status.textContent = visibleStatus;
  card.querySelector(".progress > div").style.width = `${(visualProgress * 100).toFixed(1)}%`;
  const messages = card.querySelectorAll(".jmsg");
  messages[0].textContent = `${job.mode === "disk" ? "强制硬盘" : "自动高速/容量降级"} · ${job.model_name || "未记录模型"} · ${Number(job.processed_tokens || 0).toLocaleString()} / ${Number(job.token_budget || 0).toLocaleString()} token`;
  const savedCount = (job.registered_profiles || []).length;
  messages[1].textContent = `${job.message || ""}${savedCount ? ` · 已保存 ${savedCount} 个配置` : (job.registered_profile_id ? ` · 已保存 1 个配置` : "")}`;
  const view = card.querySelector("[data-job-view]");
  view.dataset.jobView = job.id;
  view.textContent = jobViewLabel(job, detail);
  const cancel = card.querySelector("[data-job-cancel]");
  cancel.dataset.jobCancel = job.id;
  cancel.hidden = !active;
  let remove = card.querySelector("[data-job-delete]");
  if (active && remove) remove.remove();
  if (!active && !remove) {
    card.querySelector(".job-actions").insertAdjacentHTML("beforeend", `<button class="danger" data-job-delete="${job.id}">删除</button>`);
  }
}

function renderJobs(jobs) {
  const jobList = $("#jobList");
  const jobDetail = $("#jobDetail");
  // 详情成为所选任务卡的紧邻项；任务卡自身按 ID 原位更新，避免每次
  // 轮询重建节点导致进度扫光动画从第 0 帧重新闪烁。
  if (jobList.contains(jobDetail)) jobList.after(jobDetail);
  const keep = new Set(jobs.map((job) => job.id));
  jobList.querySelector(".empty-placeholder")?.remove();
  jobList.querySelectorAll(".job").forEach((card) => {
    if (!keep.has(card.dataset.jid)) card.remove();
  });
  for (const job of jobs) {
    let card = [...jobList.querySelectorAll(".job")].find((node) => node.dataset.jid === job.id);
    if (!card) {
      const holder = document.createElement("div");
      holder.innerHTML = jobCardMarkup(job);
      card = holder.firstElementChild;
    }
    updateJobCard(card, job, jobDetail);
    jobList.append(card);
  }
  if (!jobs.length) jobList.innerHTML = `<div class="empty-placeholder"><b>暂无扫描任务</b><span>选择模型和语料，完成 token 扫描后在这里查看热力图。</span></div>`;
  if (state.currentJob?.id && !jobDetail.hidden) {
    const selected = [...jobList.querySelectorAll(".job")].find((el) => el.dataset.jid === state.currentJob.id);
    if (selected) selected.after(jobDetail);
  }
  $$("#jobList .job").forEach((el) => {
    if (el.dataset.bound) return;
    el.dataset.bound = "1";
    el.addEventListener("click", () => toggleJobDetail(el.dataset.jid));
  });
  $$('[data-job-view]').forEach((button) => {
    if (button.dataset.bound) return;
    button.dataset.bound = "1";
    button.addEventListener("click", (event) => { event.stopPropagation(); toggleJobDetail(button.dataset.jobView); });
  });
  $$('[data-job-cancel]').forEach((button) => {
    if (button.dataset.bound) return;
    button.dataset.bound = "1";
    button.addEventListener("click", async (event) => {
    event.stopPropagation();
    if (!confirm("停止当前 token 扫描？已完成的整块统计会保留在运行记录中，但不能导出为正式配置。")) return;
    button.disabled = true;
    await api(`/api/training/jobs/${button.dataset.jobCancel}/cancel`, { method: "POST" });
    refreshTraining();
    });
  });
  $$('[data-job-delete]').forEach((button) => {
    if (button.dataset.bound) return;
    button.dataset.bound = "1";
    button.addEventListener("click", async (event) => {
    event.stopPropagation();
    if (!confirm("删除该训练任务记录？已保存到模型配置库的配置不会随任务删除。")) return;
    await api(`/api/training/jobs/${button.dataset.jobDelete}`, { method: "DELETE" });
    if (state.currentJob?.id === button.dataset.jobDelete) { state.currentJob = null; $("#jobDetail").hidden = true; }
    refreshTraining();
    });
  });
  const running = jobs.some((j) => j.status === "running" || j.status === "pending");
  const selectedTrainingModel = state.models.find((model) => model.path === $("#trainModelSelect")?.value);
  const routeTrainingUnavailable = !!selectedTrainingModel && selectedTrainingModel.supports_route_training === false;
  $("#trainStartBtn").disabled = running || routeTrainingUnavailable;
  if (running) setTimeout(refreshTraining, 2000);
}

async function toggleJobDetail(jid) {
  const detail = $("#jobDetail");
  if (state.currentJob?.id === jid && !detail.hidden) {
    detail.hidden = true;
    state.currentJob = null;
    refreshTraining();
    return;
  }
  await showJob(jid);
  refreshTraining();
}

async function showJob(jid) {
  const j = await api(`/api/training/jobs/${jid}`);
  state.currentJob = j;
  const d = $("#jobDetail");
  d.hidden = false;
  $("#jobDetailTitle").textContent = `任务 ${j.id}`;
  $("#jobDetailMeta").textContent =
    `模型=${j.model_name || "未记录"} (${j.model_version || "未知版本"}) · 状态=${{pending:"排队",running:"扫描中",done:"完成",failed:"失败",cancelled:"已停止"}[j.status] || j.status}\n` +
    `语料=${(j.corpus_files || []).join(", ")}\n` +
    `token=${Number(j.processed_tokens || 0).toLocaleString()} / ${Number(j.token_budget || 0).toLocaleString()} · 4096 token/块 · 对话=${Number(j.total_documents || 0).toLocaleString()} 条\n` +
    `${trainingLayerStatus(j)}\n` +
    (j.prefill_seconds ? `prefill=${Number(j.prefill_seconds).toFixed(1)} 秒 · ${Number(j.prefill_tokens_per_second || 0).toFixed(3)} token/s · 模型加载=${Number(j.model_load_seconds || 0).toFixed(1)} 秒\n` : "") +
    `激活专家=${Object.keys(j.counts || {}).length.toLocaleString()} · ${esc(j.message || "")}`;
  const done = j.status === "done";
  const hasHeatmap = Object.keys(j.counts || {}).length > 0;
  const canSaveProfile = done && hasHeatmap && (j.plan_keys || []).length > 0;
  $("#heatmapSection").hidden = !hasHeatmap;
  $(".final-profile-fields").hidden = !done;
  $(".expert-detail").hidden = !done;
  $("#coverageRange").disabled = !done;
  $("#finalProfileName").disabled = !done;
  $("#finalProfileDescription").disabled = !done;
  if (done) {
    $("#coverageRange").value = (Number(j.coverage_target || .95) * 100).toFixed(1);
    $("#coverageValue").textContent = `${(Number(j.actual_coverage || 0) * 100).toFixed(2)}%`;
    $("#coverageExperts").textContent = `${Number((j.plan_keys || []).length).toLocaleString()} 个`;
    $("#coverageDynamic").textContent = `${(Number(j.plan_bytes_mb || 0) / 1024).toFixed(2)} GiB`;
    $("#coverageFixed").textContent = `${Number(j.fixed_model_gib || 0).toFixed(2)} GiB`;
    $("#coverageTotal").textContent = `${(Number(j.fixed_model_gib || 0) + Number(j.plan_bytes_mb || 0) / 1024).toFixed(2)} GiB`;
    $("#finalProfileName").value = j.profile_name || $("#finalProfileName").value || "";
    $("#finalProfileDescription").value = j.profile_description || $("#finalProfileDescription").value || "";
  } else if (hasHeatmap) {
    $("#coverageValue").textContent = "扫描中";
    $("#coverageExperts").textContent = "等待完成";
    $("#coverageDynamic").textContent = "—";
    $("#coverageFixed").textContent = `${Number(j.fixed_model_gib || 0).toFixed(2)} GiB`;
    $("#coverageTotal").textContent = "等待选择";
  }
  if (hasHeatmap) drawExpertHeatmap(j);
  const keys = j.plan_keys || [];
  $("#jobExperts").innerHTML = keys.map((key) => `<div class="expert-cell">${esc(key)} · ${Number(j.plan_sizes_mb?.[key] || j.expert_size_mb || 0).toFixed(3)} MiB · hit ${j.counts?.[key] || 0}</div>`).join("") || `<div class="hint">尚无规划专家</div>`;
  $("#profileSaveActions").hidden = !canSaveProfile;
  $("#registerBtn").disabled = !canSaveProfile;
}

function drawExpertHeatmap(job) {
  const canvas = $("#expertHeatmap");
  const context = canvas.getContext("2d");
  const layerIds = Array.isArray(job.expert_layers) && job.expert_layers.length
    ? job.expert_layers.map(Number)
    : Array.from({ length: Number(job.layers || 1) }, (_, layer) => layer);
  const layers = layerIds.length, experts = Number(job.experts_per_layer || 1);
  const counts = job.counts || {}, selected = new Set(job.plan_keys || []);
  const layerMaxima = layerIds.map((layer) => {
    let maximum = 1;
    for (let expert = 0; expert < experts; expert++) {
      maximum = Math.max(maximum, Number(counts[`${layer}:${expert}`] || 0));
    }
    return maximum;
  });
  const cw = canvas.width / experts, ch = canvas.height / layers;
  context.clearRect(0, 0, canvas.width, canvas.height);
  for (let row = 0; row < layers; row++) {
    const layer = layerIds[row];
    for (let expert = 0; expert < experts; expert++) {
      const key = `${layer}:${expert}`, hits = Number(counts[key] || 0);
      const level = hits ? Math.log1p(hits) / Math.log1p(layerMaxima[row]) : 0;
      const hue = 18 + level * 27, light = 8 + level * 64;
      context.fillStyle = hits ? `hsl(${hue} 78% ${light}%)` : "#1a0e0b";
      context.fillRect(expert * cw, row * ch, Math.ceil(cw), Math.ceil(ch));
      if (selected.has(key) && cw >= 2) {
        context.fillStyle = "rgba(255,241,190,.38)";
        context.fillRect(expert * cw, row * ch, Math.ceil(cw), 1);
      }
    }
  }
  canvas.onmousemove = (event) => {
    const rect = canvas.getBoundingClientRect();
    const expert = Math.min(experts - 1, Math.floor((event.clientX - rect.left) / rect.width * experts));
    const row = Math.min(layers - 1, Math.floor((event.clientY - rect.top) / rect.height * layers));
    const layer = layerIds[row];
    const key = `${layer}:${expert}`, tip = $("#heatmapTip");
    tip.textContent = `L${layer} · E${expert} · ${Number(counts[key] || 0).toLocaleString()} 次${selected.has(key) ? " · 已选" : ""}`;
    tip.style.left = `${Math.min(rect.width - 130, event.clientX - rect.left + 10)}px`;
    tip.style.top = `${Math.max(4, event.clientY - rect.top - 28)}px`;
    tip.hidden = false;
  };
  canvas.onmouseleave = () => ($("#heatmapTip").hidden = true);
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

$("#trainStartBtn").addEventListener("click", async () => {
  const { files } = await api("/api/training/corpus");
  if (!files.length) return alert("先上传语料");
  const selectedFiles = files.filter((file) => state.corpusSelected.has(file.name));
  if (!selectedFiles.length) return alert("请至少勾选一个语料文件");
  const modelPath = $("#trainModelSelect").value;
  if (!modelPath) return alert("请先选择配置对应的模型");
  const body = {
    corpus_files: selectedFiles.map((f) => f.name),
    mode: document.querySelector('input[name="tmode"]:checked').value,
    token_budget: +$("#tokenBudget").value,
    model_path: modelPath,
  };
  try {
    const r = await api("/api/training/jobs", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    state.currentJob = r.job;
    primeTrainingTerminal(r.job);
    activateTab("terminal");
  } catch (e) { alert(e.message); }
});

$("#registerBtn").addEventListener("click", async () => {
  if (!state.currentJob) return;
  const name = $("#finalProfileName").value.trim();
  const description = $("#finalProfileDescription").value.trim();
  if (!name || !description) return alert("请填写配置名称和介绍后再保存");
  try {
    const r = await api(`/api/training/jobs/${state.currentJob.id}/register`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name, description }),
    });
    alert(`已保存配置：${r.profile.name}\n可继续调整覆盖率并更换名称，保存其他容量版本。`);
    await loadProfiles();
    await showJob(state.currentJob.id);
  } catch (e) { alert(e.message); }
});

let coverageTimer = null;
let coveragePlanning = false;
let coveragePendingValue = null;

async function flushCoveragePlan() {
  if (coveragePlanning || coveragePendingValue === null) return;
  if (!state.currentJob || state.currentJob.status !== "done") {
    coveragePendingValue = null;
    return;
  }
  const jobId = state.currentJob.id;
  const requestedCoverage = coveragePendingValue;
  coveragePendingValue = null;
  coveragePlanning = true;
  try {
    const result = await api(`/api/training/jobs/${jobId}/plan`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ coverage_percent: requestedCoverage }),
    });
    // A newer slider position may arrive while the previous plan is being
    // calculated.  Do not repaint with the stale result; the finally block
    // immediately submits the newest value, so the server also ends on the
    // last position instead of whichever request happened to finish last.
    if (coveragePendingValue !== null || state.currentJob?.id !== jobId) return;
    state.currentJob = result.job;
    await showJob(result.job.id);
    refreshTraining();
  } catch (error) {
    if (coveragePendingValue === null) alert(error.message);
  } finally {
    coveragePlanning = false;
    if (coveragePendingValue !== null) {
      clearTimeout(coverageTimer);
      coverageTimer = setTimeout(flushCoveragePlan, 0);
    }
  }
}

$("#coverageRange").addEventListener("input", (event) => {
  $("#coverageValue").textContent = `${Number(event.target.value).toFixed(1)}%`;
  coveragePendingValue = Number(event.target.value);
  clearTimeout(coverageTimer);
  coverageTimer = setTimeout(flushCoveragePlan, 280);
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
    pill.textContent = d.engine_ready ? "CCCP 就绪" : "CCCP 未就绪";
    pill.className = "state-pill " + (d.engine_ready ? "on" : "off");
    const auth = $("#apiAuthPill");
    auth.textContent = d.auth ? "已启用" : "未启用";
    auth.className = "state-pill " + (d.auth ? "on" : "off");
    $("#apiCopyKeyBtn").hidden = !d.auth || !state.settings?.cccp_api_key;
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
  await loadModels();
  refreshSessions();
  pollStatus();
  const h = await api("/api/health").catch(() => null);
  if (h?.version) $("#aboutVer").textContent = `v${h.version}`;
  await loadSettings().catch(() => {});
  loadSystem().catch(() => {});
  refreshHome();
  refreshDlJobs();
  checkForUpdates(false);
  const s = await api("/api/launch/status").catch(() => null);
  if (s?.instance) {
    state.instance = s.instance;
    renderChatInstance(s.instance);
  }
})();

/* ---------- 无边框自绘标题栏（仅 pywebview 原生窗口）---------- */
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
  const stickToBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
  el.textContent = lines && lines.length ? lines.join("\n") : empty;
  if (stickToBottom) el.scrollTop = el.scrollHeight;
}
async function refreshTerminal() {
  try {
    const data = await api("/api/terminal");
    const progress = data.progress || { percent: 0, label: "尚未启动", detail: "等待启动任务", phase: "idle" };
    const percent = Math.max(0, Math.min(100, Number(progress.percent || 0)));
    $("#termStage").textContent = progress.label;
    $("#termDetail").textContent = progress.detail || "";
    const indeterminate = Boolean(progress.indeterminate);
    $("#termPercent").textContent = indeterminate ? "编译中" : `${percent}%`;
    $("#termProgressBar").style.width = `${percent}%`;
    $("#termProgress").classList.toggle("indeterminate", indeterminate);
    if (indeterminate) {
      $("#termProgress").removeAttribute("aria-valuenow");
      $("#termProgress").setAttribute("aria-valuetext", `算子编译中，已用 ${progress.elapsed_s || 0} 秒`);
    } else {
      $("#termProgress").setAttribute("aria-valuenow", String(percent));
      $("#termProgress").removeAttribute("aria-valuetext");
    }
    $("#terminalStatus").dataset.state = progress.state || "idle";
    const instance = data.instance;
    const trainingJob = data.training_job;
    const memory = data.preflight?.memory;
    $("#termMeta").textContent = trainingJob
      ? `任务 ${trainingJob.id} · ${Number(trainingJob.processed_tokens || 0).toLocaleString()}/${Number(trainingJob.token_budget || 0).toLocaleString()} token · 4096 token/块`
      : instance
      ? `PID ${instance.pid} · 模型 ID ${instance.served_model_name} · port ${instance.port} · ${(instance.profiles || []).length} 个配置` +
        (memory ? ` · 运行估算 ${memory.total_estimate_gb}/${memory.device_capacity_gb ?? memory.limit_gb} GiB` : "")
      : "等待启动任务";
    renderTerm($("#termOutput"), data.lines, "暂无运行日志");
  } catch (e) { $("#termOutput").textContent = `读取失败: ${e.message}`; }
}
function startTerminal() { stopTerminal(); refreshTerminal(); termTimer = setInterval(refreshTerminal, 1500); }
function stopTerminal() { if (termTimer) { clearInterval(termTimer); termTimer = null; } }
$("#termRefreshBtn").addEventListener("click", refreshTerminal);

/* ---------- 独立推理环境（CPU / NVIDIA CUDA / AMD ROCm）---------- */
async function loadSystem() {
  const d = await api("/api/system");
  const runtimes = d.inference_runtimes || {};
  const base = `CPU ${d.cpu_count} 线程 · RAM ${d.ram_available_gb}/${d.ram_total_gb} GiB 可用 · 磁盘 ${d.disk_free_gb} GiB 可用 · 自动内存预检 · ${d.platform}`;
  const lines = ["cpu", "cuda", "amd"].map((backend) => {
    const item = runtimes[backend];
    if (!item) return `${backend.toUpperCase()}：未探测`;
    const stateText = item.ready ? "可用" : (item.installed ? "已安装/当前不可用" : "未安装");
    const details = [item.torch_version ? `torch ${item.torch_version}` : "", item.compute_runtime,
      item.device_name, item.device_memory_gb ? `${item.device_memory_gb} GiB` : ""].filter(Boolean).join(" · ");
    return `${item.label}：${stateText}${details ? ` · ${details}` : ""}\n  ${item.reason}`;
  });
  if ((d.display_adapters || []).length) lines.push(`显示适配器：${d.display_adapters.join(" | ")}`);
  $("#sysInfo").textContent = `${base}\n${lines.join("\n")}`;

  for (const selector of [$("#homeDeviceSelect"), $("#deviceSelect")]) {
    for (const option of selector.options) {
      const report = runtimes[option.value];
      option.disabled = option.value !== "cpu" && !report?.ready;
      option.title = report?.reason || "尚未完成运行环境自检";
    }
    if (selector.selectedOptions[0]?.disabled) selector.value = "cpu";
  }
}
