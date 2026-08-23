"use strict";

const REFRESH_MS = 5 * 60 * 1000;
const TIER_RANK = { high: 0, medium: 1, low: 2 };
const PROP_LABELS = { strikeouts: "Strikeouts", walks: "Walks", earned_runs: "Earned Runs" };

let state = { slate: null, sortKey: "mu", selectedKey: null, prop: "strikeouts", actionableOnly: false };

const $ = (sel) => document.querySelector(sel);
const keyOf = (p) => `${p.pitcher}-${p.game_pk}`;
const pct = (x) => (x == null ? "—" : (x * 100).toFixed(1) + "%");
const num = (x, d = 1) => (x == null ? "—" : Number(x).toFixed(d));
const signed = (x, d = 2) => (x == null ? "—" : (x >= 0 ? "+" : "") + Number(x).toFixed(d));
const americanOdds = (x) => (x == null ? "—" : (x > 0 ? "+" : "") + Math.round(x));

function actionabilityBadge(actionability, conviction) {
  if (!actionability) return "";
  const cv = conviction != null ? ` cv${Number(conviction).toFixed(1)}` : "";
  if (actionability === "lean_over") return `<span class="badge act-over">▲${cv}</span>`;
  if (actionability === "lean_under") return `<span class="badge act-under">▼${cv}</span>`;
  return `<span class="badge act-none" title="no edge (provisional threshold)">—${cv}</span>`;
}

async function fetchJSON(url) {
  const r = await fetch(url, { cache: "no-store" });
  return r.json();
}

async function loadSlate(date) {
  const params = new URLSearchParams();
  if (date) params.set("date", date);
  params.set("prop", state.prop);
  const slate = await fetchJSON(`/api/slate?${params.toString()}`);
  if (slate.error) { showBanner(slate.error); return; }
  state.slate = slate;
  render();
}

function showBanner(msg) {
  const b = $("#banner");
  if (!msg) { b.hidden = true; return; }
  b.hidden = false; b.textContent = msg;
}

function filteredPitchers() {
  const ps = state.slate.pitchers || [];
  if (!state.actionableOnly) return [...ps];
  return ps.filter((p) => {
    const a = p.line?.actionability;
    return a === "lean_over" || a === "lean_under";
  });
}

function sortedPitchers() {
  const ps = filteredPitchers();
  const k = state.sortKey;
  const val = (p) => {
    if (k === "mu") return p.mu ?? -1;
    if (k === "conviction") return p.line ? (p.line.conviction ?? -1) : -1;
    if (k === "edge") return p.line ? Math.abs(p.line.edge ?? 0) : -99;
    if (k === "p_over_line") return p.line ? (p.line.p_over ?? -1) : -1;
    if (k === "tier") return -(TIER_RANK[p.line?.tier] ?? 9);
    return p.mu ?? -1;
  };
  return ps.sort((a, b) => val(b) - val(a));
}

function renderKpis() {
  const k = state.slate.kpis;
  const age = k.model_age_days == null ? "—" : Number(k.model_age_days).toFixed(1) + "d";
  const cards = [
    ["Pitchers on slate", k.n_pitchers],
    ["Lines available", k.n_with_line],
    ["No line (sweep only)", k.n_no_line],
    ["Median vig", k.median_vig == null ? "—" : pct(k.median_vig)],
    ["Model age", age + (k.model_stale ? " ⚠" : "")],
  ];
  $("#kpis").innerHTML = cards.map(([l, v]) =>
    `<div class="kpi"><div class="label">${l}</div><div class="val mono">${v}</div></div>`).join("");
}

function renderSlate() {
  const ps = sortedPitchers();
  const filterNote = state.actionableOnly ? " · actionable only" : "";
  $("#slate-sub").textContent = `· ${ps.length} pitchers · sorted by ${$("#sort-select").selectedOptions[0].text.toLowerCase()}${filterNote}`;
  $("#slate").innerHTML = ps.map((p) => {
    const k = keyOf(p);
    const line = p.line
      ? `${num(p.line.line, 1)}`
      : `<span class="muted">—</span>`;
    const lean = p.line
      ? `<span class="${p.line.lean === "over" ? "over" : "under"}">${p.line.lean || ""} ${signed(p.line.edge)}</span>`
      : "";
    const tier = p.line ? `<span class="chip ${p.line.tier || "low"}">${p.line.tier || ""}</span>` : "";
    const actBadge = p.line ? actionabilityBadge(p.line.actionability, p.line.conviction) : "";
    return `<div class="srow ${k === state.selectedKey ? "sel" : ""}" data-key="${k}">
      <span class="nm">${p.name || "?"} <span class="muted">${p.team || ""} v ${p.opponent || ""}</span></span>
      <span class="mu mono">${num(p.mu, 2)}</span>
      <span class="ln mono">${line}</span>
      <span class="ed">${lean}</span>
      <span class="tr">${tier}</span>
      <span class="act">${actBadge}</span>
    </div>`;
  }).join("");
  $("#slate").querySelectorAll(".srow").forEach((el) =>
    el.addEventListener("click", () => { state.selectedKey = el.dataset.key; renderSlate(); renderDetail(); }));
}

function statTiles(s) {
  if (!s) return `<p class="muted">Form stats populate on the next refresh (run <code>python -m src.pipeline.refresh</code> after this pipeline update).</p>`;

  // ---- Weather / game-context chips ----
  let envChips = "";
  if (s.is_dome === 1.0) {
    envChips += `<span class="chip">🏟 Dome</span>`;
  } else if (s.temp_f != null) {
    const wx = [`${num(s.temp_f, 0)}°F`];
    if (s.wind_mph != null) wx.push(`${num(s.wind_mph, 0)} mph wind`);
    if (s.humidity != null) wx.push(`${num(s.humidity, 0)}% humidity`);
    envChips += `<span class="chip">🌤 ${wx.join(" · ")}</span>`;
  }
  if (s.game_total != null) {
    envChips += `<span class="chip">O/U ${num(s.game_total, 1)}</span>`;
  }
  if (s.is_favorite === 1.0) envChips += `<span class="chip">Favourite</span>`;
  else if (s.is_favorite === 0.0) envChips += `<span class="chip muted">Underdog</span>`;

  const envSection = envChips
    ? `<div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px">${envChips}</div>`
    : "";

  // ---- Pitcher form ----
  const formTiles = [
    ["Projected K (mu)", num(s.mu, 2)],
    ["Avg innings pitched", num(s.ip_avg_last5, 1)],
    ["Expected batters", num(s.bf_avg_last5, 1)],
    ["Avg pitches thrown", num(s.pitch_count_avg_last5, 0)],
    ["Pitcher K% L5", pct(s.k_rate_last5)],
    ["Pitcher K% season", pct(s.k_rate_season)],
    ["K% vs LHB", pct(s.k_rate_vs_LHB)],
    ["K% vs RHB", pct(s.k_rate_vs_RHB)],
    ["Whiff% L5", pct(s.whiff_rate_last5)],
    ["FB velo L5", num(s.velo_avg_last5, 1)],
    ["Rest days", num(s.rest_days, 0)],
  ];

  // ---- Plate-discipline skill features (Spec ②) ----
  const skillTiles = [
    ["SwStr% L5", pct(s.swstr_rate_last5)],
    ["CSW% L5",   pct(s.csw_rate_last5)],
    ["Putaway% L5", pct(s.putaway_rate_last5)],
    ["K-BB rate L5", pct(s.k_minus_bb_rate_last5)],
  ].filter(([, v]) => v !== "—");

  // ---- Matchup ----
  // Priority: lineup-weighted → L10 hand-split → season hand-split → raw L10 fallback.
  let oppKLabel, oppKVal;
  if (s.opponent_lineup_k_rate_vs_hand != null) {
    oppKLabel = "Opp K% vs hand (lineup)";
    oppKVal   = pct(s.opponent_lineup_k_rate_vs_hand);
  } else if (s.opponent_k_rate_vs_hand_last10 != null) {
    oppKLabel = "Opp K% vs hand (L10)";
    oppKVal   = pct(s.opponent_k_rate_vs_hand_last10);
  } else if (s.opponent_k_rate_vs_hand_season != null) {
    oppKLabel = "Opp K% vs hand (szn)";
    oppKVal   = pct(s.opponent_k_rate_vs_hand_season);
  } else {
    oppKLabel = "Opp K% L10 (↩ no hand split)";
    oppKVal   = pct(s.opponent_k_rate_last10);
  }

  const matchupTiles = [
    [oppKLabel, oppKVal],
    ["Opp K% L10 (team)", pct(s.opponent_k_rate_last10)],
    ["Park K factor", num(s.park_k_factor, 2)],
  ];

  // Ump factor only when meaningfully non-neutral (>1% deviation).
  if (s.ump_k_factor != null && Math.abs(s.ump_k_factor - 1.0) > 0.01) {
    const dir = s.ump_k_factor > 1.0 ? "↑ K-friendly" : "↓ K-suppressing";
    matchupTiles.push([`Ump factor (${dir})`, num(s.ump_k_factor, 2)]);
  }

  // ---- Booster second-opinion (Spec ④) ----
  const boosterTiles = [];
  if (s.booster_mu != null) {
    boosterTiles.push(["Booster μ", num(s.booster_mu, 2)]);
  }
  if (s.glm_booster_agreement != null) {
    const agr = s.glm_booster_agreement;
    const label = agr > 0 ? "▲ bullish" : agr < 0 ? "▼ bearish" : "↔ split";
    boosterTiles.push(["GLM·Boost agreement", label]);
  }

  const allTiles = [...formTiles, ...skillTiles, ...matchupTiles, ...boosterTiles];

  const grid = `<div class="statgrid">${allTiles.map(([l, v]) =>
    `<div class="stat"><div class="label">${l}</div><div class="val mono">${v}</div></div>`
  ).join("")}</div>`;

  const impNote = s.was_imputed
    ? `<p class="muted" style="font-size:12px;margin-top:10px">⚠ some opponent/park values fell back to the league mean (thin sample).</p>`
    : "";

  return envSection + grid + impNote;
}

function renderDetail() {
  const p = (state.slate.pitchers || []).find((x) => keyOf(x) === state.selectedKey);
  const box = $("#detail");
  if (!p) { box.className = "detail muted"; box.textContent = "Select a pitcher to see the probability ladder and decision stats."; return; }
  box.className = "detail";

  const lineThresh = p.line ? p.line.line_threshold : null;
  const ladder = (p.ladder || []).map((row) => {
    const isLine = lineThresh != null && row.threshold === lineThresh;
    const w = Math.round((row.p_over ?? 0) * 100);
    return `<div class="ladder-row ${isLine ? "lineat" : ""}">
      <span class="th">${row.threshold}+</span>
      <div class="track"><div class="fill" style="width:${w}%"></div></div>
      <span class="pct">${pct(row.p_over)}</span>
      <span class="tg">${isLine ? "← line" : (row.tier || "")}</span>
    </div>`;
  }).join("");

  let lineChips = `<span class="chip">no posted line — sweep only</span>`;
  if (p.line) {
    const leanCls = p.line.lean === "over" ? "over" : "under";
    const cvVal = p.line.conviction != null ? Number(p.line.conviction).toFixed(2) : "—";
    const pLo = pct(p.line.p_over_lo);
    const pHi = pct(p.line.p_over_hi);
    const actLabel = { lean_over: "▲ lean over", lean_under: "▼ lean under", no_action: "— no edge" }[p.line.actionability] || "—";
    const actCls = p.line.actionability === "lean_over" ? "over" : (p.line.actionability === "lean_under" ? "under" : "muted");
    const edgeVsMarket = p.line.edge != null ? signed(p.line.edge) : "—";
    lineChips = `
      <span class="chip ${leanCls}" style="background:${p.line.lean === "over" ? "var(--ok-bg)" : "var(--surface-2)"};color:${p.line.lean === "over" ? "var(--ok)" : "var(--muted)"}">lean ${p.line.lean || "—"} ${p.line.line_threshold || ""}+</span>
      <span class="chip">P(over) ${pct(p.line.p_over)}</span>
      <span class="chip">P(market) ${pct(p.line.p_market)}</span>
      <span class="chip" title="p_over − p_market (was vs a fixed 50% coinflip pre-migration)">edge vs market ${edgeVsMarket}</span>
      <span class="chip">O ${americanOdds(p.line.over_american)} / U ${americanOdds(p.line.under_american)}</span>
      <span class="chip muted" style="font-size:11px">vig ${pct(p.line.vig)}</span>
      <span class="chip">push ${num(p.line.push_mass, 2)}</span>
      <span class="chip ${p.line.tier || "low"}">tier ${p.line.tier || "—"}</span>
      <span class="chip ${actCls}" title="conviction = |p_over−market| / sd(p_over) · thresholds provisional">${actLabel} · cv ${cvVal}</span>
      <span class="chip muted" style="font-size:11px" title="P(over) at mu±eta_se bounds">P band [${pLo}, ${pHi}]</span>`;
  }

  box.innerHTML = `
    <div class="dhead">
      <div>
        <h2>${p.name || "?"}</h2>
        <div class="muted" style="font-size:13px">${p.team || ""} vs ${p.opponent || ""}${p.stats?.pitcher_throws ? " · " + p.stats.pitcher_throws + "HP" : ""}${p.line?.start_time ? " · " + p.line.start_time : ""}</div>
      </div>
      <div style="text-align:right">
        <div class="muted" style="font-size:12px">posted line</div>
        <div class="mono" style="font-weight:500">${p.line ? num(p.line.line, 1) : "—"}</div>
      </div>
    </div>
    <div class="lineblock">${lineChips}</div>
    <div class="section-label">P(over) by ${PROP_LABELS[state.prop] || state.prop} threshold</div>
    <div style="margin-bottom:14px">${ladder || '<span class="muted">no sweep</span>'}</div>
    <div class="section-label">Decision context <span class="muted">· existing windows (L5 / season)</span></div>
    ${statTiles(p.stats)}`;
}

function renderControls() {
  const propLabel = PROP_LABELS[state.prop] || state.prop;
  const title = $("#page-title");
  if (title) title.textContent = `${propLabel} projections`;

  const propSel = $("#prop-select");
  if (propSel && propSel.value !== state.prop) propSel.value = state.prop;

  const sel = $("#date-select");
  const dates = state.slate.available_dates || [state.slate.game_date];
  sel.innerHTML = dates.slice().reverse().map((d) =>
    `<option value="${d}" ${d === state.slate.game_date ? "selected" : ""}>${d}</option>`).join("");
  $("#date-pill").textContent = state.slate.game_date;
  const m = state.slate.kpis;
  $("#model-line").textContent = `Model age ${m.model_age_days == null ? "—" : Number(m.model_age_days).toFixed(1) + " days"}${m.model_stale ? " (stale — retrain due)" : ""} · source: Underdog Fantasy lines + pybaseball/StatsAPI`;
  showBanner(m.line_source_error ? `Underdog lines unavailable (${m.line_source_error}) — showing the threshold sweep only.` : null);
}

function render() {
  renderControls();
  renderKpis();
  renderSlate();
  renderDetail();
}

function init() {
  $("#refresh-mins").textContent = Math.round(REFRESH_MS / 60000) + "m";
  $("#sort-select").addEventListener("change", (e) => { state.sortKey = e.target.value; renderSlate(); });
  $("#actionable-filter").addEventListener("change", (e) => { state.actionableOnly = e.target.checked; state.selectedKey = null; renderSlate(); renderDetail(); });
  $("#date-select").addEventListener("change", (e) => { state.selectedKey = null; loadSlate(e.target.value); });
  $("#prop-select").addEventListener("change", (e) => { state.prop = e.target.value; state.selectedKey = null; loadSlate($("#date-select").value || null); });
  $("#refresh-btn").addEventListener("click", () => loadSlate($("#date-select").value || null));
  loadSlate(null);
  setInterval(() => loadSlate($("#date-select").value || null), REFRESH_MS);
}

init();
