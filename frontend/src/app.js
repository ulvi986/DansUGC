/* Ad Intelligence — single-flow dashboard (zero-build, same-origin API).
   Type an app name -> fetch its live ads -> run the agent panel -> show results. */
"use strict";

const API = ""; // same origin
const $ = (id) => document.getElementById(id);
const esc = (s) =>
  s == null ? "" : String(s).replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

const state = { current: null, busy: false };

/* ------------------------------- net ------------------------------------- */
async function api(path, opts) {
  const res = await fetch(API + path, opts);
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch (_) {}
    throw new Error(msg);
  }
  return res.status === 204 ? null : res.json();
}
function jpost(path, body) {
  return api(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

/* ------------------------------- ui utils -------------------------------- */
function toast(msg, isError) {
  const t = $("toast");
  t.textContent = msg;
  t.className = "toast show" + (isError ? " error" : "");
  setTimeout(() => (t.className = "toast"), 3400);
}
function note(msg, isError) {
  const n = $("searchNote");
  n.textContent = msg || "";
  n.className = "search-note" + (isError ? " error" : "");
}
function show(el) { $(el).classList.remove("hidden"); }
function hide(el) { $(el).classList.add("hidden"); }

function setStep(name, status /* active | done */) {
  document.querySelectorAll("#steps li").forEach((li) => {
    if (li.dataset.step === name) li.className = status;
  });
}
function resetSteps() {
  document.querySelectorAll("#steps li").forEach((li) => (li.className = ""));
}

/* ------------------------------- boot ------------------------------------ */
async function boot() {
  try {
    const h = await api("/health");
    const pill = $("modePill");
    if (h.mode === "llm") {
      pill.textContent = `${(h.provider || "llm").toUpperCase()} · ${h.model}`;
      pill.className = "pill llm";
    } else {
      pill.textContent = "Heuristic mode";
      pill.className = "pill heuristic";
    }
    if (!h.live_fetch) {
      note("Live fetch is off — set SCRAPECREATORS_API_KEY in .env.", true);
    }
  } catch (_) {
    $("modePill").textContent = "offline";
    $("modePill").className = "pill offline";
  }
  await loadRecent();
  wire();
}

function wire() {
  $("searchForm").addEventListener("submit", (e) => { e.preventDefault(); run(); });
  $("backBtn").addEventListener("click", goHome);
  document.querySelectorAll(".tab").forEach((t) =>
    t.addEventListener("click", () => switchTab(t.dataset.tab)));
}

/* ------------------------------- recent ---------------------------------- */
async function loadRecent() {
  try {
    const apps = await api("/apps");
    const box = $("recent");
    if (!apps.length) { box.innerHTML = ""; return; }
    box.innerHTML =
      '<span style="width:100%;font-size:12px;color:var(--ink-soft);margin-bottom:2px">Already analyzed</span>' +
      apps.slice(0, 8).map((a) =>
        `<button class="chip" data-app="${esc(a.app_name)}"><b>${esc(a.app_name)}</b> · ${a.ad_count} ads</button>`
      ).join("");
    box.querySelectorAll(".chip").forEach((c) =>
      c.addEventListener("click", () => {
        $("appNameInput").value = c.dataset.app;
        run(true); // already stored -> analyze directly
      }));
  } catch (_) { /* non-fatal */ }
}

/* ------------------------------- main flow ------------------------------- */
/* run(): fetch live ads for the typed name (unless already stored), then analyse. */
async function run(skipFetch) {
  if (state.busy) return;
  const name = $("appNameInput").value.trim();
  if (!name) { note("Type an app name first.", true); return; }

  state.busy = true;
  $("analyzeBtn").disabled = true;
  note("");
  hide("hero"); hide("results"); show("progress");
  resetSteps();

  try {
    if (!skipFetch) {
      $("progressTitle").textContent = `Fetching live ads for “${name}”`;
      $("progressSub").textContent = "Searching Meta & TikTok ad libraries…";
      setStep("fetch", "active");
      const f = await jpost("/fetch-ads", { app_name: name });
      setStep("fetch", "done");
      const breakdown = Object.entries(f.by_platform || {})
        .map(([k, v]) => `${k}: ${v}`).join(" · ") || "none";
      toast(`Fetched ${f.fetched_ads} ad(s) · ${breakdown}`);
    } else {
      setStep("fetch", "done");
    }

    $("progressTitle").textContent = "Analyzing the ads";
    $("progressSub").textContent = "Text, visual, feature, pattern, scoring & strategy agents…";
    setStep("analyze", "active");

    const out = await jpost("/analyze", { app_name: name, platforms: [] });

    setStep("analyze", "done");
    setStep("strategy", "done");
    await renderResult(out);
    await loadRecent();
    toast("Analysis complete.");
  } catch (e) {
    goHome();
    note("Error: " + e.message, true);
    toast(e.message, true);
  } finally {
    state.busy = false;
    $("analyzeBtn").disabled = false;
  }
}

function goHome() {
  hide("progress"); hide("results"); show("hero");
}

/* ------------------------------- render ---------------------------------- */
async function renderResult(out) {
  state.current = out;
  hide("hero"); hide("progress"); show("results");

  $("rTitle").textContent = out.app_name || "—";
  $("rSub").textContent =
    `${(out.platforms || []).join(", ") || "all platforms"} · ${out.analyzed_ads_count} ads analysed · run #${out.run_id ?? "—"}`;
  const pct = Math.round((out.confidence_score || 0) * 100);
  $("confRing").style.setProperty("--p", pct);
  $("confVal").textContent = pct + "%";

  const intel = out.intelligence || {};
  renderExecutive(intel);
  renderWinners(intel);
  renderClusters(intel);
  renderOpportunities(intel);
  renderStrategies(intel, out);
  renderMap(intel);
  renderEvidence(out);
  switchTab("executive");

  if (out.run_id != null) {
    try {
      const agents = await api(`/analysis-runs/${out.run_id}/agents`);
      renderAgents(agents);
      renderScoresInto(agents);
    } catch (_) {
      $("tab-agents").innerHTML = '<p class="muted">Agent outputs unavailable.</p>';
    }
  }
  window.scrollTo({ top: 0, behavior: "smooth" });
}

/* small shared helpers */
const jnice = (s) => String(s == null ? "" : s).replace(/_/g, " ");
function confChip(n) {
  n = Math.round(n || 0);
  const cls = n >= 75 ? "hi" : n >= 50 ? "mid" : "lo";
  return `<span class="cchip ${cls}">${n}% conf</span>`;
}
function emptyIntel() {
  return '<p class="muted">No market-intelligence layer for this run (it predates the intelligence engine, or the ad set was too small). Re-run the analysis to generate it.</p>';
}

/* ---- 8. Executive summary + Creative DNA -------------------------------- */
function renderExecutive(intel) {
  const ex = intel.executive_summary, dna = intel.creative_dna;
  if (!ex) { $("tab-executive").innerHTML = emptyIntel(); return; }
  const Q = (label, o) => !o ? "" : `
    <div class="exq">
      <div class="exq-h"><span class="exq-label">${label}</span>${confChip(o.confidence)}</div>
      <div class="exq-a">${esc(o.answer)}</div>
      ${(o.evidence || []).map((e) => `<div class="exq-ev">${esc(e)}</div>`).join("")}
    </div>`;
  const disc = ex.disclaimer || intel.disclaimer;
  $("tab-executive").innerHTML = `
    ${disc ? `<div class="disclaimer-banner">ⓘ ${esc(disc)}</div>` : ""}
    <div class="exec-headline">${esc(ex.headline)}</div>
    ${dna ? dnaCard(dna) : ""}
    <div class="exq-grid">
      ${Q("Strongest observed pattern", ex.what_is_winning)}
      ${Q("Why it scores highest (proxy)", ex.why_is_it_winning)}
      ${Q("What is saturated", ex.what_is_saturated)}
      ${Q("What is underused", ex.what_is_underused)}
      ${Q("What to test next", ex.what_to_test_next)}
      ${Q("Highest-confidence opportunity", ex.highest_confidence_opportunity)}
    </div>`;
}
function dnaCard(dna) {
  const cell = (k, o) => `<div class="dna-cell"><span class="dna-k">${k}</span><span class="dna-v">${esc(o.label)} <i>${o.support_pct}%</i></span></div>`;
  const t = dna.transformation || {};
  return `
    <div class="dna">
      <div class="dna-title">Creative DNA ${confChip(dna.confidence)}</div>
      <div class="dna-formula">${esc(dna.formula)}</div>
      <div class="dna-grid">
        ${cell("Hook", dna.hook)}${cell("Emotion", dna.emotion)}
        ${cell("Format", dna.format)}${cell("CTA", dna.cta)}
        <div class="dna-cell"><span class="dna-k">Transformation</span><span class="dna-v">${esc(t.from)} → ${esc(t.to)}</span></div>
      </div>
      ${(dna.low_frequency_signals || []).length ? `<div class="dna-lowfreq"><b>Low-frequency signals</b> (not dominant): ${(dna.low_frequency_signals).map((s) => `<span class="tag">${esc(s.label)} ${s.usage_pct}%</span>`).join("")}</div>` : ""}
    </div>`;
}

/* ---- 2. Winner patterns ------------------------------------------------- */
function renderWinners(intel) {
  const ws = intel.winner_patterns || [];
  if (!ws.length) { $("tab-winners").innerHTML = emptyIntel(); return; }
  $("tab-winners").innerHTML =
    '<p class="muted">Strongest <b>observed</b> creative patterns — ranked by creative-quality <b>proxy</b> lift and consistency, not by real performance (no spend/CTR/conversion data).</p>' +
    ws.map((w) => `
      <div class="wcard">
        <div class="wcard-h"><span class="wname">${esc(w.pattern)}</span>${confChip(w.confidence)}</div>
        <div class="wmeta">${w.ads_count} ads · ${w.frequency_pct}% · lift
          <b class="${w.score_lift >= 0 ? "pos" : "neg"}">${w.score_lift >= 0 ? "+" : ""}${w.score_lift} pts</b></div>
        <div class="wbreak">${Object.entries(w.confidence_breakdown || {}).map(([k, v]) =>
          `<span class="wb"><span class="wb-bar"><i style="width:${v}%"></i></span>${jnice(k)} <b>${v}</b></span>`).join("")}</div>
        <ul class="list">${(w.evidence || []).map((e) => `<li>${esc(e)}</li>`).join("")}</ul>
      </div>`).join("");
}

/* ---- 1. Creative clusters ----------------------------------------------- */
function renderClusters(intel) {
  const cs = intel.creative_clusters || [];
  if (!cs.length) { $("tab-clusters").innerHTML = '<p class="muted">Not enough ads to form clusters (need ≥2 sharing a hook + format).</p>'; return; }
  $("tab-clusters").innerHTML = cs.map((c) => `
    <div class="wcard">
      <div class="wcard-h"><span class="wname">${esc(c.cluster_name)}</span>${confChip(c.confidence)}</div>
      <div class="wmeta">${c.ads_count} ads · ${c.frequency_pct}% · consistency ${c.consistency}% · avg score ${c.avg_creative_score}/100</div>
      <dl class="kv">
        <dt>Dominant hook</dt><dd>${esc(jnice(c.dominant_hook))}</dd>
        <dt>Dominant CTA</dt><dd>${esc(jnice(c.dominant_cta))}</dd>
        <dt>Dominant emotion</dt><dd>${esc(jnice(c.dominant_emotion))}</dd>
        <dt>Visual structure</dt><dd>UGC ${c.visual_structure.ugc_pct}% · demo ${c.visual_structure.product_demo_pct}% · app-screen ${c.visual_structure.app_screen_pct}%</dd>
      </dl>
      <p class="muted">${esc(c.reasoning)}</p>
    </div>`).join("");
}

/* ---- 3 + 5. Opportunity gaps & saturation ------------------------------- */
function renderOpportunities(intel) {
  const gaps = intel.opportunity_gaps || [], sat = intel.market_saturation || [];
  if (!gaps.length && !sat.length) { $("tab-opportunities").innerHTML = emptyIntel(); return; }
  let html = '<div class="section-title">Underused opportunities (whitespace)</div>';
  html += gaps.length ? gaps.map((g) => `
    <div class="gap">
      <div class="wcard-h"><span class="wname">${esc(g.label)} <span class="plat">${esc(g.type)}</span></span>${confChip(g.confidence)}</div>
      <div class="wmeta">current usage ${g.current_usage_pct}%</div>
      <p>${esc(g.reason)} ${esc(g.reasoning)}</p>
    </div>`).join("") : '<p class="muted">No clear whitespace detected.</p>';
  html += '<div class="section-title">Market saturation</div>';
  html += sat.length ? sat.map((s) => `
    <div class="score-row">
      <span class="name">${esc(s.pattern)}</span>
      <span class="sbar"><i class="risk-${s.risk}" style="width:${s.saturation}%"></i></span>
      <span class="val">${s.saturation}% · ${s.risk}</span>
    </div>
    <p class="muted" style="margin:0 0 12px">${esc(s.recommendation)}</p>`).join("") : '<p class="muted">—</p>';
  $("tab-opportunities").innerHTML = html;
}

/* ---- 6. Strategy triad + classic creative brief ------------------------- */
function renderStrategies(intel, out) {
  const st = intel.strategies || {};
  const order = [["safe", "Safe"], ["winning", "Winning"], ["contrarian", "Contrarian"]];
  let html = order.map(([k]) => {
    const s = st[k]; if (!s) return "";
    return `
      <div class="scard scard-${k}">
        <div class="wcard-h"><span class="wname">${esc(s.name)}</span>${confChip(s.confidence)}</div>
        <p>${esc(s.thesis)}</p>
        <ul class="list">${(s.plays || []).map((p) => `<li>${esc(p)}</li>`).join("")}</ul>
        ${(s.evidence || []).length ? `<div class="section-title">Evidence</div>${(s.evidence || []).map((e) => `<div class="exq-ev">${esc(e)}</div>`).join("")}` : ""}
        <div class="risk">⚠ ${esc(s.risk)}</div>
      </div>`;
  }).join("");
  if (!html) html = emptyIntel();

  // structured, data-tied creative brief variants (A/B/C)
  const cb = intel.creative_brief || [];
  if (cb.length) {
    html += '<div class="section-title">Ready-to-run creative variants</div>';
    html += cb.map((v) => `
      <div class="scard">
        <div class="wcard-h"><span class="wname">${esc(v.variant)}</span><span class="plat">${esc(v.strategy)}</span></div>
        <dl class="kv">
          <dt>Format</dt><dd>${esc(v.format)}</dd>
          <dt>Hook</dt><dd>${esc(v.hook)}</dd>
          <dt>Visual opening</dt><dd>${esc(v.visual_opening)}</dd>
          <dt>Core message</dt><dd>${esc(v.core_message)}</dd>
          <dt>Proof</dt><dd>${esc(v.proof)}</dd>
          <dt>CTA</dt><dd>${esc(v.cta)}</dd>
          <dt>Why this test</dt><dd>${esc(v.why_this_test_exists)}</dd>
          <dt>Winning condition</dt><dd>${esc(v.winning_condition)}</dd>
        </dl>
        <div class="risk">⚠ ${esc(v.risk)}</div>
      </div>`).join("");
  }

  // classic LLM-generated creative brief, kept as a ready-to-run artefact
  const s = out.strategy || {};
  if (s.strategy_name || (s.hooks || []).length) {
    html += `
      <div class="section-title">Ready-to-run creative brief</div>
      <dl class="kv">
        <dt>Concept</dt><dd>${esc(s.creative_concept) || "—"}</dd>
        <dt>Target audience</dt><dd>${esc(s.target_audience) || "—"}</dd>
      </dl>
      <div class="section-title">Three hooks</div>
      ${(s.hooks || []).map((h) => `<span class="tag">${esc(h)}</span>`).join("") || '<span class="muted">—</span>'}
      <div class="section-title">Video script</div>
      ${(s.video_script || []).map((l) => `<div class="script-step">${esc(l)}</div>`).join("") || '<span class="muted">—</span>'}
      <div class="section-title">A/B test plan</div>
      <ul class="list">${(s.ab_test_plan || []).map((t) => `<li>${esc(t)}</li>`).join("") || "<li>—</li>"}</ul>`;
  }
  $("tab-strategies").innerHTML = html;
}

/* ---- 7. Ad market map --------------------------------------------------- */
function renderMap(intel) {
  const m = intel.market_map || {};
  if (!m.hooks) { $("tab-map").innerHTML = emptyIntel(); return; }
  const bars = (arr) => (arr || []).map((d) => `
    <div class="score-row">
      <span class="name">${esc(d.label || jnice(d.value))}</span>
      <span class="sbar"><i style="width:${d.pct}%"></i></span>
      <span class="val">${d.count} · ${d.pct}%</span>
    </div>`).join("") || '<p class="muted">—</p>';
  const combos = (arr) => (arr || []).map((c) => `
    <div class="pattern"><span class="pct">${c.pct}%</span>
      <span class="txt">${esc(c.combo)} <span class="muted">(${c.count} ads)</span></span></div>`).join("") || '<p class="muted">—</p>';
  const dc = m.dominant_combinations || {};
  $("tab-map").innerHTML = `
    <div class="section-title">Hooks</div>${bars(m.hooks)}
    <div class="section-title">Formats</div>${bars(m.formats)}
    <div class="section-title">Emotions</div>${bars(m.emotions)}
    <div class="section-title">CTAs</div>${bars(m.ctas)}
    <div class="section-title">Dominant hook × format</div>${combos(dc.hook_x_format)}
    <div class="section-title">Dominant format × emotion</div>${combos(dc.format_x_emotion)}
    ${(intel.platform_intelligence || []).length ? `<div class="section-title">Platform intelligence</div>` + (intel.platform_intelligence).map((pi) => `
      <div class="wcard">
        <div class="wcard-h"><span class="wname">${esc(pi.platform)} <span class="plat">${pi.ads_count} ads · ${esc(pi.signal)}</span></span></div>
        <p class="muted">${esc(pi.interpretation)}</p>
        <dl class="kv">
          <dt>Hook</dt><dd>${esc(pi.dominant_hook)}</dd>
          <dt>Format</dt><dd>${esc(pi.dominant_format)}</dd>
          <dt>CTA</dt><dd>${esc(pi.dominant_cta)}</dd>
          <dt>Proof</dt><dd>${esc(pi.dominant_proof)}</dd>
          <dt>Angle</dt><dd>${esc(pi.dominant_angle)}</dd>
        </dl>
        <ul class="list"><li><b>Replicate:</b> ${esc(pi.so_what.replicate)}</li><li><b>Avoid:</b> ${esc(pi.so_what.avoid)}</li><li><b>Test next:</b> ${esc(pi.so_what.test_next)}</li></ul>
      </div>`).join("") : ""}`;
}

/* ---- Evidence (measured frequencies, scores, limitations) --------------- */
function renderEvidence(out) {
  $("tab-evidence").innerHTML = `
    <div class="section-title">Measured frequency patterns</div>
    ${(out.winning_patterns || []).map((p) => `
      <div class="pattern">
        <span class="pct">${(p.percentage || 0).toFixed(0)}%</span>
        <span class="txt">${esc(p.statement)}${p.platform ? `<span class="plat">${esc(p.platform)}</span>` : ""}</span>
        <span class="bar"><i style="width:${Math.min(100, p.percentage || 0)}%"></i></span>
      </div>`).join("") || '<p class="muted">—</p>'}
    <div class="section-title">Average creative scores</div>
    <div id="evScores"><p class="muted">Loading…</p></div>
    <div class="section-title">Reasoning</div>
    <p class="muted">${esc(out.reasoning_summary) || "—"}</p>
    <div class="section-title">Limitations</div>
    ${(out.limitations || []).map((l) => `<div class="limitation">⚠ ${esc(l)}</div>`).join("") || '<span class="muted">None reported.</span>'}`;
}

function renderScoresInto(agents) {
  const box = $("evScores");
  if (!box) return;
  const scoring = agents.filter((a) => a.agent_name === "scoring_agent" && a.output);
  if (!scoring.length) { box.innerHTML = '<p class="muted">No scores.</p>'; return; }
  const keys = ["hook_strength", "pain_point_clarity", "visual_clarity", "product_demonstration", "emotional_trigger", "cta_strength", "platform_fit"];
  const max = { hook_strength: 20, pain_point_clarity: 15, visual_clarity: 15, product_demonstration: 15, emotional_trigger: 15, cta_strength: 10, platform_fit: 10 };
  const avg = {}; keys.forEach((k) => (avg[k] = 0));
  let total = 0;
  scoring.forEach((s) => { keys.forEach((k) => (avg[k] += s.output[k] || 0)); total += s.output.total || 0; });
  keys.forEach((k) => (avg[k] /= scoring.length));
  total /= scoring.length;
  box.innerHTML = `
    <p class="muted" title="Based on observable creative factors (hook clarity, pain-point clarity, visual clarity, product demonstration, emotional trigger, CTA strength, platform fit). Not a substitute for spend, engagement or conversion data.">Average creative-quality proxy score across ${scoring.length} ad(s): <strong style="color:var(--ink)">${total.toFixed(1)} / 100</strong> <i>ⓘ proxy, not performance</i></p>
    ${keys.map((k) => `
      <div class="score-row">
        <span class="name">${jnice(k)}</span>
        <span class="sbar"><i style="width:${Math.min(100, (avg[k] / max[k]) * 100)}%"></i></span>
        <span class="val">${avg[k].toFixed(1)} / ${max[k]}</span>
      </div>`).join("")}`;
}

function renderAgents(agents) {
  if (!agents.length) { $("tab-agents").innerHTML = '<p class="muted">No agent outputs.</p>'; return; }
  const byName = {};
  agents.forEach((a) => (byName[a.agent_name] = byName[a.agent_name] || []).push(a));
  $("tab-agents").innerHTML = Object.entries(byName).map(([name, list]) => `
    <details class="agent-card">
      <summary>${esc(name.replace(/_/g, " "))}
        <span><span class="src ${list[0].source}">${list[0].source}</span>
        <span class="muted" style="font-size:12px;margin-left:8px">${list.length} output(s)</span></span>
      </summary>
      ${list.slice(0, 8).map((a) => `
        <pre>ad #${a.ad_id ?? "—"} · conf ${(a.confidence ?? 0).toFixed(2)} · agreement ${(a.consensus_agreement ?? 1).toFixed(2)}
${esc(JSON.stringify(a.output, null, 2))}</pre>`).join("")}
      ${list.length > 8 ? `<pre>…and ${list.length - 8} more</pre>` : ""}
    </details>`).join("");
}

function switchTab(name) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("is-active", t.dataset.tab === name));
  document.querySelectorAll(".panel").forEach((p) => p.classList.add("hidden"));
  $("tab-" + name).classList.remove("hidden");
}

boot();
