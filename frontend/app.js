"use strict";

const $ = (id) => document.getElementById(id);
const state = { config: null, concepts: [], folder: null, portfolio: null };

async function api(path, body) {
  const opt = body
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    : {};
  const r = await fetch(path, opt);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch (_) {}
    throw new Error(msg);
  }
  return r.json();
}

function keys() {
  return {
    or: $("orKey").value.trim() || null,
    ncbi: $("ncbiKey").value.trim() || null,
    email: $("ncbiEmail").value.trim() || null,
    model: $("model").value.trim() || null,
  };
}

// ---------------------------------------------------------------- init
async function init() {
  const cfg = await api("/api/config");
  state.config = cfg;
  $("model").value = cfg.default_model;
  $("ncbiEmail").value = cfg.ncbi_email || "";
  setPill("orStatus", cfg.has_openrouter_key);
  setPill("ncbiStatus", cfg.has_ncbi_key);
  const box = $("domains");
  cfg.domains.forEach((d) => {
    const c = document.createElement("span");
    c.className = "chip";
    c.textContent = d;
    c.onclick = () => c.classList.toggle("on");
    box.appendChild(c);
  });
}
function setPill(id, ok) {
  const el = $(id);
  el.textContent = ok ? "env set" : "not set";
  el.className = "pill " + (ok ? "ok" : "no");
}
function selectedDomains() {
  return [...document.querySelectorAll("#domains .chip.on")].map((c) => c.textContent);
}

// ---------------------------------------------------------------- step 1: map
$("mapBtn").onclick = async () => {
  const q = $("question").value.trim();
  if (!q) return setMsg("mapMsg", "Enter a research question.", true);
  setMsg("mapMsg", "Asking the model to decompose the question…");
  $("mapBtn").disabled = true;
  try {
    const k = keys();
    const res = await api("/api/map", {
      question: q, domains: selectedDomains(),
      model: k.model, extra_context: $("extra").value.trim(), api_key: k.or,
    });
    state.concepts = res.concepts.map(normalizeConcept);
    $("mapNotes").textContent = res.notes ? "Notes: " + res.notes : "";
    renderConcepts();
    $("conceptSection").classList.remove("hidden");
    $("filterSection").classList.remove("hidden");
    $("compileSection").classList.remove("hidden");
    setMsg("mapMsg", `Model: ${res.model}. Review the ${state.concepts.length} concept blocks below.`);
  } catch (e) {
    setMsg("mapMsg", e.message, true);
  } finally {
    $("mapBtn").disabled = false;
  }
};

function normalizeConcept(c) {
  return {
    name: c.name || "Concept",
    rationale: c.rationale || "",
    explode: true,
    role: c.role || "required",
    mesh: (c.mesh || []).map((m) => ({ ...m, include: !!m.matched })),
    freetext: c.freetext || [],
  };
}

// ---------------------------------------------------------------- render concepts
function renderConcepts() {
  const root = $("concepts");
  root.innerHTML = "";
  state.concepts.forEach((c, ci) => root.appendChild(conceptCard(c, ci)));
}

function conceptCard(c, ci) {
  const el = document.createElement("div");
  el.className = "concept";

  const title = document.createElement("input");
  title.value = c.name;
  title.style.fontWeight = "700";
  title.oninput = () => (c.name = title.value);
  el.appendChild(title);

  const role = document.createElement("label");
  role.className = "inline";
  role.innerHTML = `Facet role <select style="width:auto"><option value="required">Required</option><option value="optional">Optional / sensitivity</option><option value="contextual">Context only</option></select>`;
  role.querySelector("select").value = c.role || "required";
  role.querySelector("select").onchange = (e) => (c.role = e.target.value);
  el.appendChild(role);

  if (c.rationale) {
    const r = document.createElement("p");
    r.className = "rat";
    r.textContent = c.rationale;
    el.appendChild(r);
  }

  // explode toggle
  const exp = document.createElement("label");
  exp.className = "inline";
  exp.innerHTML = `<input type="checkbox" ${c.explode ? "checked" : ""} style="width:auto"> Explode MeSH (include narrower terms)`;
  exp.querySelector("input").onchange = (e) => (c.explode = e.target.checked);
  el.appendChild(exp);

  // MeSH items
  const mh = document.createElement("div");
  mh.className = "ftlabel";
  mh.textContent = "MeSH headings";
  el.appendChild(mh);

  c.mesh.forEach((m) => el.appendChild(meshRow(c, m)));

  // free-text
  const ftl = document.createElement("div");
  ftl.className = "ftlabel";
  ftl.textContent = "Free-text terms (Title/Abstract) — one per line";
  el.appendChild(ftl);
  const ta = document.createElement("textarea");
  ta.rows = Math.min(12, Math.max(3, c.freetext.length));
  ta.value = c.freetext.join("\n");
  ta.oninput = () => (c.freetext = ta.value.split("\n").map((s) => s.trim()).filter(Boolean));
  c._ta = ta;
  el.appendChild(ta);

  const rm = document.createElement("button");
  rm.className = "ghost small";
  rm.textContent = "Remove concept";
  rm.onclick = () => { state.concepts.splice(ci, 1); renderConcepts(); };
  el.appendChild(rm);

  return el;
}

function meshRow(concept, m) {
  const row = document.createElement("div");
  row.className = "mesh-item";

  const cb = document.createElement("input");
  cb.type = "checkbox";
  cb.checked = m.include;
  cb.style.width = "auto";
  cb.onchange = () => (m.include = cb.checked);
  row.appendChild(cb);

  const q = document.createElement("span");
  q.textContent = `“${m.query}”`;
  q.className = m.matched ? "small" : "small matchbad";
  row.appendChild(q);

  const sel = document.createElement("select");
  if (!m.options.length) {
    const o = document.createElement("option");
    o.textContent = "— no MeSH match —";
    sel.appendChild(o);
    sel.disabled = true;
    cb.checked = m.include = false;
  } else {
    m.options.forEach((op) => {
      const o = document.createElement("option");
      o.value = op.dui;
      o.textContent = `${op.label} (${op.dui})`;
      if (op.dui === m.selected_dui) o.selected = true;
      sel.appendChild(o);
    });
    sel.onchange = () => (m.selected_dui = sel.value);
  }
  row.appendChild(sel);

  if (!m.matched && m.options.length) {
    const t = document.createElement("span");
    t.className = "tag matchbad";
    t.textContent = "fuzzy — confirm";
    row.appendChild(t);
  }

  if (m.options.length) {
    const pull = document.createElement("button");
    pull.className = "ghost small";
    pull.textContent = "+ entry terms";
    pull.title = "Add this descriptor's synonyms (exploded if enabled) to free-text";
    pull.onclick = async () => {
      pull.disabled = true; pull.textContent = "…";
      try {
        const res = await api("/api/expand", { dui: m.selected_dui, explode: concept.explode });
        const set = new Set(concept.freetext.map((s) => s.toLowerCase()));
        res.entry_terms.forEach((t) => { if (!set.has(t.toLowerCase())) { set.add(t.toLowerCase()); concept.freetext.push(t); } });
        if (concept._ta) { concept._ta.value = concept.freetext.join("\n"); concept._ta.rows = Math.min(14, concept.freetext.length); }
        const info = document.createElement("span");
        info.className = "expandinfo";
        info.textContent = `+${res.entry_terms.length} terms from ${res.descriptors.length} descriptor(s)`;
        row.appendChild(info);
      } catch (e) { alert(e.message); }
      finally { pull.disabled = false; pull.textContent = "+ entry terms"; }
    };
    row.appendChild(pull);
  }
  return row;
}

$("addConcept").onclick = () => {
  state.concepts.push({ name: "New concept", rationale: "", role: "required", explode: true, mesh: [], freetext: [] });
  renderConcepts();
};

// ---------------------------------------------------------------- step 3/4: compile
function collectConcepts() {
  return state.concepts.map((c) => {
    const mesh = c.mesh
      .filter((m) => m.include && m.options.length)
      .map((m) => {
        const op = m.options.find((o) => o.dui === m.selected_dui) || m.options[0];
        return op.label;
      });
    return { name: c.name, role: c.role || "required", explode: c.explode, mesh, freetext: c.freetext };
  }).filter((c) => c.mesh.length || c.freetext.length);
}

function collectFilters() {
  const csv = (id) => $(id).value.split(",").map((s) => s.trim()).filter(Boolean);
  return {
    date_from: $("dateFrom").value.trim(),
    date_to: $("dateTo").value.trim(),
    languages: csv("languages"),
    article_types_include: csv("ptInclude"),
    article_types_exclude: csv("ptExclude"),
    species: $("species").value,
    exclude_terms: csv("excludeTerms"),
    custom_include: $("customIncl").value.trim(),
  };
}

let compiled = null;
$("compileBtn").onclick = async () => {
  try {
    const concepts = collectConcepts();
    if (!concepts.length) return setMsg("searchMsg", "No concepts with terms to search.", true);
    const strict = $("strictMode").checked;
    const portfolio = await api("/api/portfolio", { concepts, filters: collectFilters(), strict });
    compiled = portfolio.primary;
    state.portfolio = portfolio;
    $("queryBox").textContent = compiled.query;
    $("queryBox").classList.remove("hidden");
    const mode = strict
      ? ` · <span class="tag matchok">strict MeSH synonyms</span>`
      : ` · <span class="tag matchbad">LLM free-text (not reproducible)</span>`;
    $("queryMeta").innerHTML = `<b>${compiled.n_concepts}</b> concept blocks · <b>${compiled.n_terms}</b> terms · query hash <b>${compiled.hash}</b>${mode}`;
    $("countBtn").disabled = false;
    $("searchBtn").disabled = false;
    const variants = portfolio.variants.length > 1 ? ` ${portfolio.variants.length} auditable query variants generated.` : "";
    setMsg("searchMsg", "Primary query compiled." + variants + " Preview the count or run the full search.");
  } catch (e) { setMsg("searchMsg", e.message, true); }
};

$("evaluateBtn").onclick = async () => {
  if (!compiled) return setMsg("searchMsg", "Compile a query first.", true);
  const known = $("knownPmids").value.split(/[\s,]+/).map((p) => p.trim()).filter(Boolean);
  if (!known.length) return setMsg("searchMsg", "Add one or more known relevant PMIDs first.", true);
  setMsg("searchMsg", "Checking known relevant PMIDs against PubMed…");
  try {
    const k = keys();
    const res = await api("/api/evaluate", { query: compiled.query, known_pmids: known, api_key: k.ncbi, email: k.email });
    const pct = (res.recall * 100).toFixed(1);
    setMsg("searchMsg", `Known-item recall: ${pct}% (${res.found_pmids.length}/${res.known_pmids.length}); missed: ${res.missed_pmids.join(", ") || "none"}. Total results: ${res.count.toLocaleString()}.`);
  } catch (e) { setMsg("searchMsg", e.message, true); }
};

$("countBtn").onclick = async () => {
  if (!compiled) return;
  setMsg("searchMsg", "Counting hits on PubMed…");
  try {
    const k = keys();
    const res = await api("/api/count", { query: compiled.query, api_key: k.ncbi, email: k.email });
    let extra = "";
    if (res.errors && Object.keys(res.errors).length) extra = " ⚠ " + JSON.stringify(res.errors);
    setMsg("searchMsg", `PubMed returns ${res.count.toLocaleString()} records.${extra}`);
  } catch (e) { setMsg("searchMsg", e.message, true); }
};

$("searchBtn").onclick = async () => {
  if (!compiled) return;
  const max = parseInt($("maxRecords").value, 10) || null;
  setMsg("searchMsg", "Running exhaustive search (fetching records)… this can take a while.");
  $("searchBtn").disabled = true;
  try {
    const k = keys();
    const res = await api("/api/search", {
      query: compiled.query, max_records: max, api_key: k.ncbi, email: k.email,
      protocol: {
        question: $("question").value.trim(),
        model: keys().model, domains: selectedDomains(),
        concepts: collectConcepts(), filters: collectFilters(),
        portfolio: state.portfolio || null,
        known_pmids: $("knownPmids").value.split(/[\s,]+/).filter(Boolean),
      },
    });
    state.folder = res.folder;
    renderResults(res);
    setMsg("searchMsg", "Done.");
  } catch (e) { setMsg("searchMsg", e.message, true); }
  finally { $("searchBtn").disabled = false; }
};

// ---------------------------------------------------------------- results
function renderResults(res) {
  $("resultsSection").classList.remove("hidden");
  const capped = res.capped ? ` (capped at ${res.fetched.toLocaleString()} of ${res.count.toLocaleString()} — raise “Max records” to fetch all)` : "";
  $("resultsSummary").innerHTML = `Total matches: <b>${res.count.toLocaleString()}</b> · fetched <b>${res.fetched.toLocaleString()}</b>${capped} · hash <b>${res.hash}</b>`;
  const tb = $("resultsTable").querySelector("tbody");
  tb.innerHTML = "";
  res.articles.forEach((a, i) => {
    const tr = document.createElement("tr");
    const doi = a.doi ? `<a href="https://doi.org/${a.doi}" target="_blank" rel="noopener">${a.doi}</a>` : "";
    tr.innerHTML =
      `<td>${i + 1}</td>` +
      `<td>${esc(a.title)}</td>` +
      `<td class="small">${esc(a.authors)}</td>` +
      `<td class="small">${esc(a.journal)}</td>` +
      `<td>${esc(a.year)}</td>` +
      `<td class="small">${doi}</td>` +
      `<td><a href="${a.url}" target="_blank" rel="noopener">${a.pmid}</a></td>`;
    tb.appendChild(tr);
  });
}
function esc(s) { return (s || "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

$("dlCsv").onclick = () => dl("csv");
$("dlJsonl").onclick = () => dl("jsonl");
$("dlProto").onclick = () => dl("protocol");
function dl(fmt) { if (state.folder) window.open(`/api/download?folder=${encodeURIComponent(state.folder)}&fmt=${fmt}`, "_blank"); }

function setMsg(id, txt, err) { const el = $(id); el.textContent = txt; el.className = "msg" + (err ? " err" : ""); }

init().catch((e) => alert("Init failed: " + e.message));
