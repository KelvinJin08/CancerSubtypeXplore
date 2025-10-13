/************** Config (same-origin API) **************/
const API_BASE = ""; // FastAPI 同源

/************** Shared constants for Part 1 **************/
const FEATURE_DIMS = { mRNA: 1000, DNA_METH: 1000, miRNA: 500 };
const COMMON_FILES = [
  "1_tr.csv","1_te.csv","1_featname.csv",
  "2_tr.csv","2_te.csv","2_featname.csv",
  "3_tr.csv","3_te.csv","3_featname.csv",
  "labels_tr.csv","labels_te.csv","README.md"
];

/************** Part 1 · Curated TCGA projects **************/
const DATASETS = [
  { id:"TCGA-BRCA", cancer:"Breast Invasive Carcinoma",
    subtypes:["BRCA.Normal","BRCA.LumA","BRCA.Her2","BRCA.LumB","BRCA.Basal","BRCA.NA"],
    train:532, test:229 },
  { id:"TCGA-LIHC", cancer:"Liver Hepatocellular Carcinoma",
    subtypes:["LIHC.iCluster:1","LIHC.iCluster:2","LIHC.iCluster:3","LIHC.NA"],
    train:130, test:57 },
  { id:"TCGA-COAD", cancer:"Colon Adenocarcinoma",
    subtypes:["GI.CIN","GI.MSI","GI.GS","GI.HM-SNV"],
    train:182, test:78 },
  { id:"TCGA-UCEC", cancer:"Uterine Corpus Endometrial Carcinoma",
    subtypes:["UCEC.CN_HIGH","UCEC.CN_LOW","UCEC.POLE","UCEC.NA","UCEC.MSI_H"],
    train:289, test:125 },
  { id:"TCGA-STAD", cancer:"Stomach Adenocarcinoma",
    subtypes:["STAD.EBV","STAD.MSI","STAD.Genome Stable","STAD.Chromosomal Instability","STAD.NA"],
    train:238, test:103 },
  { id:"TCGA-HNSC", cancer:"Head and Neck Squamous Cell Carcinoma",
    subtypes:["HNSC.Atypical","HNSC.Basal","HNSC.Classical","HNSC.Mesenchymal"],
    train:193, test:84 },

  { id:"TCGA-KIRC", cancer:"Kidney Renal Clear Cell Carcinoma",
    subtypes:["KIRC.1","KIRC.2","KIRC.3","KIRC.4","KIRC.NA"],
    train:175, test:75 },
  { id:"TCGA-READ", cancer:"Rectum Adenocarcinoma",
    subtypes:["GI.CIN","GI.GS","GI.HM-SNV","GI.MSI"],
    train:60, test:26 },
  { id:"TCGA-ESCA", cancer:"Esophageal Carcinoma",
    subtypes:["GI.CIN","GI.HM-SNV","GI.ESCC","GI.GS","GI.MSI"],
    train:111, test:48 },
  { id:"TCGA-KICH", cancer:"Kidney Chromophobe",
    subtypes:["KICH.Eosin.0","KICH.Eosin.1"],
    train:45, test:20 },
  { id:"TCGA-UCS",  cancer:"Uterine Carcinosarcoma",
    subtypes:["UCS.1","UCS.2"],
    train:39, test:18 },
  { id:"TCGA-KIRP", cancer:"Kidney Renal Papillary Cell Carcinoma",
    subtypes:["KIRP.C2a","KIRP.C1","KIRP.C2c - CIMP","KIRP.C2b"],
    train:102, test:45 },
  { id:"TCGA-ACC",  cancer:"Adrenocortical Carcinoma",
    subtypes:["ACC.CIMP-high","ACC.CIMP-low","ACC.CIMP-intermediate"],
    train:54, test:24 },
  { id:"TCGA-SKCM", cancer:"Skin Cutaneous Melanoma",
    subtypes:["SKCM.BRAF","SKCM.NF1","SKCM.RAS","SKCM.Triple Wild-Type","SKCM.NA"],
    train:45, test:18 },
  { id:"TCGA-BLCA", cancer:"Bladder Urothelial Carcinoma",
    subtypes:["BLCA.Luminal","BLCA.Basal","BLCA.Neuroendocrine","BLCA.Stroma-rich"],
    train:88, test:39 },
  { id:"TCGA-LUSC", cancer:"Lung Squamous Cell Carcinoma",
    subtypes:["LUSC.Luminal","LUSC.Basal","LUSC.Classical","LUSC.Secretory"],
    train:51, test:22 },
];

// NEW: safely read .value from an element by id
function safeVal(id, fallback=null){
  const el = document.getElementById(id);
  return el && typeof el.value !== "undefined" ? el.value : fallback;
}

/************** HTTP helpers **************/
async function jget(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} ${r.status}`);
  return await r.json();
}
async function jpost(url, payload) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!r.ok) throw new Error(`${url} ${r.status}`);
  return await r.json();
}

// === Part3: render results like Part 2 (table + CSV + scatter) ===
function renderDLResults(container, rows){
  if (!rows || rows.length === 0) {
    container.innerHTML = `<div class="card muted">No results.</div>`;
    return;
  }
  const headers = ["project","omics","model","acc","f1w","f1m","n_train","n_test"];
  const thead = `<thead><tr>${headers.map(h=>`<th>${h}</th>`).join("")}</tr></thead>`;
  const tbody = rows.map(r => `
    <tr>
      <td>${r.project}</td>
      <td>${r.omics}</td>
      <td>${r.model}</td>
      <td>${fmt(r.acc)}</td>
      <td>${fmt(r.f1w)}</td>
      <td>${fmt(r.f1m)}</td>
      <td>${r.n_train}</td>
      <td>${r.n_test}</td>
    </tr>
  `).join("");

  const csv = [headers.join(","), ...rows.map(r => headers.map(h => r[h]).join(","))].join("\n");
  const blob = new Blob([csv], {type: "text/csv"});
  const url = URL.createObjectURL(blob);

  container.innerHTML = `
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
      <div><strong>Results</strong> (${rows.length} rows)</div>
      <a class="btn" href="${url}" download="dl_results.csv">Download CSV</a>
    </div>
    <table class="table">${thead}<tbody>${tbody}</tbody></table>
    <div id="dl-chart-scatter" style="margin-top:10px"></div>
  `;

  // Scatter: ACC vs F1-macro
  const div = container.querySelector("#dl-chart-scatter");
  if (typeof Plotly !== "undefined") {
    const models = [...new Set(rows.map(r => r.model))];
    const traces = models.map(m => {
      const sub = rows.filter(r => r.model === m);
      return {
        type: "scatter",
        mode: "markers",
        name: m,
        x: sub.map(r => r.acc),
        y: sub.map(r => r.f1m),
        text: sub.map(r => `${r.project} • ${r.model} • ${r.omics}`),
        hovertemplate: "ACC=%{x:.3f}<br>F1-macro=%{y:.3f}<br>%{text}<extra></extra>",
        marker: { size: 10, line: { width: 1, color: "#111" } }
      };
    });
    Plotly.newPlot(div, traces, {
      title: { text: "ACC vs F1-macro", x: 0, y: 0.98 },
      margin: { t: 28, r: 10, b: 50, l: 50 },
      paper_bgcolor: "#0f141a",
      plot_bgcolor: "#0f141a",
      font: { color: "#e6edf3" },
      xaxis: { title: "Accuracy", rangemode: "tozero" },
      yaxis: { title: "F1-macro", rangemode: "tozero" },
      legend: { orientation: "h", y: -0.2 }
    }, {displayModeBar:false, responsive:true});
  } else {
    div.innerHTML = `<div class="muted">Plotly not found.</div>`;
  }

  function fmt(x){ if (x==null || Number.isNaN(x)) return "—"; return (typeof x==="number")? x.toFixed(4): x; }
}


// === helpers for Part3 table & scatter ===
function mountTable(container, data) {
  if (!container) return;
  if (Array.isArray(data) && data.length && !Array.isArray(data[0]) && typeof data[0] === "object") {
    const cols = Object.keys(data[0]);
    const thead = `<thead><tr>${cols.map(h=>`<th>${h}</th>`).join("")}</tr></thead>`;
    const tbody = data.map(row => `<tr>${cols.map(h=>`<td>${row[h] ?? "—"}</td>`).join("")}</tr>`).join("");
    container.innerHTML = `<table class="table">${thead}<tbody>${tbody}</tbody></table>`;
    return;
  }
  const rows = Array.isArray(data) ? data : [];
  const html = rows.map(([k,v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
  container.innerHTML = `<table class="table"><tbody>${html}</tbody></table>`;
}

function mountPointsScatter(container, points) {
  if (!container) return;
  if (!points || !points.length) {
    container.innerHTML = `<div class="muted">No points to visualize.</div>`;
    return;
  }
  const w = container.clientWidth || 720, h = 360, pad = 24;
  const xs = points.map(p=>p.x), ys = points.map(p=>p.y);
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const ymin = Math.min(...ys), ymax = Math.max(...ys);
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("width", String(w));
  svg.setAttribute("height", String(h));
  svg.style.border = "1px solid var(--border)";
  const sx = x => pad + (x - xmin) / Math.max(1e-9, (xmax - xmin)) * (w - 2*pad);
  const sy = y => h - pad - (y - ymin) / Math.max(1e-9, (ymax - ymin)) * (h - 2*pad);
  for (const p of points) {
    const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    c.setAttribute("cx", String(sx(p.x)));
    c.setAttribute("cy", String(sy(p.y)));
    c.setAttribute("r", "3");
    const ok = (p.pred !== undefined && p.label !== undefined) ? (p.pred === p.label) : null;
    c.setAttribute("fill", ok === null ? "#999" : (ok ? "#2ecc71" : "#e74c3c"));
    c.setAttribute("opacity", "0.85");
    svg.appendChild(c);
  }
  container.innerHTML = ""; container.appendChild(svg);
}



function renderDatasetTable(targetId, rows){
  const el = document.getElementById(targetId);
  if (!el) return;
  const thead = `
    <thead>
      <tr>
        <th>Dataset</th>
        <th>Cancer type</th>
        <th>Cancer subtypes</th>
        <th>#Train</th>
        <th>#Val/Test</th>
        <th style="width:90px">Details</th>
      </tr>
    </thead>`;
  const tbody = rows.map((r, idx) => `
    <tr>
      <td><span class="badge">${r.id}</span></td>
      <td>${r.cancer}</td>
      <td>${r.subtypes.join(", ")}</td>
      <td>${r.train ?? "—"}</td>
      <td>${r.test ?? "—"}</td>
      <td><button class="btn" data-toggle="row-${idx}">Expand</button></td>
    </tr>
    <tr id="row-${idx}" class="expand-row" style="display:none">
      <td colspan="6">
        <div class="expand-pane">
          <dl class="kv">
            <dt>Folder</dt>
            <dd class="path">../${r.id}/</dd>
            <dt>Feature sizes</dt>
            <dd>mRNA = ${FEATURE_DIMS.mRNA}, DNA-meth = ${FEATURE_DIMS.DNA_METH}, miRNA = ${FEATURE_DIMS.miRNA}</dd>
            <dt>Subtypes</dt>
            <dd class="subtypes">${r.subtypes.map(s=>`<span class="badge">${s}</span>`).join(" ")}</dd>
            <dt>Files included</dt>
            <dd class="filelist">${COMMON_FILES.map(f=>`<div><code>${f}</code></div>`).join("")}</dd>
          </dl>
        </div>
      </td>
    </tr>
  `).join("");

  el.innerHTML = `<table class="table">${thead}<tbody>${tbody}</tbody></table>`;
  el.querySelectorAll("button[data-toggle]").forEach(btn=>{
    btn.addEventListener("click", ()=>{
      const id = btn.getAttribute("data-toggle");
      const row = document.getElementById(id);
      const visible = row.style.display !== "none";
      row.style.display = visible ? "none" : "";
      btn.textContent = visible ? "Expand" : "Collapse";
    });
  });
}

/************** Part 2 · ML Runner **************/
async function initMLControls() {
  const metaRes = await fetch(`${API_BASE}/datasets`);
  if (!metaRes.ok) throw new Error(`/datasets ${metaRes.status}`);
  const meta = await metaRes.json();

  const dsRoot = document.getElementById("ds-root");
  dsRoot.innerHTML = meta.dataset_roots.map(k => `<option value="${k}">${k}</option>`).join("");

  const modelBox = document.getElementById("model-box");
  modelBox.classList.add("checkbox-grid");
  modelBox.innerHTML = meta.models.map(m => `
    <label><input type="checkbox" name="model" value="${m}" checked> ${m}</label>
  `).join("");

  const omicsBox = document.getElementById("omics-box");
  const combos = meta.default_combos;
  omicsBox.innerHTML = combos.map(c => `
    <label><input type="checkbox" name="combo" value="${c.join(",")}" checked> ${c.join("+")}</label>
  `).join("");

  const projectSelect = document.getElementById("project");
  async function loadProjectsForRoot(rootKey) {
    projectSelect.innerHTML = `<option>Loading...</option>`;
    const r = await fetch(`${API_BASE}/projects?root=${encodeURIComponent(rootKey)}`);
    if (!r.ok) throw new Error(`/projects?root=${rootKey} ${r.status}`);
    const data = await r.json();
    projectSelect.innerHTML = data.projects.map(p => `<option value="${p}">${p}</option>`).join("");
  }
  await loadProjectsForRoot(dsRoot.value);
  dsRoot.addEventListener("change", (e)=> loadProjectsForRoot(e.target.value));
}

function bindRun() {
  const btn = document.getElementById("btn-run");
  const status = document.getElementById("run-status");
  const out = document.getElementById("ml-results");

  btn.onclick = async () => {
    const dataset_root_key = document.getElementById("ds-root").value;
    const project = document.getElementById("project").value;
    const models = Array.from(document.querySelectorAll('#model-box input[name="model"]:checked')).map(el => el.value);
    const omics_combos = Array.from(document.querySelectorAll('#omics-box input[name="combo"]:checked')).map(el => el.value.split(","));

    if (!project || models.length === 0 || omics_combos.length === 0) {
      alert("Please select project, at least one model, and at least one omics combination.");
      return;
    }

    status.textContent = "Running...";
    btn.disabled = true;
    out.innerHTML = "";

    try {
      const res = await fetch(`${API_BASE}/run-ml`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ project, models, omics_combos, dataset_root_key })
      });
      if (!res.ok) throw new Error(`/run-ml ${res.status}`);
      const data = await res.json();
      renderMLResults(out, data.rows);
      status.textContent = "Done.";
    } catch (e) {
      console.error(e);
      status.textContent = "Error.";
      out.innerHTML = `<div class="card muted">Failed to run ML: ${e}</div>`;
    } finally {
      btn.disabled = false;
    }
  };
}

function renderMLResults(container, rows){
  if (!rows || rows.length === 0) {
    container.innerHTML = `<div class="card muted">No results.</div>`;
    return;
  }
  const headers = ["project","omics","model","acc","f1w","f1m","n_train","n_test"];
  const thead = `<thead><tr>${headers.map(h=>`<th>${h}</th>`).join("")}</tr></thead>`;
  const tbody = rows.map(r => `
    <tr>
      <td>${r.project}</td>
      <td>${r.omics}</td>
      <td>${r.model}</td>
      <td>${fmt(r.acc)}</td>
      <td>${fmt(r.f1w)}</td>
      <td>${fmt(r.f1m)}</td>
      <td>${r.n_train}</td>
      <td>${r.n_test}</td>
    </tr>
  `).join("");

  const csv = [headers.join(","), ...rows.map(r => headers.map(h => r[h]).join(","))].join("\n");
  const blob = new Blob([csv], {type: "text/csv"}); const url = URL.createObjectURL(blob);

  container.innerHTML = `
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
      <div><strong>Results</strong> (${rows.length} rows)</div>
      <a class="btn" href="${url}" download="ml_results.csv">Download CSV</a>
    </div>
    <table class="table">${thead}<tbody>${tbody}</tbody></table>
  `;

  // 仅渲染散点图
  renderScatter(rows);

  function fmt(x){ if (x==null || Number.isNaN(x)) return "—"; return (typeof x==="number")? x.toFixed(4): x; }
}

/************** Only Scatter **************/
function renderScatter(rows){
  const div = document.getElementById("chart-scatter");
  if (!div || !rows || rows.length === 0) return;

  const models = [...new Set(rows.map(r => r.model))];
  const traces = models.map(m => {
    const sub = rows.filter(r => r.model === m);
    return {
      type: "scatter",
      mode: "markers",
      name: m,
      x: sub.map(r => r.acc),
      y: sub.map(r => r.f1m),
      text: sub.map(r => `${r.project} • ${r.model} • ${r.omics}`),
      hovertemplate: "ACC=%{x:.3f}<br>F1-macro=%{y:.3f}<br>%{text}<extra></extra>",
      marker: { size: 10, line: { width: 1, color: "#111" } }
    };
  });

  const layout = {
    margin: { t: 28, r: 10, b: 50, l: 50 },
    paper_bgcolor: "#0f141a",
    plot_bgcolor: "#0f141a",
    font: { color: "#e6edf3" },
    xaxis: { title: "Accuracy", rangemode: "tozero" },
    yaxis: { title: "F1-macro", rangemode: "tozero" },
    legend: { orientation: "h", y: -0.2 }
  };

  Plotly.newPlot(div, traces, layout, {displayModeBar:false, responsive:true});
}

/************** Init **************/
window.addEventListener("DOMContentLoaded", async () => {
  try {
    renderDatasetTable("datasetTable", DATASETS);
    await initMLControls();
    bindRun();
    console.log("[OK] Part1 & Part2 initialized");
  } catch (e) {
    console.error("Init failed:", e);
    const s = document.getElementById("run-status") || document.createElement("div");
    s.id = "run-status"; s.style.margin = "10px 0"; s.textContent = "Init failed. See console.";
    document.body.prepend(s);
  }
});





/* =========================
   Part 3 · User-defined Deep Learning
   (biomarker export + auto ENSG→Gene mapping on frontend)
   ========================= */
(function(){
  // ---------- helpers ----------
  const $ = (id) => document.getElementById(id);
  const gSafeVal = (id, fb=null) =>
    (typeof window.safeVal === "function")
      ? window.safeVal(id, fb)
      : (document.getElementById(id)?.value ?? fb);

  // 前端缓存：ENSG(无版本号) -> 基因名
  const ENSG_CACHE = Object.create(null);

  // 去掉 ENSG 版本号（ENSG00000141510.12 -> ENSG00000141510）
  function stripVersion(id){
    try { return String(id).split(".")[0]; } catch(e){ return String(id); }
  }

  // 单个 ENSG 查询 MyGene.info，返回 symbol（若无则 "—"）
  async function fetchSymbolForEnsg(ensg){
    const bare = stripVersion(ensg);
    if (ENSG_CACHE[bare]) return ENSG_CACHE[bare];

    const url = `https://mygene.info/v3/query?q=ensemblgene:${encodeURIComponent(bare)}&fields=symbol&species=human`;
    try {
      const r = await fetch(url);
      if (!r.ok) throw new Error(`mygene ${r.status}`);
      const j = await r.json();
      let sym = "—";
      if (j && Array.isArray(j.hits) && j.hits.length > 0) {
        // 取第一条 hits 的 symbol
        const hit = j.hits[0];
        if (hit && hit.symbol) sym = String(hit.symbol);
      }
      ENSG_CACHE[bare] = sym || "—";
      return ENSG_CACHE[bare];
    } catch (e) {
      console.warn("MyGene lookup failed:", bare, e);
      ENSG_CACHE[bare] = "—";
      return "—";
    }
  }

  // 批量把某个 biomarker 卡片里的 Gene 列填充（仅填缺失项）
  async function populateGenesForCard(card){
    const rows = Array.from(card.querySelectorAll("tbody tr"));
    // 顺序逐个查（50 条很快），减少并发引起的限流
    for (const tr of rows) {
      const tds = tr.querySelectorAll("td");
      if (tds.length < 4) continue;
      const ensgCell = tds[1];
      const geneCell = tds[2];
      const cur = (geneCell.textContent || "").trim();
      if (cur && cur !== "—" && cur !== "...") continue; // 已有就不查
      const ensg = (ensgCell.textContent || "").trim();
      if (!ensg) continue;
      geneCell.textContent = "..."; // 占位
      const sym = await fetchSymbolForEnsg(ensg);
      geneCell.textContent = sym || "—";
    }
  }

  // （可选）生成“带 Gene 列”的 CSV 下载（前端生成）
  function mountDownloadCSVWithGene(card, filename){
    try {
      const rows = Array.from(card.querySelectorAll("tbody tr")).map(tr=>{
        const tds = tr.querySelectorAll("td");
        return {
          rank: (tds[0]?.textContent || "").trim(),
          ensg: (tds[1]?.textContent || "").trim(),
          gene: (tds[2]?.textContent || "").trim(),
          score: (tds[3]?.textContent || "").trim(),
        };
      });
      const headers = ["rank","ensembl_gene_id","symbol","score"];
      const csv = [headers.join(",")].concat(
        rows.map(r => [r.rank, r.ensg, r.gene, r.score].join(","))
      ).join("\n");
      const url = URL.createObjectURL(new Blob([csv], {type:"text/csv"}));
      const a = document.createElement("a");
      a.className = "btn";
      a.href = url;
      a.download = filename || "biomarkers_with_gene.csv";
      a.textContent = "Download CSV (with Gene)";
      return a;
    } catch(e){
      console.warn("build CSV(with gene) failed:", e);
      return null;
    }
  }

  // Part3 renderer: results table + CSV, draw ACC–F1 into the existing card (#dl-chart-scatter)
  function renderDLResultsLocal(container, rows){
    if (!container) return;

    const fmt = (x) => (x==null || Number.isNaN(x)) ? "—" : (typeof x==="number" ? x.toFixed(4) : x);

    // 清理 #dl-output 里旧版重复图
    const outHost = document.getElementById("dl-output");
    if (outHost) {
      const dups = outHost.querySelectorAll("#dl-chart-scatter");
      dups.forEach((el) => {
        const official = document.getElementById("dl-chart-scatter");
        if (official && el !== official) {
          const card = el.closest(".card");
          if (card && card.parentElement === outHost) outHost.removeChild(card);
          else el.remove();
        }
      });
    }

    if (!rows || rows.length === 0) {
      container.innerHTML = `<div class="card muted">No results.</div>`;
      const empty = document.getElementById("dl-chart-scatter");
      if (empty && typeof Plotly !== "undefined") { try { Plotly.purge(empty); } catch(e){} }
      return;
    }

    // 结果表 + CSV
    const headers = ["project","omics","model","acc","f1w","f1m","n_train","n_test"];
    const thead = `<thead><tr>${headers.map(h=>`<th>${h}</th>`).join("")}</tr></thead>`;
    const tbody = rows.map(r => `
      <tr>
        <td>${r.project}</td>
        <td>${r.omics}</td>
        <td>${r.model}</td>
        <td>${fmt(r.acc)}</td>
        <td>${fmt(r.f1w)}</td>
        <td>${fmt(r.f1m)}</td>
        <td>${r.n_train}</td>
        <td>${r.n_test}</td>
      </tr>
    `).join("");
    const csv = [headers.join(","), ...rows.map(r => headers.map(h => r[h]).join(","))].join("\n");
    const url = URL.createObjectURL(new Blob([csv], {type: "text/csv"}));

    container.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
        <div><strong>Results</strong> (${rows.length} rows)</div>
        <a class="btn" href="${url}" download="dl_results.csv">Download CSV</a>
      </div>
      <table class="table">${thead}<tbody>${tbody}</tbody></table>
    `;

    // 在 Part3 顶部占位卡片里画散点
    const div = document.getElementById("dl-chart-scatter");
    if (div && typeof Plotly !== "undefined") {
      const models = [...new Set(rows.map(r => r.model))];
      const traces = models.map(m => {
        const sub = rows.filter(r => r.model === m);
        return {
          type: "scatter", mode: "markers", name: m,
          x: sub.map(r => r.acc), y: sub.map(r => r.f1m),
          text: sub.map(r => `${r.project} • ${r.model} • ${r.omics}`),
          hovertemplate: "ACC=%{x:.3f}<br>F1-macro=%{y:.3f}<br>%{text}<extra></extra>",
          marker: { size: 10, line: { width: 1, color: "#111" } }
        };
      });
      try { Plotly.purge(div); } catch(e) {}
      Plotly.newPlot(div, traces, {
        margin: { t: 10, r: 10, b: 50, l: 50 },
        paper_bgcolor: "#0f141a",
        plot_bgcolor: "#0f141a",
        font: { color: "#e6edf3" },
        xaxis: { title: "Accuracy", rangemode: "tozero" },
        yaxis: { title: "F1-macro", rangemode: "tozero" },
        legend: { orientation: "h", y: -0.25 }
      }, {displayModeBar:false, responsive:true});
    }
  }

  // ---------- grab UI ----------
  const kind = $("dl-kind");
  const nameI = $("dl-name");
  const inDim = $("dl-input-dim");
  const nClasses = $("dl-num-classes");
  const panelMLP = $("dl-panel-mlp");
  const panelATT = $("dl-panel-att");

  const btn = $("btn-dl-train");
  const status = $("dl-status");
  const out = $("dl-output");

  const dsSelect = $("dl-dataset");
  const projectSelect = $("project");
  const rootSelect = $("ds-root");

  if (!btn || !kind) { console.warn("[Part3] UI not found, skip."); return; }

  // 输入维数/类别自动决定 → 只做提示
  if (inDim) {
    inDim.readOnly = true;
    inDim.placeholder = "auto (varies by combo)";
    inDim.title = "Determined by dataset + selected omics (see table).";
  }
  if (nClasses) {
    nClasses.readOnly = true;
    nClasses.placeholder = "auto (from labels)";
    nClasses.title = "Determined automatically from labels (train + val).";
  }

  function togglePanels(){
    if (kind.value === "simple_mlp") { panelMLP && (panelMLP.style.display=""); panelATT && (panelATT.style.display="none"); }
    else if (kind.value === "attention") { panelMLP && (panelMLP.style.display="none"); panelATT && (panelATT.style.display=""); }
    else { panelMLP && (panelMLP.style.display="none"); panelATT && (panelATT.style.display="none"); }
  }
  kind.addEventListener("change", togglePanels); togglePanels();

  function cloneProjectOptions() {
    const ds = $("dl-dataset");
    const proj = $("project");
    if (!ds || !proj) return;
    const prev = ds.value;
    ds.innerHTML = "";
    Array.from(proj.options).forEach(opt=>{
      const o = document.createElement("option");
      o.value = opt.value; o.textContent = opt.textContent;
      ds.appendChild(o);
    });
    ds.value = proj.value || prev || (ds.options[0] && ds.options[0].value);
  }
  cloneProjectOptions();

  if (projectSelect) {
    projectSelect.addEventListener("change", ()=>{ cloneProjectOptions(); syncRunName(); });
    const mo = new MutationObserver(cloneProjectOptions);
    mo.observe(projectSelect, { childList: true, subtree: true });
  }
  if (dsSelect && projectSelect) {
    dsSelect.addEventListener("change", ()=>{
      const ds = $("dl-dataset"), proj = $("project");
      if (!ds || !proj) return;
      proj.value = ds.value;
      proj.dispatchEvent(new Event("change", { bubbles: true }));
    });
  }
  if (rootSelect) {
    rootSelect.addEventListener("change", ()=>{
      const ds = $("dl-dataset");
      if (ds) ds.innerHTML = "";
      syncRunName();
    });
  }
  function syncRunName(){
    const p = (projectSelect && projectSelect.value) || (dsSelect && dsSelect.value) || "";
    if (p && (!nameI.value || nameI.value === "MyDLRun")) nameI.value = `${p}-DL`;
  }
  syncRunName();

  function getSelectedOmicsCombos() {
    return Array.from(document.querySelectorAll('#omics-box input[name="combo"]:checked'))
      .map(el => el.value.replace(/\s+/g,""));
  }

  function setProgress(cur, tot){
    const bar = $("dl-progress");
    if (!bar) return;
    cur = Math.max(0, Number(cur)||0);
    tot = Math.max(cur, Number(tot)||0);
    const pct = tot>0 ? (cur*100/tot) : 0;
    bar.style.width = pct.toFixed(1) + "%";
    bar.title = `${cur}/${tot}`;
  }

  btn.onclick = async ()=>{
    btn.disabled = true; status.textContent = "Submitting..."; out.textContent = ""; setProgress(0,100);
    try{
      const datasetId = (dsSelect && dsSelect.value) || (projectSelect && projectSelect.value) || "DEMO";
      const datasetRoot = (rootSelect && rootSelect.value) || "val";
      const combos = getSelectedOmicsCombos();

      const lr = parseFloat(gSafeVal("dl-tr-lr","0.001"));
      const wd = parseFloat(gSafeVal("dl-tr-wd","0.0001"));
      const bs = parseInt(gSafeVal("dl-tr-bs","64"));
      const epochs = parseInt(gSafeVal("dl-tr-epochs","20"));

      const base = {
        name: gSafeVal("dl-name","MyDLRun") || "MyDLRun",
        kind: gSafeVal("dl-kind","simple_mlp") || "simple_mlp",
        input_dim: 0,
        num_classes: 0,
        seed: 42
      };
      let model = {};
      if (base.kind === "simple_mlp") {
        const hidden = (gSafeVal("dl-mlp-hidden","256,128,64")||"")
          .split(",").map(s=>parseInt((s||"").trim())).filter(x=>!isNaN(x));
        model = {
          ...base,
          hidden_dims: hidden,
          dropout: parseFloat(gSafeVal("dl-mlp-dropout","0.2")),
          activation: gSafeVal("dl-mlp-act","relu"),
          batch_norm: !!$("dl-mlp-bn")?.checked
        };
      } else if (base.kind === "attention") {
        model = {
          ...base,
          d_model: parseInt(gSafeVal("dl-att-dmodel","128")),
          n_heads: parseInt(gSafeVal("dl-att-heads","4")),
          n_layers: parseInt(gSafeVal("dl-att-layers","2")),
          ff_multiplier: parseInt(gSafeVal("dl-att-ff","4")),
          dropout: parseFloat(gSafeVal("dl-att-dropout","0.1"))
        };
      } else {
        model = { ...base, kind: "linear" };
      }

      const payload = {
        model,
        train: { lr, weight_decay: wd, batch_size: bs, epochs },
        dataset_id: datasetId,
        dataset_root_key: datasetRoot,
        omics_combos: combos.length ? combos : undefined
      };

      const data = await jpost(`/model/train`, payload);
      status.textContent = `Job ${data.job_id} submitted.`;

      const poll = async ()=>{
        const d = await jget(`/model/train/${data.job_id}`);
        if (d.status === "running") {
          const cur = d.epoch || 0;
          const tot = d.epochs || epochs || 100;
          setProgress(cur, tot);
          const accTxt = (d?.metrics?.val_acc != null && d.metrics.val_acc.toFixed) ? d.metrics.val_acc.toFixed(4) : d?.metrics?.val_acc;
          status.textContent = `Epoch ${cur}/${tot}  val_acc=${accTxt}`;
          setTimeout(poll, 800);
        } else if (d.status === "succeeded") {
          const tot = d.epochs || epochs || 100;
          setProgress(tot, tot);
          const bestTxt = (d?.metrics?.best_val_acc != null && d.metrics.best_val_acc.toFixed) ? d.metrics.best_val_acc.toFixed(4) : d?.metrics?.best_val_acc;
          status.textContent = `Done. best_val_acc=${bestTxt}`;

          if (inDim) inDim.value = "auto (see table)";
          if (nClasses) nClasses.value = "auto (from labels)";

          out.innerHTML = "";

          const summary = document.createElement("div");
          out.appendChild(summary);
          // 小工具：本地格式化
          const fmt2 = (x) => (x==null || Number.isNaN(x)) ? "—" : (typeof x==="number" ? x.toFixed(6) : x);

          // 概要
          mountTable(summary, [
            ["Dataset", datasetId],
            ["Model", model.kind],
            ["LR", lr],
            ["Batch size", bs],
            ["Epochs (per combo)", epochs],
            ["Selected omics combos", (d.selected_omics && d.selected_omics.length) ? d.selected_omics.join(", ") : "-"],
            ["Best val acc (overall)", (d.metrics.best_val_acc*100).toFixed(2) + "%"]
          ]);

          // 表格 + 散点
          if (Array.isArray(d.rows) && d.rows.length) {
            const tableBox = document.createElement("div");
            tableBox.style.marginTop = "12px";
            out.appendChild(tableBox);
            renderDLResultsLocal(tableBox, d.rows);
          }

          // === Biomarkers (自动映射 Gene 列) ===
          const bmList = Array.isArray(d.biomarkers) ? d.biomarkers : [];
          const sec = document.createElement("div");
          sec.style.marginTop = "14px";
          out.appendChild(sec);

          if (bmList.length === 0) {
            const card = document.createElement("div");
            card.className = "card";
            card.style.padding = "8px 10px";
            card.innerHTML = `<div class="muted">No biomarker output. Make sure at least one selected omics combo contains <strong>1 (mRNA)</strong>.</div>`;
            sec.appendChild(card);
          } else {
            for (const bm of bmList) {
              const card = document.createElement("div");
              card.className = "card";
              card.style.marginTop = "10px";
              card.style.padding = "8px 10px 10px 10px";

              const header = document.createElement("div");
              header.style.display = "flex";
              header.style.justifyContent = "space-between";
              header.style.alignItems = "center";
              header.style.marginBottom = "6px";
              header.innerHTML = `
                <div style="font-weight:600">Top 50 mRNA biomarkers (combo: ${bm.combo ?? "-"})</div>
                <div class="muted" style="font-size:12px">Gene names via MyGene.info</div>
              `;
              card.appendChild(header);

              const tbl = document.createElement("table");
              tbl.className = "table";
              const rowsHtml = (bm.top50 || []).map(r =>
                `<tr><td>${r.rank}</td><td><code>${r.feature}</code></td><td>${r.gene ?? "—"}</td><td>${fmt2(r.score)}</td></tr>`
              ).join("");
              tbl.innerHTML = `
                <thead><tr><th>rank</th><th>ENSG</th><th>Gene</th><th>score</th></tr></thead>
                <tbody>${rowsHtml || `<tr><td colspan="4" class="muted">No mRNA slice for this combo</td></tr>`}</tbody>
              `;
              card.appendChild(tbl);

              // 右上角下载按钮（带 Gene 列，前端生成）
              const btnRow = document.createElement("div");
              btnRow.style.display = "flex";
              btnRow.style.justifyContent = "flex-end";
              btnRow.style.gap = "8px";
              btnRow.style.marginTop = "6px";
              const dn = mountDownloadCSVWithGene(card, `biomarkers_${bm.combo || "combo"}_with_gene.csv`);
              if (dn) btnRow.appendChild(dn);
              if (bm.csv) {
                const aRaw = document.createElement("a");
                aRaw.className = "btn";
                aRaw.href = bm.csv;
                aRaw.download = "";
                aRaw.textContent = "Download CSV (raw)";
                btnRow.appendChild(aRaw);
              }
              card.appendChild(btnRow);

              sec.appendChild(card);

              // 异步填充 Gene 列（仅缺失项），完成后重建“with gene”CSV按钮（以确保最新）
              populateGenesForCard(card).then(()=>{
                const old = btnRow.querySelector('a[download$="_with_gene.csv"]');
                if (old) old.remove();
                const dn2 = mountDownloadCSVWithGene(card, `biomarkers_${bm.combo || "combo"}_with_gene.csv`);
                if (dn2) btnRow.prepend(dn2);
              });
            }
          }

          btn.disabled = false;
        } else {
          if (d.status === "failed") {
            status.textContent = `Failed: ${d.message || ""}`;
            btn.disabled = false;
          } else {
            setTimeout(poll, 1000);
          }
        }
      };
      setTimeout(poll, 700);
    }catch(e){
      console.error(e); status.textContent = "Error."; out.textContent = String(e);
      btn.disabled = false;
    }
  };

  console.log("[OK] Part3 ready (auto ENSG→Gene mapping)");
})();

/* =========================
   Part 4 · Consensus Biomarkers (frontend-only, sorted by frequency)
   ========================= */
(function(){
  // ---------- helpers ----------
  const $ = (id) => document.getElementById(id);

  // CSV 简单解析器
  function parseCSV(text){
    const lines = text.replace(/\r/g,'').split('\n').filter(x=>x.trim().length>0);
    if (lines.length === 0) return {header:[], rows:[]};
    const header = lines[0].split(',').map(s=>s.trim());
    const rows = lines.slice(1).map(line => {
      const cols = line.split(',');
      const obj = {};
      header.forEach((h,i)=> obj[h] = (cols[i] ?? '').trim());
      return obj;
    });
    return {header, rows};
  }

  // 列名匹配（不区分大小写）
  function findColumn(header, candidates){
    const idx = {};
    header.forEach((h,i)=> idx[h.toLowerCase()] = i);
    for (const c of candidates){
      if (idx.hasOwnProperty(c.toLowerCase())) return {name: header[idx[c.toLowerCase()]], index: idx[c.toLowerCase()]};
    }
    return null;
  }

  // 去版本号 ENSG000001234.5 -> ENSG000001234
  function stripVersion(x){ try { return String(x).split('.')[0]; } catch(e){ return String(x); } }

  // 取文件名（不含路径）
  function basename(path){ return String(path).split(/[\\/]/).pop(); }

  // MyGene 前端缓存 + 查询
  const ENSG_CACHE = Object.create(null);
  async function fetchSymbolForEnsg(ensg){
    const bare = stripVersion(ensg);
    if (ENSG_CACHE[bare]) return ENSG_CACHE[bare];
    const url = `https://mygene.info/v3/query?q=ensemblgene:${encodeURIComponent(bare)}&fields=symbol&species=human`;
    try {
      const r = await fetch(url);
      if (!r.ok) throw new Error(`mygene ${r.status}`);
      const j = await r.json();
      let sym = "—";
      if (j && Array.isArray(j.hits) && j.hits.length > 0) {
        const hit = j.hits[0];
        if (hit && hit.symbol) sym = String(hit.symbol);
      }
      ENSG_CACHE[bare] = sym || "—";
      return ENSG_CACHE[bare];
    } catch (e) {
      console.warn("MyGene lookup failed:", bare, e);
      ENSG_CACHE[bare] = "—";
      return "—";
    }
  }

  // 把 Gene 列异步补齐
  async function populateGeneColumn(tableTbody){
    const trs = Array.from(tableTbody.querySelectorAll('tr'));
    for (const tr of trs){
      const tds = tr.querySelectorAll('td');
      if (tds.length < 6) continue;  // 预期: rank, ENSG, Gene, freq, avg_rank/avg_z, files
      const ensg = (tds[1].textContent || "").trim();
      const cell = tds[2];
      const cur = (cell.textContent || "").trim();
      if (ensg && (!cur || cur === "—" || cur === "...")){
        cell.textContent = "...";
        const sym = await fetchSymbolForEnsg(ensg);
        cell.textContent = sym || "—";
      }
    }
  }

  function fmt(x, nd=4){ if (x==null || Number.isNaN(x)) return "—"; return (typeof x==="number")? x.toFixed(nd): x; }

  // ---------- DOM ----------
  const fileInput = $("p4-file");
  const topkInput = $("p4-topk");
  const methodSel = $("p4-method");
  const mWrap = $("p4-m-wrap");
  const mInput = $("p4-m");
  const btn = $("p4-run");
  const status = $("p4-status");
  const summaryDiv = $("p4-summary");
  const downloadsDiv = $("p4-downloads");
  const tableHost = $("p4-table");
  const chartDiv = $("p4-chart");
  const fileHint  = document.getElementById("p4-file-hint");
  const fileLabel = document.getElementById("p4-file-label");

  function updateFileHint(){
  const n = fileInput && fileInput.files ? fileInput.files.length : 0;
  if (fileHint)  fileHint.textContent = n ? `${n} file(s) selected` : "No files selected";
  if (fileLabel) fileLabel.textContent = n ? `Choose files (${n})` : "Choose files";
}

  if (fileInput) {
  updateFileHint();                 // init
  fileInput.addEventListener("change", updateFileHint);
  }

  if (!fileInput || !btn) { console.warn("[Part4] UI not found"); return; }

  function syncMVisibility(){
    mWrap.style.display = (methodSel.value === "freq") ? "" : "none";
  }
  methodSel.addEventListener("change", syncMVisibility);
  syncMVisibility();

  // ---------- core: read files ----------
  async function readFiles(files, topk){
    const out = []; // [{name, rows:[{ensg,gene,rank,score}], set, rankMap, zMap}]
    for (const f of files){
      const txt = await f.text();
      const {header, rows} = parseCSV(txt);
      if (header.length === 0) continue;

      const colFeature = findColumn(header, ["ensembl_gene_id", "feature", "ENSG", "ensg"]);
      const colGene    = findColumn(header, ["gene","symbol","hgnc_symbol"]);
      const colRank    = findColumn(header, ["rank"]);
      const colScore   = findColumn(header, ["score","importance","weight","contrib"]);

      if (!colFeature) {
        console.warn(`File ${f.name}: missing feature/ensembl_gene_id column, skipped.`);
        continue;
      }

      // 规整化行
      let items = rows.map((r,i)=>{
        const ensg = stripVersion(r[colFeature.name]);
        const gene = colGene ? (r[colGene.name] || "") : "";
        const rank = colRank ? parseFloat(r[colRank.name]) : (i+1);
        const score = colScore ? parseFloat(r[colScore.name]) : NaN;
        return { ensg, gene, rank, score };
      });

      // 排序 & 取 topk
      if (colRank) items.sort((a,b)=> (a.rank - b.rank));
      else if (colScore) items.sort((a,b)=> (isNaN(b.score)?0:b.score) - (isNaN(a.score)?0:a.score));
      if (topk && topk>0) items = items.slice(0, Math.min(topk, items.length));

      // rankMap / zMap
      const set = new Set(), rankMap = new Map(), zMap = new Map();
      let scMean = 0, scStd = 1;
      const validScores = items.map(x=>x.score).filter(x=>!isNaN(x));
      if (validScores.length >= 2){
        const mu = validScores.reduce((a,b)=>a+b,0)/validScores.length;
        const sd = Math.sqrt(validScores.reduce((a,b)=>a+(b-mu)*(b-mu),0)/(validScores.length-1)) || 1;
        scMean = mu; scStd = sd;
      }
      for (const it of items){
        set.add(it.ensg);
        rankMap.set(it.ensg, it.rank);
        if (!isNaN(it.score)) zMap.set(it.ensg, (it.score - scMean)/scStd);
      }

      out.push({ name: basename(f.name), rows: items, set, rankMap, zMap });
    }
    return out;
  }

  function jaccard(aSet, bSet){
    let inter = 0;
    for (const x of aSet) if (bSet.has(x)) inter++;
    const uni = aSet.size + bSet.size - inter;
    return uni>0 ? inter/uni : 0;
  }

  // ---------- core: consensus ----------
  function buildConsensus(fileObjs, method, m){
    const n = fileObjs.length;
    const allGenes = new Set();
    fileObjs.forEach(o=> o.set.forEach(g=> allGenes.add(g)));

    // 出现次数 / 均值 rank / 均值 z
    const freq = new Map(), avgRank = new Map(), avgZ = new Map(), filesByGene = new Map();
    for (const g of allGenes){
      let f = 0, rSum = 0, zSum = 0, zCnt = 0, inFiles = [];
      for (const o of fileObjs){
        if (o.set.has(g)){
          f++;
          rSum += (o.rankMap.get(g) || 0);
          inFiles.push(o.name);
          if (o.zMap.has(g)) { zSum += o.zMap.get(g); zCnt++; }
        }
      }
      freq.set(g, f);
      if (f>0) avgRank.set(g, rSum/f);
      avgZ.set(g, (zCnt>0) ? (zSum/zCnt) : NaN);
      filesByGene.set(g, inFiles);
    }

    // 方法选择 + 候选集合
    let selected = [];
    if (method === "intersect"){
      selected = Array.from(allGenes).filter(g => freq.get(g) === n);
    } else if (method === "freq"){
      selected = Array.from(allGenes).filter(g => freq.get(g) >= Math.max(1, m));
    } else if (method === "avg_rank" || method === "avg_z"){
      selected = Array.from(allGenes); // 全部纳入，排序由下游决定
    }

    // ✅ 统一在这里按 “频率降序” 排序（次级：avg_rank 升序，其次 avg_z 降序，然后按 ENSG 字典序）
    selected.sort((a,b)=>{
      const d1 = (freq.get(b) - freq.get(a));
      if (d1 !== 0) return d1;
      const arA = avgRank.get(a); const arB = avgRank.get(b);
      if (!isNaN(arA) && !isNaN(arB)) {
        const d2 = arA - arB;
        if (d2 !== 0) return d2;
      } else if (!isNaN(arA)) {
        return -1;
      } else if (!isNaN(arB)) {
        return 1;
      }
      const azA = avgZ.get(a), azB = avgZ.get(b);
      if (!isNaN(azA) && !isNaN(azB)) {
        const d3 = azB - azA;
        if (d3 !== 0) return d3;
      } else if (!isNaN(azA)) {
        return -1;
      } else if (!isNaN(azB)) {
        return 1;
      }
      return String(a).localeCompare(String(b));
    });

    return { selected, freq, avgRank, avgZ, filesByGene, unionSize: allGenes.size };
  }

  // ---------- render ----------
  function renderSummary(fileObjs, topk, consensus, method, m){
    const n = fileObjs.length;
    // pairwise jaccard
    const pairs = [];
    for (let i=0;i<n;i++){
      for (let j=i+1;j<n;j++){
        pairs.push(jaccard(fileObjs[i].set, fileObjs[j].set));
      }
    }
    const jacAvg = pairs.length ? (pairs.reduce((a,b)=>a+b,0)/pairs.length) : 0;

    summaryDiv.innerHTML = `
      <table class="table">
        <tbody>
          <tr><td class="muted">#Files</td><td>${n}</td></tr>
          <tr><td class="muted">top-k per file</td><td>${topk}</td></tr>
          <tr><td class="muted">Union size</td><td>${consensus.unionSize}</td></tr>
          <tr><td class="muted">Selected size</td><td>${consensus.selected.length}</td></tr>
          <tr><td class="muted">Avg pairwise Jaccard (top-k)</td><td>${fmt(jacAvg,3)}</td></tr>
          <tr><td class="muted">Method</td><td>${method}${method==="freq" ? ` (m=${m})` : ""}</td></tr>
        </tbody>
      </table>
    `;
  }

  function renderConsensusTable(consensus){
    const {selected, freq, avgRank, avgZ, filesByGene} = consensus;

    // 构造表格（selected 已按频率降序）
    const headers = ["rank","ENSG","Gene","freq","avg_rank","avg_z","in_files"];
    const thead = `<thead><tr>${headers.map(h=>`<th>${h}</th>`).join('')}</tr></thead>`;

    const rowsHtml = selected.map((g, i) => {
      const files = (filesByGene.get(g)||[]).join("; ");
      const ar = avgRank.get(g);
      const az = avgZ.get(g);
      return `
        <tr>
          <td>${i+1}</td>
          <td><code>${g}</code></td>
          <td>—</td>
          <td>${freq.get(g) ?? 0}</td>
          <td>${isNaN(ar) ? "—" : fmt(ar,3)}</td>
          <td>${isNaN(az) ? "—" : fmt(az,3)}</td>
          <td style="max-width:420px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap" title="${files}">${files}</td>
        </tr>
      `;
    }).join("");

    tableHost.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
        <div><strong>Consensus biomarkers</strong> (${selected.length} rows, sorted by <em>frequency</em>)</div>
        <div id="p4-dl-wrap"></div>
      </div>
      <table class="table">${thead}<tbody id="p4-tbody">${rowsHtml || `<tr><td colspan="7" class="muted">No items</td></tr>`}</tbody></table>
    `;

    // 填充 Gene 列
    const tbody = document.getElementById("p4-tbody");
    if (tbody) populateGeneColumn(tbody);

    // 下载 CSV（带 Gene 列，按当前排序）
    const btnWrap = document.getElementById("p4-dl-wrap");
    if (btnWrap) {
      const a = buildDownloadCSVFromDOM(tbody);
      if (a) btnWrap.appendChild(a);
    }
  }

  function buildDownloadCSVFromDOM(tbody){
    try{
      const rows = Array.from(tbody.querySelectorAll('tr')).map(tr=>{
        const t = tr.querySelectorAll('td');
        return {
          rank: (t[0]?.textContent||"").trim(),
          ensg: (t[1]?.textContent||"").trim(),
          gene: (t[2]?.textContent||"").trim(),
          freq: (t[3]?.textContent||"").trim(),
          avg_rank: (t[4]?.textContent||"").trim(),
          avg_z: (t[5]?.textContent||"").trim(),
          in_files: (t[6]?.getAttribute('title') || t[6]?.textContent || "").trim()
        };
      });
      const headers = ["rank","ensembl_gene_id","symbol","freq","avg_rank","avg_z","in_files"];
      const csv = [headers.join(",")].concat(
        rows.map(r => [r.rank,r.ensg,r.gene,r.freq,r.avg_rank,r.avg_z,`"${r.in_files.replace(/"/g,'""')}"`].join(","))
      ).join("\n");
      const url = URL.createObjectURL(new Blob([csv], {type:"text/csv"}));
      const a = document.createElement("a");
      a.className = "btn";
      a.href = url;
      a.download = "consensus_biomarkers.csv";
      a.textContent = "Download CSV";
      return a;
    }catch(e){
      console.warn("build CSV failed:", e);
      return null;
    }
  }

  function renderFreqHistogram(consensus, fileCount){
    if (!chartDiv || typeof Plotly === "undefined") return;
    const cnt = new Map();
    for (const g of consensus.selected){
      const f = consensus.freq.get(g) || 0;
      cnt.set(f, (cnt.get(f)||0)+1);
    }
    const xs = Array.from(cnt.keys()).sort((a,b)=>a-b);
    const ys = xs.map(x=>cnt.get(x));
    try { Plotly.purge(chartDiv); } catch(e){}
    Plotly.newPlot(chartDiv, [{
      type: "bar",
      x: xs,
      y: ys,
      hovertemplate: "frequency=%{x}<br>#genes=%{y}<extra></extra>"
    }], {
      margin: {t: 10, r: 10, b: 50, l: 50 },
      paper_bgcolor: "#0f141a",
      plot_bgcolor: "#0f141a",
      font: { color: "#e6edf3" },
      xaxis: { title: "Frequency across files (in top-k)", dtick: 1, range: [0.5, fileCount+0.5] },
      yaxis: { title: "#Genes", rangemode: "tozero" }
    }, {displayModeBar:false, responsive:true});
  }

  // ---------- main ----------
  btn.onclick = async ()=>{
    try{
      status.textContent = "";
      tableHost.innerHTML = "";
      summaryDiv.innerHTML = "";
      chartDiv && (chartDiv.innerHTML = "");
      downloadsDiv.innerHTML = "";

      const files = Array.from(fileInput.files || []);
      if (files.length < 2) {
        alert("Please upload at least TWO CSV files (from Part 3 biomarker exports).");
        return;
      }
      const topk = parseInt(topkInput.value || "50");
      const method = methodSel.value;
      const m = parseInt(mInput.value || "2");

      status.textContent = "Parsing files...";
      const objs = await readFiles(files, topk);
      if (objs.length < 2) {
        alert("No valid files parsed. Make sure each CSV has feature/ensembl_gene_id column.");
        status.textContent = "No valid files.";
        return;
      }

      status.textContent = "Building consensus...";
      const cns = buildConsensus(objs, method, m);

      renderSummary(objs, topk, cns, method, m);
      renderConsensusTable(cns);
      renderFreqHistogram({selected: cns.selected, freq: cns.freq}, objs.length);

      // 额外：把文件名列表也导出一下，便于留痕
      const metaCsv = ["file"].concat(objs.map(o=>o.name)).join("\n");
      const url = URL.createObjectURL(new Blob([metaCsv], {type:"text/csv"}));
      const a = document.createElement("a");
      a.className = "btn";
      a.href = url;
      a.download = "consensus_inputs.csv";
      a.textContent = "Download file list";
      downloadsDiv.innerHTML = "";
      downloadsDiv.appendChild(a);

      status.textContent = "Done.";
    }catch(e){
      console.error(e);
      status.textContent = "Error.";
      alert("Consensus failed: " + e);
    }
  };

  console.log("[OK] Part4 ready (sorted by frequency)");
})();










