const fmt = new Intl.NumberFormat("en-US", { maximumFractionDigits: 3 });
const pct = new Intl.NumberFormat("en-US", { style: "percent", maximumFractionDigits: 1 });
const apiBase = "";
let activeRows = [];
let activeFile = null;

function metric(label, value) {
  return `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`;
}

function matrixCell(label, value, cls) {
  return `<div class="cell ${cls}"><span>${label}</span><strong>${value}</strong></div>`;
}

function renderMetrics(report) {
  const profile = report.dataset_profile;
  const metrics = report.selected_metrics;
  document.getElementById("metrics").innerHTML = [
    metric("Rows Analyzed", fmt.format(profile.rows)),
    metric("Bankruptcy Rate", pct.format(profile.bankruptcy_rate)),
    metric("Selected Recall", pct.format(metrics.recall)),
    metric("Average Precision", pct.format(metrics.average_precision)),
  ].join("");
}

function renderLeaderboard(report) {
  const selected = report.selected_model;
  document.getElementById("selectedModel").textContent = `Selected: ${selected}`;
  document.getElementById("leaderboard").innerHTML = report.leaderboard
    .map((row) => `
      <tr class="${row.model === selected ? "best" : ""}">
        <td>${row.model}</td>
        <td>${fmt.format(row.average_precision)}</td>
        <td>${fmt.format(row.recall)}</td>
        <td>${fmt.format(row.roc_auc)}</td>
        <td>${fmt.format(row.balanced_accuracy)}</td>
      </tr>
    `)
    .join("");
}

function renderMatrix(report) {
  const m = report.selected_metrics;
  document.getElementById("matrix").innerHTML = [
    matrixCell("True Negative", m.true_negative, "tn"),
    matrixCell("False Positive", m.false_positive, "fp"),
    matrixCell("False Negative", m.false_negative, "fn"),
    matrixCell("True Positive", m.true_positive, "tp"),
  ].join("");
}

function renderImportance(report) {
  const rows = report.feature_importance.slice(0, 12);
  const max = Math.max(...rows.map((row) => row.importance), 0.000001);
  document.getElementById("importance").innerHTML = rows
    .map((row) => `
      <div class="bar-row">
        <span title="${row.feature}">${row.feature}</span>
        <div class="track"><div class="fill" style="width:${(row.importance / max) * 100}%"></div></div>
        <strong>${fmt.format(row.importance)}</strong>
      </div>
    `)
    .join("");
}

function renderProfile(report) {
  const profile = report.dataset_profile;
  const items = [
    ["Features", profile.features],
    ["Columns", profile.columns],
    ["Missing Values", profile.missing_values],
    ["Duplicate Rows", profile.duplicate_rows],
    ["Healthy Firms", profile.class_counts["0"]],
    ["Bankrupt Firms", profile.class_counts["1"]],
  ];
  document.getElementById("profile").innerHTML = items
    .map(([label, value]) => `<div class="profile-item"><span>${label}</span><strong>${fmt.format(value)}</strong></div>`)
    .join("");
  document.getElementById("generatedAt").textContent = new Date(report.generated_at).toLocaleString();
}

function profileItems(target, items) {
  document.getElementById(target).innerHTML = items
    .map(([label, value]) => `<div class="profile-item"><span>${label}</span><strong>${value}</strong></div>`)
    .join("");
}

async function renderOps() {
  const [auditResponse, schemaResponse] = await Promise.all([
    fetch(`${apiBase}/api/audit/summary`, { cache: "no-store" }),
    fetch(`${apiBase}/api/schema`, { cache: "no-store" }),
  ]);

  if (auditResponse.ok) {
    const audit = await auditResponse.json();
    profileItems("audit", [
      ["Requests", fmt.format(audit.total_requests)],
      ["Rows Scored", fmt.format(audit.total_rows_scored)],
      ["Critical", fmt.format(audit.aggregate_risk_counts?.Critical || 0)],
      ["High", fmt.format(audit.aggregate_risk_counts?.High || 0)],
    ]);
    document.getElementById("auditStatus").textContent = "Prediction audit loaded";
  }

  if (schemaResponse.ok) {
    const schema = await schemaResponse.json();
    profileItems("schema", [
      ["Schema", schema.schema_version],
      ["Required Columns", fmt.format(schema.required_column_count)],
      ["Max Rows", fmt.format(schema.max_rows_per_request)],
      ["Target", schema.target],
    ]);
    document.getElementById("schemaStatus").textContent = "Validation contract active";
  }
}

async function boot() {
  try {
    const health = await fetch(`${apiBase}/api/health`, { cache: "no-store" }).catch(() => null);
    if (health && health.ok) {
      const state = await health.json();
      document.getElementById("status").textContent = state.model_ready ? "Prediction API online" : "API online, model loading";
    }

    const response = await fetch(`${apiBase}/api/report`, { cache: "no-store" });
    if (!response.ok) throw new Error(`Report unavailable: ${response.status}`);
    const report = await response.json();
    renderMetrics(report);
    renderLeaderboard(report);
    renderMatrix(report);
    renderImportance(report);
    renderProfile(report);
    await renderOps();
    if (!document.getElementById("status").textContent.includes("API")) {
      document.getElementById("status").textContent = "Live model report loaded";
    }
  } catch (error) {
    document.getElementById("status").textContent = error.message;
  }
}

function renderPredictions(payload) {
  document.getElementById("predictionSummary").innerHTML =
    `<span>${payload.count} rows scored with ${payload.model_name} at threshold ${fmt.format(payload.threshold)}</span>`;
  document.getElementById("predictions").innerHTML = payload.predictions
    .map((row, index) => `
      <tr>
        <td>${index + 1}</td>
        <td>${pct.format(row.bankruptcy_probability)}</td>
        <td>${row.risk_label}</td>
        <td>${row.prediction === 1 ? "Bankruptcy risk" : "No bankruptcy signal"}</td>
      </tr>
    `)
    .join("");
}

function renderDrift(payload) {
  document.getElementById("driftStatus").textContent =
    `${payload.overall_status.toUpperCase()} drift status across ${payload.rows_checked} rows`;
  document.getElementById("drift").innerHTML = payload.top_features
    .slice(0, 12)
    .map((row) => `
      <tr>
        <td title="${row.feature}">${row.feature}</td>
        <td>${row.severity}</td>
        <td>${fmt.format(row.z_shift)}</td>
        <td>${pct.format(row.out_of_range_rate)}</td>
      </tr>
    `)
    .join("");
}

async function scoreRows(rows) {
  activeRows = rows;
  activeFile = null;
  const response = await fetch(`${apiBase}/api/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ rows }),
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Prediction failed");
  renderPredictions(payload);
  renderOps().catch(() => {});
}

async function loadSample() {
  document.getElementById("predictionSummary").innerHTML = "<span>Loading sample rows</span>";
  const response = await fetch(`${apiBase}/api/sample?limit=5`, { cache: "no-store" });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Could not load sample rows");
  await scoreRows(payload.rows);
}

async function uploadCsv(file) {
  activeRows = [];
  activeFile = file;
  document.getElementById("predictionSummary").innerHTML = `<span>Scoring ${file.name}</span>`;
  const form = new FormData();
  form.append("file", file);
  const response = await fetch(`${apiBase}/api/predict`, { method: "POST", body: form });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Prediction failed");
  renderPredictions(payload);
  renderOps().catch(() => {});
}

async function checkDrift() {
  document.getElementById("driftStatus").textContent = "Checking drift";
  let response;
  if (activeFile) {
    const form = new FormData();
    form.append("file", activeFile);
    response = await fetch(`${apiBase}/api/drift`, { method: "POST", body: form });
  } else if (activeRows.length) {
    response = await fetch(`${apiBase}/api/drift`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rows: activeRows }),
    });
  } else {
    await loadSample();
    return checkDrift();
  }
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Drift check failed");
  renderDrift(payload);
}

async function downloadPredictions() {
  let response;
  if (activeFile) {
    const form = new FormData();
    form.append("file", activeFile);
    response = await fetch(`${apiBase}/api/predict.csv`, { method: "POST", body: form });
  } else if (activeRows.length) {
    response = await fetch(`${apiBase}/api/predict.csv`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rows: activeRows }),
    });
  } else {
    await loadSample();
    return downloadPredictions();
  }
  if (!response.ok) {
    const payload = await response.json();
    throw new Error(payload.error || "CSV export failed");
  }
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "bankruptcy_predictions.csv";
  link.click();
  URL.revokeObjectURL(url);
}

document.getElementById("sampleBtn").addEventListener("click", () => {
  loadSample().catch((error) => {
    document.getElementById("predictionSummary").innerHTML = `<span>${error.message}</span>`;
  });
});

document.getElementById("driftBtn").addEventListener("click", () => {
  checkDrift().catch((error) => {
    document.getElementById("driftStatus").textContent = error.message;
  });
});

document.getElementById("downloadBtn").addEventListener("click", () => {
  downloadPredictions().catch((error) => {
    document.getElementById("predictionSummary").innerHTML = `<span>${error.message}</span>`;
  });
});

document.getElementById("csvInput").addEventListener("change", (event) => {
  const [file] = event.target.files;
  if (!file) return;
  uploadCsv(file).catch((error) => {
    document.getElementById("predictionSummary").innerHTML = `<span>${error.message}</span>`;
  });
});

boot();
