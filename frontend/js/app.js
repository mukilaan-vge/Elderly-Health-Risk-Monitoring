const API_BASE = "";

const form = document.getElementById("assessmentForm");
const predictButton = document.getElementById("predictButton");
const resetButton = document.getElementById("resetButton");
const formError = document.getElementById("formError");
const emptyResult = document.getElementById("emptyResult");
const predictionResult = document.getElementById("predictionResult");
const resultState = document.getElementById("resultState");

const statusDot = document.querySelector(".status-dot");
const statusText = document.getElementById("statusText");
const modelVersion = document.getElementById("modelVersion");
const apiHealth = document.getElementById("apiHealth");
const modelLoaded = document.getElementById("modelLoaded");
const activeVersion = document.getElementById("activeVersion");
const livePill = document.getElementById("livePill");
const systemHelp = document.getElementById("systemHelp");

const riskLevel = document.getElementById("riskLevel");
const riskIcon = document.getElementById("riskIcon");
const riskBanner = document.getElementById("riskBanner");
const confidenceValue = document.getElementById("confidenceValue");
const resultModelVersion = document.getElementById("resultModelVersion");
const predictionClass = document.getElementById("predictionClass");
const probabilityList = document.getElementById("probabilityList");

const defaultValues = {};
Array.from(form.elements).forEach((element) => {
  if (element.name) defaultValues[element.name] = element.value;
});

function setApiStatus({ online, modelLoadedValue, version }) {
  statusDot.className = `status-dot ${online ? "status-online" : "status-offline"}`;
  statusText.textContent = online ? "API Connected" : "API Offline";
  const safeVersion = version || "—";
  modelVersion.textContent = safeVersion;
  apiHealth.textContent = online ? "Healthy" : "Offline";
  modelLoaded.textContent = modelLoadedValue ? "Ready" : "Unavailable";
  activeVersion.textContent = safeVersion;
  livePill.className = `live-pill ${online ? "online" : "offline"}`;
  livePill.textContent = online ? "Live" : "Offline";
  systemHelp.textContent = online
    ? `Prediction service is ready. Active model version: ${safeVersion}.`
    : "Prediction service is unavailable. Start the FastAPI Docker container and refresh this page.";
}

async function checkHealth() {
  try {
    const response = await fetch(`${API_BASE}/health`, { headers: { Accept: "application/json" } });
    if (!response.ok) throw new Error(`Health check returned ${response.status}`);
    const data = await response.json();
    setApiStatus({ online: data.status === "healthy", modelLoadedValue: data.model_loaded === true, version: data.model_version });
  } catch (error) {
    setApiStatus({ online: false, modelLoadedValue: false, version: null });
  }
}

function getPayload() {
  const data = {};
  const formData = new FormData(form);
  for (const [key, value] of formData.entries()) {
    if (value === "") throw new Error(`Please provide a value for ${key}.`);
    data[key] = Number(value);
  }
  if (data.Age < 0 || data.Age > 120) throw new Error("Age must be between 0 and 120.");
  return data;
}

function showError(message) {
  formError.textContent = message;
  formError.hidden = false;
  resultState.textContent = "Error";
  resultState.className = "result-state error";
}

function clearError() {
  formError.hidden = true;
  formError.textContent = "";
}

function riskClass(level) {
  const value = String(level || "").toLowerCase();
  if (value.includes("high")) return "high";
  if (value.includes("moderate")) return "moderate";
  return "low";
}

function renderProbabilities(probabilities = {}) {
  probabilityList.innerHTML = "";
  const order = ["Low Risk", "Moderate Risk", "High Risk"];
  const entries = order.filter((name) => Object.prototype.hasOwnProperty.call(probabilities, name));
  Object.keys(probabilities).forEach((name) => { if (!entries.includes(name)) entries.push(name); });

  entries.forEach((name) => {
    const value = Number(probabilities[name] || 0);
    const item = document.createElement("div");
    item.className = "probability-item";
    const key = name.toLowerCase().split(" ")[0];
    item.innerHTML = `
      <div class="probability-label"><span>${name}</span><span>${(value * 100).toFixed(2)}%</span></div>
      <div class="probability-track"><div class="probability-fill ${key}" style="width: 0"></div></div>`;
    probabilityList.appendChild(item);
    requestAnimationFrame(() => {
      const fill = item.querySelector(".probability-fill");
      if (fill) fill.style.width = `${Math.max(0, Math.min(100, value * 100))}%`;
    });
  });
}

function renderPrediction(data) {
  emptyResult.hidden = true;
  predictionResult.hidden = false;
  resultState.textContent = "Complete";
  resultState.className = "result-state success";

  const levelClass = riskClass(data.risk_level);
  riskBanner.className = `risk-banner ${levelClass}`;
  riskIcon.textContent = levelClass === "high" ? "!" : levelClass === "moderate" ? "~" : "✓";
  riskLevel.textContent = data.risk_level || "Unknown";
  confidenceValue.textContent = data.confidence == null ? "—" : `${(Number(data.confidence) * 100).toFixed(2)}%`;
  resultModelVersion.textContent = data.model_version || "—";
  predictionClass.textContent = data.prediction ?? "—";
  renderProbabilities(data.probabilities || {});
}

function setLoading(loading) {
  predictButton.disabled = loading;
  predictButton.classList.toggle("loading", loading);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearError();
  setLoading(true);
  try {
    const payload = getPayload();
    const response = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(payload),
    });
    let data = null;
    try { data = await response.json(); } catch (_) {}
    if (!response.ok) {
      const detail = data?.detail || `Prediction request failed (${response.status}).`;
      throw new Error(detail);
    }
    renderPrediction(data);
    setApiStatus({ online: true, modelLoadedValue: true, version: data.model_version });
  } catch (error) {
    showError(error.message || "Unable to reach the prediction service.");
  } finally {
    setLoading(false);
  }
});

resetButton.addEventListener("click", () => {
  Object.entries(defaultValues).forEach(([name, value]) => {
    const element = form.elements[name];
    if (element) element.value = value;
  });
  clearError();
  emptyResult.hidden = false;
  predictionResult.hidden = true;
  resultState.textContent = "Waiting";
  resultState.className = "result-state";
});

checkHealth();
setInterval(checkHealth, 30000);
