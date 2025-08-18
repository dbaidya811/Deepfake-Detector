const API_BASE = "http://127.0.0.1:8000";

const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const analyzeBtn = document.getElementById("analyzeBtn");
const resultSection = document.getElementById("result");
const summary = document.getElementById("summary");
const reasonsEl = document.getElementById("reasons");
const featuresEl = document.getElementById("features");
const apiStatus = document.getElementById("apiStatus");

let selectedFile = null;
let backendReady = false;

async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/health`, { cache: "no-store" });
    if (!res.ok) throw new Error("health not ok");
    const data = await res.json();
    if (data && data.status === "ok") {
      backendReady = true;
      if (apiStatus) {
        apiStatus.innerHTML = '<span class="dot dot--ok"></span> Backend online';
        apiStatus.classList.remove("warn", "err");
        apiStatus.classList.add("ok");
      }
      // enable analyze only if a file is chosen
      analyzeBtn.disabled = !selectedFile;
      return;
    }
    throw new Error("unexpected health response");
  } catch (e) {
    backendReady = false;
    if (apiStatus) {
      apiStatus.innerHTML = '<span class="dot dot--warn"></span> Waiting for backend…';
      apiStatus.classList.remove("ok", "err");
      apiStatus.classList.add("warn");
    }
    analyzeBtn.disabled = true;
  }
}

// Initial health check + polling until ready
checkHealth();
const healthTimer = setInterval(async () => {
  if (backendReady) {
    clearInterval(healthTimer);
    return;
  }
  await checkHealth();
}, 2500);

fileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = () => {
    preview.src = reader.result;
    preview.style.display = "block";
    analyzeBtn.disabled = !backendReady ? true : false;
  };
  reader.readAsDataURL(file);
});

analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  if (!backendReady) {
    alert("Backend is starting. Please wait a moment…");
    return;
  }
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Analyzing...";

  try {
    const form = new FormData();
    form.append("file", selectedFile);

    const res = await fetch(`${API_BASE}/api/detect`, {
      method: "POST",
      body: form,
    });

    if (!res.ok) {
      throw new Error(`Server error: ${res.status}`);
    }

    const data = await res.json();
    console.log(data);

    const label = data.is_ai_generated ? "AI-Generated" : "Original/Camera";
    const conf = Math.round(data.confidence * 100);
    const badgeClass = data.is_ai_generated ? "badge badge--ai" : "badge badge--real";
    summary.innerHTML = `<span class="${badgeClass}">${label}</span> <span class="conf">Confidence: ${conf}%</span>`;
    document.title = `${label} (${conf}%) · AI Detector`;

    reasonsEl.innerHTML = "";
    data.reasons.forEach((r) => {
      const li = document.createElement("li");
      li.textContent = r;
      reasonsEl.appendChild(li);
    });

    featuresEl.textContent = JSON.stringify(data.features, null, 2);

    resultSection.hidden = false;
    // re-trigger reveal animation each time
    resultSection.classList.remove("reveal");
    void resultSection.offsetWidth; // force reflow
    resultSection.classList.add("reveal");
  } catch (err) {
    alert(err);
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Analyze";
  }
});
