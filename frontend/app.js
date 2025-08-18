const API_BASE = "http://127.0.0.1:8000";

const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const analyzeBtn = document.getElementById("analyzeBtn");
const resultSection = document.getElementById("result");
const summary = document.getElementById("summary");
const reasonsEl = document.getElementById("reasons");
const featuresEl = document.getElementById("features");

let selectedFile = null;

fileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = () => {
    preview.src = reader.result;
    preview.style.display = "block";
    analyzeBtn.disabled = false;
  };
  reader.readAsDataURL(file);
});

analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
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
