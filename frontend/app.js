const updateBtn = document.getElementById('btnUpdateModels');
const updateStatus = document.getElementById('updateModelsStatus');

let lastVideo = null;
let lastAudio = null;
let lastFusion = null;
let lastLive = null;

function verdictLabel(percent) {
  if (percent >= 75) return 'Likely Deepfake';
  if (percent >= 45) return 'Suspicious';
  return 'Likely Real';
}

function setSummary(el, data) {
  if (!el || !data) return;
  const p = Math.round((data.score || 0) * 100);
  el.textContent = `${verdictLabel(p)}: ${p}%`;
}
async function postFile(url, file) {
  const fd = new FormData();
  fd.append('file', file);
  const res = await fetch(url, { method: 'POST', body: fd });
  if (!res.ok) {
    const t = await res.text();
    throw new Error(`HTTP ${res.status}: ${t}`);
  }
  return res.json();
}

function pretty(obj) {
  return JSON.stringify(obj, null, 2);
}

const videoForm = document.getElementById('videoForm');
const videoFile = document.getElementById('videoFile');
const videoResult = document.getElementById('videoResult');
const videoSummary = document.getElementById('videoSummary');
const reportVideoBtn = document.getElementById('reportVideo');

const audioForm = document.getElementById('audioForm');
const audioFile = document.getElementById('audioFile');
const audioResult = document.getElementById('audioResult');
const audioSummary = document.getElementById('audioSummary');
const reportAudioBtn = document.getElementById('reportAudio');

const liveSummary = document.getElementById('liveSummary');
const reportLiveBtn = document.getElementById('reportLive');

videoForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  if (!videoFile.files.length) return;
  const file = videoFile.files[0];
  const btn = videoForm.querySelector('button');
  btn.disabled = true;
  videoResult.textContent = 'Analyzing video...';
  try {
    const data = await postFile('/analyze/video', file);
    lastVideo = data;
    setSummary(videoSummary, data);
    if (reportVideoBtn) reportVideoBtn.disabled = false;
    videoResult.textContent = pretty(data);
  } catch (err) {
    videoResult.textContent = String(err);
  } finally {
    btn.disabled = false;
  }
});

audioForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  if (!audioFile.files.length) return;
  const file = audioFile.files[0];
  const btn = audioForm.querySelector('button');
  btn.disabled = true;
  audioResult.textContent = 'Analyzing audio...';
  try {
    const data = await postFile('/analyze/audio', file);
    lastAudio = data;
    setSummary(audioSummary, data);
    if (reportAudioBtn) reportAudioBtn.disabled = false;
    audioResult.textContent = pretty(data);
  } catch (err) {
    audioResult.textContent = String(err);
  } finally {
    btn.disabled = false;
  }
});

// Live detection via WebSocket
const startLiveBtn = document.getElementById('startLive');
const stopLiveBtn = document.getElementById('stopLive');
const liveVideo = document.getElementById('liveVideo');
const liveResult = document.getElementById('liveResult');

let ws = null;
let mediaStream = null;
let sendTimer = null;
let canvas = null;
let ctx = null;

async function startLive() {
  try {
    if (!canvas) {
      canvas = document.createElement('canvas');
      canvas.width = 384; // match backend resize
      canvas.height = 216;
      ctx = canvas.getContext('2d');
    }
    mediaStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 360 }, audio: false });
    liveVideo.srcObject = mediaStream;
    const wsUrl = (location.origin.replace(/^http/, 'ws')) + '/ws/realtime';
    ws = new WebSocket(wsUrl);
    ws.onopen = () => {
      liveResult.textContent = 'Live started.';
      stopLiveBtn.disabled = false;
      startLiveBtn.disabled = true;
      // send frames at ~2 FPS
      sendTimer = setInterval(() => {
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        if (!liveVideo.videoWidth) return;
        const scale = Math.min(canvas.width / liveVideo.videoWidth, canvas.height / liveVideo.videoHeight);
        const dw = Math.floor(liveVideo.videoWidth * scale);
        const dh = Math.floor(liveVideo.videoHeight * scale);
        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(liveVideo, 0, 0, dw, dh);
        canvas.toBlob((blob) => {
          if (blob && ws && ws.readyState === WebSocket.OPEN) {
            blob.arrayBuffer().then((buf) => ws.send(buf));
          }
        }, 'image/jpeg', 0.7);
      }, 500);
    };
    ws.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data);
        lastLive = data;
        setSummary(liveSummary, data);
        if (reportLiveBtn) reportLiveBtn.disabled = false;
        liveResult.textContent = JSON.stringify(data, null, 2);
      } catch {
        // ignore
      }
    };
    ws.onerror = () => {
      liveResult.textContent = 'WebSocket error';
    };
    ws.onclose = () => {
      liveResult.textContent += '\nClosed.';
      stopLiveBtn.disabled = true;
      startLiveBtn.disabled = false;
    };
  } catch (err) {
    liveResult.textContent = 'Live start failed: ' + err.message;
    await stopLive();
  }
}

async function stopLive() {
  if (sendTimer) {
    clearInterval(sendTimer);
    sendTimer = null;
  }
  if (ws) {
    try { ws.close(); } catch {}
    ws = null;
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach(t => t.stop());
    mediaStream = null;
  }
  if (liveVideo) liveVideo.srcObject = null;
}

if (startLiveBtn && stopLiveBtn) {
  startLiveBtn.addEventListener('click', startLive);
  stopLiveBtn.addEventListener('click', stopLive);
}

// Fusion analysis upload
const fusionForm = document.getElementById('fusionForm');
const fusionVideo = document.getElementById('fusionVideo');
const fusionAudio = document.getElementById('fusionAudio');
const fusionResult = document.getElementById('fusionResult');
const fusionSummary = document.getElementById('fusionSummary');
const reportFusionBtn = document.getElementById('reportFusion');

if (fusionForm) {
  fusionForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    fusionResult.textContent = 'Analyzing fusion...';
    const formData = new FormData();
    if (fusionVideo.files[0]) formData.append('video', fusionVideo.files[0]);
    if (fusionAudio.files[0]) formData.append('audio', fusionAudio.files[0]);
    try {
      const res = await fetch('/analyze/fusion', { method: 'POST', body: formData });
      const json = await res.json();
      lastFusion = json;
      setSummary(fusionSummary, json);
      if (reportFusionBtn) reportFusionBtn.disabled = false;
      fusionResult.textContent = JSON.stringify(json, null, 2);
    } catch (err) {
      fusionResult.textContent = 'Error: ' + err.message;
    }
  });
}

async function submitReport(kind, data) {
  try {
    const user_note = prompt('Optional: add a note for this report');
    const payload = { type: kind, result: data, user_note };
    const res = await fetch('/report', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    const j = await res.json();
    alert(j.ok ? 'Report submitted' : ('Report error: ' + (j.error || 'unknown')));
  } catch (e) {
    alert('Report failed: ' + e.message);
  }
}

if (reportVideoBtn) reportVideoBtn.addEventListener('click', () => lastVideo && submitReport('video', lastVideo));
if (reportAudioBtn) reportAudioBtn.addEventListener('click', () => lastAudio && submitReport('audio', lastAudio));
if (reportFusionBtn) reportFusionBtn.addEventListener('click', () => lastFusion && submitReport('fusion', lastFusion));
if (reportLiveBtn) reportLiveBtn.addEventListener('click', () => lastLive && submitReport('live', lastLive));

if (updateBtn) {
  updateBtn.addEventListener('click', async () => {
    try {
      updateBtn.disabled = true;
      updateStatus.textContent = 'Updating...';
      const res = await fetch('/admin/update_models', { method: 'POST' });
      const j = await res.json();
      updateStatus.textContent = res.ok ? 'Updated' : ('Error: ' + (j.error || res.status));
    } catch (e) {
      updateStatus.textContent = 'Error: ' + e.message;
    } finally {
      updateBtn.disabled = false;
      setTimeout(() => updateStatus.textContent = '', 4000);
    }
  });
}
