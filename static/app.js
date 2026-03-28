/* ═══════════════════════════════════════════════════════
   WeaponShield AI – app.js
   Handles: drag-and-drop, file preview, async upload,
            loading animation, result rendering.
═══════════════════════════════════════════════════════ */

// ── DOM refs ──────────────────────────────────────────
const fileInput    = document.getElementById('fileInput');
const dropZone     = document.getElementById('dropZone');
const dropIcon     = document.getElementById('dropIcon');
const dropText     = document.getElementById('dropText');
const filePreview  = document.getElementById('filePreview');
const fileName     = document.getElementById('fileName');
const fileMeta     = document.getElementById('fileMeta');
const fileTypeIcon = document.getElementById('fileTypeIcon');
const analyzeBtn   = document.getElementById('analyzeBtn');
const loadingPanel = document.getElementById('loadingPanel');
const loadingTitle = document.getElementById('loadingTitle');
const loadingSub   = document.getElementById('loadingSub');
const progressBar  = document.getElementById('progressBar');
const resultPanel  = document.getElementById('resultPanel');
const threatCard   = document.getElementById('threatCard');
const threatIcon   = document.getElementById('threatIcon');
const threatStatus = document.getElementById('threatStatus');
const threatDetail = document.getElementById('threatDetail');
const resultGrid   = document.getElementById('resultGrid');
const uploadPanel  = document.querySelector('.upload-panel');

let selectedFile = null;

// ── Helpers ───────────────────────────────────────────
function formatBytes(bytes) {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function getFileIcon(mimeType) {
  if (!mimeType) return '📄';
  if (mimeType.startsWith('image/')) return '🖼️';
  if (mimeType.startsWith('video/')) return '🎥';
  return '📄';
}

// ── File Selection Handler ────────────────────────────
function handleFileSelected(file) {
  if (!file) return;

  // Validate type
  const allowed = ['image/jpeg', 'image/png', 'image/jpg',
                   'video/mp4', 'video/avi', 'video/quicktime', 'video/x-msvideo'];
  if (!allowed.includes(file.type) && !file.name.match(/\.(jpg|jpeg|png|mp4|avi|mov)$/i)) {
    showToast('⚠️ Unsupported file type. Please upload an image or video.', 'warn');
    return;
  }

  selectedFile = file;

  // Update drop zone look
  dropIcon.textContent = getFileIcon(file.type);
  dropText.textContent = 'File ready for analysis';

  // Show preview card
  fileName.textContent     = file.name;
  fileMeta.textContent     = `${formatBytes(file.size)}  ·  ${file.type || 'unknown type'}`;
  fileTypeIcon.textContent = getFileIcon(file.type);
  filePreview.style.display = 'flex';

  // Enable button
  analyzeBtn.disabled = false;

  // Reset result area
  resultPanel.style.display = 'none';
}

// ── Drag & Drop ───────────────────────────────────────
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) handleFileSelected(file);
});

fileInput.addEventListener('change', (e) => {
  handleFileSelected(e.target.files[0]);
});

// ── Clear File ────────────────────────────────────────
function clearFile() {
  selectedFile = null;
  fileInput.value = '';
  filePreview.style.display  = 'none';
  resultPanel.style.display  = 'none';
  analyzeBtn.disabled        = true;
  dropIcon.textContent       = '☁️';
  dropText.textContent       = 'Drag & drop your file here';
}

// ── Reset Full UI ─────────────────────────────────────
function resetUI() {
  clearFile();
  resultPanel.style.display  = 'none';
  uploadPanel.style.display  = 'block';
}

// ── Loading Progress Sim ───────────────────────────────
let _progressInterval = null;
function startFakeProgress() {
  progressBar.style.width = '0%';
  let pct = 0;
  _progressInterval = setInterval(() => {
    if (pct < 80) {
      pct += Math.random() * 6;
      progressBar.style.width = Math.min(pct, 80) + '%';
    }
  }, 200);
}
function finishProgress() {
  clearInterval(_progressInterval);
  progressBar.style.transition = 'width 0.4s ease';
  progressBar.style.width = '100%';
}

// ── Toast Notification ────────────────────────────────
function showToast(message, type = 'info') {
  const existing = document.getElementById('__toast');
  if (existing) existing.remove();

  const toast = document.createElement('div');
  toast.id = '__toast';
  toast.textContent = message;
  const colors = { info: '#6c63ff', warn: '#f59e0b', error: '#f43f5e', success: '#22d3a5' };
  Object.assign(toast.style, {
    position: 'fixed', bottom: '24px', left: '50%',
    transform: 'translateX(-50%)',
    background: colors[type] || colors.info,
    color: '#fff', padding: '12px 24px',
    borderRadius: '8px', fontSize: '0.9rem',
    fontWeight: '600', zIndex: '9999',
    boxShadow: '0 8px 32px rgba(0,0,0,0.4)',
    animation: 'fadeIn 0.3s ease',
    fontFamily: 'Inter, sans-serif'
  });
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 3500);
}

// ── Main Detection Function ───────────────────────────
async function startDetection() {
  if (!selectedFile) {
    showToast('⚠️ Please select a file first.', 'warn');
    return;
  }

  // Switch UI states
  uploadPanel.style.display  = 'none';
  loadingPanel.style.display = 'block';
  resultPanel.style.display  = 'none';

  const isVideo = selectedFile.type.startsWith('video/');
  loadingTitle.textContent = isVideo
    ? '🎥 Processing video frames…'
    : '🖼️ Analyzing image…';
  loadingSub.textContent = isVideo
    ? 'YOLO is scanning each frame — this may take a moment.'
    : 'Running YOLO object detection on the image.';

  startFakeProgress();

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const response = await fetch('/detect', {
      method: 'POST',
      body: formData
    });

    finishProgress();

    if (!response.ok) {
      throw new Error(`Server responded with status ${response.status}`);
    }

    const data = await response.json();
    console.log('[WeaponShield] Detection response:', data);

    // Short pause so progress reaches 100%
    await new Promise(r => setTimeout(r, 450));

    loadingPanel.style.display = 'none';

    if (data.error) {
      showToast(`❌ ${data.error}`, 'error');
      uploadPanel.style.display = 'block';
      return;
    }

    renderResult(data);

  } catch (err) {
    finishProgress();
    await new Promise(r => setTimeout(r, 350));
    loadingPanel.style.display = 'none';
    uploadPanel.style.display  = 'block';
    console.error('[WeaponShield] Error:', err);
    showToast(`❌ Detection failed: ${err.message}`, 'error');
  }
}

// ── Weapon type → emoji map ────────────────────────────
const WEAPON_ICONS = {
  'rifle'   : '🔫',
  'long gun': '🔫',
  'pistol'  : '🔫',
  'handgun' : '🔫',
  'knife'   : '🗡️',
  'blade'   : '🗡️',
  'bat'     : '🏏',
  'club'    : '🏏',
  'grenade' : '💣',
  'firearm' : '🔫',
  'weapon'  : '⚠️',
};

function getWeaponIcon(type) {
  if (!type) return '⚠️';
  const key = type.toLowerCase();
  for (const [word, icon] of Object.entries(WEAPON_ICONS)) {
    if (key.includes(word)) return icon;
  }
  return '🔫';
}

// ── Render Result ─────────────────────────────────────
function renderResult(data) {
  const summary       = data.result || {};
  const weaponFound   = summary.weapon_detected === true;
  // weapon_name = specific predicted type  (e.g. "Rifle / Long Gun")
  // weapon_generic = raw model label       (e.g. "weapon")
  const weaponType    = summary.weapon_name   || 'No weapon detected';
  const frameCount    = data.frame_count ?? 1;
  const confidenceRaw = summary.confidence;
  const confidencePct = confidenceRaw != null
    ? (confidenceRaw * 100).toFixed(1) + '%'
    : '—';

  // ── Threat card ─────────────────────────────────────
  threatCard.className = 'threat-card ' + (weaponFound ? 'danger' : 'safe');

  if (weaponFound) {
    const icon = getWeaponIcon(weaponType);
    threatIcon.textContent = icon;
    threatStatus.innerHTML =
      `<span style="font-size:0.65em;opacity:0.7;font-weight:500;letter-spacing:1px;">WEAPON DETECTED</span>` +
      `<br/>${weaponType}`;
    threatDetail.textContent =
      `Predicted type: ${weaponType}  ·  Confidence: ${confidencePct}`;
  } else {
    threatIcon.textContent   = '✅';
    threatStatus.textContent = 'No Weapon Detected';
    threatDetail.textContent = 'The uploaded media was fully scanned and appears safe.';
  }

  // ── Stats grid ──────────────────────────────────────
  const isVideo  = selectedFile && selectedFile.type.startsWith('video/');
  const statsData = [
    { val: weaponFound ? '⚠️ THREAT' : '✅ SAFE',  label: 'Verdict' },
    { val: frameCount,                               label: isVideo ? 'Frames Scanned' : 'Images Processed' },
    { val: confidencePct,                            label: 'Detection Confidence' },
    { val: weaponFound ? weaponType : '—',           label: 'Predicted Weapon Type' },
  ];

  resultGrid.innerHTML = statsData.map(s => `
    <div class="result-stat">
      <div class="result-stat-val">${s.val}</div>
      <div class="result-stat-label">${s.label}</div>
    </div>
  `).join('');

  resultPanel.style.display = 'flex';
}
