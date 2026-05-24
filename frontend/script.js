const API_BASE = 'http://localhost:8000';

const STAGE_IDS = {
  extract: 'pStage1',
  batch: 'pStage2',
  infer: 'pStage3',
  recompose: 'pStage4',
};

const state = {
  jobId: null,
  processJobId: null,
  outputUrl: null,
  sseSource: null,
  isPlaying: false,
  isMuted: true,
  isLooping: false,
  videoDuration: 0,
  currentStage: null,
};

const $ = id => document.getElementById(id);

const dom = {
  uploadZone: $('uploadZone'),
  dropTarget: $('dropTarget'),
  fileInput: $('fileInput'),
  browseBtn: $('browseBtn'),
  uploadProgressWrap: $('uploadProgressWrap'),
  uploadProgressFill: $('uploadProgressFill'),
  uploadProgressLabel: $('uploadProgressLabel'),

  workspace: $('workspace'),
  toolPanel: $('toolPanel'),

  fileChipName: $('fileChipName'),
  resetBtn: $('resetBtn'),
  metaDuration: $('metaDuration'),
  metaResolution: $('metaResolution'),
  metaFps: $('metaFps'),

  tabTool1: $('tabTool1'),
  tabTool2: $('tabTool2'),
  configTool1: $('configTool1'),
  configTool2: $('configTool2'),

  removalPrompt: $('removalPrompt'),
  removalDilate: $('removalDilate'),
  removalDilateVal: $('removalDilateVal'),
  temporalSmoothing: $('temporalSmoothing'),

  stylePrompt: $('stylePrompt'),
  styleStrength: $('styleStrength'),
  styleStrengthVal: $('styleStrengthVal'),
  inferenceSteps: $('inferenceSteps'),
  inferenceStepsVal: $('inferenceStepsVal'),
  flowGuidance: $('flowGuidance'),
  flickerSuppression: $('flickerSuppression'),

  processBtn: $('processBtn'),
  useReplicate: $('useReplicate'),

  videoOriginal: $('videoOriginal'),
  videoProcessed: $('videoProcessed'),
  pendingState: $('pendingState'),
  overlayProcessed: $('overlayProcessed'),
  overlayTagProcessed: $('overlayTagProcessed'),
  syncPlayBtn: $('syncPlayBtn'),
  muteBtn: $('muteBtn'),
  loopBtn: $('loopBtn'),

  timelineTrack: $('timelineTrack'),
  timelineFill: $('timelineFill'),
  timelineThumb: $('timelineThumb'),
  timeCurrentLabel: $('timeCurrentLabel'),
  timeTotalLabel: $('timeTotalLabel'),

  progressOverlay: $('progressOverlay'),
  progressFill: $('progressFill'),
  progressPct: $('progressPct'),
  progressStatus: $('progressStatus'),
  progressDetail: $('progressDetail'),

  exportBar: $('exportBar'),
  exportMeta: $('exportMeta'),
  downloadBtn: $('downloadBtn'),
  shareBtn: $('shareBtn'),

  statusDot: $('statusDot'),
  statusLabel: $('statusLabel'),

  toastContainer: $('toastContainer'),
};

async function checkServerHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) });
    const data = await res.json();

    if (data.status === 'ok') {
      dom.statusDot.className = 'status-dot online';
      dom.statusLabel.textContent = 'Server online';
    } else {
      throw new Error('non-ok');
    }
  } catch {
    dom.statusDot.className = 'status-dot offline';
    dom.statusLabel.textContent = 'Server offline';
    showToast('Cannot reach backend. Is uvicorn running?', 'error');
  }
}

function initUploadZone() {
  dom.dropTarget.addEventListener('click', () => dom.fileInput.click());

  dom.browseBtn.addEventListener('click', e => {
    e.stopPropagation();
    dom.fileInput.click();
  });

  dom.dropTarget.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      dom.fileInput.click();
    }
  });

  dom.fileInput.addEventListener('change', e => {
    if (e.target.files[0]) handleFileSelect(e.target.files[0]);
  });

  ['dragenter', 'dragover'].forEach(evt =>
    dom.dropTarget.addEventListener(evt, e => {
      e.preventDefault();
      dom.dropTarget.classList.add('drag-over');
    })
  );

  ['dragleave', 'dragend', 'drop'].forEach(evt =>
    dom.dropTarget.addEventListener(evt, e => {
      e.preventDefault();
      dom.dropTarget.classList.remove('drag-over');
    })
  );

  dom.dropTarget.addEventListener('drop', e => {
    const file = e.dataTransfer?.files[0];
    if (file) handleFileSelect(file);
  });
}

function handleFileSelect(file) {
  const ALLOWED = ['video/mp4', 'video/quicktime', 'video/webm'];
  const MAX_MB = 500;

  if (!ALLOWED.includes(file.type)) {
    showToast(`Unsupported format: ${file.type}. Use MP4, MOV, or WebM.`, 'error');
    return;
  }

  if (file.size > MAX_MB * 1024 * 1024) {
    showToast(`File too large (${(file.size / 1e6).toFixed(0)} MB). Max is ${MAX_MB} MB.`, 'error');
    return;
  }

  dom.uploadProgressWrap.hidden = false;
  dom.fileChipName.textContent = file.name;

  const localUrl = URL.createObjectURL(file);
  dom.videoOriginal.src = localUrl;
  dom.videoOriginal.load();

  const formData = new FormData();
  formData.append('file', file);

  const xhr = new XMLHttpRequest();

  xhr.upload.addEventListener('progress', e => {
    if (e.lengthComputable) {
      const pct = Math.round((e.loaded / e.total) * 100);
      dom.uploadProgressFill.style.width = `${pct}%`;
      dom.uploadProgressLabel.textContent = `Uploading… ${pct}%`;
    }
  });

  xhr.addEventListener('load', () => {
    dom.uploadProgressWrap.hidden = true;

    if (xhr.status === 200) {
      let resp;

      try {
        resp = JSON.parse(xhr.responseText);
      } catch {
        showToast('Unexpected response from server.', 'error');
        return;
      }

      state.jobId = resp.job_id || resp.id || resp.jobId;
      onUploadSuccess(file.name, resp);
    } else {
      showToast(`Upload failed (HTTP ${xhr.status}).`, 'error');
    }
  });

  xhr.addEventListener('error', () => showToast('Network error during upload.', 'error'));
  xhr.addEventListener('abort', () => showToast('Upload cancelled.', 'info'));

  xhr.open('POST', `${API_BASE}/upload`);
  xhr.send(formData);
}

function onUploadSuccess(filename, resp) {
  dom.uploadZone.hidden = true;
  dom.workspace.hidden = false;

  if (resp.duration_s !== undefined) {
    dom.metaDuration.textContent = formatTime(resp.duration_s);
    state.videoDuration = resp.duration_s;
    dom.timeTotalLabel.textContent = formatTime(resp.duration_s);
  }

  if (resp.width && resp.height) {
    dom.metaResolution.textContent = `${resp.width}×${resp.height}`;
  }

  if (resp.fps) {
    dom.metaFps.textContent = `${resp.fps}`;
  }

  dom.processBtn.disabled = false;

  showToast(`${filename} uploaded successfully.`, 'success');

  dom.videoOriginal.addEventListener('loadedmetadata', onOriginalMetaLoaded, { once: true });
}

function onOriginalMetaLoaded() {
  const v = dom.videoOriginal;

  if (!state.videoDuration && v.duration) {
    state.videoDuration = v.duration;
    dom.metaDuration.textContent = formatTime(v.duration);
    dom.timeTotalLabel.textContent = formatTime(v.duration);
  }

  if (v.videoWidth && v.videoHeight && dom.metaResolution.textContent === '—') {
    dom.metaResolution.textContent = `${v.videoWidth}×${v.videoHeight}`;
  }
}

function initToolTabs() {
  [dom.tabTool1, dom.tabTool2].forEach(tab => {
    tab.addEventListener('click', () => {
      [dom.tabTool1, dom.tabTool2].forEach(t => {
        t.classList.remove('active');
        t.setAttribute('aria-selected', 'false');
      });

      [dom.configTool1, dom.configTool2].forEach(p => p.classList.remove('active'));

      tab.classList.add('active');
      tab.setAttribute('aria-selected', 'true');

      const panelId = tab.getAttribute('aria-controls');
      document.getElementById(panelId).classList.add('active');
    });
  });
}

function initSliderLabels() {
  dom.removalDilate.addEventListener('input', () => {
    dom.removalDilateVal.textContent = `${dom.removalDilate.value} px`;
  });

  dom.styleStrength.addEventListener('input', () => {
    dom.styleStrengthVal.textContent = parseFloat(dom.styleStrength.value).toFixed(2);
  });

  dom.inferenceSteps.addEventListener('input', () => {
    dom.inferenceStepsVal.textContent = dom.inferenceSteps.value;
  });
}

function collectToolParams() {
  const activeTool = document.querySelector('.tool-tab.active')?.dataset.tool;

  if (activeTool === 'object_removal') {
    return {
      tool: 'object_removal',
      params: {
        prompt: dom.removalPrompt.value.trim(),
        detection_backend: document.querySelector('input[name="detectionBackend"]:checked')?.value ?? 'grounding_dino',
        inpaint_model: document.querySelector('input[name="inpaintModel"]:checked')?.value ?? 'propainter',
        temporal_smoothing: dom.temporalSmoothing.checked,
        mask_dilation_px: parseInt(dom.removalDilate.value, 10),
      },
    };
  }

  if (activeTool === 'style_transfer') {
    return {
      tool: 'style_transfer',
      params: {
        prompt: dom.stylePrompt.value.trim(),
        style_backbone: document.querySelector('input[name="styleBackbone"]:checked')?.value ?? 'svd',
        style_strength: parseFloat(dom.styleStrength.value),
        flow_guidance: dom.flowGuidance.checked,
        flicker_suppression: dom.flickerSuppression.checked,
        inference_steps: parseInt(dom.inferenceSteps.value, 10),
      },
    };
  }

  return null;
}

async function startProcessing() {
  if (!state.jobId) {
    showToast('No video uploaded yet.', 'error');
    return;
  }

  const toolParams = collectToolParams();

  if (!toolParams) {
    showToast('Could not determine active tool.', 'error');
    return;
  }

  if (toolParams.tool === 'object_removal' && !toolParams.params.prompt) {
    showToast('Please describe what to remove.', 'error');
    dom.removalPrompt.focus();
    return;
  }

  if (toolParams.tool === 'style_transfer' && !toolParams.params.prompt) {
    showToast('Please enter a style prompt.', 'error');
    dom.stylePrompt.focus();
    return;
  }

  dom.processBtn.disabled = true;
  showProgressOverlay();

  try {
    const res = await fetch(`${API_BASE}/process`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        job_id: state.jobId,
        use_replicate: dom.useReplicate.checked,
        ...toolParams,
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail ?? `HTTP ${res.status}`);
    }

    const { job_id } = await res.json();
    state.processJobId = job_id;

    listenForProgress(job_id);
  } catch (err) {
    hideProgressOverlay();
    dom.processBtn.disabled = false;
    showToast(`Pipeline failed to start: ${err.message}`, 'error');
  }
}

function listenForProgress(jobId) {
  if (state.sseSource) {
    state.sseSource.close();
    state.sseSource = null;
  }

  const url = `${API_BASE}/status/${jobId}`;
  const source = new EventSource(url);
  state.sseSource = source;

  source.addEventListener('progress', e => {
    try {
      const data = JSON.parse(e.data);
      handleProgressEvent(data);
    } catch {}
  });

  source.addEventListener('complete', e => {
    try {
      const data = JSON.parse(e.data);
      handleComplete(data);
    } catch {}

    source.close();
    state.sseSource = null;
  });

  source.addEventListener('error', e => {
    let msg = 'Pipeline error.';

    if (e.data) {
      try {
        msg = JSON.parse(e.data).message ?? msg;
      } catch {}
    }

    source.close();
    state.sseSource = null;
    hideProgressOverlay();
    dom.processBtn.disabled = false;
    showToast(msg, 'error');
  });

  source.onerror = () => {
    if (source.readyState === EventSource.CLOSED) {
      return;
    }

    showToast('Lost connection to server. Retrying…', 'info');
  };
}

function handleProgressEvent(data) {
  const { stage, pct, message, detail } = data;

  if (stage && stage !== state.currentStage) {
    if (state.currentStage) {
      const prevDotId = STAGE_IDS[state.currentStage];

      if (prevDotId) {
        const prevEl = $(prevDotId);
        prevEl?.classList.remove('active');
        prevEl?.classList.add('done');
      }
    }

    state.currentStage = stage;

    const currDotId = STAGE_IDS[stage];

    if (currDotId) {
      $(currDotId)?.classList.add('active');
    }
  }

  if (typeof pct === 'number') {
    const clamped = Math.max(0, Math.min(100, pct));
    dom.progressFill.style.width = `${clamped}%`;
    dom.progressPct.textContent = `${Math.round(clamped)}%`;
  }

  if (message) dom.progressStatus.textContent = message;
  if (detail !== undefined) dom.progressDetail.textContent = detail ?? '';
}

function handleComplete(data) {
  Object.values(STAGE_IDS).forEach(id => {
    const el = $(id);
    el?.classList.remove('active');
    el?.classList.add('done');
  });

  handleProgressEvent({ pct: 100, message: 'Done!', detail: '' });

  setTimeout(() => {
    hideProgressOverlay();

    const outputUrl = `${API_BASE}/outputs/${state.processJobId}.mp4`;
    state.outputUrl = outputUrl;

    dom.videoProcessed.src = outputUrl;
    dom.videoProcessed.load();
    dom.videoProcessed.hidden = false;
    dom.pendingState.hidden = true;
    dom.overlayProcessed.hidden = false;

    const toolLabel = data.tool === 'object_removal'
      ? 'OBJECT REMOVAL'
      : data.tool === 'style_transfer'
        ? 'STYLE TRANSFER'
        : 'OUTPUT';

    dom.overlayTagProcessed.textContent = toolLabel;

    showExportBar(data);

    dom.processBtn.disabled = false;
    showToast('Processing complete! Your video is ready.', 'success');
  }, 600);
}

function showProgressOverlay() {
  Object.values(STAGE_IDS).forEach(id => {
    const el = $(id);
    el?.classList.remove('active', 'done');
  });

  state.currentStage = null;

  dom.progressFill.style.width = '0%';
  dom.progressPct.textContent = '0%';
  dom.progressStatus.textContent = 'Initialising pipeline…';
  dom.progressDetail.textContent = '';
  dom.progressOverlay.hidden = false;
  dom.exportBar.hidden = true;
}

function hideProgressOverlay() {
  dom.progressOverlay.hidden = true;
}

function showExportBar(data) {
  dom.exportBar.hidden = false;

  const parts = [];

  if (data.tool) parts.push(data.tool.replace('_', ' '));
  if (data.duration_s) parts.push(`${formatTime(data.duration_s)}`);

  dom.exportMeta.textContent = parts.join(' · ') || 'Processed video ready';
}

function initVideoPlayers() {
  dom.syncPlayBtn.addEventListener('click', togglePlayback);

  document.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

    if (e.code === 'Space') {
      e.preventDefault();
      togglePlayback();
    }
  });

  dom.muteBtn.addEventListener('click', () => {
    state.isMuted = !state.isMuted;
    dom.videoOriginal.muted = state.isMuted;
    dom.videoProcessed.muted = state.isMuted;
    dom.muteBtn.classList.toggle('active', !state.isMuted);
  });

  dom.loopBtn.addEventListener('click', () => {
    state.isLooping = !state.isLooping;
    dom.videoOriginal.loop = state.isLooping;
    dom.videoProcessed.loop = state.isLooping;
    dom.loopBtn.classList.toggle('active', state.isLooping);
  });

  dom.videoOriginal.addEventListener('timeupdate', updateScrubber);

  dom.videoOriginal.addEventListener('ended', () => {
    state.isPlaying = false;
    updatePlayBtn();
  });

  dom.timelineTrack.addEventListener('click', seekToClick);

  dom.timelineTrack.addEventListener('mousemove', e => {
    if (e.buttons === 1) seekToClick(e);
  });
}

function togglePlayback() {
  if (state.isPlaying) {
    dom.videoOriginal.pause();
    dom.videoProcessed.pause();
    state.isPlaying = false;
  } else {
    if (!dom.videoProcessed.hidden) {
      dom.videoProcessed.currentTime = dom.videoOriginal.currentTime;
    }

    dom.videoOriginal.play().catch(() => {});

    if (!dom.videoProcessed.hidden) {
      dom.videoProcessed.play().catch(() => {});
    }

    state.isPlaying = true;
  }

  updatePlayBtn();
}

function updatePlayBtn() {
  dom.syncPlayBtn.innerHTML = state.isPlaying
    ? `<svg viewBox="0 0 20 20" fill="none" aria-label="Pause">
         <rect x="5" y="3" width="4" height="14" rx="1" fill="currentColor"/>
         <rect x="11" y="3" width="4" height="14" rx="1" fill="currentColor"/>
       </svg>`
    : `<svg viewBox="0 0 20 20" fill="none" aria-label="Play">
         <polygon points="5,3 17,10 5,17" fill="currentColor"/>
       </svg>`;
}

function updateScrubber() {
  const v = dom.videoOriginal;
  const pct = v.duration ? (v.currentTime / v.duration) * 100 : 0;

  dom.timelineFill.style.width = `${pct}%`;
  dom.timelineThumb.style.left = `${pct}%`;
  dom.timelineTrack.setAttribute('aria-valuenow', Math.round(pct));
  dom.timeCurrentLabel.textContent = formatTime(v.currentTime);
}

function seekToClick(e) {
  const rect = dom.timelineTrack.getBoundingClientRect();
  const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
  const t = pct * (dom.videoOriginal.duration || 0);

  dom.videoOriginal.currentTime = t;

  if (!dom.videoProcessed.hidden) {
    dom.videoProcessed.currentTime = t;
  }
}

function initExportActions() {
  dom.downloadBtn.addEventListener('click', () => {
    if (!state.outputUrl) return;

    const a = document.createElement('a');
    a.href = state.outputUrl;
    a.download = `flowedit_output_${state.processJobId}.mp4`;
    a.style.display = 'none';

    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  });

  dom.shareBtn.addEventListener('click', async () => {
    if (!state.outputUrl) return;

    try {
      await navigator.clipboard.writeText(state.outputUrl);
      showToast('Link copied to clipboard.', 'success');
    } catch {
      showToast('Could not copy link — check browser permissions.', 'error');
    }
  });
}

function initReset() {
  dom.resetBtn.addEventListener('click', () => {
    if (state.sseSource) {
      state.sseSource.close();
      state.sseSource = null;
    }

    dom.videoOriginal.src = '';
    dom.videoProcessed.src = '';
    dom.videoProcessed.hidden = true;
    dom.pendingState.hidden = false;
    dom.overlayProcessed.hidden = true;

    Object.assign(state, {
      jobId: null,
      processJobId: null,
      outputUrl: null,
      sseSource: null,
      isPlaying: false,
      videoDuration: 0,
      currentStage: null,
    });

    dom.fileInput.value = '';

    dom.workspace.hidden = true;
    dom.uploadZone.hidden = false;

    ['metaDuration', 'metaResolution', 'metaFps'].forEach(k => {
      dom[k].textContent = '—';
    });

    dom.exportBar.hidden = true;
    dom.progressOverlay.hidden = true;
    dom.processBtn.disabled = true;
    dom.timelineFill.style.width = '0%';
    dom.timelineThumb.style.left = '0%';
    dom.timeCurrentLabel.textContent = '0:00';
    dom.timeTotalLabel.textContent = '0:00';

    updatePlayBtn();
  });
}

function showToast(message, type = 'info', duration = 4000) {
  const icons = {
    success: '✓',
    error: '✕',
    info: 'ℹ',
  };

  const toast = document.createElement('div');
  toast.className = `toast toast--${type}`;
  toast.innerHTML = `<span class="toast-icon">${icons[type] ?? 'ℹ'}</span>
                     <span>${escapeHtml(message)}</span>`;

  dom.toastContainer.appendChild(toast);

  setTimeout(() => {
    toast.classList.add('removing');
    toast.addEventListener('animationend', () => toast.remove(), { once: true });
  }, duration);
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function formatTime(secs) {
  if (!isFinite(secs) || secs < 0) return '0:00';

  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60).toString().padStart(2, '0');

  return `${m}:${s}`;
}

document.addEventListener('DOMContentLoaded', () => {
  checkServerHealth();

  initUploadZone();
  initToolTabs();
  initSliderLabels();
  initVideoPlayers();
  initExportActions();
  initReset();

  dom.processBtn.addEventListener('click', startProcessing);

  setInterval(checkServerHealth, 30_000);
});