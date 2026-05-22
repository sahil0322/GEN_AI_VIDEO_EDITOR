/**
 * FlowEdit — script.js
 *
 * Responsibilities:
 *   1. Upload zone  — drag/drop + click, XHR upload with progress to POST /upload
 *   2. Tool config  — reads all sidebar form values into a params object
 *   3. Pipeline     — POST /process to kick off backend pipeline
 *   4. SSE listener — connects to GET /status/{job_id}, drives all progress UI
 *   5. Video player — loads both video slots, syncs playback, shared scrubber
 *   6. Export       — download button, copy link
 *
 * Backend API surface expected (FastAPI):
 *   POST /upload              → { job_id: string, filename: string, duration_s: number }
 *   POST /process             → { job_id: string }     (starts async task)
 *   GET  /status/{job_id}     → SSE stream (see handleSSEEvent for event schema)
 *   GET  /outputs/{job_id}    → served MP4 file
 *   GET  /health              → { status: "ok" }
 */

/* ============================================================
   CONFIG
   ============================================================ */
const API_BASE = 'http://localhost:8000';  // FastAPI dev server

/** Map of SSE pipeline stages → progress-dot element IDs */
const STAGE_IDS = {
  extract:    'pStage1',
  batch:      'pStage2',
  infer:      'pStage3',
  recompose:  'pStage4',
};

/* ============================================================
   STATE
   ============================================================ */
const state = {
  jobId:          null,     // set after /upload succeeds
  processJobId:   null,     // set after /process succeeds (may differ if re-used)
  outputUrl:      null,     // final video URL for download
  sseSource:      null,     // active EventSource reference
  isPlaying:      false,
  isMuted:        true,
  isLooping:      false,
  videoDuration:  0,
  currentStage:   null,
};

/* ============================================================
   DOM REFERENCES
   (all queried once at init, never re-queried in hot paths)
   ============================================================ */
const $ = id => document.getElementById(id);

const dom = {
  // Upload zone
  uploadZone:          $('uploadZone'),
  dropTarget:          $('dropTarget'),
  fileInput:           $('fileInput'),
  browseBtn:           $('browseBtn'),
  uploadProgressWrap:  $('uploadProgressWrap'),
  uploadProgressFill:  $('uploadProgressFill'),
  uploadProgressLabel: $('uploadProgressLabel'),

  // Workspace
  workspace:           $('workspace'),
  toolPanel:           $('toolPanel'),

  // File chip + meta
  fileChipName:        $('fileChipName'),
  resetBtn:            $('resetBtn'),
  metaDuration:        $('metaDuration'),
  metaResolution:      $('metaResolution'),
  metaFps:             $('metaFps'),

  // Tool tabs
  tabTool1:            $('tabTool1'),
  tabTool2:            $('tabTool2'),
  configTool1:         $('configTool1'),
  configTool2:         $('configTool2'),

  // Tool 1 fields
  removalPrompt:       $('removalPrompt'),
  removalDilate:       $('removalDilate'),
  removalDilateVal:    $('removalDilateVal'),
  temporalSmoothing:   $('temporalSmoothing'),

  // Tool 2 fields
  stylePrompt:         $('stylePrompt'),
  styleStrength:       $('styleStrength'),
  styleStrengthVal:    $('styleStrengthVal'),
  inferenceSteps:      $('inferenceSteps'),
  inferenceStepsVal:   $('inferenceStepsVal'),
  flowGuidance:        $('flowGuidance'),
  flickerSuppression:  $('flickerSuppression'),

  // Process button
  processBtn:          $('processBtn'),
  useReplicate:        $('useReplicate'),

  // Video players
  videoOriginal:       $('videoOriginal'),
  videoProcessed:      $('videoProcessed'),
  pendingState:        $('pendingState'),
  overlayProcessed:    $('overlayProcessed'),
  overlayTagProcessed: $('overlayTagProcessed'),
  syncPlayBtn:         $('syncPlayBtn'),
  muteBtn:             $('muteBtn'),
  loopBtn:             $('loopBtn'),

  // Timeline
  timelineTrack:       $('timelineTrack'),
  timelineFill:        $('timelineFill'),
  timelineThumb:       $('timelineThumb'),
  timeCurrentLabel:    $('timeCurrentLabel'),
  timeTotalLabel:      $('timeTotalLabel'),

  // Progress overlay
  progressOverlay:     $('progressOverlay'),
  progressFill:        $('progressFill'),
  progressPct:         $('progressPct'),
  progressStatus:      $('progressStatus'),
  progressDetail:      $('progressDetail'),

  // Export bar
  exportBar:           $('exportBar'),
  exportMeta:          $('exportMeta'),
  downloadBtn:         $('downloadBtn'),
  shareBtn:            $('shareBtn'),

  // Status indicator
  statusDot:           $('statusDot'),
  statusLabel:         $('statusLabel'),

  // Toast container
  toastContainer:      $('toastContainer'),
};


/* ============================================================
   1. SERVER HEALTH CHECK
   ============================================================ */

/** Pings /health and updates the header status indicator. */
async function checkServerHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) });
    const data = await res.json();
    if (data.status === 'ok') {
      dom.statusDot.className   = 'status-dot online';
      dom.statusLabel.textContent = 'Server online';
    } else {
      throw new Error('non-ok');
    }
  } catch {
    dom.statusDot.className     = 'status-dot offline';
    dom.statusLabel.textContent = 'Server offline';
    showToast('Cannot reach backend. Is uvicorn running?', 'error');
  }
}


/* ============================================================
   2. UPLOAD ZONE
   ============================================================ */

function initUploadZone() {
  // Click → open file picker
  dom.dropTarget.addEventListener('click', () => dom.fileInput.click());
  dom.browseBtn.addEventListener('click',  e => { e.stopPropagation(); dom.fileInput.click(); });

  // Keyboard accessibility (Enter / Space on focusable drop target)
  dom.dropTarget.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); dom.fileInput.click(); }
  });

  // File input change
  dom.fileInput.addEventListener('change', e => {
    if (e.target.files[0]) handleFileSelect(e.target.files[0]);
  });

  // Drag & drop events
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

/**
 * Validates the selected file then POSTs it to /upload via XHR
 * (XHR used instead of fetch so we get real upload progress events).
 *
 * @param {File} file
 */
function handleFileSelect(file) {
  const ALLOWED = ['video/mp4', 'video/quicktime', 'video/webm'];
  const MAX_MB  = 500;

  if (!ALLOWED.includes(file.type)) {
    showToast(`Unsupported format: ${file.type}. Use MP4, MOV, or WebM.`, 'error');
    return;
  }
  if (file.size > MAX_MB * 1024 * 1024) {
    showToast(`File too large (${(file.size / 1e6).toFixed(0)} MB). Max is ${MAX_MB} MB.`, 'error');
    return;
  }

  // Show upload progress UI
  dom.uploadProgressWrap.hidden = false;
  dom.fileChipName.textContent  = file.name;

  // Preview the original video locally before upload completes
  const localUrl = URL.createObjectURL(file);
  dom.videoOriginal.src = localUrl;
  dom.videoOriginal.load();

  const formData = new FormData();
  formData.append('file', file);

  const xhr = new XMLHttpRequest();

  xhr.upload.addEventListener('progress', e => {
    if (e.lengthComputable) {
      const pct = Math.round((e.loaded / e.total) * 100);
      dom.uploadProgressFill.style.width    = `${pct}%`;
      dom.uploadProgressLabel.textContent   = `Uploading… ${pct}%`;
    }
  });

  xhr.addEventListener('load', () => {
    dom.uploadProgressWrap.hidden = true;

    if (xhr.status === 200) {
      let resp;
      try { resp = JSON.parse(xhr.responseText); }
      catch { showToast('Unexpected response from server.', 'error'); return; }

      state.jobId = resp.job_id || resp.id || resp.jobId;
      onUploadSuccess(file.name, resp);
    } else {
      showToast(`Upload failed (HTTP ${xhr.status}).`, 'error');
    }
  });

  xhr.addEventListener('error',  () => showToast('Network error during upload.', 'error'));
  xhr.addEventListener('abort',  () => showToast('Upload cancelled.', 'info'));

  xhr.open('POST', `${API_BASE}/upload`);
  xhr.send(formData);
}

/**
 * Called when the server has accepted the file.
 * Transitions the UI from the upload zone to the workspace.
 *
 * @param {string}  filename
 * @param {object}  resp  - server response: { job_id, filename, duration_s, fps, width, height }
 */
function onUploadSuccess(filename, resp) {
  // Show workspace, hide upload zone
  dom.uploadZone.hidden  = true;
  dom.workspace.hidden   = false;

  // Populate meta row
  if (resp.duration_s !== undefined) {
    dom.metaDuration.textContent   = formatTime(resp.duration_s);
    state.videoDuration            = resp.duration_s;
    dom.timeTotalLabel.textContent = formatTime(resp.duration_s);
  }
  if (resp.width && resp.height) {
    dom.metaResolution.textContent = `${resp.width}×${resp.height}`;
  }
  if (resp.fps) {
    dom.metaFps.textContent = `${resp.fps}`;
  }

  // Enable the process button now that we have a job_id
  dom.processBtn.disabled = false;

  showToast(`${filename} uploaded successfully.`, 'success');

  // Wire up video metadata once the local preview loads
  dom.videoOriginal.addEventListener('loadedmetadata', onOriginalMetaLoaded, { once: true });
}

/** Populates meta fields from the local video element (fallback if server doesn't return them). */
function onOriginalMetaLoaded() {
  const v = dom.videoOriginal;
  if (!state.videoDuration && v.duration) {
    state.videoDuration            = v.duration;
    dom.metaDuration.textContent   = formatTime(v.duration);
    dom.timeTotalLabel.textContent = formatTime(v.duration);
  }
  if (v.videoWidth && v.videoHeight && dom.metaResolution.textContent === '—') {
    dom.metaResolution.textContent = `${v.videoWidth}×${v.videoHeight}`;
  }
}


/* ============================================================
   3. TOOL TABS
   ============================================================ */

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


/* ============================================================
   4. LIVE-UPDATE SLIDER LABELS
   ============================================================ */

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


/* ============================================================
   5. COLLECT TOOL PARAMETERS
   ============================================================ */

/**
 * Reads all sidebar form controls and returns a structured params object
 * that will be sent verbatim in the POST /process request body.
 *
 * The backend pipeline/orchestrator.py unpacks these into the correct
 * model-specific config objects.
 *
 * @returns {{ tool: string, params: object }}
 */
function collectToolParams() {
  const activeTool = document.querySelector('.tool-tab.active')?.dataset.tool;

  if (activeTool === 'object_removal') {
    return {
      tool: 'object_removal',
      params: {
        prompt:             dom.removalPrompt.value.trim(),
        detection_backend:  document.querySelector('input[name="detectionBackend"]:checked')?.value ?? 'grounding_dino',
        inpaint_model:      document.querySelector('input[name="inpaintModel"]:checked')?.value ?? 'propainter',
        /**
         * temporal_smoothing flag is read here and forwarded to the backend.
         * When true, mask_propagator.py enables RAFT-based mask warping
         * (optical_flow.py → warp_mask_temporal) instead of per-frame SAM re-inference.
         */
        temporal_smoothing: dom.temporalSmoothing.checked,
        mask_dilation_px:   parseInt(dom.removalDilate.value, 10),
      },
    };
  }

  if (activeTool === 'style_transfer') {
    return {
      tool: 'style_transfer',
      params: {
        prompt:             dom.stylePrompt.value.trim(),
        style_backbone:     document.querySelector('input[name="styleBackbone"]:checked')?.value ?? 'svd',
        style_strength:     parseFloat(dom.styleStrength.value),
        /**
         * flow_guidance enables flow_guidance.py in the style_transfer tool.
         * When true, the previous styled frame is warped using RAFT's forward
         * flow field and passed as ControlNet conditioning for the next frame.
         * This is the primary mechanism for inter-frame colour continuity.
         */
        flow_guidance:      dom.flowGuidance.checked,
        /**
         * flicker_suppression enables flicker_suppressor.py.
         * When true, a per-pixel flow-consistency score blends the new diffusion
         * output with the warped previous frame, eliminating the strobing effect
         * that occurs when each frame is styled independently.
         */
        flicker_suppression: dom.flickerSuppression.checked,
        inference_steps:    parseInt(dom.inferenceSteps.value, 10),
      },
    };
  }

  return null;
}


/* ============================================================
   6. PROCESS BUTTON — trigger pipeline
   ============================================================ */

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

  // Validate required fields
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
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        job_id:       state.jobId,
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

    // Start SSE listener for real-time pipeline progress
    listenForProgress(job_id);

  } catch (err) {
    hideProgressOverlay();
    dom.processBtn.disabled = false;
    showToast(`Pipeline failed to start: ${err.message}`, 'error');
  }
}


/* ============================================================
   7. SSE — real-time pipeline progress
   ============================================================ */

/**
 * Opens a Server-Sent Events connection to GET /status/{jobId}.
 *
 * Expected SSE event schema from the backend (sse-starlette):
 *   event: progress
 *   data: {
 *     stage:   "extract" | "batch" | "infer" | "recompose",
 *     pct:     0–100,
 *     message: string,         // e.g. "Processing frame 42 / 120"
 *     detail:  string,         // e.g. "Running ProPainter batch 3/15"  (optional)
 *   }
 *
 *   event: complete
 *   data: { output_path: string, duration_s: number, tool: string }
 *
 *   event: error
 *   data: { message: string }
 *
 * @param {string} jobId
 */
function listenForProgress(jobId) {
  // Close any stale connection
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
    } catch { /* malformed JSON — skip */ }
  });

  source.addEventListener('complete', e => {
    try {
      const data = JSON.parse(e.data);
      handleComplete(data);
    } catch { /* malformed JSON — skip */ }
    source.close();
    state.sseSource = null;
  });

  source.addEventListener('error', e => {
    let msg = 'Pipeline error.';
    if (e.data) {
      try { msg = JSON.parse(e.data).message ?? msg; } catch { /* ignore */ }
    }
    source.close();
    state.sseSource = null;
    hideProgressOverlay();
    dom.processBtn.disabled = false;
    showToast(msg, 'error');
  });

  // Fallback: native EventSource onerror (connection lost)
  source.onerror = () => {
    if (source.readyState === EventSource.CLOSED) {
      // Server closed cleanly — handled above by 'complete'/'error'
      return;
    }
    showToast('Lost connection to server. Retrying…', 'info');
  };
}

/**
 * Handles a single SSE "progress" event payload.
 * Updates: stage indicators, progress bar, status text.
 *
 * @param {{ stage: string, pct: number, message: string, detail?: string }} data
 */
function handleProgressEvent(data) {
  const { stage, pct, message, detail } = data;

  // Advance pipeline stage indicators
  if (stage && stage !== state.currentStage) {
    // Mark previous stages as done
    if (state.currentStage) {
      const prevDotId = STAGE_IDS[state.currentStage];
      if (prevDotId) {
        const prevEl = $(prevDotId);
        prevEl?.classList.remove('active');
        prevEl?.classList.add('done');
      }
    }
    state.currentStage = stage;

    // Activate current stage
    const currDotId = STAGE_IDS[stage];
    if (currDotId) {
      $(currDotId)?.classList.add('active');
    }
  }

  // Update progress bar
  if (typeof pct === 'number') {
    const clamped = Math.max(0, Math.min(100, pct));
    dom.progressFill.style.width   = `${clamped}%`;
    dom.progressPct.textContent    = `${Math.round(clamped)}%`;
  }

  if (message) dom.progressStatus.textContent = message;
  if (detail  !== undefined) dom.progressDetail.textContent = detail ?? '';
}

/**
 * Handles the SSE "complete" event.
 * Loads the processed video, hides the progress overlay, shows the export bar.
 *
 * @param {{ output_path: string, duration_s: number, tool: string }} data
 */
function handleComplete(data) {
  // Mark all stages done
  Object.values(STAGE_IDS).forEach(id => {
    const el = $(id);
    el?.classList.remove('active');
    el?.classList.add('done');
  });

  // Set progress to 100%
  handleProgressEvent({ pct: 100, message: 'Done!', detail: '' });

  // Short delay so the user sees 100% before the overlay disappears
  setTimeout(() => {
    hideProgressOverlay();

    // Construct the playback URL (FastAPI serves outputs/ as a static route)
   const outputUrl = `${API_BASE}/outputs/${state.processJobId}.mp4`;
    state.outputUrl  = outputUrl;

    // Load processed video
    dom.videoProcessed.src = outputUrl;
    dom.videoProcessed.load();
    dom.videoProcessed.hidden    = false;
    dom.pendingState.hidden      = true;
    dom.overlayProcessed.hidden  = false;

    // Update overlay tag to the tool name
    const toolLabel = data.tool === 'object_removal' ? 'OBJECT REMOVAL'
                    : data.tool === 'style_transfer'  ? 'STYLE TRANSFER'
                    : 'OUTPUT';
    dom.overlayTagProcessed.textContent = toolLabel;

    // Show export bar
    showExportBar(data);

    dom.processBtn.disabled = false;
    showToast('Processing complete! Your video is ready.', 'success');

  }, 600);
}


/* ============================================================
   8. PROGRESS OVERLAY helpers
   ============================================================ */

function showProgressOverlay() {
  // Reset all stage dots
  Object.values(STAGE_IDS).forEach(id => {
    const el = $(id);
    el?.classList.remove('active', 'done');
  });
  state.currentStage = null;

  dom.progressFill.style.width   = '0%';
  dom.progressPct.textContent    = '0%';
  dom.progressStatus.textContent = 'Initialising pipeline…';
  dom.progressDetail.textContent = '';
  dom.progressOverlay.hidden     = false;
  dom.exportBar.hidden           = true;
}

function hideProgressOverlay() {
  dom.progressOverlay.hidden = true;
}


/* ============================================================
   9. EXPORT BAR
   ============================================================ */

function showExportBar(data) {
  dom.exportBar.hidden = false;

  const parts = [];
  if (data.tool)       parts.push(data.tool.replace('_', ' '));
  if (data.duration_s) parts.push(`${formatTime(data.duration_s)}`);
  dom.exportMeta.textContent = parts.join(' · ') || 'Processed video ready';
}


/* ============================================================
   10. VIDEO PLAYER — sync + scrubber
   ============================================================ */

function initVideoPlayers() {
  // Play / pause sync button
  dom.syncPlayBtn.addEventListener('click', togglePlayback);

  // Space bar toggles playback
  document.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    if (e.code === 'Space') { e.preventDefault(); togglePlayback(); }
  });

  // Mute toggle
  dom.muteBtn.addEventListener('click', () => {
    state.isMuted = !state.isMuted;
    dom.videoOriginal.muted   = state.isMuted;
    dom.videoProcessed.muted  = state.isMuted;
    dom.muteBtn.classList.toggle('active', !state.isMuted);
  });

  // Loop toggle
  dom.loopBtn.addEventListener('click', () => {
    state.isLooping = !state.isLooping;
    dom.videoOriginal.loop   = state.isLooping;
    dom.videoProcessed.loop  = state.isLooping;
    dom.loopBtn.classList.toggle('active', state.isLooping);
  });

  // Keep scrubber in sync with the original video's timeupdate
  dom.videoOriginal.addEventListener('timeupdate', updateScrubber);

  // When original video ends, sync processed video position too
  dom.videoOriginal.addEventListener('ended', () => {
    state.isPlaying = false;
    updatePlayBtn();
  });

  // Allow clicking on the timeline to seek
  dom.timelineTrack.addEventListener('click', seekToClick);
  dom.timelineTrack.addEventListener('mousemove', e => {
    if (e.buttons === 1) seekToClick(e); // drag-seek
  });
}

function togglePlayback() {
  if (state.isPlaying) {
    dom.videoOriginal.pause();
    dom.videoProcessed.pause();
    state.isPlaying = false;
  } else {
    // Sync processed video to same timestamp before playing
    if (!dom.videoProcessed.hidden) {
      dom.videoProcessed.currentTime = dom.videoOriginal.currentTime;
    }
    dom.videoOriginal.play().catch(() => {});
    if (!dom.videoProcessed.hidden) dom.videoProcessed.play().catch(() => {});
    state.isPlaying = true;
  }
  updatePlayBtn();
}

function updatePlayBtn() {
  // Swap the play icon for a pause icon when playing
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
  const v   = dom.videoOriginal;
  const pct = v.duration ? (v.currentTime / v.duration) * 100 : 0;
  dom.timelineFill.style.width         = `${pct}%`;
  dom.timelineThumb.style.left         = `${pct}%`;
  dom.timelineTrack.setAttribute('aria-valuenow', Math.round(pct));
  dom.timeCurrentLabel.textContent     = formatTime(v.currentTime);
}

function seekToClick(e) {
  const rect = dom.timelineTrack.getBoundingClientRect();
  const pct  = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
  const t    = pct * (dom.videoOriginal.duration || 0);

  dom.videoOriginal.currentTime  = t;
  if (!dom.videoProcessed.hidden) dom.videoProcessed.currentTime = t;
}


/* ============================================================
   11. EXPORT — download + share
   ============================================================ */

function initExportActions() {
  dom.downloadBtn.addEventListener('click', () => {
    if (!state.outputUrl) return;

    // Trigger download via a temporary anchor element
    const a       = document.createElement('a');
    a.href        = state.outputUrl;
    a.download    = `flowedit_output_${state.processJobId}.mp4`;
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


/* ============================================================
   12. RESET — start over
   ============================================================ */

function initReset() {
  dom.resetBtn.addEventListener('click', () => {
    // Close any live SSE connection
    if (state.sseSource) {
      state.sseSource.close();
      state.sseSource = null;
    }

    // Reset videos
    dom.videoOriginal.src    = '';
    dom.videoProcessed.src   = '';
    dom.videoProcessed.hidden = true;
    dom.pendingState.hidden   = false;
    dom.overlayProcessed.hidden = true;

    // Reset state
    Object.assign(state, {
      jobId: null, processJobId: null, outputUrl: null,
      sseSource: null, isPlaying: false, videoDuration: 0, currentStage: null,
    });

    // Reset file input
    dom.fileInput.value = '';

    // Hide workspace, show upload zone
    dom.workspace.hidden   = true;
    dom.uploadZone.hidden  = false;

    // Reset meta + export
    ['metaDuration', 'metaResolution', 'metaFps'].forEach(k => { dom[k].textContent = '—'; });
    dom.exportBar.hidden     = true;
    dom.progressOverlay.hidden = true;
    dom.processBtn.disabled  = true;
    dom.timelineFill.style.width = '0%';
    dom.timelineThumb.style.left = '0%';
    dom.timeCurrentLabel.textContent = '0:00';
    dom.timeTotalLabel.textContent   = '0:00';
    updatePlayBtn();
  });
}


/* ============================================================
   13. TOAST NOTIFICATIONS
   ============================================================ */

/**
 * Shows a temporary toast notification.
 *
 * @param {string} message
 * @param {'success'|'error'|'info'} type
 * @param {number} duration  ms before auto-dismiss (default 4000)
 */
function showToast(message, type = 'info', duration = 4000) {
  const icons = { success: '✓', error: '✕', info: 'ℹ' };

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

/** Minimal HTML escape to prevent XSS in toast messages. */
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}


/* ============================================================
   14. UTILITIES
   ============================================================ */

/**
 * Formats a number of seconds as "M:SS".
 * @param {number} secs
 * @returns {string}
 */
function formatTime(secs) {
  if (!isFinite(secs) || secs < 0) return '0:00';
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60).toString().padStart(2, '0');
  return `${m}:${s}`;
}


/* ============================================================
   15. BOOTSTRAP — wire everything up on DOMContentLoaded
   ============================================================ */

document.addEventListener('DOMContentLoaded', () => {
  // Check server health (non-blocking)
  checkServerHealth();

  // Initialise all UI modules
  initUploadZone();
  initToolTabs();
  initSliderLabels();
  initVideoPlayers();
  initExportActions();
  initReset();

  // Process button click handler
  dom.processBtn.addEventListener('click', startProcessing);

  // Re-check server health every 30s
  setInterval(checkServerHealth, 30_000);
});
