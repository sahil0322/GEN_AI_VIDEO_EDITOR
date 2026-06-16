# FlowEdit — Generative AI Video Enhancement Platform

A full-stack video editing platform powered by latent diffusion models and
optical flow. Upload an MP4, apply AI-driven object removal or style transfer,
and download a temporally-consistent output — all from a browser UI.

---

## Architecture

```
Browser (HTML/CSS/JS)
  │  POST /upload       → FastAPI
  │  POST /process      → FastAPI → BackgroundTask
  │  GET  /status/{id}  ← SSE stream (real-time progress)
  │  GET  /outputs/{id} ← processed MP4

FastAPI pipeline
  Stage 1: extractor.py   → FFmpeg → frames/ + audio.aac
  Stage 2: batcher.py     → List[List[Path]]
  Stage 3: AI models      → processed frames
  Stage 4: recomposer.py  → FFmpeg → output.mp4

Tool 1 — Object Removal
  GroundingDINO (text → bbox) → SAM (bbox → mask)
  → mask_propagator (RAFT warp across frames)
  → ProPainter / E2FGVI (temporal inpainting)

Tool 2 — Style Transfer
  flow_guidance (RAFT warp prev styled frame as init)
  → style_engine (SVD or ControlNet img2img)
  → flicker_suppressor (consistency_score blend)
```

---

## Requirements

| Requirement | Minimum | Notes |
|---|---|---|
| Python | 3.10+ | |
| FFmpeg | 5.0+ | Must be on PATH |
| CUDA GPU | 8 GB VRAM | Tool 1 only; 12 GB for SVD |
| Disk | 25 GB free | For weights + storage |
| RAM | 16 GB | |

No GPU? Set `USE_REPLICATE_FALLBACK=true` in `.env` and add your
[Replicate API token](https://replicate.com/account/api-tokens).

---

## Quick Start

```bash
# 1. Clone
git clone <repo-url> flowedit && cd flowedit

# 2. One-command setup (installs all deps + downloads weights ~4 GB)
chmod +x setup.sh && ./setup.sh

# 3. Add your Replicate token if no local GPU
echo "REPLICATE_API_TOKEN=r8_your_token_here" >> .env

# 4. Start
make run

# 5. Open browser
open http://localhost:5500
```

---

## Manual Setup (step by step)

### System packages
```bash
# Ubuntu / Debian
sudo apt install ffmpeg python3.10 python3-pip

# macOS
brew install ffmpeg python@3.10
```

### Python dependencies
```bash
pip install -r requirements.txt
```

### Source-only packages
```bash
# RAFT optical flow
git clone https://github.com/princeton-vl/RAFT.git && pip install -e RAFT/

# ProPainter video inpainting
git clone https://github.com/sczhou/ProPainter.git && pip install -e ProPainter/

# GroundingDINO object detection
git clone https://github.com/IDEA-Research/GroundingDINO.git && pip install -e GroundingDINO/

# Segment Anything Model
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### Model weights
```bash
python3 download_weights.py
```

Downloads (~4 GB total):

| File | Size | Purpose |
|---|---|---|
| `weights/sam_vit_h_4b8939.pth` | 2.4 GB | SAM segmentation |
| `weights/groundingdino_swint_ogc.pth` | 700 MB | Text-to-bbox detection |
| `weights/raft-things.pth` | 20 MB | Optical flow |
| `weights/ProPainter.pth` | 800 MB | Video inpainting |

HuggingFace models (SVD, ControlNet, SD 1.5) download automatically on
first use via `diffusers` and cache in `~/.cache/huggingface/`.

---

## Running

```bash
make run          # backend (port 8000) + frontend (port 5500)
make api          # backend only
make frontend     # frontend only
```

- **UI:**   http://localhost:5500
- **API:**  http://localhost:8000
- **Docs:** http://localhost:8000/docs  (Swagger UI)

---

## Usage

### Tool 1 — Object Removal

1. Upload an MP4 (drag & drop or browse)
2. Select **Object Removal** tab
3. Type what to remove: `"the person on the right"`
4. Choose detection backend:
   - **GroundingDINO** — text description → auto-detect → SAM mask *(recommended)*
   - **SAM + bounding box** — draw a box on the first frame
5. Choose inpainting model:
   - **ProPainter** — best quality, needs 6 GB VRAM
   - **E2FGVI** — faster, 4 GB VRAM
6. Toggle **Temporal mask smoothing** (recommended) — uses RAFT to propagate
   the mask frame-by-frame instead of re-running SAM on every frame
7. Click **Run pipeline**

### Tool 2 — Style Transfer

1. Upload an MP4
2. Select **Style Transfer** tab
3. Enter a style prompt: `"a snowy winter landscape, cinematic, 8k"`
4. Choose backbone:
   - **Stable Video Diffusion** — most temporally consistent, 12 GB VRAM
   - **ControlNet** — better text control, 4 GB VRAM
5. Adjust **Style strength** (0.1 = subtle, 1.0 = total transformation)
6. Keep **Optical flow guidance** ON — warps previous styled frame as diffusion
   init to prevent inter-frame colour jumps
7. Keep **Flicker suppression** ON — blends new output with warped-previous
   using per-pixel RAFT consistency score to eliminate strobing
8. Click **Run pipeline**

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload` | Upload video → `{job_id, fps, width, height, duration_s}` |
| `POST` | `/process` | Start pipeline → `{job_id}` |
| `GET`  | `/status/{job_id}` | SSE stream: `progress` / `complete` / `error` events |
| `GET`  | `/outputs/{job_id}` | Download processed MP4 |
| `GET`  | `/health` | `{status, gpu_available, gpu_name}` |

### SSE progress event
```json
{
  "event": "progress",
  "stage": "infer",
  "pct": 42,
  "message": "Processing frame 50 / 120",
  "detail": "ProPainter batch 6 / 15"
}
```

### SSE complete event
```json
{
  "event": "complete",
  "output_path": "storage/outputs/a3f7c2d1.mp4",
  "duration_s": 12.4,
  "tool": "object_removal"
}
```

---

## Testing

```bash
make test          # full suite
make test-fast     # skip GPU-dependent tests
make test-api      # route tests only
make test-pipeline # extractor + temporal engine tests only
```

Tests use synthetic videos — no real footage needed. GPU tests are marked
`@pytest.mark.slow` and skipped with `make test-fast`.

---

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8000` | Backend server port |
| `CORS_ORIGINS` | `http://localhost:5500` | Allowed frontend origins |
| `MAX_UPLOAD_MB` | `500` | Upload file size limit |
| `INFER_BATCH_SIZE` | `8` | Frames per inference batch (lower if OOM) |
| `RAFT_BATCH_SIZE` | `4` | Frame pairs per RAFT call |
| `USE_REPLICATE_FALLBACK` | `false` | Route heavy models to cloud |
| `REPLICATE_API_TOKEN` | *(empty)* | Required if `USE_REPLICATE_FALLBACK=true` |
| `SAM_CHECKPOINT` | `weights/sam_vit_h_4b8939.pth` | SAM weight path |
| `RAFT_CHECKPOINT` | `weights/raft-things.pth` | RAFT weight path |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` |

---

## Troubleshooting

**`CUDA out of memory`**
Lower `INFER_BATCH_SIZE` in `.env` (try 4, then 2).
Switch Tool 2 backbone to ControlNet (4 GB) instead of SVD (12 GB).

**`FFmpeg not found`**
Ensure `ffmpeg` is on your system PATH:
`which ffmpeg` — if empty: `sudo apt install ffmpeg` or `brew install ffmpeg`.

**`SAM checkpoint not found`**
Run `python3 download_weights.py` from the project root.

**`GroundingDINO finds no objects`**
Lower `box_threshold` in `segmenter.py` (default 0.35 → try 0.20).
Or use the bounding-box mode (draw manually on the frame).

**Flickering in style transfer output**
Ensure both **Optical flow guidance** and **Flicker suppression** are enabled.
Increase diffusion steps (25 → 40). Lower style strength slightly (0.75 → 0.55).

**Frontend can't reach backend**
Check `CORS_ORIGINS` in `.env` includes the frontend URL exactly.
Confirm backend is running: `curl http://localhost:8000/health`.

---

## Project Structure

```
flowedit/
├── main.py                     FastAPI entry point
├── requirements.txt
├── setup.sh / run.sh / Makefile
├── download_weights.py
├── app/
│   ├── api/routes/             upload.py · process.py · status.py
│   ├── core/                   config.py · job_store.py
│   ├── pipeline/               extractor · batcher · recomposer · orchestrator
│   │   └── temporal/           optical_flow.py (RAFT) · flow_utils.py
│   ├── models/
│   │   ├── object_removal/     segmenter · mask_propagator · inpainter
│   │   └── style_transfer/     style_engine · flow_guidance · flicker_suppressor
│   └── schemas/                job.py · requests.py
├── frontend/                   index.html · styles.css · script.js
├── storage/                    uploads/ frames/ processed/ outputs/
├── weights/                    model checkpoints
└── tests/                      conftest · test_extractor · test_temporal · test_api
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML5, CSS3, Vanilla JS, Server-Sent Events |
| Backend | FastAPI, uvicorn, sse-starlette, Pydantic v2 |
| Video processing | FFmpeg, OpenCV |
| Optical flow | RAFT (princeton-vl) |
| Segmentation | SAM ViT-H (Meta), GroundingDINO (IDEA-Research) |
| Inpainting | ProPainter (sczhou), E2FGVI (MCG-NKU) |
| Style transfer | Stable Video Diffusion, ControlNet + SD 1.5 (HuggingFace) |
| Cloud GPU | Replicate API |
| Deep learning | PyTorch 2.3, HuggingFace diffusers 0.27 |
