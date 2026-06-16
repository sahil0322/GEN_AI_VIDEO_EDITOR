#!/usr/bin/env bash
# ==============================================================================
# setup.sh — One-command FlowEdit environment setup
#
# Run once after cloning:
#   chmod +x setup.sh && ./setup.sh
#
# What it does:
#   1. Checks for required system tools (ffmpeg, Python 3.10+)
#   2. Installs Python dependencies from requirements.txt
#   3. Clones and installs source-only packages (RAFT, ProPainter, GroundingDINO, SAM)
#   4. Downloads model weight checkpoints
#   5. Creates storage directories
#   6. Copies .env.example → .env
# ==============================================================================

set -euo pipefail   # exit on error, unset variable, or pipe failure

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[setup]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC}   $*"; }
error() { echo -e "${RED}[error]${NC}  $*" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── 1. System checks ──────────────────────────────────────────────────────────
info "Checking system requirements…"

command -v python3 &>/dev/null || error "Python 3 not found. Install Python 3.10+."
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
IFS='.' read -r PY_MAJOR PY_MINOR <<< "$PY_VER"
(( PY_MAJOR >= 3 && PY_MINOR >= 10 )) || error "Python 3.10+ required. Found: $PY_VER"
info "Python $PY_VER ✓"

command -v ffmpeg &>/dev/null  || error "ffmpeg not found.\n  Ubuntu: sudo apt install ffmpeg\n  macOS:  brew install ffmpeg"
command -v ffprobe &>/dev/null || error "ffprobe not found (usually ships with ffmpeg)."
info "FFmpeg $(ffmpeg -version 2>&1 | head -1 | awk '{print $3}') ✓"

# Check for GPU (non-fatal)
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
    info "GPU detected: $GPU_NAME ✓"
else
    warn "No CUDA GPU detected. Models will run on CPU (very slow)."
    warn "Set USE_REPLICATE_FALLBACK=true in .env to use cloud GPU."
fi

# ── 2. Python dependencies ────────────────────────────────────────────────────
info "Installing Python dependencies from requirements.txt…"
pip install --quiet -r requirements.txt
info "requirements.txt installed ✓"

# ── 3. Source-only installs ───────────────────────────────────────────────────
info "Installing source packages…"

# RAFT optical flow
if [ ! -d "RAFT" ]; then
    info "Cloning RAFT…"
    git clone --quiet https://github.com/princeton-vl/RAFT.git
fi
pip install --quiet -e RAFT/
info "RAFT ✓"

# ProPainter video inpainting
if [ ! -d "ProPainter" ]; then
    info "Cloning ProPainter…"
    git clone --quiet https://github.com/sczhou/ProPainter.git
fi
pip install --quiet -e ProPainter/
info "ProPainter ✓"

# GroundingDINO
if [ ! -d "GroundingDINO" ]; then
    info "Cloning GroundingDINO…"
    git clone --quiet https://github.com/IDEA-Research/GroundingDINO.git
fi
pip install --quiet -e GroundingDINO/
info "GroundingDINO ✓"

# Segment Anything Model
pip install --quiet git+https://github.com/facebookresearch/segment-anything.git
info "SAM ✓"

# ── 4. Download model weights ─────────────────────────────────────────────────
info "Downloading model weights…"
python3 download_weights.py
info "Weights downloaded ✓"

# ── 5. Create storage directories ─────────────────────────────────────────────
info "Creating storage directories…"
mkdir -p storage/{uploads,frames,processed,outputs} weights
info "Directories created ✓"

# ── 6. Environment file ───────────────────────────────────────────────────────
if [ ! -f ".env" ]; then
    cp .env.example .env
    info ".env created from .env.example — edit it to add your REPLICATE_API_TOKEN"
else
    info ".env already exists — skipping"
fi

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  FlowEdit setup complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  Start the server:   make run"
echo "  Open the UI:        http://localhost:5500"
echo "  API docs:           http://localhost:8000/docs"
echo ""
