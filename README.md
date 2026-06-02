# FlowEdit (GEN-AI-VIDEO-EDITOR)

🚧 **Status: Active Development (Work in Progress)** 🚧

A local, open-source AI video editing pipeline built to solve the biggest problem in generative video: **flickering and temporal inconsistency**.

Instead of processing videos frame-by-frame (which causes textures and lighting to jitter wildly), this pipeline uses optical flow tracking to force AI models to maintain consistency across time.

## Architecture & Current Progress

This project is being built from the ground up. Here is what is currently working in the `main` branch:

* **Real-time SSE Infrastructure:** A decoupled vanilla JS frontend and async FastAPI backend connected via Server-Sent Events (SSE) for zero-latency progress tracking.
* **Lossless Pipeline:** Automated FFmpeg integration that strips variable-framerate video into exact PNG batches, preserves the audio track, and recompiles the final edit without dropping frames or desyncing sound.
* **Temporal Consistency Engine (Step 5 Completed):** Integrated **RAFT (Recurrent All-Pairs Field Transforms)** to compute forward and backward optical flow fields. This allows us to track exact pixel movement and penalize the AI when it creates unstable hallucinations.

## Roadmap: What's Next?

The core architecture is finished. We are currently wiring up the specific AI editing tools:

- [ ] **Tool 1: Prompt-Based Object Removal**
  - Text-to-segmentation using **GroundingDINO + SAM (Segment Anything Model)**.
  - Using RAFT flow fields to propagate the mask smoothly across camera movements.
  - Passing the masked video to **ProPainter** for seamless background inpainting.
- [ ] **Tool 2: Flow-Guided Style Transfer**
  - Stable Diffusion / ControlNet style application.
  - Using the consistency engine to blend warped past-frames with new frames, eliminating the "strobing" effect common in AI video filters.

## Running Locally

*Note: As this is in active development, setup instructions will evolve. You will need a CUDA-capable NVIDIA GPU.*

**1. Clone the repository**

```bash
git clone [https://github.com/sahil0322/GEN_AI_VIDEO_EDITOR.git](https://github.com/sahil0322/GEN_AI_VIDEO_EDITOR.git)
cd GEN_AI_VIDEO_EDITOR
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Start the services**

You need two terminals.

Terminal 1 (Backend):

```bash
uvicorn main:app --reload --port 8000
```

Terminal 2 (Frontend):

```bash
cd frontend
python -m http.server 5500
```

Open `http://localhost:5500` in your browser.

## Folder Structure

```text
├── app/
│   ├── api/          # FastAPI routes and SSE status streams
│   ├── core/         # Job state management and config
│   └── pipeline/     # The video brain: extractors, batchers, and RAFT flow
├── frontend/         # Pure HTML/CSS/JS UI
├── storage/          # Temporary dir for frames, audio, and processed batches
└── weights/          # (Gitignored) Folder for massive .pth model files
```

## Contributing

Since this is a highly experimental pipeline, major architectural changes are happening frequently. If you want to contribute to the optical flow logic or FFmpeg wrappers, please feel free to open an issue to discuss it first ❤️
