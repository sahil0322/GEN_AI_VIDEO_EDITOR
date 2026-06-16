# ==============================================================================
# tests/test_extractor.py
# Tests for pipeline Stage 1 (extractor.py) and Stage 4 (recomposer.py).
# All tests use synthetic video — no real footage needed.
# ==============================================================================

from __future__ import annotations

import asyncio
from pathlib import Path

import cv2
import numpy as np
import pytest

from tests.conftest import make_frames


# ── extractor tests ────────────────────────────────────────────────────────────

class TestMetadataReading:
    def test_reads_fps(self, test_video_path):
        from app.pipeline.extractor import _read_metadata
        meta = _read_metadata(test_video_path)
        assert abs(meta["fps"] - 24.0) < 1.0, f"Expected ~24 fps, got {meta['fps']}"

    def test_reads_resolution(self, test_video_path):
        from app.pipeline.extractor import _read_metadata
        meta = _read_metadata(test_video_path)
        assert meta["width"]  == 320
        assert meta["height"] == 240

    def test_reads_frame_count(self, test_video_path):
        from app.pipeline.extractor import _read_metadata
        meta = _read_metadata(test_video_path)
        assert meta["frame_count"] == 24

    def test_computes_duration(self, test_video_path):
        from app.pipeline.extractor import _read_metadata
        meta = _read_metadata(test_video_path)
        assert abs(meta["duration_s"] - 1.0) < 0.1

    def test_missing_file_raises(self, tmp_path):
        from app.pipeline.extractor import _read_metadata
        with pytest.raises(RuntimeError, match="could not open"):
            _read_metadata(tmp_path / "nonexistent.mp4")


class TestFrameExtraction:
    def test_extracts_png_frames(self, test_video_path, tmp_path):
        """FFmpeg should write exactly frame_count PNG files."""
        from app.pipeline.extractor import extract

        frames_dir, _, meta = asyncio.run(
            extract(test_video_path, "testjob", progress_cb=None)
        )
        pngs = sorted(frames_dir.glob("frame_*.png"))
        assert len(pngs) > 0, "No frames extracted"
        # Allow ±2 frames for VFR rounding
        assert abs(len(pngs) - meta["frame_count"]) <= 2

    def test_frames_are_valid_images(self, test_video_path, tmp_path):
        from app.pipeline.extractor import extract

        frames_dir, _, _ = asyncio.run(
            extract(test_video_path, "testjob2", progress_cb=None)
        )
        pngs = sorted(frames_dir.glob("frame_*.png"))
        first = cv2.imread(str(pngs[0]))
        assert first is not None
        assert first.shape == (240, 320, 3)

    def test_frames_numbered_sequentially(self, test_video_path):
        from app.pipeline.extractor import extract

        frames_dir, _, _ = asyncio.run(
            extract(test_video_path, "testjob3", progress_cb=None)
        )
        pngs = sorted(frames_dir.glob("frame_*.png"))
        nums = [int(p.stem.split("_")[1]) for p in pngs]
        assert nums == list(range(1, len(nums) + 1)), "Frame numbers not sequential"

    def test_missing_upload_raises(self, tmp_path):
        from app.pipeline.extractor import extract
        with pytest.raises(FileNotFoundError):
            asyncio.run(extract(tmp_path / "ghost.mp4", "x"))

    def test_progress_callback_called(self, test_video_path):
        from app.pipeline.extractor import extract

        events = []
        async def cb(pct, msg, detail=""): events.append((pct, msg))

        asyncio.run(extract(test_video_path, "testjob_cb", progress_cb=cb))
        assert len(events) >= 2
        pcts = [e[0] for e in events]
        assert pcts[-1] >= pcts[0], "Progress % should be non-decreasing"


class TestAudioDetection:
    def test_no_audio_detected_in_synthetic(self, test_video_path):
        """Synthetic video has no audio track."""
        from app.pipeline.extractor import _has_audio
        # Synthetic mp4v video has no audio — should return False
        result = _has_audio(test_video_path)
        assert result is False


# ── batcher tests ──────────────────────────────────────────────────────────────

class TestBatcher:
    def test_batch_count(self, test_video_path, tmp_path):
        from app.pipeline.extractor import extract
        from app.pipeline.batcher import make_batches

        frames_dir, _, meta = asyncio.run(extract(test_video_path, "btest1"))
        batches = make_batches(frames_dir, batch_size=8, overlap=0)
        n_frames = meta["frame_count"]
        assert len(batches) == (n_frames + 7) // 8  # ceil division

    def test_overlap_produces_extra_frames_per_batch(self, test_video_path):
        from app.pipeline.extractor import extract
        from app.pipeline.batcher import make_batches

        frames_dir, _, _ = asyncio.run(extract(test_video_path, "btest2"))
        batches_no_overlap = make_batches(frames_dir, batch_size=8, overlap=0)
        batches_overlap    = make_batches(frames_dir, batch_size=8, overlap=2)
        # With overlap, there should be more batches (smaller step size)
        assert len(batches_overlap) >= len(batches_no_overlap)

    def test_empty_dir_raises(self, tmp_path):
        from app.pipeline.batcher import make_batches
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="No frame"):
            make_batches(empty, batch_size=8)

    def test_output_slice_first_batch(self):
        from app.pipeline.batcher import output_slice
        batch = [Path(f"frame_{i:04d}.png") for i in range(8)]
        # First batch — overlap doesn't remove any frames
        result = output_slice(0, batch, overlap=2)
        assert result == batch

    def test_output_slice_subsequent_batch(self):
        from app.pipeline.batcher import output_slice
        batch = [Path(f"frame_{i:04d}.png") for i in range(8)]
        result = output_slice(1, batch, overlap=2)
        assert result == batch[2:]  # first 2 discarded


# ── recomposer tests ───────────────────────────────────────────────────────────

class TestRecomposer:
    def test_recompose_produces_mp4(self, tmp_path):
        """Write 8 synthetic PNG frames → recompose → verify MP4 exists."""
        from app.pipeline.recomposer import recompose

        proc_dir = tmp_path / "processed"
        proc_dir.mkdir()
        out_path = tmp_path / "output.mp4"

        # Write 8 random PNG frames
        for i in range(1, 9):
            frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(proc_dir / f"frame_{i:06d}.png"), frame)

        result = asyncio.run(
            recompose(proc_dir, audio_path=None, output_path=out_path, fps=24.0)
        )
        assert result.exists()
        assert result.stat().st_size > 1000  # non-trivial file

    def test_recompose_video_is_readable(self, tmp_path):
        from app.pipeline.recomposer import recompose

        proc_dir = tmp_path / "proc2"
        proc_dir.mkdir()
        out_path = tmp_path / "out2.mp4"

        for i in range(1, 9):
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(proc_dir / f"frame_{i:06d}.png"), frame)

        asyncio.run(recompose(proc_dir, None, out_path, fps=24.0))
        cap = cv2.VideoCapture(str(out_path))
        assert cap.isOpened()
        assert cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0
        cap.release()

    def test_empty_processed_dir_raises(self, tmp_path):
        from app.pipeline.recomposer import recompose
        empty = tmp_path / "empty_proc"
        empty.mkdir()
        with pytest.raises(ValueError, match="No processed frames"):
            asyncio.run(recompose(empty, None, tmp_path / "out.mp4", fps=24.0))
