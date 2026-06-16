# ==============================================================================
# tests/test_temporal.py
# Tests for optical flow utilities — all CPU, no RAFT model needed.
# ==============================================================================

from __future__ import annotations

import numpy as np
import pytest

from app.pipeline.temporal import flow_utils


# ── warp_frame ─────────────────────────────────────────────────────────────────

class TestWarpFrame:
    def test_zero_flow_is_identity(self):
        """Zero displacement flow should return the frame unchanged."""
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        flow  = np.zeros((64, 64, 2), dtype=np.float32)
        out   = flow_utils.warp_frame(frame, flow)
        np.testing.assert_array_equal(out, frame)

    def test_output_same_shape(self):
        frame = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        flow  = np.zeros((128, 128, 2), dtype=np.float32)
        out   = flow_utils.warp_frame(frame, flow)
        assert out.shape == frame.shape

    def test_shift_right_by_10(self):
        """Constant rightward flow of 10px should shift content left in output."""
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[:, 20:30, :] = 255    # white stripe at column 20-29

        flow        = np.zeros((64, 64, 2), dtype=np.float32)
        flow[..., 0] = -10.0   # negative = output looks 10px to the right
        out = flow_utils.warp_frame(frame, flow)
        # The white stripe should have moved right
        assert out[:, 30:40, :].mean() > 100

    def test_float_frame_passthrough(self):
        frame = np.random.rand(32, 32, 3).astype(np.float32)
        flow  = np.zeros((32, 32, 2), dtype=np.float32)
        out   = flow_utils.warp_frame(frame, flow)
        assert out.dtype == np.float32

    def test_single_channel_mask(self):
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[20:40, 20:40] = 1.0
        flow = np.zeros((64, 64, 2), dtype=np.float32)
        out  = flow_utils.warp_frame(mask, flow)
        assert out.shape == (64, 64)


# ── consistency_score ──────────────────────────────────────────────────────────

class TestConsistencyScore:
    def test_zero_flow_scores_near_one(self):
        """Perfect cycle-consistency (zero flow in both directions) → score ≈ 1."""
        zero = np.zeros((64, 64, 2), dtype=np.float32)
        score = flow_utils.consistency_score(zero, zero)
        assert score.shape == (64, 64)
        assert score.mean() > 0.95, f"Expected ~1.0, got {score.mean():.3f}"

    def test_opposing_flows_score_near_zero(self):
        """Contradictory flows: fwd=(+10,0), bwd=(+10,0) → cycle error = 20px → low score."""
        fwd = np.zeros((32, 32, 2), dtype=np.float32)
        bwd = np.zeros((32, 32, 2), dtype=np.float32)
        fwd[..., 0] = 10.0    # large rightward forward flow
        bwd[..., 0] = 10.0    # same direction backward → cycle error ≠ 0
        score = flow_utils.consistency_score(fwd, bwd)
        assert score.mean() < 0.5, f"Expected low score, got {score.mean():.3f}"

    def test_score_in_range(self):
        fwd = np.random.randn(32, 32, 2).astype(np.float32) * 5
        bwd = np.random.randn(32, 32, 2).astype(np.float32) * 5
        score = flow_utils.consistency_score(fwd, bwd)
        assert score.min() >= 0.0
        assert score.max() <= 1.0

    def test_output_shape(self):
        fwd = np.zeros((48, 64, 2), dtype=np.float32)
        bwd = np.zeros((48, 64, 2), dtype=np.float32)
        score = flow_utils.consistency_score(fwd, bwd)
        assert score.shape == (48, 64)


# ── blend_frames ───────────────────────────────────────────────────────────────

class TestBlendFrames:
    def test_alpha_one_returns_frame_a(self):
        a = np.full((32, 32, 3), 200, dtype=np.uint8)
        b = np.full((32, 32, 3), 50,  dtype=np.uint8)
        out = flow_utils.blend_frames(a, b, alpha=1.0)
        np.testing.assert_allclose(out.astype(float), a.astype(float), atol=1)

    def test_alpha_zero_returns_frame_b(self):
        a = np.full((32, 32, 3), 200, dtype=np.uint8)
        b = np.full((32, 32, 3), 50,  dtype=np.uint8)
        out = flow_utils.blend_frames(a, b, alpha=0.0)
        np.testing.assert_allclose(out.astype(float), b.astype(float), atol=1)

    def test_alpha_half_is_midpoint(self):
        a = np.full((32, 32, 3), 200, dtype=np.uint8)
        b = np.full((32, 32, 3), 100, dtype=np.uint8)
        out = flow_utils.blend_frames(a, b, alpha=0.5)
        np.testing.assert_allclose(out.astype(float), 150.0, atol=1)

    def test_per_pixel_alpha(self):
        a    = np.full((4, 4, 3), 200, dtype=np.uint8)
        b    = np.full((4, 4, 3), 100, dtype=np.uint8)
        alpha = np.zeros((4, 4), dtype=np.float32)
        alpha[:, :2] = 1.0   # left half → frame_a; right half → frame_b
        out = flow_utils.blend_frames(a, b, alpha=alpha)
        assert out[:, 0, 0].mean() > 180  # left half ≈ frame_a
        assert out[:, 3, 0].mean() < 120  # right half ≈ frame_b

    def test_output_is_uint8(self):
        a   = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
        b   = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
        out = flow_utils.blend_frames(a, b, alpha=0.5)
        assert out.dtype == np.uint8


# ── dilate_mask / soften_mask ─────────────────────────────────────────────────

class TestMaskHelpers:
    def test_dilate_expands_mask(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[30:34, 30:34] = 255   # 4×4 square
        dilated = flow_utils.dilate_mask(mask, dilation_px=4)
        assert dilated.sum() > mask.sum()

    def test_dilate_zero_is_noop(self):
        mask    = np.random.randint(0, 2, (32, 32), dtype=np.uint8) * 255
        dilated = flow_utils.dilate_mask(mask, dilation_px=0)
        np.testing.assert_array_equal(dilated, mask)

    def test_soften_reduces_hard_edges(self):
        mask   = np.zeros((64, 64), dtype=np.float32)
        mask[20:44, 20:44] = 1.0
        soft   = flow_utils.soften_mask(mask, kernel_size=5, sigma=2.0)
        # Boundary pixels should be between 0 and 1
        assert soft[19, 20] > 0.0
        assert soft[19, 20] < 1.0
