# Copyright (c) 2026 Eric Cooper.
"""The benchmark's plumbing, with a stand-in detector (CI doesn't install SpeciesNet)."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from camera_mcp.bench import capacity, clip_frames, time_detector


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs the ffmpeg binary")
def test_frames_are_spread_across_a_clip(tmp_path):
    clip = tmp_path / "c.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=10:d=4",
                    "-pix_fmt", "yuv420p", str(clip)], check=True)
    frames = clip_frames(str(clip), 4)
    assert len(frames) == 4 and frames[0].shape == (180, 320, 3)
    assert not (frames[0] == frames[-1]).all()  # different moments, not one frame four times


def test_timing_skips_the_warm_up_and_times_both_modes():
    class Stand:
        calls = []

        def analyze(self, frame, tiles=False, **kw):
            self.calls.append(tiles)
            return [1, 2] if tiles else [1]

    det = Stand()
    whole, tiled = time_detector(det, ["f1", "f2", "f3"])
    assert det.calls == [False] + [False] * 3 + [True] * 3  # one warm-up, then three of each
    assert whole.boxes == [1, 1, 1] and tiled.boxes == [2, 2, 2] and len(whole.seconds) == 3


def test_capacity():
    assert capacity(0.25) == 4.0 and capacity(4.0) == 0.25 and capacity(1.0, frames_per_second=0.5) == 2.0
