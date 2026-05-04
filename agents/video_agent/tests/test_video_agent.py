"""
agents/video_agent/tests/test_video_agent.py
─────────────────────────────────────────────
Unit tests for Phase 3 — Video Generation.

Tests all pure/low-dependency methods without running MoviePy:
  - _calc_duration()
  - _gradient_bg()
  - _find_char_image()
  - _build_timeline()
  - _load_edit_settings()
  - _MOOD_GRADIENT / _TIME_BRIGHTNESS coverage

Run:  pytest agents/video_agent/tests/test_video_agent.py -v
"""

import json
import os
import sys

import numpy as np
import pytest
from PIL import Image

ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, ROOT)

from agents.video_agent.agent import (
    VideoGenerationAgent,
    _load_edit_settings,
    _MOOD_GRADIENT,
    _TIME_BRIGHTNESS,
    MIN_SCENE_DUR,
    SCENE_END_BUFFER,
    VIDEO_W,
    VIDEO_H,
)


@pytest.fixture
def agent():
    return VideoGenerationAgent()


# ── _calc_duration ─────────────────────────────────────────────────────────────

class TestCalcDuration:

    def test_empty_segments_returns_min_duration(self, agent):
        assert agent._calc_duration({}) == MIN_SCENE_DUR
        assert agent._calc_duration({"dialogue_segments": []}) == MIN_SCENE_DUR

    def test_single_segment_adds_buffer(self, agent):
        scene = {"dialogue_segments": [{"end_ms": 5000}]}
        expected = 5000 / 1000 + SCENE_END_BUFFER
        assert agent._calc_duration(scene) == pytest.approx(expected)

    def test_uses_max_end_ms_across_segments(self, agent):
        scene = {"dialogue_segments": [
            {"end_ms": 3000},
            {"end_ms": 7000},
            {"end_ms": 2000},
        ]}
        expected = 7000 / 1000 + SCENE_END_BUFFER
        assert agent._calc_duration(scene) == pytest.approx(expected)

    def test_very_short_dialogue_clamped_to_min(self, agent):
        scene = {"dialogue_segments": [{"end_ms": 100}]}
        result = agent._calc_duration(scene)
        assert result >= MIN_SCENE_DUR

    def test_long_dialogue_not_clamped(self, agent):
        scene = {"dialogue_segments": [{"end_ms": 60_000}]}
        result = agent._calc_duration(scene)
        assert result == pytest.approx(60.0 + SCENE_END_BUFFER)


# ── _gradient_bg ───────────────────────────────────────────────────────────────

class TestGradientBg:

    def test_creates_png_file(self, agent, tmp_path):
        out = str(tmp_path / "bg.png")
        result = agent._gradient_bg("dramatic", "DAY", out)
        assert os.path.exists(result)

    def test_returns_save_path(self, agent, tmp_path):
        out = str(tmp_path / "bg.png")
        result = agent._gradient_bg("dramatic", "DAY", out)
        assert result == out

    def test_output_dimensions(self, agent, tmp_path):
        out = str(tmp_path / "bg.png")
        agent._gradient_bg("hopeful", "DAY", out)
        img = Image.open(out)
        assert img.size == (VIDEO_W, VIDEO_H)

    def test_unknown_mood_uses_default_gradient(self, agent, tmp_path):
        out = str(tmp_path / "bg.png")
        result = agent._gradient_bg("nonexistent_mood", "DAY", out)
        assert os.path.exists(result)

    def test_night_is_darker_than_day(self, agent, tmp_path):
        day_path   = str(tmp_path / "day.png")
        night_path = str(tmp_path / "night.png")
        agent._gradient_bg("dramatic", "DAY",   day_path)
        agent._gradient_bg("dramatic", "NIGHT", night_path)
        day_mean   = np.array(Image.open(day_path)).mean()
        night_mean = np.array(Image.open(night_path)).mean()
        assert night_mean < day_mean

    def test_all_moods_generate_without_error(self, agent, tmp_path):
        for mood in _MOOD_GRADIENT:
            out = str(tmp_path / f"{mood}.png")
            agent._gradient_bg(mood, "DAY", out)
            assert os.path.exists(out)

    def test_all_times_of_day_generate_without_error(self, agent, tmp_path):
        for tod in _TIME_BRIGHTNESS:
            out = str(tmp_path / f"{tod}.png")
            agent._gradient_bg("dramatic", tod, out)
            assert os.path.exists(out)


# ── _find_char_image ───────────────────────────────────────────────────────────

class TestFindCharImage:

    def test_exact_key_match(self, agent):
        images = {"Alice": "/path/alice.png", "Bob": "/path/bob.png"}
        assert agent._find_char_image("Alice", images) == "/path/alice.png"

    def test_no_match_returns_none(self, agent):
        images = {"Alice": "/path/alice.png"}
        assert agent._find_char_image("Carol", images) is None

    def test_case_insensitive_normalized_match(self, agent):
        images = {"Alice Smith": "/path/alice.png"}
        assert agent._find_char_image("alice smith", images) == "/path/alice.png"

    def test_ignores_spaces_and_dots_in_normalization(self, agent):
        images = {"Dr. Smith": "/path/smith.png"}
        assert agent._find_char_image("DrSmith", images) == "/path/smith.png"

    def test_prefix_match_shorter_query(self, agent):
        # "Alice" should match "Alice Mercer"
        images = {"Alice Mercer": "/path/alice.png"}
        assert agent._find_char_image("Alice", images) == "/path/alice.png"

    def test_prefix_match_longer_query(self, agent):
        # "Alice Mercer" should match "Alice" entry
        images = {"Alice": "/path/alice.png"}
        assert agent._find_char_image("Alice Mercer", images) == "/path/alice.png"

    def test_stem_prefix_match(self, agent):
        images = {"__stem__detective_james": "/path/james.png"}
        assert agent._find_char_image("Detective", images) == "/path/james.png"

    def test_word_overlap_match(self, agent):
        images = {"James Watson": "/path/james.png"}
        # "Watson" is > 4 chars, so it triggers word-overlap match
        assert agent._find_char_image("Watson", images) == "/path/james.png"

    def test_empty_images_dict_returns_none(self, agent):
        assert agent._find_char_image("Alice", {}) is None

    def test_exact_match_takes_precedence_over_prefix(self, agent):
        images = {
            "Alice": "/path/exact.png",
            "Alice Mercer": "/path/prefix.png",
        }
        assert agent._find_char_image("Alice", images) == "/path/exact.png"


# ── _build_timeline ────────────────────────────────────────────────────────────

class TestBuildTimeline:

    def _seg(self, seg_id, start_ms, end_ms, speaker="Hero",
             line="Hello.", emotion="neutral", audio_file="/a.mp3"):
        return {
            "segment_id": seg_id, "start_ms": start_ms, "end_ms": end_ms,
            "speaker": speaker, "line": line, "emotion": emotion,
            "audio_file": audio_file,
        }

    def test_converts_ms_to_seconds(self, agent):
        segs = [self._seg("s1", 0, 3000)]
        result = agent._build_timeline(segs)
        assert result[0]["start"] == pytest.approx(0.0)
        assert result[0]["end"]   == pytest.approx(3.0)

    def test_all_required_fields_present(self, agent):
        segs = [self._seg("s1", 0, 2000)]
        item = agent._build_timeline(segs)[0]
        for field in ("segment_id", "start", "end", "speaker", "text", "emotion", "audio_file"):
            assert field in item, f"Missing field: {field}"

    def test_filters_out_ellipsis_lines(self, agent):
        segs = [
            self._seg("s1", 0, 800, line="..."),
            self._seg("s2", 800, 2000, line="Real line."),
        ]
        result = agent._build_timeline(segs)
        assert len(result) == 1
        assert result[0]["text"] == "Real line."

    def test_filters_out_blank_lines(self, agent):
        segs = [self._seg("s1", 0, 800, line="   ")]
        result = agent._build_timeline(segs)
        assert result == []

    def test_preserves_order(self, agent):
        segs = [
            self._seg("s1", 0,    1000, line="First."),
            self._seg("s2", 1000, 2500, line="Second."),
            self._seg("s3", 2500, 4000, line="Third."),
        ]
        result = agent._build_timeline(segs)
        texts = [r["text"] for r in result]
        assert texts == ["First.", "Second.", "Third."]

    def test_empty_segments_returns_empty_list(self, agent):
        assert agent._build_timeline([]) == []

    def test_text_field_maps_from_line(self, agent):
        segs = [self._seg("s1", 0, 1000, line="Spoken words.")]
        result = agent._build_timeline(segs)
        assert result[0]["text"] == "Spoken words."

    def test_emotion_defaults_to_neutral(self, agent):
        seg = self._seg("s1", 0, 1000)
        del seg["emotion"]
        result = agent._build_timeline([seg])
        assert result[0]["emotion"] == "neutral"


# ── _load_edit_settings ────────────────────────────────────────────────────────

class TestLoadEditSettings:

    def test_returns_empty_dict_when_file_missing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = _load_edit_settings()
        assert result == {}

    def test_reads_valid_json(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        settings_dir = tmp_path / "data" / "outputs"
        settings_dir.mkdir(parents=True)
        (settings_dir / "edit_settings.json").write_text(
            '{"show_subtitles": false, "speed": 1.5}', encoding="utf-8"
        )
        result = _load_edit_settings()
        assert result == {"show_subtitles": False, "speed": 1.5}

    def test_returns_empty_dict_on_malformed_json(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        settings_dir = tmp_path / "data" / "outputs"
        settings_dir.mkdir(parents=True)
        (settings_dir / "edit_settings.json").write_text("not json!", encoding="utf-8")
        result = _load_edit_settings()
        assert result == {}

    def test_subtitles_default_accessible(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = _load_edit_settings()
        # Default is True when not set (accessed as .get("show_subtitles", True))
        assert result.get("show_subtitles", True) is True


# ── Constants coverage ─────────────────────────────────────────────────────────

class TestConstants:

    def test_all_moods_in_gradient_dict(self):
        expected = {"dramatic","tense","mysterious","hopeful","melancholic","comedic","romantic","fantasy"}
        assert expected.issubset(_MOOD_GRADIENT.keys())

    def test_all_times_in_brightness_dict(self):
        assert set(_TIME_BRIGHTNESS.keys()) == {"DAY", "DAWN", "DUSK", "NIGHT"}

    def test_brightness_ordering(self):
        assert _TIME_BRIGHTNESS["DAY"] > _TIME_BRIGHTNESS["DAWN"]
        assert _TIME_BRIGHTNESS["DAWN"] > _TIME_BRIGHTNESS["DUSK"]
        assert _TIME_BRIGHTNESS["DUSK"] > _TIME_BRIGHTNESS["NIGHT"]

    def test_gradient_tuples_are_rgb_pairs(self):
        for mood, (top, bot) in _MOOD_GRADIENT.items():
            assert len(top) == 3 and len(bot) == 3, f"{mood}: expected (R,G,B) pairs"

    def test_video_dimensions_widescreen(self):
        assert VIDEO_W / VIDEO_H == pytest.approx(16 / 9, rel=0.1)
