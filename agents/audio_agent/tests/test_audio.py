"""
agents/audio_agent/tests/test_audio.py
────────────────────────────────────────
Unit tests for Phase 2 — Audio Generation.

Run:  pytest agents/audio_agent/tests/test_audio.py -v
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from mcp.tools.audio_tools.tts_tool import TTSEngine
from mcp.tools.audio_tools.bgm_tool import BGMEngine, _MOOD_PARAMS
from agents.audio_agent.agent import AudioGenerationAgent

NEUTRAL_VOICE = {
    "voice_style":    "neutral",
    "speaking_speed": "normal",
    "pitch":          "medium",
    "tts_description": "A neutral voice.",
}

SAMPLE_HANDOFF = {
    "title": "Test Story",
    "voice_configs": [
        {"character_name": "Hero",    **NEUTRAL_VOICE},
        {"character_name": "Villain", **{**NEUTRAL_VOICE, "voice_style": "deep"}},
    ],
    "segments": [
        {"segment_id": "scene1_line1", "scene_id": 1, "speaker": "Hero",
         "line": "We must act now.", "emotion": "neutral", "mood": "dramatic"},
        {"segment_id": "scene1_line2", "scene_id": 1, "speaker": "Villain",
         "line": "It is already too late.", "emotion": "angry", "mood": "dramatic"},
        {"segment_id": "scene2_line1", "scene_id": 2, "speaker": "Hero",
         "line": "...", "emotion": "neutral", "mood": "tense"},
        {"segment_id": "scene2_line2", "scene_id": 2, "speaker": "Hero",
         "line": "There must be another way.", "emotion": "sad", "mood": "tense"},
    ],
    "music_moods": [
        {"scene_id": 1, "mood": "dramatic",  "tone": "dark",        "duration_seconds": 5},
        {"scene_id": 2, "mood": "tense",     "tone": "suspenseful", "duration_seconds": 5},
    ],
}


@pytest.fixture
def audio_dir(tmp_path):
    d = tmp_path / "audio"
    d.mkdir()
    return d


@pytest.fixture
def handoff_file(tmp_path):
    path = tmp_path / "phase2_audio_handoff.json"
    path.write_text(json.dumps(SAMPLE_HANDOFF), encoding="utf-8")
    return str(path)


class TestTTSEngine:
    def test_synthesise_produces_file(self, audio_dir):
        engine   = TTSEngine()
        out      = str(audio_dir / "hello.mp3")
        duration = engine.synthesize("Hello world.", NEUTRAL_VOICE, out)
        assert os.path.exists(out) and os.path.getsize(out) > 0
        assert duration > 0

    def test_silence_for_ellipsis(self, audio_dir):
        engine   = TTSEngine()
        out      = str(audio_dir / "silence.mp3")
        duration = engine.synthesize("...", NEUTRAL_VOICE, out)
        assert duration == 800

    def test_silence_for_empty_string(self, audio_dir):
        engine   = TTSEngine()
        out      = str(audio_dir / "empty.mp3")
        duration = engine.synthesize("", NEUTRAL_VOICE, out)
        assert duration == 800

    @pytest.mark.parametrize("style", ["deep", "soft", "authoritative", "neutral", "raspy"])
    def test_all_voice_styles_produce_files(self, audio_dir, style):
        engine   = TTSEngine()
        out      = str(audio_dir / f"{style}.mp3")
        duration = engine.synthesize("Testing voice.", {**NEUTRAL_VOICE, "voice_style": style}, out)
        assert os.path.exists(out) and duration > 0

    def test_slow_speaking_speed(self, audio_dir):
        engine     = TTSEngine()
        out_normal = str(audio_dir / "normal.mp3")
        out_slow   = str(audio_dir / "slow.mp3")
        engine.synthesize("Testing speed.", NEUTRAL_VOICE, out_normal)
        engine.synthesize("Testing speed.", {**NEUTRAL_VOICE, "speaking_speed": "slow"}, out_slow)
        assert os.path.getsize(out_slow) >= os.path.getsize(out_normal)

    def test_creates_parent_dirs(self, tmp_path):
        engine = TTSEngine()
        nested = str(tmp_path / "a" / "b" / "c" / "out.mp3")
        engine.synthesize("Hello.", NEUTRAL_VOICE, nested)
        assert os.path.exists(nested)


class TestBGMEngine:
    @pytest.mark.parametrize("mood", list(_MOOD_PARAMS.keys()))
    def test_all_moods_generate_file(self, audio_dir, mood):
        engine = BGMEngine()
        out    = str(audio_dir / f"{mood}.mp3")
        path   = engine.generate(mood, 3, out)
        assert os.path.exists(path) and os.path.getsize(path) > 0

    def test_unknown_mood_falls_back(self, audio_dir):
        engine = BGMEngine()
        path   = engine.generate("nonexistent_mood", 3, str(audio_dir / "unknown.mp3"))
        assert os.path.exists(path)

    def test_duration_approximately_correct(self, audio_dir):
        from pydub import AudioSegment
        engine = BGMEngine()
        path   = engine.generate("dramatic", 10, str(audio_dir / "timed.mp3"))
        audio  = AudioSegment.from_file(path)
        assert 9000 <= len(audio) <= 11000

    def test_short_duration(self, audio_dir):
        engine = BGMEngine()
        path   = engine.generate("hopeful", 2, str(audio_dir / "short.mp3"))
        assert os.path.exists(path)


class TestAudioGenerationAgent:
    @pytest.fixture(autouse=True)
    def patch_dirs(self, tmp_path, monkeypatch):
        import agents.audio_agent.agent as mod
        monkeypatch.setattr(mod, "DIALOGUE_DIR",    str(tmp_path / "dialogue"))
        monkeypatch.setattr(mod, "BGM_DIR",         str(tmp_path / "bgm"))
        monkeypatch.setattr(mod, "TIMING_MANIFEST", str(tmp_path / "timing_manifest.json"))

    def test_run_produces_manifest(self, handoff_file):
        manifest = AudioGenerationAgent().run(handoff_file)
        assert "title" in manifest and "scenes" in manifest and "flat_segments" in manifest

    def test_manifest_has_correct_scene_count(self, handoff_file):
        assert AudioGenerationAgent().run(handoff_file)["total_scenes"] == 2

    def test_manifest_has_correct_segment_count(self, handoff_file):
        assert len(AudioGenerationAgent().run(handoff_file)["flat_segments"]) == 4

    def test_timing_is_sequential_within_scene(self, handoff_file):
        segs = [s for s in AudioGenerationAgent().run(handoff_file)["flat_segments"] if s["scene_id"] == 1]
        assert segs[1]["start_ms"] == segs[0]["end_ms"]

    def test_all_audio_files_created(self, handoff_file):
        for seg in AudioGenerationAgent().run(handoff_file)["flat_segments"]:
            assert os.path.exists(seg["audio_file"])

    def test_all_bgm_files_created(self, handoff_file):
        for scene in AudioGenerationAgent().run(handoff_file)["scenes"]:
            assert os.path.exists(scene["bgm_file"])

    def test_segment_fields_present(self, handoff_file):
        required = {"segment_id","scene_id","speaker","line","audio_file","start_ms","end_ms","duration_ms","emotion"}
        for seg in AudioGenerationAgent().run(handoff_file)["flat_segments"]:
            assert required.issubset(seg.keys())

    def test_end_ms_greater_than_start_ms(self, handoff_file):
        for seg in AudioGenerationAgent().run(handoff_file)["flat_segments"]:
            assert seg["end_ms"] >= seg["start_ms"]

    def test_silence_handled_for_ellipsis(self, handoff_file):
        ellipsis_seg = next(s for s in AudioGenerationAgent().run(handoff_file)["flat_segments"] if s["line"] == "...")
        assert ellipsis_seg["duration_ms"] == 800

    def test_rerun_clears_old_files(self, handoff_file, tmp_path):
        agent = AudioGenerationAgent()
        agent.run(handoff_file)
        first  = set(os.listdir(tmp_path / "dialogue"))
        agent.run(handoff_file)
        second = set(os.listdir(tmp_path / "dialogue"))
        assert first == second
