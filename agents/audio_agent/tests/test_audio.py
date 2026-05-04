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


# ── _fuzzy_voice_match ─────────────────────────────────────────────────────────

class TestFuzzyVoiceMatch:
    VOICE_MAP = {
        "Hero":         {"voice_style": "neutral"},
        "Dr. Smith":    {"voice_style": "authoritative"},
        "Alice Mercer": {"voice_style": "soft"},
    }

    def _agent(self):
        return AudioGenerationAgent()

    def test_exact_match(self):
        result = self._agent()._fuzzy_voice_match("Hero", self.VOICE_MAP)
        assert result == {"voice_style": "neutral"}

    def test_case_insensitive_exact(self):
        result = self._agent()._fuzzy_voice_match("hero", self.VOICE_MAP)
        assert result is not None

    def test_dots_and_spaces_stripped(self):
        result = self._agent()._fuzzy_voice_match("DrSmith", self.VOICE_MAP)
        assert result == {"voice_style": "authoritative"}

    def test_prefix_match(self):
        # "Alice" should match "Alice Mercer"
        result = self._agent()._fuzzy_voice_match("Alice", self.VOICE_MAP)
        assert result == {"voice_style": "soft"}

    def test_word_overlap_match(self):
        # "Mercer" (5 chars) overlaps with "Alice Mercer"
        result = self._agent()._fuzzy_voice_match("Mercer", self.VOICE_MAP)
        assert result == {"voice_style": "soft"}

    def test_no_match_returns_none(self):
        result = self._agent()._fuzzy_voice_match("Unknown Character", self.VOICE_MAP)
        assert result is None

    def test_empty_voice_map_returns_none(self):
        result = self._agent()._fuzzy_voice_match("Hero", {})
        assert result is None


# ── _build_speaker_voices ──────────────────────────────────────────────────────

class TestBuildSpeakerVoices:

    def _agent(self):
        return AudioGenerationAgent()

    def test_each_speaker_gets_a_voice(self):
        segments  = [
            {"speaker": "Hero"},
            {"speaker": "Villain"},
        ]
        voice_map = {}
        result = self._agent()._build_speaker_voices(segments, voice_map)
        assert "Hero" in result and "Villain" in result

    def test_unique_styles_assigned(self):
        segments = [{"speaker": s} for s in ["A", "B", "C", "D"]]
        voices   = self._agent()._build_speaker_voices(segments, {})
        styles   = [voices[s]["voice_style"] for s in ["A", "B", "C", "D"]]
        assert len(styles) == len(set(styles)), "Each speaker should have a unique voice style"

    def test_preferred_style_used_when_available(self):
        segments  = [{"speaker": "Hero"}]
        voice_map = {"Hero": {"voice_style": "deep", "speaking_speed": "normal"}}
        result    = self._agent()._build_speaker_voices(segments, voice_map)
        assert result["Hero"]["voice_style"] == "deep"

    def test_collision_reassigned_to_unused_style(self):
        # Both characters prefer "deep" — second one should get a different style
        segments  = [{"speaker": "A"}, {"speaker": "B"}]
        voice_map = {
            "A": {"voice_style": "deep"},
            "B": {"voice_style": "deep"},
        }
        result = self._agent()._build_speaker_voices(segments, voice_map)
        assert result["A"]["voice_style"] != result["B"]["voice_style"]

    def test_speaking_speed_inherited_from_voice_config(self):
        segments  = [{"speaker": "Hero"}]
        voice_map = {"Hero": {"voice_style": "soft", "speaking_speed": "slow"}}
        result    = self._agent()._build_speaker_voices(segments, voice_map)
        assert result["Hero"]["speaking_speed"] == "slow"

    def test_unknown_speaker_defaults_to_neutral_speed(self):
        segments = [{"speaker": "Mystery"}]
        result   = self._agent()._build_speaker_voices(segments, {})
        assert result["Mystery"]["speaking_speed"] == "normal"

    def test_speaker_order_preserved(self):
        speakers = ["Alpha", "Beta", "Gamma"]
        segments = [{"speaker": s} for s in speakers]
        result   = self._agent()._build_speaker_voices(segments, {})
        assert list(result.keys()) == speakers


# ── _generate_bgm (mood-arc logic) ────────────────────────────────────────────

class TestGenerateBgmMoodArc:

    @pytest.fixture(autouse=True)
    def patch_dirs(self, tmp_path, monkeypatch):
        import agents.audio_agent.agent as mod
        monkeypatch.setattr(mod, "BGM_DIR", str(tmp_path / "bgm"))

    def test_varied_moods_preserved_as_is(self, tmp_path):
        music_moods = [
            {"scene_id": 1, "mood": "dramatic",   "duration_seconds": 3},
            {"scene_id": 2, "mood": "mysterious",  "duration_seconds": 3},
            {"scene_id": 3, "mood": "hopeful",     "duration_seconds": 3},
        ]
        agent = AudioGenerationAgent()
        result = agent._generate_bgm(music_moods)
        assert result[0]["mood"] == "dramatic"
        assert result[1]["mood"] == "mysterious"
        assert result[2]["mood"] == "hopeful"

    def test_mono_mood_triggers_arc_reassignment(self, tmp_path):
        music_moods = [
            {"scene_id": i + 1, "mood": "dramatic", "duration_seconds": 3}
            for i in range(4)
        ]
        agent  = AudioGenerationAgent()
        result = agent._generate_bgm(music_moods)
        moods  = [r["mood"] for r in result]
        assert len(set(moods)) > 1, "All-same-mood should trigger arc for variety"

    def test_bgm_files_created(self, tmp_path):
        music_moods = [{"scene_id": 1, "mood": "tense", "duration_seconds": 3}]
        result = AudioGenerationAgent()._generate_bgm(music_moods)
        assert os.path.exists(result[0]["audio_file"])

    def test_entries_have_required_fields(self, tmp_path):
        music_moods = [{"scene_id": 1, "mood": "dramatic", "duration_seconds": 3}]
        result = AudioGenerationAgent()._generate_bgm(music_moods)
        for entry in result:
            assert {"scene_id", "mood", "audio_file", "duration_ms"}.issubset(entry.keys())


# ── _build_manifest ────────────────────────────────────────────────────────────

class TestBuildManifest:

    def _timing(self, scene_id, start_ms, end_ms):
        return {
            "segment_id": f"scene{scene_id}_line1",
            "scene_id":   scene_id,
            "speaker":    "Hero",
            "line":       "Hello.",
            "emotion":    "neutral",
            "audio_file": f"/audio/{scene_id}.mp3",
            "start_ms":   start_ms,
            "end_ms":     end_ms,
            "duration_ms": end_ms - start_ms,
        }

    def _bgm(self, scene_id, mood="dramatic"):
        return {
            "scene_id":   scene_id,
            "mood":       mood,
            "audio_file": f"/bgm/scene{scene_id}.mp3",
            "duration_ms": 30_000,
        }

    def test_manifest_structure(self):
        agent    = AudioGenerationAgent()
        manifest = agent._build_manifest("T", [self._timing(1, 0, 3000)], [self._bgm(1)])
        assert "title" in manifest
        assert "total_scenes" in manifest
        assert "scenes" in manifest
        assert "flat_segments" in manifest

    def test_scene_count_correct(self):
        agent    = AudioGenerationAgent()
        timings  = [self._timing(1, 0, 2000), self._timing(2, 0, 3000)]
        bgm      = [self._bgm(1), self._bgm(2)]
        manifest = agent._build_manifest("T", timings, bgm)
        assert manifest["total_scenes"] == 2

    def test_scenes_sorted_by_id(self):
        agent    = AudioGenerationAgent()
        timings  = [self._timing(3, 0, 1000), self._timing(1, 0, 1000)]
        bgm      = [self._bgm(3), self._bgm(1)]
        manifest = agent._build_manifest("T", timings, bgm)
        ids      = [s["scene_id"] for s in manifest["scenes"]]
        assert ids == sorted(ids)

    def test_total_duration_is_max_of_dialogue_and_bgm(self):
        agent   = AudioGenerationAgent()
        timing  = self._timing(1, 0, 5000)   # dialogue ends at 5s
        bgm     = self._bgm(1)               # BGM is 30s
        manifest = agent._build_manifest("T", [timing], [bgm])
        assert manifest["scenes"][0]["total_duration_ms"] == 30_000

    def test_flat_segments_contains_all_entries(self):
        agent   = AudioGenerationAgent()
        timings = [self._timing(1, 0, 1000), self._timing(1, 1000, 2500)]
        manifest = agent._build_manifest("T", timings, [self._bgm(1)])
        assert len(manifest["flat_segments"]) == 2

    def test_title_preserved(self):
        agent    = AudioGenerationAgent()
        manifest = agent._build_manifest("My Story", [self._timing(1, 0, 1000)], [self._bgm(1)])
        assert manifest["title"] == "My Story"
