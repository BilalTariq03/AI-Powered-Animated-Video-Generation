"""
agents/edit_agent/tests/test_edit_agent.py
───────────────────────────────────────────
Unit tests for Phase 5 — Intelligent Edit & Undo System.

Covers:
  - _parse_scope()          : scope string parsing
  - EditExecutor.execute()  : every handler's inputs, outputs, and error paths
  - StateManager            : snapshot, revert, history

Run:  pytest agents/edit_agent/tests/test_edit_agent.py -v
"""

import json
import os
import sys

import pytest

ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, ROOT)

import agents.edit_agent.executor as executor_mod
from agents.edit_agent.executor import EditExecutor, _parse_scope
from agents.edit_agent.state_manager import StateManager


# ── _parse_scope ───────────────────────────────────────────────────────────────

class TestParseScope:

    def test_all_string(self):
        assert _parse_scope("all") == ("all", None)

    def test_empty_string_defaults_to_all(self):
        assert _parse_scope("") == ("all", None)

    def test_none_defaults_to_all(self):
        assert _parse_scope(None) == ("all", None)

    def test_character_scope(self):
        assert _parse_scope("character:Alice") == ("character", "Alice")

    def test_character_scope_with_spaces(self):
        assert _parse_scope("character: Alice") == ("character", "Alice")

    def test_scene_scope_returns_int(self):
        assert _parse_scope("scene:3") == ("scene", 3)

    def test_scene_scope_invalid_number(self):
        kind, value = _parse_scope("scene:abc")
        assert kind == "scene" and value is None

    def test_case_insensitive_prefixes(self):
        assert _parse_scope("Character:Bob")[0] == "character"
        assert _parse_scope("Scene:2")[0] == "scene"


# ── EditExecutor fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def p2_file(tmp_path):
    data = {
        "voice_configs": [
            {"character_name": "Hero",   "voice_style": "neutral",      "speaking_speed": "normal"},
            {"character_name": "Villain","voice_style": "deep",         "speaking_speed": "slow"},
        ],
        "music_moods": [
            {"scene_id": 1, "mood": "dramatic", "duration_seconds": 30},
            {"scene_id": 2, "mood": "tense",    "duration_seconds": 30},
        ],
    }
    p = tmp_path / "p2.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return str(p)


@pytest.fixture
def p3_file(tmp_path):
    data = {
        "scenes": [
            {"scene_id": 1, "image_generation_prompt": "A dark forest.", "mood": "dramatic"},
            {"scene_id": 2, "image_generation_prompt": "A bright city.",  "mood": "hopeful"},
        ],
        "character_visuals": [
            {"name": "Hero",   "image_prompt": "A brave hero portrait."},
            {"name": "Villain","image_prompt": "A cunning villain portrait."},
        ],
    }
    p = tmp_path / "p3.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return str(p)


@pytest.fixture
def timing_file(tmp_path):
    data = {
        "scenes": [
            {
                "scene_id": 1,
                "dialogue_segments": [
                    {"segment_id": "s1", "start_ms": 0,    "end_ms": 3000, "duration_ms": 3000},
                    {"segment_id": "s2", "start_ms": 3000, "end_ms": 5000, "duration_ms": 2000},
                ],
                "total_duration_ms": 5000,
            },
        ],
        "flat_segments": [],
    }
    p = tmp_path / "timing.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return str(p)


@pytest.fixture
def patch_executor(monkeypatch, tmp_path, p2_file, p3_file, timing_file):
    """Redirect all executor path constants to tmp_path."""
    monkeypatch.setattr(executor_mod, "PHASE2_HANDOFF",  p2_file)
    monkeypatch.setattr(executor_mod, "PHASE3_HANDOFF",  p3_file)
    monkeypatch.setattr(executor_mod, "TIMING_MANIFEST", timing_file)
    monkeypatch.setattr(executor_mod, "IMAGES_DIR",      str(tmp_path / "images"))
    monkeypatch.setattr(executor_mod, "EDIT_SETTINGS",   str(tmp_path / "edit_settings.json"))
    os.makedirs(str(tmp_path / "images"), exist_ok=True)
    return tmp_path


# ── EditExecutor — unknown action ──────────────────────────────────────────────

class TestExecuteUnknownAction:

    def test_unknown_action_returns_error(self, patch_executor):
        result = EditExecutor().execute({"intent": "fly_to_moon", "scope": "all", "parameters": {}})
        assert result["ok"] is False
        assert "Unknown" in result["message"]
        assert result["phase_to_rerun"] is None


# ── change_voice_tone ──────────────────────────────────────────────────────────

class TestChangeVoiceTone:

    def test_all_scope_updates_all_voices(self, patch_executor):
        result = EditExecutor().execute({
            "intent": "change_voice_tone",
            "scope": "all",
            "parameters": {"tone": "deep"},
        })
        assert result["ok"] is True
        assert result["phase_to_rerun"] == 2
        data = json.loads(open(executor_mod.PHASE2_HANDOFF).read())
        for vc in data["voice_configs"]:
            assert vc["voice_style"] == "deep"

    def test_character_scope_updates_only_target(self, patch_executor):
        EditExecutor().execute({
            "intent": "change_voice_tone",
            "scope": "character:Hero",
            "parameters": {"tone": "raspy"},
        })
        data = json.loads(open(executor_mod.PHASE2_HANDOFF).read())
        hero_vc    = next(v for v in data["voice_configs"] if v["character_name"] == "Hero")
        villain_vc = next(v for v in data["voice_configs"] if v["character_name"] == "Villain")
        assert hero_vc["voice_style"] == "raspy"
        assert villain_vc["voice_style"] == "deep"  # unchanged

    def test_invalid_tone_falls_back_to_neutral(self, patch_executor):
        EditExecutor().execute({
            "intent": "change_voice_tone",
            "scope": "all",
            "parameters": {"tone": "alien_voice"},
        })
        data = json.loads(open(executor_mod.PHASE2_HANDOFF).read())
        assert all(v["voice_style"] == "neutral" for v in data["voice_configs"])

    def test_missing_handoff_file_returns_error(self, monkeypatch, tmp_path):
        monkeypatch.setattr(executor_mod, "PHASE2_HANDOFF", str(tmp_path / "missing.json"))
        result = EditExecutor().execute({
            "intent": "change_voice_tone", "scope": "all", "parameters": {"tone": "deep"},
        })
        assert result["ok"] is False

    def test_no_character_match_returns_error(self, patch_executor):
        result = EditExecutor().execute({
            "intent": "change_voice_tone",
            "scope": "character:Ghost",
            "parameters": {"tone": "soft"},
        })
        assert result["ok"] is False


# ── make_scene_darker / make_scene_lighter ─────────────────────────────────────

class TestMakeSceneVisual:

    def test_darker_appends_modifier_to_prompt(self, patch_executor):
        EditExecutor().execute({
            "intent": "make_scene_darker",
            "scope": "scene:1",
            "parameters": {},
        })
        data = json.loads(open(executor_mod.PHASE3_HANDOFF).read())
        s1   = next(s for s in data["scenes"] if s["scene_id"] == 1)
        assert "shadow" in s1["image_generation_prompt"].lower() or "dark" in s1["image_generation_prompt"].lower()

    def test_lighter_appends_modifier_to_prompt(self, patch_executor):
        EditExecutor().execute({
            "intent": "make_scene_lighter",
            "scope": "scene:1",
            "parameters": {},
        })
        data = json.loads(open(executor_mod.PHASE3_HANDOFF).read())
        s1   = next(s for s in data["scenes"] if s["scene_id"] == 1)
        assert "bright" in s1["image_generation_prompt"].lower() or "light" in s1["image_generation_prompt"].lower()

    def test_returns_phase_3_rerun(self, patch_executor):
        result = EditExecutor().execute({
            "intent": "make_scene_darker", "scope": "all", "parameters": {},
        })
        assert result["ok"] is True
        assert result["phase_to_rerun"] == 3

    def test_deletes_cached_background_image(self, patch_executor, tmp_path):
        bg = tmp_path / "images" / "scene1_bg.png"
        bg.write_text("fake image")
        EditExecutor().execute({
            "intent": "make_scene_darker", "scope": "scene:1", "parameters": {},
        })
        assert not bg.exists()

    def test_missing_handoff_returns_error(self, monkeypatch, tmp_path):
        monkeypatch.setattr(executor_mod, "PHASE3_HANDOFF", str(tmp_path / "missing.json"))
        result = EditExecutor().execute({
            "intent": "make_scene_darker", "scope": "all", "parameters": {},
        })
        assert result["ok"] is False


# ── add_background_music ───────────────────────────────────────────────────────

class TestAddBackgroundMusic:

    def test_all_scope_updates_all_scene_moods(self, patch_executor):
        EditExecutor().execute({
            "intent": "add_background_music",
            "scope": "all",
            "parameters": {"mood": "mysterious"},
        })
        data = json.loads(open(executor_mod.PHASE2_HANDOFF).read())
        assert all(mm["mood"] == "mysterious" for mm in data["music_moods"])

    def test_scene_scope_updates_only_target(self, patch_executor):
        EditExecutor().execute({
            "intent": "add_background_music",
            "scope": "scene:1",
            "parameters": {"mood": "hopeful"},
        })
        data = json.loads(open(executor_mod.PHASE2_HANDOFF).read())
        s1   = next(m for m in data["music_moods"] if m["scene_id"] == 1)
        s2   = next(m for m in data["music_moods"] if m["scene_id"] == 2)
        assert s1["mood"] == "hopeful"
        assert s2["mood"] == "tense"   # unchanged

    def test_invalid_mood_falls_back_to_dramatic(self, patch_executor):
        EditExecutor().execute({
            "intent": "add_background_music",
            "scope": "all",
            "parameters": {"mood": "psychedelic"},
        })
        data = json.loads(open(executor_mod.PHASE2_HANDOFF).read())
        assert all(mm["mood"] == "dramatic" for mm in data["music_moods"])

    def test_fantasy_mood_accepted(self, patch_executor):
        result = EditExecutor().execute({
            "intent": "add_background_music",
            "scope": "all",
            "parameters": {"mood": "fantasy"},
        })
        assert result["ok"] is True
        data = json.loads(open(executor_mod.PHASE2_HANDOFF).read())
        assert all(mm["mood"] == "fantasy" for mm in data["music_moods"])

    def test_returns_phase_2_rerun(self, patch_executor):
        result = EditExecutor().execute({
            "intent": "add_background_music", "scope": "all", "parameters": {"mood": "tense"},
        })
        assert result["phase_to_rerun"] == 2


# ── subtitle toggle ────────────────────────────────────────────────────────────

class TestSubtitleToggle:

    def test_remove_subtitle_sets_false(self, patch_executor, tmp_path):
        EditExecutor().execute({"intent": "remove_subtitle", "scope": "all", "parameters": {}})
        data = json.loads(open(executor_mod.EDIT_SETTINGS).read())
        assert data["show_subtitles"] is False

    def test_add_subtitle_sets_true(self, patch_executor, tmp_path):
        EditExecutor().execute({"intent": "remove_subtitle", "scope": "all", "parameters": {}})
        EditExecutor().execute({"intent": "add_subtitle",    "scope": "all", "parameters": {}})
        data = json.loads(open(executor_mod.EDIT_SETTINGS).read())
        assert data["show_subtitles"] is True

    def test_returns_phase_3_rerun(self, patch_executor):
        result = EditExecutor().execute({"intent": "remove_subtitle", "scope": "all", "parameters": {}})
        assert result["phase_to_rerun"] == 3

    def test_settings_file_created(self, patch_executor, tmp_path):
        settings_path = tmp_path / "edit_settings.json"
        assert not settings_path.exists()
        EditExecutor().execute({"intent": "remove_subtitle", "scope": "all", "parameters": {}})
        assert settings_path.exists()


# ── change_character_design ────────────────────────────────────────────────────

class TestChangeCharacterDesign:

    def test_updates_image_prompt(self, patch_executor):
        EditExecutor().execute({
            "intent": "change_character_design",
            "scope": "character:Hero",
            "parameters": {"description": "anime style warrior"},
        })
        data = json.loads(open(executor_mod.PHASE3_HANDOFF).read())
        hero = next(c for c in data["character_visuals"] if c["name"] == "Hero")
        assert "anime" in hero["image_prompt"]

    def test_deletes_cached_portrait(self, patch_executor, tmp_path, monkeypatch):
        portrait = tmp_path / "images" / "hero.png"
        portrait.write_text("fake portrait")
        # Stub regeneration so the portrait is NOT recreated — isolates the delete behaviour
        monkeypatch.setattr(EditExecutor, "_regenerate_character_image", lambda *_: None)
        EditExecutor().execute({
            "intent": "change_character_design",
            "scope": "character:Hero",
            "parameters": {"description": "sci-fi armor"},
        })
        assert not portrait.exists()

    def test_regenerates_portrait_after_delete(self, patch_executor, monkeypatch):
        calls = []
        monkeypatch.setattr(
            EditExecutor, "_regenerate_character_image",
            lambda self, name, prompt: calls.append((name, prompt)) or None,
        )
        EditExecutor().execute({
            "intent": "change_character_design",
            "scope": "character:Hero",
            "parameters": {"description": "sci-fi armor"},
        })
        assert len(calls) == 1
        assert calls[0][0] == "Hero"
        assert "sci-fi armor" in calls[0][1]

    def test_all_scope_updates_all_characters(self, patch_executor):
        EditExecutor().execute({
            "intent": "change_character_design",
            "scope": "all",
            "parameters": {"description": "steampunk style"},
        })
        data = json.loads(open(executor_mod.PHASE3_HANDOFF).read())
        for char in data["character_visuals"]:
            assert "steampunk" in char["image_prompt"]

    def test_returns_phase_3_rerun(self, patch_executor):
        result = EditExecutor().execute({
            "intent": "change_character_design",
            "scope": "all",
            "parameters": {"description": "noir style"},
        })
        assert result["phase_to_rerun"] == 3

    def test_no_match_returns_error(self, patch_executor):
        result = EditExecutor().execute({
            "intent": "change_character_design",
            "scope": "character:Ghost",
            "parameters": {"description": "spooky"},
        })
        assert result["ok"] is False


# ── adjust_speed ───────────────────────────────────────────────────────────────

class TestAdjustSpeed:

    def test_speed_up_shortens_timing(self, patch_executor):
        original_end = 3000
        EditExecutor().execute({
            "intent": "speed_up_scene",
            "scope": "scene:1",
            "parameters": {"factor": 2.0},
        })
        data = json.loads(open(executor_mod.TIMING_MANIFEST).read())
        seg = data["scenes"][0]["dialogue_segments"][0]
        assert seg["end_ms"] == int(original_end / 2)

    def test_slow_down_extends_timing(self, patch_executor):
        original_end = 3000
        EditExecutor().execute({
            "intent": "slow_down_scene",
            "scope": "scene:1",
            "parameters": {"factor": 2.0},
        })
        data = json.loads(open(executor_mod.TIMING_MANIFEST).read())
        seg = data["scenes"][0]["dialogue_segments"][0]
        assert seg["end_ms"] == int(original_end * 2)

    def test_returns_phase_3_rerun(self, patch_executor):
        result = EditExecutor().execute({
            "intent": "speed_up_scene", "scope": "scene:1", "parameters": {"factor": 1.5},
        })
        assert result["phase_to_rerun"] == 3

    def test_missing_manifest_returns_error(self, monkeypatch, tmp_path):
        monkeypatch.setattr(executor_mod, "TIMING_MANIFEST", str(tmp_path / "missing.json"))
        result = EditExecutor().execute({
            "intent": "speed_up_scene", "scope": "scene:1", "parameters": {"factor": 1.5},
        })
        assert result["ok"] is False


# ── regenerate_script ──────────────────────────────────────────────────────────

class TestRegenerateScript:

    def test_returns_phase_1_rerun(self, patch_executor):
        result = EditExecutor().execute({
            "intent": "regenerate_script", "scope": "all", "parameters": {},
        })
        assert result["ok"] is True
        assert result["phase_to_rerun"] == 1


# ── StateManager ───────────────────────────────────────────────────────────────

@pytest.fixture
def state_mgr(tmp_path, monkeypatch):
    """StateManager rooted in tmp_path via CWD change."""
    monkeypatch.chdir(tmp_path)
    return StateManager()


class TestStateManagerSnapshot:

    def test_snapshot_returns_version_string(self, state_mgr):
        version = state_mgr.snapshot("initial run")
        assert version == "v001"

    def test_sequential_snapshots_increment(self, state_mgr):
        v1 = state_mgr.snapshot("first")
        v2 = state_mgr.snapshot("second")
        assert v1 == "v001"
        assert v2 == "v002"

    def test_snapshot_without_tracked_files_succeeds(self, state_mgr):
        version = state_mgr.snapshot("empty state")
        assert version.startswith("v")

    def test_snapshot_copies_tracked_files(self, state_mgr, tmp_path):
        outputs = tmp_path / "data" / "outputs"
        outputs.mkdir(parents=True)
        (outputs / "scene_manifest.json").write_text('{"test": 1}', encoding="utf-8")
        version   = state_mgr.snapshot("with manifest")
        asset_dir = tmp_path / "data" / "state_versions" / "assets" / version
        assert asset_dir.exists()

    def test_description_stored_in_history(self, state_mgr):
        state_mgr.snapshot("my description")
        h = state_mgr.history()
        assert h[0]["description"] == "my description"

    def test_has_video_false_when_no_video(self, state_mgr):
        state_mgr.snapshot("no video")
        h = state_mgr.history()
        assert h[0]["has_video"] is False


class TestStateManagerHistory:

    def test_empty_db_returns_empty_list(self, state_mgr):
        assert state_mgr.history() == []

    def test_newest_first_ordering(self, state_mgr):
        state_mgr.snapshot("first")
        state_mgr.snapshot("second")
        state_mgr.snapshot("third")
        h = state_mgr.history()
        assert h[0]["description"] == "third"
        assert h[-1]["description"] == "first"

    def test_history_entry_has_required_fields(self, state_mgr):
        state_mgr.snapshot("test")
        entry = state_mgr.history()[0]
        for field in ("id", "version", "timestamp", "description", "has_video"):
            assert field in entry

    def test_version_field_format(self, state_mgr):
        state_mgr.snapshot("test")
        version = state_mgr.history()[0]["version"]
        assert version.startswith("v") and len(version) == 4  # "v001"


class TestStateManagerRevert:

    def test_unknown_version_raises_value_error(self, state_mgr):
        with pytest.raises(ValueError, match="not found"):
            state_mgr.revert("v999")

    def test_revert_restores_file_content(self, state_mgr, tmp_path):
        outputs = tmp_path / "data" / "outputs"
        outputs.mkdir(parents=True)
        manifest = outputs / "scene_manifest.json"
        manifest.write_text('{"original": true}', encoding="utf-8")

        version = state_mgr.snapshot("before edit")

        manifest.write_text('{"modified": true}', encoding="utf-8")
        assert json.loads(manifest.read_text())["modified"] is True

        state_mgr.revert(version)
        restored = json.loads(manifest.read_text())
        assert restored.get("original") is True

    def test_revert_returns_true_on_success(self, state_mgr, tmp_path):
        version = state_mgr.snapshot("snap")
        result  = state_mgr.revert(version)
        assert result is True

    def test_revert_missing_asset_dir_raises(self, state_mgr, tmp_path):
        state_mgr.snapshot("snap")
        # Manually delete the asset directory
        import shutil
        asset_dir = tmp_path / "data" / "state_versions" / "assets" / "v001"
        shutil.rmtree(str(asset_dir))
        with pytest.raises(FileNotFoundError):
            state_mgr.revert("v001")
