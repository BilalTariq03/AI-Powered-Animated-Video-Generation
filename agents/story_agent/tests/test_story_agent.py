"""
agents/story_agent/tests/test_story_agent.py
─────────────────────────────────────────────
Unit tests for Phase 1 — Story & Script Generation.

Covers:
  - ScriptwriterAgent._all_character_names()
  - ScriptwriterAgent._enrich()
  - validate_phase1_output()  (Pydantic schema)

Run:  pytest agents/story_agent/tests/test_story_agent.py -v
"""

import os
import sys
from unittest.mock import MagicMock

import pytest

ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, ROOT)

from shared.schemas.schema import validate_phase1_output


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_agent():
    """Return a ScriptwriterAgent with all external I/O bypassed."""
    from agents.story_agent.agent import ScriptwriterAgent
    agent = ScriptwriterAgent.__new__(ScriptwriterAgent)
    agent.chat = MagicMock(return_value="A generated synopsis long enough to pass validation.")
    agent.invoke_tool = MagicMock(return_value=MagicMock(success=False, data=None))
    return agent


def _base_scene(
    *,
    scene_id=1,
    mood="dramatic",
    tone="neutral",
    time_of_day="DAY",
    duration_seconds=30,
    chars=("Hero", "Villain"),
    image_prompt="A dark misty forest at dawn with tall ancient trees and dense fog.",
):
    return {
        "scene_id": scene_id,
        "location": "A forest",
        "time_of_day": time_of_day,
        "duration_seconds": duration_seconds,
        "mood": mood,
        "tone": tone,
        "characters": list(chars),
        "action": "The characters meet.",
        "dialogue": [
            {"speaker": "Hero",   "line": "We must act now.", "emotion": "neutral", "visual_cue": "Med."},
            {"speaker": "Villain","line": "Too late.",        "emotion": "angry",   "visual_cue": "CU."},
        ],
        "visual_notes": "Cinematic composition.",
        "image_generation_prompt": image_prompt,
    }


def _valid_character(name="Hero"):
    return {
        "name": name,
        "role": "protagonist",
        "age_range": "30s",
        "gender": "male",
        "personality_traits": ["brave"],
        "appearance": {
            "build": "athletic", "hair": "dark", "eyes": "brown",
            "clothing_style": "casual", "distinguishing_features": "",
        },
        "voice_profile": {
            "voice_style": "neutral", "speaking_speed": "normal", "pitch": "medium",
            "emotion_range": ["neutral"], "tts_description": "A calm, measured voice.",
        },
        "reference_style": "cinematic realism",
        "image_prompt": f"Portrait of {name}, cinematic realism, photorealistic, detailed photography.",
        "scenes_appeared": [1],
    }


def _long_synopsis():
    return "A story about two characters who must make hard choices in a dark world."


# ── _all_character_names ───────────────────────────────────────────────────────

class TestAllCharacterNames:

    def test_extracts_string_characters(self):
        agent = _make_agent()
        parsed = {"scenes": [{"characters": ["Alice", "Bob"]}]}
        assert set(agent._all_character_names(parsed)) == {"Alice", "Bob"}

    def test_extracts_dict_characters_by_name_key(self):
        agent = _make_agent()
        parsed = {"scenes": [{"characters": [{"name": "Alice"}]}]}
        assert "Alice" in agent._all_character_names(parsed)

    def test_extracts_dict_characters_by_character_key(self):
        agent = _make_agent()
        parsed = {"scenes": [{"characters": [{"character": "Bob"}]}]}
        assert "Bob" in agent._all_character_names(parsed)

    def test_deduplicates_across_scenes(self):
        agent = _make_agent()
        parsed = {
            "scenes": [
                {"characters": ["Alice", "Bob"]},
                {"characters": ["Alice", "Carol"]},
            ]
        }
        names = agent._all_character_names(parsed)
        assert len(names) == 3
        assert set(names) == {"Alice", "Bob", "Carol"}

    def test_empty_parsed_returns_empty(self):
        agent = _make_agent()
        assert agent._all_character_names({}) == []
        assert agent._all_character_names({"scenes": []}) == []

    def test_mixed_string_and_dict_entries(self):
        agent = _make_agent()
        parsed = {"scenes": [{"characters": ["Alice", {"name": "Bob"}]}]}
        names = agent._all_character_names(parsed)
        assert "Alice" in names and "Bob" in names


# ── _enrich ────────────────────────────────────────────────────────────────────

class TestEnrich:

    def _parsed(self, **scene_kwargs):
        return {
            "story": {"title": "T", "synopsis": _long_synopsis()},
            "scenes": [_base_scene(**scene_kwargs)],
        }

    def test_coerces_string_scene_id_to_int(self):
        agent = _make_agent()
        result = agent._enrich(self._parsed(scene_id="2"), {}, "prompt")
        assert result["scenes"][0]["scene_id"] == 2

    def test_invalid_scene_id_uses_index(self):
        agent = _make_agent()
        result = agent._enrich(self._parsed(scene_id="abc"), {}, "prompt")
        assert result["scenes"][0]["scene_id"] == 1  # index 0 + 1

    def test_invalid_mood_replaced_by_arc_mood(self):
        valid = {"tense","mysterious","hopeful","dramatic","comedic","romantic","melancholic","fantasy"}
        agent = _make_agent()
        result = agent._enrich(self._parsed(mood="spooky"), {}, "prompt")
        assert result["scenes"][0]["mood"] in valid

    def test_valid_mood_preserved(self):
        agent = _make_agent()
        result = agent._enrich(self._parsed(mood="hopeful"), {}, "prompt")
        assert result["scenes"][0]["mood"] == "hopeful"

    def test_fantasy_mood_preserved(self):
        agent = _make_agent()
        result = agent._enrich(self._parsed(mood="fantasy"), {}, "prompt")
        assert result["scenes"][0]["mood"] == "fantasy"

    def test_invalid_tone_falls_back_to_neutral(self):
        agent = _make_agent()
        result = agent._enrich(self._parsed(tone="boring"), {}, "prompt")
        assert result["scenes"][0]["tone"] == "neutral"

    def test_valid_tone_preserved(self):
        agent = _make_agent()
        result = agent._enrich(self._parsed(tone="dark"), {}, "prompt")
        assert result["scenes"][0]["tone"] == "dark"

    def test_lowercase_time_of_day_uppercased(self):
        agent = _make_agent()
        result = agent._enrich(self._parsed(time_of_day="night"), {}, "prompt")
        assert result["scenes"][0]["time_of_day"] == "NIGHT"

    def test_invalid_time_of_day_replaced_with_day(self):
        agent = _make_agent()
        result = agent._enrich(self._parsed(time_of_day="evening"), {}, "prompt")
        assert result["scenes"][0]["time_of_day"] == "DAY"

    def test_duration_clamped_to_120(self):
        agent = _make_agent()
        result = agent._enrich(self._parsed(duration_seconds=200), {}, "prompt")
        assert result["scenes"][0]["duration_seconds"] == 120

    def test_duration_within_limit_preserved(self):
        agent = _make_agent()
        result = agent._enrich(self._parsed(duration_seconds=45), {}, "prompt")
        assert result["scenes"][0]["duration_seconds"] == 45

    def test_dict_characters_normalized_to_strings(self):
        agent = _make_agent()
        parsed = self._parsed()
        parsed["scenes"][0]["characters"] = [{"name": "Alice"}, "Bob"]
        result = agent._enrich(parsed, {}, "prompt")
        for c in result["scenes"][0]["characters"]:
            assert isinstance(c, str)

    def test_empty_dialogue_line_replaced(self):
        agent = _make_agent()
        parsed = self._parsed()
        parsed["scenes"][0]["dialogue"][0]["line"] = "   "
        result = agent._enrich(parsed, {}, "prompt")
        assert result["scenes"][0]["dialogue"][0]["line"] != ""

    def test_single_dialogue_line_gets_fallback_added(self):
        agent = _make_agent()
        parsed = self._parsed()
        parsed["scenes"][0]["dialogue"] = [parsed["scenes"][0]["dialogue"][0]]
        result = agent._enrich(parsed, {}, "prompt")
        assert len(result["scenes"][0]["dialogue"]) >= 2

    def test_invalid_emotion_normalized_to_neutral(self):
        agent = _make_agent()
        parsed = self._parsed()
        parsed["scenes"][0]["dialogue"][0]["emotion"] = "ecstatic"
        result = agent._enrich(parsed, {}, "prompt")
        assert result["scenes"][0]["dialogue"][0]["emotion"] == "neutral"

    def test_visual_cue_defaulted_when_missing(self):
        agent = _make_agent()
        parsed = self._parsed()
        del parsed["scenes"][0]["dialogue"][0]["visual_cue"]
        result = agent._enrich(parsed, {}, "prompt")
        assert result["scenes"][0]["dialogue"][0].get("visual_cue")

    def test_flat_story_structure_promoted_to_nested(self):
        """LLM sometimes puts title/genre at top level instead of under 'story'."""
        agent = _make_agent()
        parsed = {"title": "Flat Title", "genre": "Action", "scenes": [_base_scene()]}
        result = agent._enrich(parsed, {"genre": "Action", "themes": ["conflict"]}, "prompt")
        assert "story" in result
        assert result["story"]["title"] == "Flat Title"

    def test_story_defaults_filled_from_story_meta(self):
        agent = _make_agent()
        parsed = {"story": {"synopsis": _long_synopsis()}, "scenes": [_base_scene()]}
        result = agent._enrich(parsed, {"genre": "Sci-Fi", "themes": ["exploration"]}, "prompt")
        assert result["story"]["genre"] == "Sci-Fi"
        assert result["story"]["themes"] == ["exploration"]

    def test_short_synopsis_triggers_llm_call(self):
        agent = _make_agent()
        parsed = {
            "story": {"title": "T", "synopsis": "Short."},
            "scenes": [_base_scene()],
        }
        agent._enrich(parsed, {}, "a story prompt")
        agent.chat.assert_called()


# ── validate_phase1_output ─────────────────────────────────────────────────────

class TestValidatePhaseOneOutput:

    def _valid_data(self):
        return {
            "story": {
                "title": "Test Story", "genre": "Drama",
                "synopsis": _long_synopsis(),
                "themes": ["redemption", "loss"],
                "arc": "intro -> end",
                "estimated_duration_seconds": 60,
            },
            "scenes": [{
                "scene_id": 1, "location": "Forest clearing", "time_of_day": "DAY",
                "duration_seconds": 30, "mood": "dramatic", "tone": "dark",
                "characters": ["Hero"], "action": "Hero contemplates.",
                "dialogue": [
                    {"speaker": "Hero", "line": "I cannot do this.", "emotion": "sad",    "visual_cue": "CU."},
                    {"speaker": "Hero", "line": "But I must.",       "emotion": "neutral", "visual_cue": "Wide."},
                ],
                "visual_notes": "Moody forest lighting.",
                "image_generation_prompt": "Dense misty forest with dramatic side lighting at dawn.",
            }],
            "characters": [_valid_character("Hero")],
        }

    def test_valid_input_passes(self):
        obj, errors = validate_phase1_output(self._valid_data())
        assert obj is not None and errors == []

    def test_returns_phase_one_output_instance(self):
        from shared.schemas.schema import PhaseOneOutput
        obj, _ = validate_phase1_output(self._valid_data())
        assert isinstance(obj, PhaseOneOutput)

    def test_short_synopsis_fails(self):
        data = self._valid_data()
        data["story"]["synopsis"] = "Too short."
        obj, errors = validate_phase1_output(data)
        assert obj is None
        assert any("synopsis" in e for e in errors)

    def test_empty_scenes_list_fails(self):
        data = self._valid_data()
        data["scenes"] = []
        _, errors = validate_phase1_output(data)
        assert errors

    def test_single_dialogue_line_fails(self):
        data = self._valid_data()
        data["scenes"][0]["dialogue"] = [data["scenes"][0]["dialogue"][0]]
        _, errors = validate_phase1_output(data)
        assert errors

    def test_duration_over_120_fails(self):
        data = self._valid_data()
        data["scenes"][0]["duration_seconds"] = 150
        _, errors = validate_phase1_output(data)
        assert errors

    def test_unknown_character_referenced_in_scene_fails(self):
        data = self._valid_data()
        data["scenes"][0]["characters"] = ["Ghost"]
        data["scenes"][0]["dialogue"][0]["speaker"] = "Ghost"
        data["scenes"][0]["dialogue"][1]["speaker"] = "Ghost"
        _, errors = validate_phase1_output(data)
        assert errors  # Ghost not in characters list

    def test_duplicate_scene_ids_fail(self):
        data = self._valid_data()
        data["scenes"].append({**data["scenes"][0]})  # same scene_id=1 twice
        _, errors = validate_phase1_output(data)
        assert errors

    def test_invalid_mood_fails(self):
        data = self._valid_data()
        data["scenes"][0]["mood"] = "spooky"
        _, errors = validate_phase1_output(data)
        assert errors

    def test_invalid_time_of_day_fails(self):
        data = self._valid_data()
        data["scenes"][0]["time_of_day"] = "evening"
        _, errors = validate_phase1_output(data)
        assert errors

    def test_short_image_prompt_fails(self):
        data = self._valid_data()
        data["scenes"][0]["image_generation_prompt"] = "Too short"
        _, errors = validate_phase1_output(data)
        assert errors

    def test_fantasy_mood_valid(self):
        """Regression: fantasy was added as a valid mood value."""
        data = self._valid_data()
        data["scenes"][0]["mood"] = "fantasy"
        data["characters"][0]["role"] = "protagonist"  # ensure char still in list
        obj, errors = validate_phase1_output(data)
        assert obj is not None, f"fantasy mood should be valid; errors: {errors}"

    def test_all_valid_moods_accepted(self):
        valid_moods = ["tense","mysterious","hopeful","dramatic","comedic","romantic","melancholic","fantasy"]
        for mood in valid_moods:
            data = self._valid_data()
            data["scenes"][0]["mood"] = mood
            obj, errors = validate_phase1_output(data)
            assert obj is not None, f"Mood '{mood}' should be valid; errors: {errors}"

    def test_all_valid_time_of_day_accepted(self):
        for tod in ["DAY", "NIGHT", "DAWN", "DUSK"]:
            data = self._valid_data()
            data["scenes"][0]["time_of_day"] = tod
            obj, errors = validate_phase1_output(data)
            assert obj is not None, f"time_of_day '{tod}' should be valid; errors: {errors}"

    def test_image_prompt_at_30_chars_passes(self):
        data = self._valid_data()
        data["scenes"][0]["image_generation_prompt"] = "A" * 30
        obj, errors = validate_phase1_output(data)
        assert obj is not None, f"30-char prompt should pass; errors: {errors}"

    def test_error_list_empty_on_success(self):
        _, errors = validate_phase1_output(self._valid_data())
        assert errors == []

    def test_missing_required_field_returns_errors(self):
        data = self._valid_data()
        del data["story"]["genre"]
        _, errors = validate_phase1_output(data)
        assert errors
