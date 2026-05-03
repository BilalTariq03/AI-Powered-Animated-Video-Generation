"""
shared/schemas/schema.py
─────────────────────────
Pydantic schema definitions for Phase 1 output.
All phases validate against these models.
"""

from __future__ import annotations
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

MoodType      = Literal["tense", "mysterious", "hopeful", "dramatic", "comedic", "romantic", "melancholic"]
ToneType      = Literal["dark", "light", "neutral", "suspenseful", "uplifting"]
EmotionType   = Literal["neutral", "angry", "sad", "excited", "fearful", "surprised", "disgusted", "happy"]
TimeOfDay     = Literal["DAY", "NIGHT", "DAWN", "DUSK"]
RoleType      = Literal["protagonist", "antagonist", "supporting", "minor"]
VoiceStyle    = Literal["deep", "soft", "raspy", "high-pitched", "neutral", "authoritative", "whispery"]
SpeakingSpeed = Literal["slow", "normal", "fast"]
PitchType     = Literal["low", "medium", "high"]


class StoryModel(BaseModel):
    title:                      str       = Field(..., min_length=1)
    genre:                      str       = Field(..., min_length=1)
    synopsis:                   str       = Field(..., min_length=20)
    themes:                     List[str] = Field(..., min_length=1)
    arc:                        str
    estimated_duration_seconds: int       = Field(..., gt=0)

    @field_validator("themes")
    @classmethod
    def themes_non_empty(cls, v):
        if not v:
            raise ValueError("themes must contain at least one item")
        return v


class DialogueLine(BaseModel):
    speaker:    str         = Field(..., min_length=1)
    line:       str         = Field(..., min_length=1)
    emotion:    EmotionType = "neutral"
    visual_cue: str         = Field(..., min_length=1)


class SceneModel(BaseModel):
    scene_id:                int
    location:                str            = Field(..., min_length=1)
    time_of_day:             TimeOfDay
    duration_seconds:        int            = Field(..., gt=0, le=120)
    mood:                    MoodType
    tone:                    ToneType
    characters:              List[str]      = Field(..., min_length=1)
    action:                  str            = Field(..., min_length=1)
    dialogue:                List[DialogueLine] = Field(..., min_length=2)
    visual_notes:            str            = Field(..., min_length=1)
    image_generation_prompt: str            = Field(..., min_length=20)

    @field_validator("dialogue")
    @classmethod
    def min_two_dialogue_lines(cls, v):
        if len(v) < 2:
            raise ValueError("Each scene must have at least 2 dialogue lines")
        return v


class AppearanceModel(BaseModel):
    build:                   str = Field(..., min_length=1)
    hair:                    str = Field(..., min_length=1)
    eyes:                    str = Field(..., min_length=1)
    clothing_style:          str = Field(..., min_length=1)
    distinguishing_features: str = ""


class VoiceProfileModel(BaseModel):
    voice_style:     VoiceStyle    = "neutral"
    speaking_speed:  SpeakingSpeed = "normal"
    pitch:           PitchType     = "medium"
    emotion_range:   List[EmotionType] = Field(default_factory=lambda: ["neutral"])
    tts_description: str           = Field(..., min_length=10)

    @field_validator("emotion_range")
    @classmethod
    def emotion_range_non_empty(cls, v):
        return v if v else ["neutral"]


class CharacterModel(BaseModel):
    name:               str              = Field(..., min_length=1)
    role:               RoleType
    age_range:          str              = Field(..., min_length=1)
    gender:             str              = Field(..., min_length=1)
    personality_traits: List[str]        = Field(..., min_length=1)
    appearance:         AppearanceModel
    voice_profile:      VoiceProfileModel
    reference_style:    str              = Field(..., min_length=1)
    image_prompt:       str              = Field(..., min_length=30)
    scenes_appeared:    List[int]        = Field(..., min_length=1)


class PhaseOneOutput(BaseModel):
    story:      StoryModel           = Field(..., description="Narrative summary and metadata")
    scenes:     List[SceneModel]     = Field(..., min_length=1)
    characters: List[CharacterModel] = Field(..., min_length=1)

    @model_validator(mode="after")
    def character_names_consistent_with_scenes(self) -> "PhaseOneOutput":
        known = {c.name for c in self.characters}
        for scene in self.scenes:
            for name in scene.characters:
                if name not in known:
                    raise ValueError(
                        f"Scene {scene.scene_id} references unknown character '{name}'. "
                        f"Known: {known}"
                    )
        return self

    @model_validator(mode="after")
    def scene_ids_unique(self) -> "PhaseOneOutput":
        ids = [s.scene_id for s in self.scenes]
        if len(ids) != len(set(ids)):
            raise ValueError("scene_id values must be unique")
        return self


def validate_phase1_output(data: dict) -> tuple[Optional[PhaseOneOutput], list[str]]:
    from pydantic import ValidationError
    try:
        return PhaseOneOutput(**data), []
    except ValidationError as e:
        errors = [f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors()]
        return None, errors
