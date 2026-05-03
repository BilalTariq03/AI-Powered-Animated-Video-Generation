"""
agents/audio_agent/agent.py
────────────────────────────
Phase 2 audio generation pipeline agent.

Reads:  data/outputs/phase2_audio_handoff.json  (from Phase 1)
Writes: data/outputs/audio/dialogue/<segment_id>.mp3
        data/outputs/audio/bgm/scene<N>_bgm.mp3
        data/outputs/audio/timing_manifest.json
"""

import json
import os
import shutil

from mcp.tools.audio_tools.tts_tool import TTSEngine
from mcp.tools.audio_tools.bgm_tool import BGMEngine
from config import DIALOGUE_DIR, BGM_DIR, TIMING_MANIFEST


class AudioGenerationAgent:
    name = "AudioGenerationAgent"

    def __init__(self, elevenlabs_api_key: str = ""):
        self.tts = TTSEngine(elevenlabs_api_key)
        self.bgm = BGMEngine()

    def run(self, handoff_path: str) -> dict:
        print(f"\n[{self.name}] Loading handoff: {handoff_path}")
        with open(handoff_path, encoding="utf-8") as f:
            handoff = json.load(f)

        voice_map   = {vc["character_name"]: vc for vc in handoff.get("voice_configs", [])}
        segments    = handoff.get("segments", [])
        music_moods = handoff.get("music_moods", [])

        for d in (DIALOGUE_DIR, BGM_DIR):
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d)

        print(f"[{self.name}] Synthesising {len(segments)} dialogue segments...")
        timing_entries = self._synthesise_dialogue(segments, voice_map)

        print(f"[{self.name}] Generating {len(music_moods)} BGM tracks...")
        bgm_entries = self._generate_bgm(music_moods)

        manifest = self._build_manifest(handoff.get("title", ""), timing_entries, bgm_entries)
        os.makedirs(os.path.dirname(TIMING_MANIFEST), exist_ok=True)
        with open(TIMING_MANIFEST, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print(f"\n[{self.name}] Complete.")
        print(f"   Dialogue : {len(timing_entries)} files -> {DIALOGUE_DIR}/")
        print(f"   BGM      : {len(bgm_entries)} files -> {BGM_DIR}/")
        print(f"   Manifest : {TIMING_MANIFEST}")
        return manifest

    def _synthesise_dialogue(self, segments: list, voice_map: dict) -> list:
        entries   = []
        cursor_ms: dict[int, int] = {}

        for seg in segments:
            seg_id      = seg["segment_id"]
            scene_id    = seg["scene_id"]
            speaker     = seg["speaker"]
            line        = seg["line"]
            voice_config = voice_map.get(speaker, {"voice_style": "neutral", "speaking_speed": "normal"})
            out_path    = os.path.join(DIALOGUE_DIR, f"{seg_id}.mp3")

            print(f"[{self.name}]   {seg_id}: {speaker!r} -> {line[:50]!r}...")
            duration_ms = self.tts.synthesize(line, voice_config, out_path)

            start_ms = cursor_ms.get(scene_id, 0)
            end_ms   = start_ms + duration_ms
            cursor_ms[scene_id] = end_ms

            entries.append({
                "segment_id":  seg_id,
                "scene_id":    scene_id,
                "speaker":     speaker,
                "line":        line,
                "emotion":     seg.get("emotion", "neutral"),
                "audio_file":  out_path,
                "start_ms":    start_ms,
                "end_ms":      end_ms,
                "duration_ms": duration_ms,
            })
        return entries

    # Fallback arc applied when Phase 1 assigns the same mood to every scene
    _MOOD_ARC = ["mysterious", "tense", "dramatic", "melancholic", "hopeful",
                 "tense", "dramatic", "romantic"]

    def _generate_bgm(self, music_moods: list) -> list:
        # If all scenes share one mood (common when script is regenerated with old Phase 1)
        # auto-assign a varied arc so BGM is distinct per scene.
        all_moods = [mm.get("mood", "dramatic") for mm in music_moods]
        if len(set(all_moods)) == 1:
            print(f"[{self.name}] All scenes share mood '{all_moods[0]}' - applying mood arc for variety.")
            for i, mm in enumerate(music_moods):
                mm["mood"] = self._MOOD_ARC[i % len(self._MOOD_ARC)]

        entries = []
        for mm in music_moods:
            scene_id = mm["scene_id"]
            mood     = mm.get("mood", "dramatic")
            duration = mm.get("duration_seconds", 30)
            out_path = os.path.join(BGM_DIR, f"scene{scene_id}_bgm.mp3")
            print(f"[{self.name}]   Scene {scene_id} BGM: mood={mood}, {duration}s")
            actual_path = self.bgm.generate(mood, duration, out_path)
            entries.append({"scene_id": scene_id, "mood": mood,
                            "audio_file": actual_path, "duration_ms": duration * 1000})
        return entries

    def _build_manifest(self, title: str, timing_entries: list, bgm_entries: list) -> dict:
        bgm_by_scene = {b["scene_id"]: b for b in bgm_entries}
        scenes_map: dict[int, list] = {}
        for entry in timing_entries:
            scenes_map.setdefault(entry["scene_id"], []).append(entry)

        scene_list = []
        for scene_id, segs in sorted(scenes_map.items()):
            bgm      = bgm_by_scene.get(scene_id, {})
            dia_end  = max((s["end_ms"] for s in segs), default=0)
            total_ms = max(dia_end, bgm.get("duration_ms", 30_000))
            scene_list.append({
                "scene_id":          scene_id,
                "dialogue_segments": segs,
                "bgm_file":          bgm.get("audio_file", ""),
                "bgm_mood":          bgm.get("mood", ""),
                "total_duration_ms": total_ms,
            })

        return {
            "title":         title,
            "total_scenes":  len(scene_list),
            "scenes":        scene_list,
            "flat_segments": timing_entries,
        }
