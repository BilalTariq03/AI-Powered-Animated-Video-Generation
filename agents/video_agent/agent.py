"""
agents/video_agent/agent.py
────────────────────────────
Phase 3 — Video Generation Agent.

Per-scene pipeline:
  1. Generate background (HuggingFace FLUX) with gradient fallback
  2. Ken Burns zoom+pan on background
  3. Pre-render Wav2Lip lip-sync clip per dialogue segment (if available)
  4. Character switching — active speaker shown large; others dimmed at edge
  5. Body sway + head-bob animation when Wav2Lip not available
  6. Emotion tint on portrait during each line
  7. Subtitle bar timed to timing_manifest
  8. BGM + dialogue audio mixed and trimmed to actual dialogue length
  9. 0.5s fade-in / fade-out transitions

Scene duration = last dialogue end_ms + 2s buffer (not padded 30s from Phase 1).
"""

import json
import os

import numpy as np
from PIL import Image, ImageDraw

from config import (
    HF_API_KEY, HF_IMAGE_MODEL, IMAGES_DIR,
    VIDEO_DIR, SCENES_DIR, LIPSYNC_DIR, FINAL_VIDEO,
    PHASE3_HANDOFF, TIMING_MANIFEST,
)

VIDEO_W          = 1280
VIDEO_H          = 720
FPS              = 24
SCENE_END_BUFFER = 2.0
MIN_SCENE_DUR    = 4.0
FADE_DUR         = 0.5

CHAR_W, CHAR_H   = 180, 230   # active speaker portrait size
SIDE_W, SIDE_H   = 90,  115   # other characters (smaller, dimmed)

_MOOD_GRADIENT = {
    "dramatic":    ((15,  0,  0),  (70, 15, 15)),
    "tense":       ((0,  10, 30),  (25, 35, 55)),
    "mysterious":  ((10,  0, 30),  (35, 10, 55)),
    "hopeful":     ((20, 40, 80),  (70, 110, 150)),
    "melancholic": ((15, 15, 35),  (50, 50, 80)),
    "comedic":     ((30, 70, 30),  (90, 150, 70)),
    "romantic":    ((55, 15, 35),  (110, 55, 75)),
}
_TIME_BRIGHTNESS = {"DAY": 1.0, "DAWN": 0.75, "DUSK": 0.65, "NIGHT": 0.45}

_EMOTION_TINT = {
    "angry":     (255, 160, 160, 55),
    "sad":       (160, 190, 255, 55),
    "happy":     (255, 255, 160, 45),
    "excited":   (255, 220, 120, 50),
    "fearful":   (180, 255, 200, 45),
    "surprised": (220, 180, 255, 50),
    "disgusted": (190, 220, 160, 45),
    "neutral":   None,
}


class VideoGenerationAgent:

    def run(self, phase3_path: str = PHASE3_HANDOFF,
            timing_path: str = TIMING_MANIFEST) -> str:
        print(f"\n[VideoAgent] Loading handoff: {phase3_path}")
        handoff = _load(phase3_path)
        timing  = _load(timing_path)

        timing_by_scene = {s["scene_id"]: s for s in timing.get("scenes", [])}
        char_images     = self._resolve_char_images(handoff.get("character_visuals", []))

        for d in (SCENES_DIR, LIPSYNC_DIR, IMAGES_DIR):
            os.makedirs(d, exist_ok=True)

        # Wav2Lip availability check
        from agents.video_agent import lip_sync
        use_lipsync = lip_sync.is_available()
        print(f"[VideoAgent] Wav2Lip: {'ENABLED' if use_lipsync else 'NOT SET UP (run setup_wav2lip.py)'}")

        print(f"[VideoAgent] Generating {len(handoff['scenes'])} scene backgrounds...")
        bg_paths = self._prepare_backgrounds(handoff["scenes"])

        scene_clips = []
        for scene in handoff["scenes"]:
            sid          = scene["scene_id"]
            timing_scene = timing_by_scene.get(sid, {})
            duration     = self._calc_duration(timing_scene)

            print(f"[VideoAgent] Rendering scene {sid}  ({duration:.1f}s)...")
            clip = self._render_scene(
                scene, timing_scene, bg_paths.get(sid),
                char_images, duration, use_lipsync,
            )
            clip = clip.fadein(FADE_DUR).fadeout(FADE_DUR)

            scene_path = os.path.join(SCENES_DIR, f"scene{sid}.mp4")
            clip.write_videofile(
                scene_path, fps=FPS, codec="libx264",
                audio_codec="aac", verbose=False, logger=None,
            )
            scene_clips.append(clip)
            print(f"[VideoAgent]   Saved -> {scene_path}")

        print("[VideoAgent] Concatenating scenes...")
        from moviepy.editor import concatenate_videoclips
        final = concatenate_videoclips(scene_clips, method="compose")
        final.write_videofile(
            FINAL_VIDEO, fps=FPS, codec="libx264",
            audio_codec="aac", verbose=False, logger=None,
        )
        print(f"[VideoAgent] Done -> {FINAL_VIDEO}")
        return FINAL_VIDEO

    # ── Duration ──────────────────────────────────────────────────────────────

    def _calc_duration(self, timing_scene: dict) -> float:
        segs = timing_scene.get("dialogue_segments", [])
        if segs:
            last_end_ms = max(s["end_ms"] for s in segs)
            return max(last_end_ms / 1000 + SCENE_END_BUFFER, MIN_SCENE_DUR)
        return MIN_SCENE_DUR

    # ── Background generation ─────────────────────────────────────────────────

    def _prepare_backgrounds(self, scenes: list) -> dict:
        paths = {}
        for scene in scenes:
            sid     = scene["scene_id"]
            bg_path = os.path.join(IMAGES_DIR, f"scene{sid}_bg.png")
            if os.path.exists(bg_path):
                print(f"[VideoAgent]   Scene {sid} bg: cached")
                paths[sid] = bg_path
                continue
            print(f"[VideoAgent]   Scene {sid} bg: generating...")
            path = self._hf_generate(scene.get("image_generation_prompt", ""), bg_path)
            if not path:
                print(f"[VideoAgent]   Scene {sid} bg: HF failed, using gradient.")
                path = self._gradient_bg(scene.get("mood", "dramatic"),
                                         scene.get("time_of_day", "DAY"), bg_path)
            paths[sid] = path
        return paths

    def _hf_generate(self, prompt: str, save_path: str) -> str | None:
        import io, urllib.parse, requests
        full = (f"{prompt}, cinematic background, no people, "
                "no characters, highly detailed, wide angle")

        # ── Primary: Pollinations.ai (free, no key needed) ────────────────
        try:
            encoded = urllib.parse.quote(full)
            url     = (f"https://image.pollinations.ai/prompt/{encoded}"
                       f"?width={VIDEO_W}&height={VIDEO_H}&model=flux&nologo=true")
            resp = requests.get(url, timeout=90)
            if resp.status_code == 200 and resp.content:
                img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                img.resize((VIDEO_W, VIDEO_H), Image.LANCZOS).save(save_path)
                return save_path
            print(f"[VideoAgent] Pollinations returned {resp.status_code}")
        except Exception as e:
            print(f"[VideoAgent] Pollinations error: {e}")

        # ── Fallback: HuggingFace (requires HF_API_KEY) ───────────────────
        if not HF_API_KEY:
            return None
        import time
        url     = f"https://router.huggingface.co/hf-inference/models/{HF_IMAGE_MODEL}"
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        try:
            resp = requests.post(url, headers=headers, json={"inputs": full}, timeout=120)
            if resp.status_code == 503:
                wait = resp.json().get("estimated_time", 30) if resp.content else 30
                time.sleep(min(wait, 60))
                resp = requests.post(url, headers=headers, json={"inputs": full}, timeout=120)
            if resp.status_code != 200:
                print(f"[VideoAgent] HF API returned {resp.status_code}: {resp.text[:200]}")
                return None
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            img.resize((VIDEO_W, VIDEO_H), Image.LANCZOS).save(save_path)
            return save_path
        except Exception as e:
            print(f"[VideoAgent] HF error: {e}")
            return None

    def _gradient_bg(self, mood: str, time_of_day: str, save_path: str) -> str:
        top, bot   = _MOOD_GRADIENT.get(mood, ((20, 20, 20), (60, 60, 60)))
        brightness = _TIME_BRIGHTNESS.get(time_of_day, 1.0)
        arr        = np.zeros((VIDEO_H, VIDEO_W, 3), dtype=np.float32)
        for c in range(3):
            arr[:, :, c] = np.linspace(top[c], bot[c], VIDEO_H).reshape(-1, 1) * brightness
        Image.fromarray(arr.clip(0, 255).astype(np.uint8)).save(save_path)
        return save_path

    # ── Scene rendering ───────────────────────────────────────────────────────

    def _render_scene(self, scene: dict, timing_scene: dict,
                      bg_path: str | None, char_images: dict,
                      duration: float, use_lipsync: bool):
        from moviepy.editor import VideoClip
        from agents.video_agent import lip_sync

        segments    = timing_scene.get("dialogue_segments", [])
        bgm_file    = timing_scene.get("bgm_file", "")
        scene_id    = scene["scene_id"]
        chars_in    = scene.get("characters_in_scene", [])
        scene_label = scene.get("location", "")

        # ── Pre-render Wav2Lip clips per segment ──────────────────────────
        timeline = self._build_timeline(segments)
        if use_lipsync:
            print(f"[VideoAgent]   Running Wav2Lip on {len(timeline)} segments...")
            for item in timeline:
                speaker    = item["speaker"]
                face_path  = self._find_char_image(speaker, char_images)
                audio_path = item.get("audio_file", "")
                if not face_path or not audio_path or not os.path.exists(audio_path):
                    continue
                out_mp4 = os.path.join(
                    LIPSYNC_DIR, f"scene{scene_id}_{item['segment_id']}.mp4"
                )
                if os.path.exists(out_mp4):
                    print(f"[VideoAgent]     {item['segment_id']}: cached")
                else:
                    print(f"[VideoAgent]     {item['segment_id']}: generating...")
                    lip_sync.generate(face_path, audio_path, out_mp4)

                if os.path.exists(out_mp4):
                    frames = lip_sync.load_frames(out_mp4)
                    if frames:
                        item["lipsync_frames"] = frames

        # ── Load and oversized background for Ken Burns ───────────────────
        bg_arr = self._load_bg(bg_path, scene)

        # ── Load character images: chars_in + every speaker in timeline ───
        # Key scene_char_imgs by the actual speaker name used in timeline
        # so _draw_characters can look up by item["speaker"] directly.
        all_char_names = list(chars_in)
        for item in timeline:
            if item["speaker"] not in all_char_names:
                all_char_names.append(item["speaker"])

        scene_char_imgs = {}
        for name in all_char_names:
            path = self._find_char_image(name, char_images)
            if path and os.path.exists(path) and name not in scene_char_imgs:
                scene_char_imgs[name] = Image.open(path).convert("RGBA")
        # Deduplicate: if two names resolved to the same image file, drop the
        # chars_in version so timeline speaker names are used for lookup.
        seen_paths: set[str] = set()
        deduped: dict = {}
        # Prefer timeline speaker names over chars_in names
        timeline_speakers = [it["speaker"] for it in timeline]
        ordered = timeline_speakers + [n for n in all_char_names if n not in timeline_speakers]
        for name in ordered:
            if name in scene_char_imgs:
                path = self._find_char_image(name, char_images)
                if path not in seen_paths:
                    seen_paths.add(path)
                    deduped[name] = scene_char_imgs[name]
        scene_char_imgs = deduped

        def make_frame(t: float) -> np.ndarray:
            # 1 — Ken Burns: zoom from 1.0 → 1.08, pan right+down
            progress = t / max(duration, 1)
            zoom     = 1.0 + 0.08 * progress
            new_w    = int(VIDEO_W / zoom)
            new_h    = int(VIDEO_H / zoom)
            sh, sw   = bg_arr.shape[:2]
            ox       = int(max(sw - new_w, 0) * progress)
            oy       = int(max(sh - new_h, 0) * progress)
            crop     = bg_arr[oy:oy + new_h, ox:ox + new_w]
            frame    = np.array(
                Image.fromarray(crop).resize((VIDEO_W, VIDEO_H), Image.BILINEAR)
            )
            img = Image.fromarray(frame).convert("RGBA")

            # 2 — Find active dialogue item
            active = next(
                (it for it in timeline if it["start"] <= t < it["end"]), None
            )
            active_speaker = active["speaker"] if active else (chars_in[0] if chars_in else None)

            # 3 — Draw all scene characters; active speaker is large, rest small + dimmed
            self._draw_characters(img, t, active, active_speaker,
                                  scene_char_imgs, chars_in)

            # 4 — Scene location label (first 2.5 s, fades out)
            if t < 2.5 and scene_label:
                fade_a = int(200 * min(1.0, (2.5 - t) / 0.6))
                label_ov = Image.new("RGBA", (VIDEO_W, VIDEO_H), (0, 0, 0, 0))
                ld = ImageDraw.Draw(label_ov)
                bw = min(50 + len(scene_label) * 10, VIDEO_W - 40)
                ld.rectangle([(20, 20), (bw, 58)], fill=(0, 0, 0, fade_a))
                ld.text((30, 28), scene_label, fill=(220, 220, 180, fade_a))
                img = Image.alpha_composite(img, label_ov)

            # 5 — Subtitle bar
            if active:
                sub = Image.new("RGBA", (VIDEO_W, VIDEO_H), (0, 0, 0, 0))
                sd  = ImageDraw.Draw(sub)
                sd.rectangle([(0, VIDEO_H - 90), (VIDEO_W, VIDEO_H)],
                              fill=(0, 0, 0, 175))
                sd.text((30, VIDEO_H - 82),
                        f'[{active["speaker"]}]  •  {active["emotion"]}',
                        fill=(140, 200, 255, 255))
                sd.text((30, VIDEO_H - 58), active["text"],
                        fill=(255, 255, 255, 255))
                img = Image.alpha_composite(img, sub)

            return np.array(img.convert("RGB"))

        video = VideoClip(make_frame, duration=duration)
        audio = self._mix_audio(bgm_file, segments, duration)
        if audio:
            video = video.set_audio(audio)
        return video

    # ── Character drawing ─────────────────────────────────────────────────────

    def _draw_characters(self, img: Image.Image, t: float,
                         active: dict | None, active_speaker: str | None,
                         scene_char_imgs: dict, chars_in: list):
        if not scene_char_imgs:
            return

        # Active speaker drawn large at bottom-left
        if active_speaker and active_speaker in scene_char_imgs:
            char_pil = scene_char_imgs[active_speaker].copy()

            # Emotion tint
            if active:
                tint = _EMOTION_TINT.get(active.get("emotion", "neutral"))
                if tint:
                    tl = Image.new("RGBA", char_pil.size, tint)
                    char_pil = Image.alpha_composite(char_pil.convert("RGBA"), tl)

            # Wav2Lip frame override when speaking — overlay animated face on portrait
            if active and "lipsync_frames" in active:
                frames   = active["lipsync_frames"]
                t_in_seg = t - active["start"]
                fidx     = min(int(t_in_seg * FPS), len(frames) - 1)
                lip_frame = Image.fromarray(frames[fidx]).convert("RGBA")

                # Base: full portrait letterboxed into CHAR_W × CHAR_H
                orig_w, orig_h = char_pil.size
                fit_ratio = min(CHAR_W / max(orig_w, 1), CHAR_H / max(orig_h, 1))
                base_w = int(orig_w * fit_ratio)
                base_h = int(orig_h * fit_ratio)
                base = char_pil.resize((base_w, base_h), Image.LANCZOS)
                canvas = Image.new("RGBA", (CHAR_W, CHAR_H), (0, 0, 0, 0))
                bx = (CHAR_W - base_w) // 2
                by = (CHAR_H - base_h) // 2
                canvas.paste(base, (bx, by), base)

                # Overlay lipsync frame on face region (upper 65% of portrait area)
                face_h_limit = int(base_h * 0.65)
                lf_w, lf_h = lip_frame.size
                lf_ratio = min(base_w / max(lf_w, 1), face_h_limit / max(lf_h, 1))
                lf_nw = int(lf_w * lf_ratio)
                lf_nh = int(lf_h * lf_ratio)
                lip_frame = lip_frame.resize((lf_nw, lf_nh), Image.LANCZOS)
                lx = bx + (base_w - lf_nw) // 2
                canvas.paste(lip_frame, (lx, by), lip_frame)
                char_pil = canvas
            else:
                # Body animation: breathing + talking pulse
                breathe = 1.0 + 0.015 * np.sin(2 * np.pi * 0.2 * t)
                if active:
                    # Head bob: ±4px vertical at 1.5 Hz
                    bob = int(4 * np.sin(2 * np.pi * 1.5 * t))
                    # Talking scale: ±3% at 3 Hz
                    talk_scale = 1.0 + 0.03 * np.sin(2 * np.pi * 3.0 * t)
                    scale = breathe * talk_scale
                else:
                    bob   = 0
                    scale = breathe

                # Aspect-ratio-preserving resize (letterbox into CHAR_W × CHAR_H)
                orig_w, orig_h = char_pil.size
                fit_ratio = min(CHAR_W / max(orig_w, 1), CHAR_H / max(orig_h, 1)) * scale
                pw = int(orig_w * fit_ratio)
                ph = int(orig_h * fit_ratio)
                char_pil = char_pil.resize((pw, ph), Image.LANCZOS)

                # Pad/crop back to CHAR_W x CHAR_H so position stays fixed
                canvas = Image.new("RGBA", (CHAR_W, CHAR_H), (0, 0, 0, 0))
                ox = (CHAR_W - pw) // 2
                oy = (CHAR_H - ph) // 2 + bob
                canvas.paste(char_pil, (ox, oy), char_pil)
                char_pil = canvas

            # Glow border: blue when speaking, grey when idle
            border_col = (100, 180, 255, 190) if active else (160, 160, 160, 110)
            cx = 20
            cy = VIDEO_H - CHAR_H - 100   # above subtitle bar
            border = Image.new("RGBA", (CHAR_W + 6, CHAR_H + 6), border_col)
            img.paste(border,   (cx - 3,   cy - 3),   border)
            img.paste(char_pil, (cx,        cy),       char_pil)

            # Speaker name tag below portrait
            tag_ov = Image.new("RGBA", (VIDEO_W, VIDEO_H), (0, 0, 0, 0))
            td = ImageDraw.Draw(tag_ov)
            if active_speaker:
                td.rectangle([(cx, cy + CHAR_H + 2), (cx + CHAR_W, cy + CHAR_H + 22)],
                              fill=(0, 0, 0, 160))
                short = (active_speaker[:16] + "…") if len(active_speaker) > 16 else active_speaker
                td.text((cx + 4, cy + CHAR_H + 4), short, fill=(220, 220, 255, 255))
            img.alpha_composite(tag_ov)

        # Other characters (non-speaking) shown small + dimmed on right edge
        others = [n for n in scene_char_imgs if n != active_speaker]
        for i, name in enumerate(others[:3]):
            raw = scene_char_imgs[name]
            ow, oh = raw.size
            side_ratio = min(SIDE_W / max(ow, 1), SIDE_H / max(oh, 1))
            sw = int(ow * side_ratio)
            sh = int(oh * side_ratio)
            side_canvas = Image.new("RGBA", (SIDE_W, SIDE_H), (0, 0, 0, 0))
            side_img = raw.resize((sw, sh), Image.LANCZOS)
            side_canvas.paste(side_img, ((SIDE_W - sw) // 2, (SIDE_H - sh) // 2), side_img)
            other_pil = side_canvas
            # Dim non-speaking characters
            dimmed = Image.new("RGBA", other_pil.size, (0, 0, 0, 0))
            dimmed.paste(other_pil, mask=other_pil)
            fade  = Image.new("RGBA", other_pil.size, (0, 0, 0, 120))
            dimmed = Image.alpha_composite(dimmed, fade)

            rx = VIDEO_W - SIDE_W - 20
            ry = VIDEO_H - CHAR_H - 100 + i * (SIDE_H + 8)
            if ry + SIDE_H < VIDEO_H - 90:
                img.paste(dimmed, (rx, ry), dimmed)

    # ── Audio mixing ──────────────────────────────────────────────────────────

    def _mix_audio(self, bgm_file: str, segments: list, duration: float):
        from moviepy.editor import AudioFileClip, CompositeAudioClip
        clips = []
        if bgm_file and os.path.exists(bgm_file):
            clips.append(
                AudioFileClip(bgm_file).set_duration(duration).volumex(0.28)
            )
        for seg in segments:
            af = seg.get("audio_file", "")
            if af and os.path.exists(af):
                clips.append(AudioFileClip(af).set_start(seg["start_ms"] / 1000))
        return CompositeAudioClip(clips) if clips else None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_timeline(self, segments: list) -> list:
        return [
            {
                "segment_id": s["segment_id"],
                "start":      s["start_ms"] / 1000,
                "end":        s["end_ms"]   / 1000,
                "speaker":    s["speaker"],
                "text":       s["line"],
                "emotion":    s.get("emotion", "neutral"),
                "audio_file": s.get("audio_file", ""),
            }
            for s in segments
            if s.get("line") and s["line"].strip() not in ("", "...")
        ]

    def _load_bg(self, bg_path: str | None, scene: dict) -> np.ndarray:
        if bg_path and os.path.exists(bg_path):
            img = Image.open(bg_path).convert("RGB")
        else:
            tmp  = os.path.join(IMAGES_DIR, "_tmp_bg.png")
            path = self._gradient_bg(scene.get("mood", "dramatic"),
                                     scene.get("time_of_day", "DAY"), tmp)
            img  = Image.open(path).convert("RGB")
        return np.array(img.resize((VIDEO_W + 100, VIDEO_H + 100), Image.LANCZOS))

    def _resolve_char_images(self, char_visuals: list) -> dict:
        result = {}
        for c in char_visuals:
            name = c["name"]
            safe = name.lower().replace(" ", "_")
            path = os.path.join(IMAGES_DIR, f"{safe}.png")
            if os.path.exists(path):
                result[name] = path
        # also scan images dir directly so any portrait PNG is indexable
        if os.path.isdir(IMAGES_DIR):
            for fname in os.listdir(IMAGES_DIR):
                if not fname.endswith(".png") or fname.startswith("scene"):
                    continue
                full = os.path.join(IMAGES_DIR, fname)
                stem = fname[:-4]  # strip .png
                # store by stem so fuzzy lookup can hit it
                if stem not in result.values():
                    result[f"__stem__{stem}"] = full
        return result

    def _find_char_image(self, name: str, char_images: dict) -> str | None:
        """Return image path for name using exact then fuzzy matching."""
        if name in char_images:
            return char_images[name]

        def _norm(s: str) -> str:
            return s.lower().replace(" ", "").replace(".", "").replace("_", "")

        name_norm = _norm(name)
        # exact normalised match
        for key, path in char_images.items():
            if key.startswith("__stem__"):
                continue
            if _norm(key) == name_norm:
                return path
        # prefix match: "Detective James" matches "Detective Jameson"
        for key, path in char_images.items():
            if key.startswith("__stem__"):
                # stem is like "detective_jameson"
                stem_norm = _norm(key[8:])
                if stem_norm.startswith(name_norm) or name_norm.startswith(stem_norm):
                    return path
            else:
                kn = _norm(key)
                if kn.startswith(name_norm) or name_norm.startswith(kn):
                    return path
        # word overlap: any word > 4 chars matches
        name_words = {w for w in name.lower().split() if len(w) > 4}
        for key, path in char_images.items():
            key_clean = key[8:] if key.startswith("__stem__") else key
            key_words  = set(key_clean.lower().replace("_", " ").split())
            if name_words & key_words:
                return path
        return None


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
