"""
mcp/tools/audio_tools/tts_tool.py
───────────────────────────────────
TTS engine — tier order:
  1. ElevenLabs  — paid cloud, highest quality  (ELEVENLABS_API_KEY + paid plan)
  2. Kokoro TTS  — local neural, very natural, free, no API key  ← default
  3. Edge TTS    — Microsoft cloud neural, free fallback
  4. gTTS        — Google basic, last resort
"""

import io
import os
import tempfile

# ── Kokoro voice mapping ───────────────────────────────────────────────────────
# voice_style → (kokoro_voice_name, lang_code, speed_multiplier)
_KOKORO_VOICES: dict[str, tuple[str, str]] = {
    "neutral":       ("am_echo",    "a"),   # clear American male
    "deep":          ("am_onyx",    "a"),   # deep American male
    "authoritative": ("am_eric",    "a"),   # firm American male
    "soft":          ("af_sarah",   "a"),   # soft American female
    "high-pitched":  ("af_nova",    "a"),   # bright American female
    "raspy":         ("bm_george",  "b"),   # British male, distinct timbre
    "whispery":      ("af_heart",   "a"),   # warm, breathy American female
}

_KOKORO_SPEED = {"slow": 0.8, "normal": 1.0, "fast": 1.2}

# ── ElevenLabs voice IDs (paid plan only) ─────────────────────────────────────
_EL_VOICE_IDS = {
    "deep":          "ErXwobaYiN019PkySvjV",
    "authoritative": "VR6AewLTigWG4xSOukaG",
    "soft":          "EXAVITQu4vr4xnSDxMaL",
    "neutral":       "21m00Tcm4TlvDq8ikWAM",
    "high-pitched":  "AZnzlk1XvdvUeBnXmlld",
    "raspy":         "CYw3kZ02Hs0563khs1Fj",
    "whispery":      "EXAVITQu4vr4xnSDxMaL",
}

# ── Edge TTS voice mapping (cloud fallback) ────────────────────────────────────
_EDGE_VOICES: dict[str, tuple[str, str, str]] = {
    "neutral":       ("en-US-GuyNeural",     "+0%",  "+0Hz"),
    "deep":          ("en-US-SteffanNeural", "-5%",  "-5Hz"),
    "authoritative": ("en-US-EricNeural",    "+0%",  "+0Hz"),
    "soft":          ("en-US-JennyNeural",   "-5%",  "+2Hz"),
    "high-pitched":  ("en-AU-NatashaNeural", "+5%",  "+8Hz"),
    "raspy":         ("en-GB-RyanNeural",    "-5%",  "-3Hz"),
    "whispery":      ("en-US-AriaNeural",    "-10%", "+0Hz"),
}

# Cache KPipeline instances (model load is expensive — do it once per process)
_kokoro_pipelines: dict[str, object] = {}


def _get_kokoro_pipeline(lang_code: str):
    if lang_code not in _kokoro_pipelines:
        from kokoro import KPipeline
        _kokoro_pipelines[lang_code] = KPipeline(lang_code=lang_code)
    return _kokoro_pipelines[lang_code]


class TTSEngine:
    def __init__(self, elevenlabs_api_key: str = ""):
        self._el_key         = elevenlabs_api_key
        self._use_elevenlabs = bool(elevenlabs_api_key)

    def synthesize(self, text: str, voice_config: dict, output_path: str) -> int:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        if not text.strip() or text.strip() == "...":
            return self._write_silence(output_path, 800)

        if self._use_elevenlabs:
            duration = self._elevenlabs(text, voice_config, output_path)
            if duration:
                return duration
            print("[TTSEngine] ElevenLabs failed - falling back to Kokoro.")

        duration = self._kokoro(text, voice_config, output_path)
        if duration:
            return duration

        print("[TTSEngine] Kokoro failed - falling back to Edge TTS.")
        duration = self._edge_tts(text, voice_config, output_path)
        if duration:
            return duration

        print("[TTSEngine] Edge TTS failed - falling back to gTTS.")
        return self._gtts(text, voice_config, output_path)

    # ── Kokoro (local neural, free) ───────────────────────────────────────────

    def _kokoro(self, text: str, voice_config: dict, output_path: str) -> int | None:
        try:
            import numpy as np

            style     = voice_config.get("voice_style", "neutral")
            speed_key = voice_config.get("speaking_speed", "normal")
            voice_name, lang_code = _KOKORO_VOICES.get(style, _KOKORO_VOICES["neutral"])
            speed     = _KOKORO_SPEED.get(speed_key, 1.0)

            pipeline    = _get_kokoro_pipeline(lang_code)
            audio_parts = []
            for _, _, audio in pipeline(text, voice=voice_name, speed=speed):
                audio_parts.append(audio)

            if not audio_parts:
                return None

            combined  = np.concatenate(audio_parts)
            int16     = (combined * 32767).astype(np.int16)

            from pydub import AudioSegment
            seg = AudioSegment(int16.tobytes(), frame_rate=24000, sample_width=2, channels=1)
            seg.export(output_path, format="mp3")
            return len(seg)
        except Exception as e:
            print(f"[TTSEngine] Kokoro exception: {e}")
            return None

    # ── ElevenLabs (paid cloud) ───────────────────────────────────────────────

    def _elevenlabs(self, text: str, voice_config: dict, output_path: str) -> int | None:
        import requests
        style    = voice_config.get("voice_style", "neutral")
        voice_id = _EL_VOICE_IDS.get(style, "21m00Tcm4TlvDq8ikWAM")
        url      = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers  = {"Accept": "audio/mpeg", "Content-Type": "application/json",
                    "xi-api-key": self._el_key}
        payload  = {"text": text, "model_id": "eleven_multilingual_v2",
                    "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}}
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=30)
            if resp.status_code != 200:
                print(f"[TTSEngine] ElevenLabs HTTP {resp.status_code}: {resp.text[:150]}")
                return None
            with open(output_path, "wb") as f:
                f.write(resp.content)
            return self._duration_ms(output_path)
        except Exception as e:
            print(f"[TTSEngine] ElevenLabs exception: {e}")
            return None

    # ── Edge TTS (cloud neural fallback) ─────────────────────────────────────

    def _edge_tts(self, text: str, voice_config: dict, output_path: str) -> int | None:
        try:
            import asyncio, shutil
            import edge_tts
            style      = voice_config.get("voice_style", "neutral")
            speed_key  = voice_config.get("speaking_speed", "normal")
            voice_name, base_rate, pitch = _EDGE_VOICES.get(style, _EDGE_VOICES["neutral"])
            rate       = {
                "slow": "-15%", "fast": "+15%"
            }.get(speed_key, base_rate)

            communicate = edge_tts.Communicate(text, voice_name, rate=rate, pitch=pitch)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name
            asyncio.run(communicate.save(tmp_path))
            shutil.move(tmp_path, output_path)
            return self._duration_ms(output_path)
        except Exception as e:
            print(f"[TTSEngine] Edge TTS exception: {e}")
            return None

    # ── gTTS (last resort) ────────────────────────────────────────────────────

    def _gtts(self, text: str, voice_config: dict, output_path: str) -> int:
        from gtts import gTTS
        slow = voice_config.get("speaking_speed") == "slow"
        tts  = gTTS(text=text, lang="en", slow=slow)
        buf  = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        with open(output_path, "wb") as f:
            f.write(buf.read())
        return self._duration_ms(output_path)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _duration_ms(self, path: str) -> int:
        try:
            from pydub import AudioSegment
            return len(AudioSegment.from_file(path))
        except Exception:
            return 2000

    def _write_silence(self, path: str, duration_ms: int) -> int:
        try:
            from pydub import AudioSegment
            AudioSegment.silent(duration=duration_ms).export(path, format="mp3")
        except Exception:
            open(path, "wb").close()
        return duration_ms
