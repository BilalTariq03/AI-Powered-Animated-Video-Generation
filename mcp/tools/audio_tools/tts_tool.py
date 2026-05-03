"""
mcp/tools/audio_tools/tts_tool.py
───────────────────────────────────
TTS engine — gTTS (free default) + ElevenLabs (optional cloud).
Used by agents/audio_agent/agent.py.
"""

import io
import os

_GTTS_TLD = {
    "deep":         "co.uk",
    "raspy":        "co.uk",
    "authoritative":"com",
    "neutral":      "com",
    "high-pitched": "com.au",
    "soft":         "com.au",
    "whispery":     "com.au",
}

_EL_VOICE_IDS = {
    "deep":          "ErXwobaYiN019PkySvjV",
    "authoritative": "VR6AewLTigWG4xSOukaG",
    "soft":          "EXAVITQu4vr4xnSDxMaL",
    "neutral":       "21m00Tcm4TlvDq8ikWAM",
    "high-pitched":  "AZnzlk1XvdvUeBnXmlld",
    "raspy":         "CYw3kZ02Hs0563khs1Fj",
    "whispery":      "EXAVITQu4vr4xnSDxMaL",
}


class TTSEngine:
    def __init__(self, elevenlabs_api_key: str = ""):
        self._el_key          = elevenlabs_api_key
        self._use_elevenlabs  = bool(elevenlabs_api_key)

    def synthesize(self, text: str, voice_config: dict, output_path: str) -> int:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if not text.strip() or text.strip() == "...":
            return self._write_silence(output_path, 800)

        if self._use_elevenlabs:
            duration = self._elevenlabs(text, voice_config, output_path)
            if duration:
                return duration
            print("[TTSEngine] ElevenLabs failed - falling back to gTTS.")

        return self._gtts(text, voice_config, output_path)

    def _gtts(self, text: str, voice_config: dict, output_path: str) -> int:
        from gtts import gTTS
        style = voice_config.get("voice_style", "neutral")
        slow  = voice_config.get("speaking_speed") == "slow"
        tld   = _GTTS_TLD.get(style, "com")
        tts   = gTTS(text=text, lang="en", tld=tld, slow=slow)
        buf   = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        with open(output_path, "wb") as f:
            f.write(buf.read())
        return self._duration_ms(output_path)

    def _elevenlabs(self, text: str, voice_config: dict, output_path: str) -> int | None:
        import requests
        style    = voice_config.get("voice_style", "neutral")
        voice_id = _EL_VOICE_IDS.get(style, "21m00Tcm4TlvDq8ikWAM")
        speed    = {"slow": 0.8, "normal": 1.0, "fast": 1.2}.get(
                       voice_config.get("speaking_speed", "normal"), 1.0)
        url      = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers  = {"Accept": "audio/mpeg", "Content-Type": "application/json",
                    "xi-api-key": self._el_key}
        payload  = {"text": text, "model_id": "eleven_monolingual_v1",
                    "voice_settings": {"stability": 0.5, "similarity_boost": 0.75, "speed": speed}}
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=30)
            if resp.status_code != 200:
                return None
            with open(output_path, "wb") as f:
                f.write(resp.content)
            return self._duration_ms(output_path)
        except Exception as e:
            print(f"[TTSEngine] ElevenLabs exception: {e}")
            return None

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
