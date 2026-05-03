"""
mcp/tools/audio_tools/bgm_tool.py
───────────────────────────────────
Ambient BGM generator using numpy harmonic synthesis.
No external API needed.
"""

import os
import wave

import numpy as np

SAMPLE_RATE = 22050

_MOOD_PARAMS: dict[str, tuple[float, list[float], float]] = {
    "dramatic":    (80,  [1.0, 1.5, 2.0, 3.0],       0.30),
    "tense":       (110, [1.0, 1.78, 2.37, 3.17],     0.25),
    "mysterious":  (60,  [1.0, 1.5, 2.5, 4.0],        0.20),
    "hopeful":     (440, [1.0, 1.25, 1.5, 2.0],       0.15),
    "melancholic": (220, [1.0, 1.5, 1.78, 2.67],      0.20),
    "comedic":     (330, [1.0, 1.25, 1.5, 2.0, 2.5],  0.15),
    "romantic":    (280, [1.0, 1.5, 2.0, 2.5],        0.18),
}


class BGMEngine:
    def generate(self, mood: str, duration_seconds: int, output_path: str) -> str:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        params = _MOOD_PARAMS.get(mood, _MOOD_PARAMS["dramatic"])
        audio  = self._render(*params, duration_seconds)
        return self._save(audio, output_path)

    def _render(self, base_freq, harmonics, amplitude, duration_seconds) -> np.ndarray:
        n     = int(SAMPLE_RATE * duration_seconds)
        t     = np.linspace(0, duration_seconds, n, endpoint=False)
        audio = np.zeros(n, dtype=np.float32)
        for i, ratio in enumerate(harmonics):
            freq     = base_freq * ratio
            harm_amp = amplitude / (i + 1)
            lfo      = 0.5 + 0.5 * np.sin(2 * np.pi * 0.08 * t + i)
            audio   += (harm_amp * lfo * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        fade = min(int(SAMPLE_RATE * 2), n // 4)
        audio[:fade]  *= np.linspace(0, 1, fade, dtype=np.float32)
        audio[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * amplitude
        return audio

    def _save(self, audio: np.ndarray, output_path: str) -> str:
        audio_int = (audio * 32767).astype(np.int16)
        try:
            from pydub import AudioSegment
            seg = AudioSegment(audio_int.tobytes(), frame_rate=SAMPLE_RATE, sample_width=2, channels=1)
            seg.export(output_path, format="mp3")
            return output_path
        except Exception:
            wav_path = output_path.replace(".mp3", ".wav")
            with wave.open(wav_path, "w") as wf:
                wf.setnchannels(1); wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE); wf.writeframes(audio_int.tobytes())
            return wav_path
