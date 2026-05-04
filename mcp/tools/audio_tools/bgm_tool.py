"""
mcp/tools/audio_tools/bgm_tool.py
───────────────────────────────────
Ambient BGM generator using numpy additive synthesis.
Each mood is distinct in pitch, timbre (harmonic stack), tempo pulse, and LFO rate.
No sawtooth — pure sine/triangle to avoid aliasing clicks.
"""

import os
import wave

import numpy as np

SAMPLE_RATE = 22050

# Per-mood parameters:
#   base_freq   – fundamental note (Hz) — all > 150 so they're audible on laptop speakers
#   harmonics   – partial ratios stacked on the fundamental (defines "colour")
#   amplitude   – master gain (0..1)
#   lfo_hz      – slow volume swell rate (Hz)
#   pulse_bpm   – rhythmic beat (0 = no pulse, purely ambient)
#   use_triangle – True → warmer triangle wave; False → sine
_MOOD_PARAMS: dict[str, dict] = {
    "dramatic":    dict(base_freq=220, harmonics=[1.0, 2.0, 3.0, 4.0, 6.0],     amplitude=0.30, lfo_hz=0.10, pulse_bpm=56,  use_triangle=False),
    "tense":       dict(base_freq=185, harmonics=[1.0, 1.78, 2.67, 3.56, 4.45], amplitude=0.25, lfo_hz=0.28, pulse_bpm=88,  use_triangle=False),
    "mysterious":  dict(base_freq=165, harmonics=[1.0, 1.50, 3.00, 6.00],       amplitude=0.20, lfo_hz=0.04, pulse_bpm=0,   use_triangle=False),
    "hopeful":     dict(base_freq=330, harmonics=[1.0, 1.25, 1.50, 2.00, 2.50], amplitude=0.18, lfo_hz=0.18, pulse_bpm=96,  use_triangle=True),
    "melancholic": dict(base_freq=196, harmonics=[1.0, 1.50, 1.78, 2.67],       amplitude=0.20, lfo_hz=0.07, pulse_bpm=38,  use_triangle=True),
    "comedic":     dict(base_freq=392, harmonics=[1.0, 1.25, 1.50, 2.00, 2.50], amplitude=0.18, lfo_hz=0.35, pulse_bpm=120, use_triangle=True),
    "romantic":    dict(base_freq=261, harmonics=[1.0, 1.50, 2.00, 2.50, 3.00], amplitude=0.18, lfo_hz=0.06, pulse_bpm=48,  use_triangle=True),
    # Pentatonic-flavoured partials (1.0, 1.25, 1.5, 1.875) evoke harp/bell timbre
    "fantasy":     dict(base_freq=293, harmonics=[1.0, 1.25, 1.50, 1.875, 2.50, 3.00], amplitude=0.20, lfo_hz=0.08, pulse_bpm=60,  use_triangle=True),
}


class BGMEngine:
    def generate(self, mood: str, duration_seconds: int, output_path: str) -> str:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        p     = _MOOD_PARAMS.get(mood, _MOOD_PARAMS["dramatic"])
        audio = self._render(duration_seconds, **p)
        return self._save(audio, output_path)

    def _render(self, duration_seconds: int, base_freq: float, harmonics: list,
                amplitude: float, lfo_hz: float, pulse_bpm: int,
                use_triangle: bool) -> np.ndarray:
        n     = int(SAMPLE_RATE * duration_seconds)
        t     = np.linspace(0, duration_seconds, n, endpoint=False)
        audio = np.zeros(n, dtype=np.float32)

        for i, ratio in enumerate(harmonics):
            freq     = base_freq * ratio
            harm_amp = amplitude / (i + 1)
            # Each partial gets a slightly different LFO phase for natural movement
            lfo      = 0.6 + 0.4 * np.sin(2 * np.pi * lfo_hz * t + i * 1.2)
            if use_triangle:
                partial = (2.0 / np.pi) * np.arcsin(np.sin(2 * np.pi * freq * t))
            else:
                partial = np.sin(2 * np.pi * freq * t)
            audio += (harm_amp * lfo * partial).astype(np.float32)

        # Smooth rhythmic pulse — raised-cosine envelope so it never clicks
        if pulse_bpm > 0:
            pulse_hz    = pulse_bpm / 60.0
            # depth 0.30 → volume swings between 0.70 and 1.00
            pulse_env   = 1.0 - 0.30 * (0.5 + 0.5 * np.cos(2 * np.pi * pulse_hz * t))
            audio      *= pulse_env.astype(np.float32)

        # Fade in / out over 1.5 s to avoid start/end clicks
        fade = min(int(SAMPLE_RATE * 1.5), n // 4)
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
            seg = AudioSegment(audio_int.tobytes(), frame_rate=SAMPLE_RATE,
                               sample_width=2, channels=1)
            seg.export(output_path, format="mp3")
            return output_path
        except Exception:
            wav_path = output_path.replace(".mp3", ".wav")
            with wave.open(wav_path, "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio_int.tobytes())
            return wav_path
