"""
agents/video_agent/lip_sync.py
────────────────────────────────
Wav2Lip wrapper.

generate(face_image_path, audio_mp3_path, output_mp4_path)
  -> True on success, False if Wav2Lip is not set up or fails.

load_frames(video_path)
  -> list of RGB numpy arrays (all frames preloaded into memory)
"""

import os
import sys
import subprocess

THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
WAV2LIP_DIR = os.path.join(THIS_DIR, "Wav2Lip")
CKPT_DIR    = os.path.join(WAV2LIP_DIR, "checkpoints")


def _crop_face_region(image_path: str) -> str:
    """Return a version of the portrait cropped to the upper 65% (face area).
    Saves alongside the original as *_facecrop.png; returns original on error."""
    try:
        from PIL import Image
        img  = Image.open(image_path)
        w, h = img.size
        cropped = img.crop((0, 0, w, int(h * 0.65)))
        out = image_path.replace(".png", "_facecrop.png")
        cropped.save(out)
        return out
    except Exception:
        return image_path


def _find_checkpoint() -> str | None:
    """Return the first .pth / .pt file in checkpoints/ that is >10 MB."""
    if not os.path.isdir(CKPT_DIR):
        return None
    for fname in os.listdir(CKPT_DIR):
        if fname.endswith((".pth", ".pt")):
            full = os.path.join(CKPT_DIR, fname)
            if os.path.getsize(full) > 10_000_000:
                return full
    return None


def is_available() -> bool:
    return (
        os.path.exists(os.path.join(WAV2LIP_DIR, "inference.py"))
        and _find_checkpoint() is not None
    )


CHECKPOINT = _find_checkpoint() or os.path.join(CKPT_DIR, "wav2lip_gan.pth")


def generate(face_image_path: str, audio_mp3_path: str,
             output_mp4_path: str) -> bool:
    """Run Wav2Lip on face image + audio -> lip-synced MP4."""
    if not is_available():
        return False

    os.makedirs(os.path.dirname(output_mp4_path) or ".", exist_ok=True)

    # Convert MP3 → 16kHz mono WAV (Wav2Lip requirement)
    audio_wav = audio_mp3_path.replace(".mp3", "_liptmp.wav")
    try:
        from pydub import AudioSegment
        (AudioSegment.from_mp3(audio_mp3_path)
         .set_frame_rate(16000)
         .set_channels(1)
         .export(audio_wav, format="wav"))
    except Exception as e:
        print(f"[LipSync] Audio conversion failed: {e}")
        return False

    checkpoint = _find_checkpoint()
    if not checkpoint:
        return False

    # Crop portrait to upper 65% so the face fills more of the frame,
    # improving S3FD detection accuracy on full-body AI portraits.
    face_image_path = _crop_face_region(face_image_path)

    # Wav2Lip runs from its own directory — all paths must be absolute
    face_image_path = os.path.abspath(face_image_path)
    audio_wav       = os.path.abspath(audio_wav)
    output_mp4_path = os.path.abspath(output_mp4_path)
    checkpoint      = os.path.abspath(checkpoint)

    try:
        result = subprocess.run(
            [
                sys.executable, "inference.py",
                "--checkpoint_path", checkpoint,
                "--face",            face_image_path,
                "--audio",           audio_wav,
                "--outfile",         output_mp4_path,
                "--static",          "True",
                "--fps",             "24",
                "--resize_factor",   "1",
                "--pads",            "0", "15", "0", "0",
            ],
            cwd=WAV2LIP_DIR,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            print(f"[LipSync] Wav2Lip error:\n{result.stderr[-500:]}")
            return False
        return os.path.exists(output_mp4_path)
    except subprocess.TimeoutExpired:
        print("[LipSync] Wav2Lip timed out (>5 min per segment)")
        return False
    except Exception as e:
        print(f"[LipSync] Exception: {e}")
        return False
    finally:
        if os.path.exists(audio_wav):
            os.remove(audio_wav)


def load_frames(video_path: str) -> list:
    """Load every frame of a video as an RGB numpy array."""
    try:
        import cv2
        cap    = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame[:, :, ::-1].copy())   # BGR → RGB
        cap.release()
        return frames
    except Exception as e:
        print(f"[LipSync] Frame load failed: {e}")
        return []
