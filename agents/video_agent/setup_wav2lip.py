"""
agents/video_agent/setup_wav2lip.py
─────────────────────────────────────
One-time setup script for Wav2Lip.

Run:  python agents/video_agent/setup_wav2lip.py

What it does:
  1. Clones Wav2Lip from GitHub into agents/video_agent/Wav2Lip/
  2. Downloads wav2lip_gan.pth checkpoint (~400 MB)
  3. Installs Wav2Lip + face_alignment requirements
"""

import os
import sys
import subprocess

THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
WAV2LIP_DIR = os.path.join(THIS_DIR, "Wav2Lip")
CKPT_DIR    = os.path.join(WAV2LIP_DIR, "checkpoints")
CKPT_PATH   = os.path.join(CKPT_DIR, "wav2lip_gan.pth")


def step(msg: str):
    print(f"\n[setup] {msg}")


def _download_checkpoint() -> bool:
    os.makedirs(CKPT_DIR, exist_ok=True)

    # ── attempt 1: HuggingFace Hub ─────────────────────────────────────────
    step("Trying HuggingFace Hub...")
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id="camenduru/wav2lip",
            filename="wav2lip_gan.pth",
            local_dir=CKPT_DIR,
            local_dir_use_symlinks=False,
        )
        if os.path.exists(CKPT_PATH) and os.path.getsize(CKPT_PATH) > 1_000_000:
            print(f"[setup] Saved to: {CKPT_PATH}")
            return True
    except Exception as e:
        print(f"[setup] HuggingFace attempt failed: {e}")

    # ── attempt 2: gdown (Google Drive) ────────────────────────────────────
    step("Trying gdown (Google Drive)...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"],
                       check=True)
        import gdown
        # Widely circulated community mirror of wav2lip_gan.pth
        gdown.download(
            id="1rbLJq5h6P-fzqLtN0rvN3FVTWJ4T5_WY",
            output=CKPT_PATH,
            quiet=False,
        )
        if os.path.exists(CKPT_PATH) and os.path.getsize(CKPT_PATH) > 1_000_000:
            return True
    except Exception as e:
        print(f"[setup] gdown attempt failed: {e}")

    # ── manual fallback ─────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("MANUAL DOWNLOAD REQUIRED")
    print("=" * 62)
    print("Automatic download failed. Please:")
    print("  1. Open: https://github.com/Rudrabha/Wav2Lip#getting-the-weights")
    print("  2. Download the 'Wav2Lip + GAN' model")
    print(f"  3. Place the file at:\n     {CKPT_PATH}")
    print("  4. Re-run this script.")
    print("=" * 62)
    return False


def main():
    # ── 1. Clone repo ──────────────────────────────────────────────────────
    if not os.path.exists(os.path.join(WAV2LIP_DIR, "inference.py")):
        step("Cloning Wav2Lip repository...")
        subprocess.run(
            ["git", "clone",
             "https://github.com/Rudrabha/Wav2Lip.git",
             WAV2LIP_DIR],
            check=True,
        )
    else:
        step("Wav2Lip repo already present.")

    # ── 2. Check / Download checkpoint ────────────────────────────────────
    existing = next(
        (os.path.join(CKPT_DIR, f) for f in os.listdir(CKPT_DIR)
         if f.endswith((".pth", ".pt")) and os.path.getsize(os.path.join(CKPT_DIR, f)) > 10_000_000),
        None
    ) if os.path.isdir(CKPT_DIR) else None

    if existing:
        step(f"Checkpoint found: {existing}  ({os.path.getsize(existing)/1024/1024:.0f} MB)")
    else:
        if not _download_checkpoint():
            print("\n[setup] Incomplete — checkpoint missing.")
            print("[setup] Phase 3 will fall back to animated portrait.")
            return

    # ── 3. Install requirements ────────────────────────────────────────────
    # Wav2Lip's requirements.txt pins numpy==1.17.1 which is incompatible with
    # Python 3.12. We already have compatible versions installed; just add the
    # packages that are genuinely missing.
    step("Installing Wav2Lip runtime dependencies (skipping pinned requirements.txt)...")
    packages = ["face_alignment", "torchvision", "tqdm", "scipy", "librosa"]
    for pkg in packages:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", pkg],
            capture_output=True,
        )
        status = "ok" if result.returncode == 0 else "FAILED"
        print(f"[setup]   {pkg}: {status}")

    print("\n[setup] Wav2Lip setup complete!")
    print("[setup] Now run: python agents/video_agent/run.py")


if __name__ == "__main__":
    main()
