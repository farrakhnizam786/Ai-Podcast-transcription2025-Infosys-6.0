import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

BASE_DIR = os.getenv("BASE_DIR")
if BASE_DIR is None:
    BASE_DIR = ROOT_DIR  # fallback if .env fails
BASE_DIR = Path(BASE_DIR)

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

def transcribe_audio(audio_path, language=None):
    audio_path = Path(audio_path)
    # Example: dummy transcription for now
    # Replace with actual Whisper call
    transcript_text = f"Transcribed text of {audio_path.name}"
    return transcript_text
