import os
import random
from pathlib import Path
from dotenv import load_dotenv

import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")

PROJECT_ROOT = os.getenv("PROJECT_ROOT")
RAW_AUDIO_DIR = os.getenv("RAW_AUDIO_DIR")
PROCESSED_AUDIO_DIR = os.getenv("PROCESSED_AUDIO_DIR")
TARGET_SAMPLE_RATE = int(os.getenv("TARGET_SAMPLE_RATE", 16000))

if not PROJECT_ROOT:
    raise RuntimeError("PROJECT_ROOT not set in .env")

PROJECT_ROOT = Path(PROJECT_ROOT)
RAW_AUDIO_PATH = PROJECT_ROOT / RAW_AUDIO_DIR
PROCESSED_AUDIO_PATH = PROJECT_ROOT / PROCESSED_AUDIO_DIR


RAW_AUDIO_PATH.mkdir(parents=True, exist_ok=True)
PROCESSED_AUDIO_PATH.mkdir(parents=True, exist_ok=True)

def process_audio_file(input_path: Path, output_path: Path):
    audio, sr = librosa.load(input_path, sr=TARGET_SAMPLE_RATE, mono=True)

    # Volume normalization
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak

    sf.write(output_path, audio, TARGET_SAMPLE_RATE, subtype="PCM_16")



processed_count = 0
skipped_count = 0
processed_files = []

audio_files = list(RAW_AUDIO_PATH.glob("*.*"))

print(f"\nFound {len(audio_files)} audio files\n")

for audio_file in tqdm(audio_files, desc="Processing Audio"):
    output_file = PROCESSED_AUDIO_PATH / f"{audio_file.stem}.wav"

    # 🔁 SKIP LOGIC
    if output_file.exists():
        skipped_count += 1
        continue

    try:
        process_audio_file(audio_file, output_file)
        processed_files.append((audio_file, output_file))
        processed_count += 1
    except Exception as e:
        print(f"❌ Error processing {audio_file.name}: {e}")

print("\n✅ PREPROCESSING COMPLETE")
print(f"Processed files : {processed_count}")
print(f"Skipped files   : {skipped_count}")

if processed_files:
    original, processed = random.choice(processed_files)

    orig_audio, orig_sr = librosa.load(original, sr=None, mono=False)
    proc_audio, proc_sr = librosa.load(processed, sr=None, mono=True)

    print("\n🔍 VERIFICATION CHECK (Random File)")
    print(f"File name       : {original.name}")
    print(f"Original SR     : {orig_sr}")
    print(f"Processed SR    : {proc_sr}")
    print(f"Original Ch     : {'Mono' if orig_audio.ndim == 1 else 'Stereo'}")
    print(f"Processed Ch    : Mono")
    print(f"Original Size   : {original.stat().st_size / 1024:.2f} KB")
    print(f"Processed Size  : {processed.stat().st_size / 1024:.2f} KB")
else:
    print("\n⚠ No new files processed, verification skipped.")
