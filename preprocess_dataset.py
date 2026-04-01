import os
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()

# Use environment variables for paths, with fallbacks
# Preprocessing takes raw audio (AUDIO_DIR) and saves to processed (PROCESSED_DIR)
INPUT_DIR = os.getenv("AUDIO_DIR", r"D:\farrakh important\internship_project infosys\data\audio")
OUTPUT_DIR = os.getenv("PROCESSED_DIR", r"D:\farrakh important\internship_project infosys\data\processed")

# Settings for Whisper AI
TARGET_SR = 16000  # Whisper expects 16kHz audio
# ---------------------

def verify_conversion(input_files, output_dir):
    """Picks one file and prints the Before vs After stats"""
    if not input_files:
        return

    # Pick a random file to test
    test_file = random.choice(input_files)
    processed_file = Path(output_dir) / (test_file.stem + ".wav")

    if not processed_file.exists():
        return

    print("\n🔎 --- VERIFICATION: Spot Check ---")
    
    try:
        # Load Original
        y_orig, sr_orig = librosa.load(str(test_file), sr=None, mono=False)
        channels_orig = "Mono" if len(y_orig.shape) == 1 else "Stereo"
        size_orig = os.path.getsize(test_file) / 1024

        # Load Processed
        y_proc, sr_proc = librosa.load(str(processed_file), sr=None, mono=False)
        channels_proc = "Mono" if len(y_proc.shape) == 1 else "Stereo"
        size_proc = os.path.getsize(processed_file) / 1024

        print(f"File: {test_file.name}")
        print(f"{'Property':<15} | {'Original (MP3)':<20} | {'Processed (WAV)':<20}")
        print("-" * 65)
        print(f"{'Sample Rate':<15} | {sr_orig} Hz {'(Too High)' if sr_orig > 16000 else '':<10} | {sr_proc} Hz (Perfect for AI)")
        print(f"{'Channels':<15} | {channels_orig:<20} | {channels_proc:<20}")
        print(f"{'File Size':<15} | {size_orig:.1f} KB {'(Compressed)':<12} | {size_proc:.1f} KB (Uncompressed)")
        print("-" * 65)
        print("✅ The AI will now process this file much faster.")
    except Exception as e:
        print(f"⚠️ Verification failed: {e}")

def preprocess_all():
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all audio files (MP3, WAV, FLAC)
    audio_files = list(input_path.glob("*.mp3")) + \
                  list(input_path.glob("*.wav")) + \
                  list(input_path.glob("*.flac"))
    
    if not audio_files:
        print(f"❌ No audio files found in {INPUT_DIR}")
        return

    print(f"🔍 Found {len(audio_files)} files. Checking for new files...")
    print(f"🎯 Target: 16kHz, Mono, Normalized WAV")

    success_count = 0

    for audio_file in tqdm(audio_files):
        try:
            # Construct the expected output path first
            output_filename = audio_file.stem + ".wav"
            save_path = output_path / output_filename


            y, sr = librosa.load(str(audio_file), sr=TARGET_SR, mono=True)

    
            if np.max(np.abs(y)) > 0:
                y = y / np.max(np.abs(y))


            sf.write(save_path, y, sr)
            success_count += 1

        except Exception as e:
            print(f"\n Error processing {audio_file.name}: {e}")

    print(f"\n Preprocessing Complete!")
    
    print(f" Successfully processed {success_count} files.")
    print(f" Clean data saved to: {OUTPUT_DIR}")
    

    if success_count > 0:
        verify_conversion(audio_files, OUTPUT_DIR)

if __name__ == "__main__":
    preprocess_all()