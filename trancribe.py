import os
import whisper
import warnings
import json
import torch
import time
import shutil
import sys
from pathlib import Path
from tqdm import tqdm

# Configuration
INPUT_DIR = r"D:\farrakh important\internship_project infosys\data\processed_audio"
OUTPUT_DIR = r"D:\farrakh important\internship_project infosys\data\transcripts"
MODEL_SIZE = "base"

def check_ffmpeg():
    """Checks if FFmpeg is installed and accessible."""
    if not shutil.which("ffmpeg"):
        print("\nCRITICAL ERROR: FFmpeg is not found in your system PATH.")
        print("Whisper requires FFmpeg to run.")
        sys.exit(1)
    print("FFmpeg detected.")

def save_transcript(result, file_stem, output_dir):
    """Saves readable text and structured JSON with timestamps."""
    txt_path = output_dir / f"{file_stem}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(result["text"].strip())

    json_path = output_dir / f"{file_stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

def generate_summary(text, num_sentences=3):
    """Generates a simple extractive summary."""
    sentences = text.split('.')
    summary = ". ".join(sentences[:num_sentences]) + "."
    return summary

def transcribe_all():
    check_ffmpeg()

    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    # Hardware Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Hardware detected: {device.upper()}")
    
    print(f"Loading Whisper model ('{MODEL_SIZE}')...")
    try:
        model = whisper.load_model(MODEL_SIZE, device=device)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    audio_files = list(input_path.glob("*.wav"))
    if not audio_files:
        print(f"No .wav files found in {INPUT_DIR}")
        return

    print(f"Found {len(audio_files)} files.")

    success_count = 0
    skipped_count = 0

    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}] Processing: {audio_file.name}")
        
        json_file = output_path / f"{audio_file.stem}.json"
        transcript_text = ""

        try:
            # Skip existing transcripts
            if json_file.exists():
                print(f"   Skip: Found existing transcript.")
                skipped_count += 1
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    transcript_text = data["text"]
            else:
                print("   Transcribing...")
                start_time = time.time()
                result = model.transcribe(str(audio_file), fp16=False)
                save_transcript(result, audio_file.stem, output_path)
                transcript_text = result["text"]
                
                duration = time.time() - start_time
                print(f"   Done in {duration:.2f}s.")
                success_count += 1

            # Preview and Summary
            print("\n   Transcript Preview:")
            preview = transcript_text[:200].replace('\n', ' ')
            print(f"   \"{preview}...\"")

            while True:
                user_input = input(f"   Generate short summary for '{audio_file.name}'? (y/n/q=quit): ").lower().strip()
                
                if user_input == 'y':
                    summary = generate_summary(transcript_text)
                    print(f"\n   SUMMARY:\n   {summary}\n")
                    break
                elif user_input == 'n':
                    break
                elif user_input == 'q':
                    return
                else:
                    print("   Please type 'y' for yes or 'n' for no.")

        except Exception as e:
            print(f"\nError handling {audio_file.name}: {e}")

    print("\nProcessing Finished.")
    print(f"New: {success_count} | Existing: {skipped_count}")

if __name__ == "__main__":
    transcribe_all()