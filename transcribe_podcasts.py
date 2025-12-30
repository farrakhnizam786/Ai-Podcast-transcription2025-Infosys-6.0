import os
import whisper
import warnings
import json
import torch
import time
import shutil
import sys
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

INPUT_DIR = r"D:\farrakh important\internship_project infosys\data\processed"
OUTPUT_DIR = r"D:\farrakh important\internship_project infosys\data\transcripts"
SUMMARY_DIR = r"D:\farrakh important\internship_project infosys\data\summaries"
MODEL_SIZE = "base"

def check_ffmpeg():
    if not shutil.which("ffmpeg"):
        print("\nCRITICAL ERROR: FFmpeg is not found in your system PATH.")
        print("Whisper requires FFmpeg to run.")
        sys.exit(1)
    print("FFmpeg detected.")

def save_transcript(result, file_stem, output_dir):
    txt_path = output_dir / f"{file_stem}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(result["text"].strip())

    json_path = output_dir / f"{file_stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

def generate_summary(text):
    try:
        from transformers import pipeline
        import logging
        
        logging.getLogger("transformers").setLevel(logging.ERROR)
        print("   Generating smart summary (DistilBART)...")
        
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        chunk = text[:3500]
        summary_result = summarizer(chunk, max_length=200, min_length=60, do_sample=False)
        
        return summary_result[0]['summary_text']

    except ImportError:
        print("   Warning: 'transformers' library not found. Using fallback summary.")
        all_sentences = [s.strip() for s in text.split('.') if s.strip()]
        selected_sentences = all_sentences[:25]
        return ". ".join(selected_sentences) + "."
        
    except Exception as e:
        print(f"   Warning: Smart summarization failed: {e}")
        return text[:1000] + "..."

def save_summary(summary, file_stem):
    summary_path = Path(SUMMARY_DIR)
    summary_path.mkdir(parents=True, exist_ok=True)
    
    file_path = summary_path / f"{file_stem}_summary.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(summary)
    
    print(f"   Summary saved to: {file_path}")

def interactive_summary_mode(output_dir):
    print("\n--- Interactive Summary Mode ---")
    
    while True:
        choice = input("\nDo you want to get a summary for a file? (y/n): ").lower().strip()
        
        if choice == 'n':
            break
        elif choice == 'y':
            file_name = input("   Enter the file name (e.g., '1001'): ").strip()
            file_stem = Path(file_name).stem
            json_path = output_dir / f"{file_stem}.json"
            
            if json_path.exists():
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        text = data["text"]
                        summary = generate_summary(text)
                        save_summary(summary, file_stem)
                except Exception as e:
                    print(f"   Error reading file: {e}")
            else:
                print(f"   Transcript not found for '{file_stem}'.")
        else:
            print("   Please type 'y' or 'n'.")

def transcribe_all():
    check_ffmpeg()

    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Hardware detected: {device.upper()}")
    
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

    success_count = 0

    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}] Processing: {audio_file.name}")
        
        try:
            audio_info = sf.info(str(audio_file))
            total_duration_sec = audio_info.duration
        except Exception:
            total_duration_sec = 0

        try:
            print("   Transcribing...")
            start_time = time.time()
            result = model.transcribe(str(audio_file), fp16=False)
            save_transcript(result, audio_file.stem, output_path)
            
            duration = time.time() - start_time
            print(f"   Done in {duration:.2f}s.")
            success_count += 1
            last_timestamp = result["segments"][-1]["end"] if "segments" in result and result["segments"] else 0

            if total_duration_sec > 0:
                coverage_pct = min((last_timestamp / total_duration_sec) * 100, 100.0)
                print(f"   Quality Check: {coverage_pct:.1f}% coverage.")
                if coverage_pct < 95:
                    print("      Warning: Possible incomplete transcription at end.")

        except Exception as e:
            print(f"\nError handling {audio_file.name}: {e}")

    print("\nProcessing Finished.")
    print(f"Processed: {success_count} files")
    interactive_summary_mode(output_path)

if __name__ == "__main__":
    transcribe_all()