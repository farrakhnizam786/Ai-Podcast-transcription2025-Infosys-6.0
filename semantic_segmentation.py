import os
import json
import numpy as np
import nltk
import warnings
import math
import zipfile
from pathlib import Path
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

INPUT_DIR = r"D:\farrakh important\internship_project infosys\data\transcripts"
OUTPUT_DIR = r"D:\farrakh important\internship_project infosys\data\semantic_segments"

# --- TUNING SETTINGS (ADJUSTED FOR BETTER SEGMENTATION) ---
WINDOW_SIZE = 2          # Lowered to 2 to catch faster topic shifts
SIMILARITY_THRESHOLD = 0.65  # Raised to 0.65 to force more splits
SUMMARY_MODEL = "sshleifer/distilbart-cnn-12-6"

def setup_resources():
    resources = ['punkt', 'punkt_tab', 'stopwords']
    for r in resources:
        try:
            if 'punkt' in r:
                nltk.data.find(f'tokenizers/{r}')
            else:
                nltk.data.find(f'corpora/{r}')
        except (LookupError, zipfile.BadZipFile, OSError):
            print(f"Downloading missing or corrupted resource: {r}...")
            try:
                nltk.download(r, quiet=True, force=True)
            except Exception as e:
                print(f"Error downloading {r}: {e}")

def format_time(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

def extract_keywords(text, top_n=5):
    try:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=top_n)
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        
        scores = tfidf_matrix.toarray().flatten()
        sorted_indices = scores.argsort()[::-1]
        
        keywords = [feature_names[i] for i in sorted_indices[:top_n]]
        return ", ".join(keywords)
    except ValueError:
        return "None"

def segment_transcript_with_time(json_data):
    segments_raw = json_data['segments']
    
    if len(segments_raw) < WINDOW_SIZE * 2:
        return [{
            "start": segments_raw[0]['start'],
            "end": segments_raw[-1]['end'],
            "text": json_data['text']
        }]

    texts = [s['text'] for s in segments_raw]

    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return []

    similarities = []
    for i in range(len(texts) - WINDOW_SIZE):
        vec1 = tfidf_matrix[i : i + WINDOW_SIZE].mean(axis=0)
        vec2 = tfidf_matrix[i + 1 : i + 1 + WINDOW_SIZE].mean(axis=0)
        sim = cosine_similarity(np.asarray(vec1), np.asarray(vec2))[0][0]
        similarities.append(sim)

    cut_indices = [0]
    
    for i, sim in enumerate(similarities):
        if sim < SIMILARITY_THRESHOLD:
            is_dip = True
            if i > 0 and similarities[i-1] < sim: is_dip = False
            if i < len(similarities)-1 and similarities[i+1] < sim: is_dip = False
            
            if is_dip:
                cut_point = i + int(WINDOW_SIZE / 2)
                cut_indices.append(cut_point)

    cut_indices.append(len(segments_raw))

    final_topics = []
    
    for i in range(len(cut_indices) - 1):
        start_idx = cut_indices[i]
        end_idx = cut_indices[i+1]
        
        chunk = segments_raw[start_idx:end_idx]
        if not chunk: continue

        topic_text = " ".join([s['text'].strip() for s in chunk])
        start_time = chunk[0]['start']
        end_time = chunk[-1]['end']
        
        final_topics.append({
            "start": start_time,
            "end": end_time,
            "text": topic_text
        })
        
    return final_topics

def generate_summary(text, summarizer):
    try:
        clean_text = text[:3000]
        # Estimate word count to adjust summary length dynamically
        word_count = len(clean_text.split())
        
        # If very short, return original text instead of trying to summarize
        if word_count < 30:
            return clean_text

        # Dynamic max_length to avoid "input shorter than max_length" warnings
        # We target a summary that is shorter than the input (e.g., 80% of input length), capped at 120
        dynamic_max = int(min(120, word_count * 0.8))
        # Ensure min_length is smaller than max_length
        dynamic_min = int(min(40, dynamic_max * 0.5))
        
        # Safety bounds to prevent model errors with 0 length
        dynamic_max = max(10, dynamic_max)
        dynamic_min = max(5, dynamic_min)

        result = summarizer(clean_text, max_length=dynamic_max, min_length=dynamic_min, do_sample=False)
        return result[0]['summary_text']
    except Exception as e:
        return f"Summary unavailable."

def process_semantic_segmentation():
    setup_resources()
    
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"🧠 Loading Summarization Model ({SUMMARY_MODEL})...")
    try:
        summarizer = pipeline("summarization", model=SUMMARY_MODEL)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    files = list(input_path.glob("*.json"))
    files.sort()

    print(f"📂 Found {len(files)} transcripts to analyze.")

    for file_path in tqdm(files, desc="Processing Topics"):
        output_file = output_path / f"{file_path.stem}_topics.txt"
        
        # Always remove old file to force regeneration with new settings
        if output_file.exists():
             os.remove(output_file) 

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)

            topic_segments = segment_transcript_with_time(json_data)

            final_report = []
            final_report.append(f"=== TOPIC ANALYSIS: {file_path.stem} ===\n")
            final_report.append(f"Detected {len(topic_segments)} major topics.\n")
            
            for i, topic in enumerate(topic_segments):
                summary = generate_summary(topic['text'], summarizer)
                
                keywords = extract_keywords(topic['text'])
                
                t_start = format_time(topic['start'])
                t_end = format_time(topic['end'])
                
                # Generate a short headline from the summary
                headline = summary.split('.')[0]
                if len(headline) > 60:
                    headline = headline[:60] + "..."
                
                final_report.append(f"🔹 TOPIC {i+1} [{t_start} - {t_end}]: {headline}")
                final_report.append(f"   SUMMARY: {summary}")
                final_report.append(f"   KEYWORDS: {keywords}")
                final_report.append("-" * 40 + "\n")

            with open(output_file, "w", encoding="utf-8") as f:
                f.write("\n".join(final_report))

        except Exception as e:
            print(f"\n❌ Error processing {file_path.name}: {e}")

    print(f"\n✅ Segmentation Complete. Output in: {OUTPUT_DIR}")

if __name__ == "__main__":
    process_semantic_segmentation()