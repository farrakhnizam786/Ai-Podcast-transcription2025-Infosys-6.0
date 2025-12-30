import os
import json
import shutil
import warnings
import numpy as np
import librosa
import soundfile as sf
import torch
import nltk
import whisper
import requests
import zipfile
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from nltk.sentiment import SentimentIntensityAnalyzer

# Suppress warnings
warnings.filterwarnings("ignore")

def setup_directories(base_dir):
    # Convert to Path object for safety
    base_path = Path(base_dir)
    
    dirs = {
        "audio": base_path / "audio",
        "processed": base_path / "processed_audio",
        "transcripts": base_path / "transcripts",
        "summary": base_path / "short_summary",
        "topics": base_path / "semantic_segments",
        "keywords": base_path / "keywords",
        "sentiment": base_path / "sentiment_data"
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs

def setup_nltk():
    resources = ['punkt', 'punkt_tab', 'vader_lexicon', 'stopwords']
    for r in resources:
        try:
            if 'punkt' in r:
                nltk.data.find(f'tokenizers/{r}')
            elif 'vader' in r:
                nltk.data.find(f'sentiment/{r}')
            else:
                nltk.data.find(f'corpora/{r}')
        except (LookupError, zipfile.BadZipFile, OSError):
            print(f"Downloading missing or corrupted resource: {r}...")
            try:
                nltk.download(r, quiet=True, force=True)
            except Exception as e:
                print(f"Error downloading {r}: {e}")

# --- HELPER FUNCTIONS ---
def format_time(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

def extract_keywords_text(text, top_n=5):
    """Extract keywords from a single text block (used for topics)."""
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

# --- 1. PREPROCESSING ---
def preprocess_audio(input_path, output_path):
    target_sr = 16000
    try:
        # Load with librosa (handles mp3, m4a, wav)
        y, sr = librosa.load(str(input_path), sr=target_sr, mono=True)
        # Normalize volume
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
        sf.write(str(output_path), y, sr)
        return True
    except Exception as e:
        print(f"Error preprocessing: {e}")
        return False

# --- 2. TRANSCRIPTION & SUMMARY ---
def transcribe_and_summarize(audio_path, transcript_path, summary_path, model_size="base"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Transcribe
    model = whisper.load_model(model_size, device=device)
    result = model.transcribe(str(audio_path), fp16=False)
    
    # Save Transcript (TXT)
    with open(str(transcript_path).replace('.json', '.txt'), "w", encoding="utf-8") as f:
        f.write(result["text"].strip())
    
    # Save Transcript (JSON)
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)
        
    # Generate Global Summary
    text = result["text"]
    summary = "Summary could not be generated."
    try:
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        
        clean_text = text[:3500]
        word_count = len(clean_text.split())
        
        if word_count < 30:
            summary = clean_text
        else:
            # Dynamic length based on input size
            dynamic_max = int(min(150, word_count * 0.8))
            dynamic_min = int(min(50, dynamic_max * 0.5))
            dynamic_max = max(10, dynamic_max)
            dynamic_min = max(5, dynamic_min)

            summary_result = summarizer(clean_text, max_length=dynamic_max, min_length=dynamic_min, do_sample=False)
            summary = summary_result[0]['summary_text']
            
    except Exception as e:
        print(f"Summary AI Error: {e}. Using fallback.")
        sentences = text.split('.')
        summary = ". ".join(sentences[:15]) + "."
        
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)
    except Exception as e:
        print(f"Error saving summary: {e}")

# --- 3. SENTIMENT ---
def analyze_sentiment(transcript_path, output_path):
    setup_nltk()
    sia = SentimentIntensityAnalyzer()
    
    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if "segments" not in data:
            return

        timeline = []
        for segment in data["segments"]:
            text = segment["text"].strip()
            scores = sia.polarity_scores(text)
            comp = scores['compound']
            label = "Positive" if comp >= 0.05 else "Negative" if comp <= -0.05 else "Neutral"
            
            timeline.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": text,
                "score": comp,
                "label": label
            })
            
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(timeline, f, indent=4)
            
    except Exception as e:
        print(f"Sentiment Analysis Error: {e}")

# --- 4. KEYWORDS ---
def extract_keywords(transcript_dir, output_dir):
    files = list(Path(transcript_dir).glob("*.json"))
    if not files: return

    documents = []
    filenames = []
    
    for p in files:
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                documents.append(data["text"])
                filenames.append(p.stem)
        except: pass
            
    if not documents: return

    try:
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000)
        tfidf_matrix = vectorizer.fit_transform(documents)
        feature_names = vectorizer.get_feature_names_out()
        
        for i, filename in enumerate(filenames):
            scores = tfidf_matrix[i].toarray().flatten()
            top_indices = scores.argsort()[-10:][::-1]
            
            keywords = []
            for idx in top_indices:
                if scores[idx] > 0:
                    keywords.append(f"{feature_names[idx]}")
            
            out_file = Path(output_dir) / f"{filename}_keywords.txt"
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(f"=== KEYWORDS FOR: {filename} ===\n")
                f.write("\n".join(keywords))
    except ValueError:
        pass 

# --- 5. TOPIC SEGMENTATION ---
def segment_topics(transcript_path, output_path):
    setup_nltk()
    
    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
            
        segments_raw = json_data.get('segments', [])
        
        # Fallback if no segments found or very short
        if len(segments_raw) < 5:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"=== TOPIC ANALYSIS: {Path(transcript_path).stem} ===\n")
                f.write(f"🔹 TOPIC 1 [00:00 - End]: {json_data['text'][:200]}... (Audio too short)\n")
            return

        # Prepare text for segmentation
        texts = [s['text'] for s in segments_raw]
        WINDOW_SIZE = 2
        SIMILARITY_THRESHOLD = 0.65

        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
        except ValueError:
            return

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

        # Initialize Summarizer
        try:
            summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        except:
            summarizer = None

        final_report = []
        final_report.append(f"=== TOPIC ANALYSIS: {Path(transcript_path).stem} ===\n")
        final_report.append(f"Detected {len(cut_indices)-1} major topics.\n")

        for i in range(len(cut_indices) - 1):
            start_idx = cut_indices[i]
            end_idx = cut_indices[i+1]
            
            chunk = segments_raw[start_idx:end_idx]
            if not chunk: continue

            topic_text = " ".join([s['text'].strip() for s in chunk])
            start_time = chunk[0]['start']
            end_time = chunk[-1]['end']
            
            # Generate Summary & Headline
            summary = "Summary unavailable"
            headline = "Topic " + str(i+1)
            
            if summarizer:
                try:
                    clean_text = topic_text[:3000]
                    if len(clean_text) > 30:
                        res = summarizer(clean_text, max_length=120, min_length=40, do_sample=False)
                        summary = res[0]['summary_text']
                        headline = summary.split('.')[0]
                        if len(headline) > 60: headline = headline[:60] + "..."
                except: pass
            
            keywords = extract_keywords_text(topic_text)
            t_start = format_time(start_time)
            t_end = format_time(end_time)
            
            final_report.append(f"🔹 TOPIC {i+1} [{t_start} - {t_end}]: {headline}")
            final_report.append(f"   SUMMARY: {summary}")
            final_report.append(f"   KEYWORDS: {keywords}")
            final_report.append("-" * 40 + "\n")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(final_report))
            
    except Exception as e:
        print(f"Topic Error: {e}")
        # Fallback
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"=== TOPICS (Fallback) ===\nError processing topics: {e}\n")

# --- MASTER PIPELINE ---
def process_new_upload(file_input, base_dir, is_url=False):
    # Dynamic path setup
    dirs = setup_directories(base_dir)
    
    # 1. Handle Input (File Object vs URL)
    if is_url:
        url_filename = file_input.split("/")[-1].split("?")[0]
        if not url_filename or not url_filename.endswith(('.mp3', '.wav', '.m4a')):
            url_filename = "downloaded_audio.mp3"
        file_stem = Path(url_filename).stem
        
        raw_path = dirs["audio"] / url_filename
        try:
            response = requests.get(file_input, stream=True)
            response.raise_for_status()
            with open(raw_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        except Exception as e:
            return f"Download failed: {e}"
    else:
        file_stem = Path(file_input.name).stem
        raw_path = dirs["audio"] / file_input.name
        with open(raw_path, "wb") as f:
            f.write(file_input.getbuffer())
        
    # 2. Preprocess
    proc_path = dirs["processed"] / f"{file_stem}.wav"
    if not preprocess_audio(raw_path, proc_path):
        return "Preprocessing failed."
        
    # 3. Transcribe & Summary
    trans_path = dirs["transcripts"] / f"{file_stem}.json"
    summ_path = dirs["summary"] / f"{file_stem}_summary.txt"
    transcribe_and_summarize(proc_path, trans_path, summ_path)
    
    # 4. Sentiment
    sent_path = dirs["sentiment"] / f"{file_stem}_sentiment.json"
    analyze_sentiment(trans_path, sent_path)
    
    # 5. Topics
    top_path = dirs["topics"] / f"{file_stem}_topics.txt"
    segment_topics(trans_path, top_path)
    
    # 6. Update Keywords (Global)
    extract_keywords(dirs["transcripts"], dirs["keywords"])
    
    return "Success"