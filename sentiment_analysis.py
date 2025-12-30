import os
import json
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from pathlib import Path
from tqdm import tqdm

# Configuration
INPUT_DIR = r"D:\farrakh important\internship_project infosys\data\transcripts"
OUTPUT_DIR = r"D:\farrakh important\internship_project infosys\data\sentiment_data"

def setup_nltk():
    """Initializes NLTK sentiment resources."""
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)

def analyze_sentiment_for_file(file_path, sia, output_path):
    """Processes transcript JSON to calculate segment-level sentiment."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "segments" not in data:
            print(f"Warning: No segments in {file_path.name}. Skipping.")
            return

        sentiment_timeline = []
        
        for segment in data["segments"]:
            text = segment["text"].strip()
            start = segment["start"]
            end = segment["end"]
            
            # VADER compound score (-1 to 1)
            scores = sia.polarity_scores(text)
            compound_score = scores['compound']
            
            if compound_score >= 0.05:
                label = "Positive"
            elif compound_score <= -0.05:
                label = "Negative"
            else:
                label = "Neutral"

            sentiment_timeline.append({
                "start": start,
                "end": end,
                "text": text,
                "score": compound_score,
                "label": label
            })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(sentiment_timeline, f, indent=4)

    except Exception as e:
        print(f"Error analyzing {file_path.name}: {e}")

def process_sentiment_analysis():
    setup_nltk()
    
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    sia = SentimentIntensityAnalyzer()
    files = list(input_path.glob("*.json"))
    
    success_count = 0
    for file_path in tqdm(files, desc="Analyzing Sentiment"):
        output_file = output_path / f"{file_path.stem}_sentiment.json"
        
        if output_file.exists():
            continue

        analyze_sentiment_for_file(file_path, sia, output_file)
        success_count += 1

    print(f"\nSentiment Analysis Complete.")
    print(f"Processed {success_count} files. Output folder: {OUTPUT_DIR}")

if __name__ == "__main__":
    process_sentiment_analysis()