import os
import json
import warnings
from pathlib import Path
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()

# Use environment variables for paths, with fallbacks
# Keywords are extracted FROM transcripts and saved TO the keywords directory
INPUT_DIR = os.getenv("TRANSCRIPT_DIR", r"D:\farrakh important\internship_project infosys\data\transcripts")
OUTPUT_DIR = os.getenv("KEYWORD_DIR", r"D:\farrakh important\internship_project infosys\data\keywords")
TOP_N_KEYWORDS = 10

# Suppress warnings
warnings.filterwarnings("ignore")

def extract_keywords(text, top_n=5):
    """
    Extracts keywords from a single text block using TF-IDF.
    """
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

def process_keyword_extraction():
    """
    Processes all transcript files and extracts keywords for each.
    """
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    files = list(input_path.glob("*.json"))
    
    if not files:
        print(f"❌ No transcript files found in {INPUT_DIR}")
        return

    print(f"📂 Found {len(files)} transcripts to analyze.")

    for file_path in tqdm(files, desc="Extracting Keywords"):
        output_file = output_path / f"{file_path.stem}_keywords.txt"
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                text = data["text"]
            
            # Extract keywords from the full text
            keywords = extract_keywords(text, top_n=TOP_N_KEYWORDS)
            
            # Save keywords to file
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"=== KEYWORDS FOR: {file_path.stem} ===\n")
                # Format keywords as a list
                keyword_list = keywords.split(", ")
                f.write("\n".join(keyword_list))

        except Exception as e:
            print(f"\n❌ Error processing {file_path.name}: {e}")

    print(f"\n✅ Keyword Extraction Complete. Output in: {OUTPUT_DIR}")

if __name__ == "__main__":
    process_keyword_extraction()