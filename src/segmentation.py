import json
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from pathlib import Path
import os

nltk.download("punkt", quiet=True)

load_dotenv()
BASE_DIR = Path(os.getenv("PROJECT_ROOT"))

def segment_topics(transcript_path, threshold=0.65):
    data = json.loads(transcript_path.read_text())
    sentences = nltk.sent_tokenize(data["text"])

    if len(sentences) < 6:
        return []

    tfidf = TfidfVectorizer(stop_words="english").fit_transform(sentences)
    similarities = []

    for i in range(len(sentences) - 1):
        sim = cosine_similarity(tfidf[i], tfidf[i+1])[0][0]
        similarities.append(sim)

    segments = []
    current = []

    for i, sim in enumerate(similarities):
        current.append(sentences[i])
        if sim < threshold:
            segments.append(" ".join(current))
            current = []

    current.append(sentences[-1])
    segments.append(" ".join(current))

    out_dir = BASE_DIR / "segments" / transcript_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, seg in enumerate(segments):
        (out_dir / f"topic_{i+1}.txt").write_text(seg, encoding="utf-8")

    return segments
