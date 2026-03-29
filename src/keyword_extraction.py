import json
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_keywords(transcript_path, k=10):
    text = json.loads(transcript_path.read_text())["text"]
    vec = TfidfVectorizer(stop_words="english")
    tfidf = vec.fit_transform([text]).toarray()[0]
    words = vec.get_feature_names_out()

    return [w for w, _ in sorted(zip(words, tfidf), key=lambda x: x[1], reverse=True)[:k]]
