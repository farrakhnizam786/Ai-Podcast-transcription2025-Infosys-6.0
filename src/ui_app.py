import sys
from pathlib import Path
import streamlit as st
import json, os
from datetime import datetime
from dotenv import load_dotenv

# Add the 'src' folder to sys.path so imports work
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
sys.path.append(str(SRC_DIR))

# Now imports will work
from preprocessing import preprocess_audio
from transcription import transcribe_audio
from segmentation import segment_topics
from summarization import summarize
from keyword_extraction import extract_keywords

# Load .env
load_dotenv(ROOT_DIR / ".env")
BASE_DIR = os.getenv("BASE_DIR")
if BASE_DIR is None:
    st.error("BASE_DIR not set in .env")
    st.stop()
BASE_DIR = Path(BASE_DIR)

INDEX_FILE = BASE_DIR / "index.json"


def load_index():
    return json.loads(INDEX_FILE.read_text()) if INDEX_FILE.exists() else []

def save_index(entry):
    data = load_index()
    if not any(e["name"] == entry["name"] for e in data):
        data.append(entry)
        INDEX_FILE.write_text(json.dumps(data, indent=2))

st.set_page_config(layout="wide")
st.title("AI Podcast Transcription Dashboard")

tab1, tab2 = st.tabs(["New Processing", "Processed Dashboard"])

with tab1:
    op = st.radio("Operation Type", [
        "Transcribe Only",
        "Transcribe + Summary",
        "Full Pipeline"
    ])

    lang = st.selectbox("Language", ["Auto", "English", "Hindi"])
    file = st.file_uploader("Upload Audio", type=["mp3","wav","m4a"])

    if st.button("🚀 Start") and file:
        raw_dir = BASE_DIR / os.getenv("AUDIO_RAW", "audio_raw")
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / file.name
        raw_path.write_bytes(file.getbuffer())

        with st.spinner("Preprocessing audio..."):
            processed_path = preprocess_audio(raw_path)

        with st.spinner("Transcribing audio..."):
            transcript_text = transcribe_audio(processed_path, None if lang=="Auto" else lang.lower())

        summary_text, keywords_list, segments_list = "", [], []

        if op != "Transcribe Only":
            with st.spinner("Generating summary..."):
                summary_text = summarize(transcript_text)
            st.subheader("Summary")
            st.write(summary_text)

        if op == "Full Pipeline":
            with st.spinner("Extracting topics and keywords..."):
                segments_list = segment_topics(transcript_text)
                keywords_list = extract_keywords(transcript_text)
            st.subheader("Keywords")
            st.write(keywords_list)

        save_index({
            "name": raw_path.stem,
            "language": lang,
            "operation": op,
            "keywords": keywords_list,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M")
        })

        st.success("Processing completed")

with tab2:
    st.subheader("Processed Audio Dashboard")
    search = st.text_input("Search by name or keyword").strip().lower()

    for item in load_index()[::-1]:
        if search == "" or search in item["name"].lower() or any(search in k.lower() for k in item["keywords"]):
            with st.expander(f"{item['name']} ({item['operation']})"):
                st.write(f"Language: {item['language']}")
                st.write(f"Processed At: {item['time']}")
                st.write(f"Keywords: {', '.join(item['keywords']) if item['keywords'] else 'N/A'}")

                transcript_file = BASE_DIR / os.getenv("TRANSCRIPTS", "transcripts") / f"{item['name']}.json"
                if transcript_file.exists():
                    transcript_json = json.loads(transcript_file.read_text())
                    st.text_area("Transcript", transcript_json.get("text", ""), height=200)
