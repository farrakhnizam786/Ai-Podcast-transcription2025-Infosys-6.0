import streamlit as st
import os
import json
import pandas as pd
import plotly.express as px
import re
import time
import io
import librosa
import soundfile as sf
import inspect
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from pathlib import Path

st.set_page_config(
    page_title="Podcast AI Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    import podcast_backend
except ModuleNotFoundError as e:
    # Check if the missing module is actually our backend file
    if getattr(e, 'name', '') in ['podcast_backend', 'backend']:
        st.error("Critical Error: Backend file not found.")
        st.info("Please ensure 'podcast_backend.py' is in the same folder as this script.")
        st.stop()
    else:
        # Otherwise, a library is missing! Let's show which one.
        st.error(f"❌ Missing a required library: **{getattr(e, 'name', str(e))}**")
        st.info(f"Please open your terminal and install it. For example: `pip install {getattr(e, 'name', '')}`")
        st.stop()
except Exception as e:
    st.error(f"❌ Error loading backend: {e}")
    st.stop()

CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR / "podcast_data"

TRANSCRIPT_DIR = BASE_DIR / "transcripts"
SUMMARY_DIR = BASE_DIR / "short_summary"
TOPIC_DIR = BASE_DIR / "semantic_segments"
KEYWORD_DIR = BASE_DIR / "keywords"
SENTIMENT_DIR = BASE_DIR / "sentiment_data"
AUDIO_DIR = BASE_DIR / "audio"

for d in [BASE_DIR, TRANSCRIPT_DIR, SUMMARY_DIR, TOPIC_DIR, KEYWORD_DIR, SENTIMENT_DIR, AUDIO_DIR]:
    os.makedirs(d, exist_ok=True)

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #4F46E5; font-weight: 700; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #6B7280; margin-bottom: 2rem; }
    .metric-card { background-color: #F3F4F6; color: #1F2937; padding: 1rem; border-radius: 8px; border-left: 5px solid #4F46E5; margin-bottom: 1rem; }
    .highlight { background-color: #FEF3C7; color: #000000; padding: 0.1rem 0.3rem; border-radius: 4px; font-weight: 500; }
    
    .sentiment-positive { color: #10B981; font-weight: bold; }
    .sentiment-negative { color: #EF4444; font-weight: bold; }
    .sentiment-neutral { color: #6B7280; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---

@st.cache_resource
def get_audio_slice(file_path, start_sec, end_sec):
    """
    Loads only the specific segment of audio into memory.
    This ensures the player stops exactly at end_sec.
    """
    try:
        # Calculate duration
        duration = end_sec - start_sec
        if duration <= 0: return None

        # Load specific slice using librosa (efficient)
        y, sr = librosa.load(str(file_path), sr=None, offset=start_sec, duration=duration)
        
        # Write to memory buffer as WAV
        buffer = io.BytesIO()
        sf.write(buffer, y, sr, format='WAV')
        return buffer.getvalue()
    except Exception as e:
        print(f"Error slicing audio: {e}")
        return None

def get_file_list(query):
    if not TRANSCRIPT_DIR.exists(): return []
    files = [f.replace(".json", "") for f in os.listdir(TRANSCRIPT_DIR) if f.endswith(".json")]
    if not query: return files
    
    matches = []
    query = query.lower()
    for f in files:
        try:
            t_text = (TRANSCRIPT_DIR / f"{f}.json").read_text(encoding='utf-8').lower()
            k_text = (KEYWORD_DIR / f"{f}_keywords.txt").read_text(encoding='utf-8').lower() if (KEYWORD_DIR / f"{f}_keywords.txt").exists() else ""
            if query in f.lower() or query in t_text or query in k_text:
                matches.append(f)
        except: pass
    return matches

def load_data(folder, filename, is_json=False):
    path = Path(folder) / filename
    if path.exists():
        try:
            text = path.read_text(encoding='utf-8')
            return json.loads(text) if is_json else text
        except: return None
    return None

def get_audio_path(filename_stem):
    for ext in ['.mp3', '.wav', '.m4a', '.flac']:
        path = AUDIO_DIR / f"{filename_stem}{ext}"
        if path.exists():
            return path
    return None

def parse_topics(raw_text):
    topics = []
    current_topic = {}
    lines = raw_text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        header_match = re.match(r"🔹 TOPIC \d+\s+\[(.*?)\s+-\s+(.*?)\]:\s+(.*)", line)
        
        if header_match:
            if current_topic: topics.append(current_topic)
            current_topic = {
                "start_str": header_match.group(1),
                "end_str": header_match.group(2),
                "title": header_match.group(3),
                "summary": "",
                "keywords": "",
                "start_sec": sum(x * int(t) for x, t in zip([60, 1], header_match.group(1).split(":"))),
                "end_sec": sum(x * int(t) for x, t in zip([60, 1], header_match.group(2).split(":")))
            }
        elif line.startswith("SUMMARY:"):
            if current_topic: current_topic["summary"] = line.replace("SUMMARY:", "").strip()
        elif line.startswith("KEYWORDS:"):
            if current_topic: current_topic["keywords"] = line.replace("KEYWORDS:", "").strip()
            
    if current_topic: topics.append(current_topic)
    return topics

def get_sentiment_label_for_segment(sentiment_data, start_sec, end_sec):
    if not sentiment_data: return "Neutral", "gray"
    try:
        df = pd.DataFrame(sentiment_data)
        mask = (df['start'] >= start_sec) & (df['end'] <= end_sec)
        segment_df = df[mask]
        
        if segment_df.empty:
            return "Neutral", "gray"
            
        avg_score = segment_df['score'].mean()
        if avg_score > 0.05: return "Positive", "green"
        elif avg_score < -0.05: return "Negative", "red"
        return "Neutral", "gray"
    except:
        return "Neutral", "gray"

# --- MAIN UI ---
st.markdown('<div class="main-header">AI Podcast Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated Transcription, Segmentation, Sentiment Analysis, and Summarization.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Control Panel")
    
    st.subheader("1. Add Content")
    input_tab1, input_tab2 = st.tabs(["Upload", "URL"])
    uploaded_file = None
    audio_url = ""
    
    with input_tab1:
        uploaded_file = st.file_uploader("Upload Audio", type=["mp3", "wav", "m4a", "flac"])
    with input_tab2:
        audio_url = st.text_input("Paste Audio URL", placeholder="https://example.com/episode.mp3")

    st.divider()

    st.subheader("2. Analysis Features")
    run_sentiment = st.checkbox("Sentiment Analysis", value=True)
    run_topics = st.checkbox("Topic Segmentation", value=True)
    
    st.divider()
    
    if st.button("Start Processing", type="primary", use_container_width=True):
        source = uploaded_file if uploaded_file else audio_url
        is_url = bool(audio_url) and not uploaded_file
        
        if source:
            with st.spinner("Processing..."):
                try:
                    # Dynamically check backend arguments to prevent crashes
                    sig = inspect.signature(podcast_backend.process_new_upload)
                    if 'enable_sentiment' in sig.parameters:
                        status = podcast_backend.process_new_upload(
                            source, 
                            str(BASE_DIR), 
                            is_url=is_url,
                            enable_sentiment=run_sentiment,
                            enable_topics=run_topics
                        )
                    else:
                        st.warning("⚠️ Using legacy backend: Feature toggles ignored.")
                        status = podcast_backend.process_new_upload(
                            source, 
                            str(BASE_DIR), 
                            is_url=is_url
                        )
                    
                    if status == "Success":
                        st.success("Done! Refreshing...")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"Error: {status}")
                except Exception as e:
                    st.error(f"Critical Error: {str(e)}")
        else:
            st.warning("Please upload a file or paste a link.")

    st.divider()
    st.subheader("Search Archive")
    search_query = st.text_input("Filter podcasts by keyword:", placeholder="e.g. climate, tech...")

available_files = get_file_list(search_query)

if not available_files:
    if search_query:
        st.warning(f"No podcasts found matching '{search_query}'.")
    else:
        st.info("Welcome! Upload a podcast to unlock AI features.")
    st.stop()

selected_podcast = available_files[0]
if len(available_files) > 1:
     selected_podcast = st.selectbox("Select Episode", available_files, index=0)

tab_overview, tab_sentiment, tab_transcript = st.tabs([
    "Overview", 
    "Sentiment Analysis", 
    "Transcript & Search"
])

with tab_overview:
    col_sum, col_key = st.columns([2, 1])
    
    with col_sum:
        st.markdown("### Smart Summary")
        summary_text = load_data(SUMMARY_DIR, f"{selected_podcast}_summary.txt")
        if summary_text:
            st.markdown(f'<div class="metric-card">{summary_text}</div>', unsafe_allow_html=True)
        else:
            st.info("Summary not available.")
            
    with col_key:
        st.markdown("### Topic Cloud")
        
        # Load the FULL transcript text to generate a rich word cloud
        transcript_json = load_data(TRANSCRIPT_DIR, f"{selected_podcast}.json", is_json=True)
        text_content = transcript_json.get('text', '') if transcript_json else ""
        
        if text_content:
            try:
                # Generate Word Cloud
                # This automatically sizes words by frequency (Big = High Count)
                wordcloud = WordCloud(
                    width=400, 
                    height=400, 
                    background_color='white',
                    colormap='viridis',
                    max_words=50
                ).generate(text_content)

                # Display using Matplotlib
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis("off")
                # Transparent background for better UI integration
                fig.patch.set_alpha(0) 
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Could not generate word cloud: {e}")
        else:
            st.caption("No text available for Word Cloud.")

    st.divider()
    st.markdown("### Topic Segments")
    topics_text = load_data(TOPIC_DIR, f"{selected_podcast}_topics.txt")
    audio_path = get_audio_path(selected_podcast)
    sentiment_data = load_data(SENTIMENT_DIR, f"{selected_podcast}_sentiment.json", is_json=True)
    
    if topics_text:
        parsed_topics = parse_topics(topics_text)
        
        for t in parsed_topics:
            sent_label, sent_color = get_sentiment_label_for_segment(sentiment_data, t['start_sec'], t['end_sec'])
            
            with st.expander(f"{t['start_str']} - {t['end_str']} | {t['title']} | Sentiment: :{sent_color}[{sent_label}]"):
                st.markdown(f"**Summary:** {t['summary']}")
                st.caption(f"**Keywords:** {t['keywords']}")
                
                c_audio, c_dl = st.columns([3, 1])
                with c_audio:
                    if audio_path:
                        audio_clip = get_audio_slice(audio_path, t['start_sec'], t['end_sec'])
                        if audio_clip:
                            st.audio(audio_clip, format='audio/wav')
                        else:
                            st.caption("Audio segment error.")
                with c_dl:
                    dl_text = f"Title: {t['title']}\nTime: {t['start_str']} - {t['end_str']}\n\nSummary:\n{t['summary']}\n\nKeywords:\n{t['keywords']}"
                    st.download_button("Info", dl_text, file_name=f"Topic_{t['start_sec']}.txt", key=f"dl_{t['start_sec']}")
    else:
        st.info("Topic segmentation not available for this episode.")

with tab_sentiment:
    st.markdown("### Emotional Journey")
    sentiment_data = load_data(SENTIMENT_DIR, f"{selected_podcast}_sentiment.json", is_json=True)

    if sentiment_data:
        df = pd.DataFrame(sentiment_data)
        if not df.empty:
            m1, m2 = st.columns(2)
            avg_score = df['score'].mean()
            tone = "Positive" if avg_score > 0.05 else "Negative" if avg_score < -0.05 else "Neutral"
            m1.metric("Overall Tone", tone, delta=f"{avg_score:.2f}")
            m2.metric("Data Points", len(df))

            df['Trend'] = df['score'].rolling(window=10, min_periods=1).mean()
            c1, c2 = st.columns([3, 1])
            with c1:
                fig = px.bar(df, x="start", y="score", color="label",
                             color_discrete_map={'Positive': '#10B981', 'Negative': '#EF4444', 'Neutral': '#D1D5DB'},
                             title="Sentiment Flow vs Trend")
                fig.add_scatter(x=df['start'], y=df['Trend'], mode='lines', name='Trend', line=dict(color='#4F46E5', width=3))
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                pie = px.pie(df, names='label', color='label', 
                             color_discrete_map={'Positive': '#10B981', 'Negative': '#EF4444', 'Neutral': '#D1D5DB'},
                             hole=0.4)
                pie.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0), height=200)
                st.plotly_chart(pie, use_container_width=True)
    else:
        st.warning("Sentiment analysis not available for this episode.")

with tab_transcript:
    st.markdown("### Full Transcript")
    
    highlight_term = st.text_input("Highlight keyword in text:", value=search_query if search_query else "")
    transcript_json = load_data(TRANSCRIPT_DIR, f"{selected_podcast}.json", is_json=True)
    
    if transcript_json:
        full_text = transcript_json.get('text', '')
        if highlight_term:
            try:
                pattern = re.compile(re.escape(highlight_term), re.IGNORECASE)
                highlighted_text = pattern.sub(lambda m: f'<span class="highlight">{m.group(0)}</span>', full_text)
                st.markdown(f'<div style="background-color: white; padding: 20px; border-radius: 10px; border: 1px solid #ddd; height: 500px; overflow-y: scroll; color: black;">{highlighted_text}</div>', unsafe_allow_html=True)
                count = len(re.findall(pattern, full_text))
                st.caption(f"Found {count} occurrences of '{highlight_term}'")
            except Exception as e:
                st.warning(f"Highlight error: {e}")
                st.text_area("Content", full_text, height=500)
        else:
            st.text_area("Content", full_text, height=500)
    else:
        st.error("Transcript file missing.")