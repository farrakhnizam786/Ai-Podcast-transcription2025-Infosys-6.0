import streamlit as st
import os
import json
import pandas as pd
import plotly.express as px
import re
import time
# Import your backend logic file
import podcast_backend

# --- CONFIGURATION ---
BASE_DIR = r"D:\farrakh important\internship_project infosys\podcast_data"
TRANSCRIPT_DIR = os.path.join(BASE_DIR, "transcripts")
SUMMARY_DIR = os.path.join(BASE_DIR, "short_summary")
TOPIC_DIR = os.path.join(BASE_DIR, "semantic_segments")
KEYWORD_DIR = os.path.join(BASE_DIR, "keywords")
SENTIMENT_DIR = os.path.join(BASE_DIR, "sentiment_data")

# --- PAGE SETUP ---
st.set_page_config(
    page_title="Podcast AI Analytics",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #4F46E5; font-weight: 700; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #6B7280; margin-bottom: 2rem; }
    
    /* Fix for White Background Text Visibility */
    .metric-card {
        background-color: #F3F4F6;
        color: #1F2937 !important; /* Force dark text color */
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #4F46E5;
        margin-bottom: 1rem;
    }
    
    .highlight { background-color: #FEF3C7; color: #000000; padding: 0.1rem 0.3rem; border-radius: 4px; font-weight: 500; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; border-radius: 4px 4px 0 0; gap: 1px; padding-top: 10px; padding-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown('<div class="main-header">🎙️ AI Podcast Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated Transcription, Segmentation, Sentiment Analysis, and Summarization.</div>', unsafe_allow_html=True)

# --- SIDEBAR: CONTROLS ---
with st.sidebar:
    st.header("⚙️ Control Panel")
    
    st.subheader("1. Add Content")
    input_tab1, input_tab2 = st.tabs(["📤 Upload", "🔗 URL"])
    
    uploaded_file = None
    audio_url = ""
    
    with input_tab1:
        uploaded_file = st.file_uploader("Upload Audio", type=["mp3", "wav", "m4a", "flac"])
    with input_tab2:
        audio_url = st.text_input("Paste Audio URL", placeholder="https://example.com/episode.mp3")

    st.divider()

    st.subheader("2. Analysis Features")
    
    # Language Selector
    language = st.selectbox("Language", ["English", "Spanish", "French", "German", "Hindi", "Japanese"])
    translate_opt = st.checkbox("Translate to English", value=False)
    
    c1, c2 = st.columns(2)
    with c1:
        st.checkbox("Sentiment Analysis", value=True)
    with c2:
        st.checkbox("Topic Segmentation", value=True)
        
    st.divider()
    
    if st.button("🚀 Start Processing", type="primary", use_container_width=True):
        source = uploaded_file if uploaded_file else audio_url
        is_url = bool(audio_url) and not uploaded_file
        
        if source:
            with st.spinner("🤖 AI is working... This may take a few minutes."):
                try:
                    status = podcast_backend.process_new_upload(source, BASE_DIR, is_url=is_url)
                    if status == "Success":
                        st.success("✅ Done! Refreshing...")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"Error: {status}")
                except Exception as e:
                    st.error(f"Critical Error: {str(e)}")
        else:
            st.warning("Please upload a file or paste a link.")

    st.divider()
    st.subheader("🔎 Search Archive")
    search_query = st.text_input("Filter podcasts by keyword:", placeholder="e.g. climate, tech...")

# --- FUNCTIONS ---
def get_file_list(query):
    if not os.path.exists(TRANSCRIPT_DIR): return []
    files = [f.replace(".json", "") for f in os.listdir(TRANSCRIPT_DIR) if f.endswith(".json")]
    if not query: return files
    
    matches = []
    query = query.lower()
    for f in files:
        t_path = os.path.join(TRANSCRIPT_DIR, f"{f}.json")
        k_path = os.path.join(KEYWORD_DIR, f"{f}_keywords.txt")
        found = False
        if os.path.exists(t_path):
            try:
                if query in json.load(open(t_path, 'r', encoding='utf-8')).get('text', '').lower(): found = True
            except: pass
        if not found and os.path.exists(k_path):
            try:
                if query in open(k_path, 'r', encoding='utf-8').read().lower(): found = True
            except: pass
        if found: matches.append(f)
    return matches

def load_data(folder, filename, is_json=False):
    path = os.path.join(folder, filename)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f) if is_json else f.read()
    return None

def time_to_seconds(time_str):
    """Converts MM:SS to seconds (int)"""
    try:
        parts = time_str.split(':')
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except:
        return 0
    return 0

def parse_topics(raw_text):
    """Parses Topic File into Structured Data with Timestamps"""
    topics = []
    current_topic = {}
    lines = raw_text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # Regex for: 🔹 TOPIC 1 [00:00 - 05:00]: Heading
        header_match = re.match(r"🔹 TOPIC \d+\s+\[(.*?)\s+-\s+(.*?)\]:\s+(.*)", line)
        
        if header_match:
            if current_topic: topics.append(current_topic)
            current_topic = {
                "start_str": header_match.group(1),
                "end_str": header_match.group(2),
                "start_sec": time_to_seconds(header_match.group(1)),
                "end_sec": time_to_seconds(header_match.group(2)),
                "title": header_match.group(3),
                "summary": "",
                "keywords": ""
            }
        elif line.startswith("SUMMARY:"):
            if current_topic: current_topic["summary"] = line.replace("SUMMARY:", "").strip()
        elif line.startswith("KEYWORDS:"):
            if current_topic: current_topic["keywords"] = line.replace("KEYWORDS:", "").strip()
            
    if current_topic: topics.append(current_topic)
    return topics

# --- MAIN LOGIC ---
available_files = get_file_list(search_query)

if not available_files:
    if search_query:
        st.warning(f"No podcasts found matching '{search_query}'.")
    else:
        st.info("👋 Welcome! Upload your first podcast to begin.")
    st.stop()

selected_podcast = st.selectbox("Select Episode", available_files, index=0)

tab_overview, tab_analysis, tab_transcript = st.tabs(["📌 Overview", "📊 Visualization", "📜 Transcript & Search"])

# --- TAB 1: OVERVIEW ---
with tab_overview:
    col_sum, col_key = st.columns([2, 1])
    
    with col_sum:
        st.markdown("### 📝 Smart Summary")
        summary_text = load_data(SUMMARY_DIR, f"{selected_podcast}_summary.txt")
        if not summary_text:
            transcript_json = load_data(TRANSCRIPT_DIR, f"{selected_podcast}.json", is_json=True)
            if transcript_json:
                summary_text = transcript_json.get('text', '')[:500] + "..."
                st.caption("⚠️ AI Summary pending. Showing transcript preview:")
        
        if summary_text:
            st.markdown(f'<div class="metric-card">{summary_text}</div>', unsafe_allow_html=True)
        else:
            st.info("No content available for summary.")
            
    with col_key:
        st.markdown("### 🔑 Top Keywords")
        keywords_text = load_data(KEYWORD_DIR, f"{selected_podcast}_keywords.txt")
        if keywords_text:
            kws = [line.strip() for line in keywords_text.splitlines() if not line.startswith("===")]
            for kw in kws[:10]: st.caption(f"🏷️ {kw}")
        else:
            st.info("Keywords pending...")

    st.divider()
    st.markdown("### 📚 Topic Segments & Analysis")
    
    topics_text = load_data(TOPIC_DIR, f"{selected_podcast}_topics.txt")
    sentiment_data = load_data(SENTIMENT_DIR, f"{selected_podcast}_sentiment.json", is_json=True)
    
    if topics_text:
        parsed_topics = parse_topics(topics_text)
        
        if parsed_topics:
            for i, t in enumerate(parsed_topics):
                with st.expander(f"⏱️ {t['start_str']} - {t['end_str']} | {t['title']}"):
                    st.markdown(f"**Summary:** {t['summary']}")
                    st.caption(f"**Keywords:** {t['keywords']}")
                    
                    # --- TOPIC VISUALIZATION BUTTON ---
                    if st.button(f"📊 Visualize Topic {i+1}", key=f"topic_btn_{i}"):
                        if sentiment_data:
                            df_topic = pd.DataFrame(sentiment_data)
                            # Filter data for this topic time range
                            mask = (df_topic['start'] >= t['start_sec']) & (df_topic['end'] <= t['end_sec'])
                            df_filtered = df_topic.loc[mask]
                            
                            if not df_filtered.empty:
                                c1, c2 = st.columns([3, 1])
                                with c1:
                                    fig_t = px.bar(df_filtered, x="start", y="score", color="label",
                                                 color_discrete_map={'Positive': '#10B981', 'Negative': '#EF4444', 'Neutral': '#D1D5DB'},
                                                 title=f"Sentiment Flow: {t['title']}")
                                    st.plotly_chart(fig_t, use_container_width=True)
                                with c2:
                                    pie_t = px.pie(df_filtered, names='label', color='label',
                                                 color_discrete_map={'Positive': '#10B981', 'Negative': '#EF4444', 'Neutral': '#D1D5DB'},
                                                 hole=0.4, title="Topic Mood")
                                    st.plotly_chart(pie_t, use_container_width=True)
                            else:
                                st.warning("No sentiment data found in this segment.")
                        else:
                            st.warning("Sentiment data missing.")
        else:
            st.text_area("Detected Topics", topics_text, height=300)
    else:
        st.warning("Topic segmentation not available.")

# --- TAB 2: VISUALIZATION (GLOBAL) ---
with tab_analysis:
    st.markdown("### 📈 Full Episode Emotional Journey")
    if sentiment_data:
        df = pd.DataFrame(sentiment_data)
        df['Trend'] = df['score'].rolling(window=20, min_periods=1).mean()
        
        m1, m2, m3 = st.columns(3)
        avg_score = df['score'].mean()
        tone = "Positive" if avg_score > 0.05 else "Negative" if avg_score < -0.05 else "Neutral"
        m1.metric("Overall Tone", tone, delta=f"{avg_score:.2f}")
        m2.metric("Duration", f"{int(df['end'].max() // 60)} mins")
        m3.metric("Data Points", len(df))

        col_chart, col_pie = st.columns([3, 1])
        with col_chart:
            fig = px.bar(df, x="start", y="score", color="label",
                         color_discrete_map={'Positive': '#10B981', 'Negative': '#EF4444', 'Neutral': '#D1D5DB'},
                         labels={'start': 'Time (s)', 'score': 'Sentiment Intensity'},
                         title="Sentiment Flow vs. Trend Line", opacity=0.4)
            fig.add_scatter(x=df['start'], y=df['Trend'], mode='lines', name='Trend', line=dict(color='#4F46E5', width=3))
            fig.update_layout(plot_bgcolor='white', height=400)
            st.plotly_chart(fig, use_container_width=True)
            
        with col_pie:
            st.markdown("**Mood Distribution**")
            pie_fig = px.pie(df, names='label', color='label',
                             color_discrete_map={'Positive': '#10B981', 'Negative': '#EF4444', 'Neutral': '#D1D5DB'},
                             hole=0.4)
            pie_fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0), height=250)
            st.plotly_chart(pie_fig, use_container_width=True)
    else:
        st.warning("Sentiment data not found.")

# --- TAB 3: TRANSCRIPT ---
with tab_transcript:
    st.markdown("### 📜 Full Transcript")
    highlight_term = st.text_input("Highlight keyword in text:", value=search_query if search_query else "")
    transcript_json = load_data(TRANSCRIPT_DIR, f"{selected_podcast}.json", is_json=True)
    
    if transcript_json:
        full_text = transcript_json.get('text', '')
        if highlight_term:
            pattern = re.compile(re.escape(highlight_term), re.IGNORECASE)
            highlighted_text = pattern.sub(lambda m: f'<span class="highlight">{m.group(0)}</span>', full_text)
            st.markdown(f'<div style="background-color: white; padding: 20px; border-radius: 10px; border: 1px solid #ddd; height: 500px; overflow-y: scroll; color: black;">{highlighted_text}</div>', unsafe_allow_html=True)
            count = len(re.findall(pattern, full_text))
            st.caption(f"Found {count} occurrences of '{highlight_term}'")
        else:
            st.text_area("Content", full_text, height=500)
    else:
        st.error("Transcript file missing.")