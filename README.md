 Automated Podcast Transcription & Topic Segmentation

The Automated Podcast Transcription & Topic Segmentation project is an end-to-end AI system designed to unlock the value hidden in long-form audio. By applying advanced Speech Processing and NLP techniques, this tool automatically transcribes podcasts, segments them into coherent chapters, analyzes emotional tone, and provides a searchable, interactive dashboard for navigation.

 Project Objectives

Transcription (Speech-to-Text)

Convert long podcast audio files into accurate text using OpenAI's Whisper model.

Generate precise timestamps for each transcribed segment to enable timeline visualization.

Support noisy, multi-speaker, real-world audio.

Topic Segmentation

Detect shifts in content and break the transcript into meaningful chapters.

Use TF-IDF Vectorization and Cosine Similarity (or transformer models like BERT) to mathematically identify topic boundaries.

Summarization & Intelligence

Generate abstractive summaries for each segment using DistilBART (Hugging Face).

Extract unique Keywords using TF-IDF ranking.

Analyze emotional tone using VADER Sentiment Analysis.

UI for Navigation & Visualization

Provide a Streamlit dashboard for uploading and processing audio.

Visualize segment-level analytics like sentiment trends over time with interactive Plotly charts.

Allow users to jump to specific topics and search transcripts instantly.

 System Architecture

graph TD
    A[Audio Input (File/URL)] --> B(Preprocessing: 16kHz Mono WAV);
    B --> C{AI Pipeline};
    C --> D[Transcription: OpenAI Whisper];
    C --> E[Sentiment: NLTK VADER];
    D --> F[NLP Processing];
    F --> G[Topic Segmentation: TF-IDF + Cosine Sim];
    F --> H[Summarization: DistilBART];
    F --> I[Keyword Extraction: TF-IDF];
    G --> J[Dashboard: Streamlit + Plotly];
    H --> J;
    I --> J;
    E --> J;


 Tech Stack

Core

Python 3.9+

OpenAI Whisper: State-of-the-art ASR model.

Librosa / Soundfile: Audio loading, resampling, and normalization.

FFmpeg: System-level audio processing engine.

NLP & AI

Hugging Face Transformers: sshleifer/distilbart-cnn-12-6 for summarization.

Scikit-Learn: TF-IDF Vectorizer, Cosine Similarity.

NLTK: VADER for sentiment, Punkt for sentence tokenization.

Visualization & UI

Streamlit: Interactive web application framework.

Plotly Express: Interactive charts (Sentiment Timeline, Pie Charts).

Storage

JSON: Structured storage for transcripts, segments, and analysis data.

TXT: Human-readable summaries and logs.

 Project Structure

project/
│
├── app.py                     # Frontend: Streamlit Dashboard UI
├── podcast_backend.py         # Backend: Master AI Logic Pipeline
├── download_kaggle_subset.py  # Utility: Data acquisition script
├── requirements.txt           # Dependency management
├── README.md                  # Project Documentation
││
└── data/              # Auto-generated Data Storage
    ├── audio/                 # Raw Input Audio
    ├── processed_audio/       # 16kHz WAVs
    ├── transcripts/           # JSON Transcripts with timestamps
    ├── semantic_segments/     # Topic Segmentation Reports
    ├── sentiment_data/        # Sentiment scores for graphing
    ├── short_summary/         # AI Summaries
    └── keywords/              # Extracted Keywords
└── src/              # Auto-generated Data Storage
    ├── podcast_backend/                 # Raw Input Audio
    ├── preprocessed_data/       # 16kHz WAVs
    ├── transcripts_podcast/           # JSON Transcripts with timestamps
    ├── semantic_segments/     # Topic Segmentation Reports
    ├── keyword_analysis/        # Sentiment scores for graphing

Project Workflow & Milestones

The project implementation follows an 8-week modular roadmap:

Milestone 1

Week 1: Project Initialization and Dataset Acquisition

Define project scope, system objectives, and expected outcomes.

Download and explore podcast datasets (e.g., TED Talks), including audio and transcripts.

Week 2: Audio Preprocessing and Speech-to-Text

Implement audio cleaning (resampling, normalization).

Apply speech-to-text models (Whisper) and validate initial transcription quality.

Milestone 2

Week 3: Topic Segmentation Implementation

Develop and evaluate topic segmentation algorithms (TF-IDF Windowing/Cosine Similarity).

Extract segment keywords and create initial summaries.

Week 4: User Interface and Indexing

Design transcript navigation tools with segment jumping features.

Implement basic search and keyword filtering in the Streamlit app.

Milestone 3

Week 5: Visualization and Detail Enhancements

Add interactive timelines with sentiment analysis and keyword displays.

Polish segment summaries and display formatting (e.g., visual cards, trend lines).

Week 6: System Testing and Feedback Collection

Test with diverse podcast samples to ensure robustness.

Iterate based on user feedback and improve transcription and segmentation parameters.

Milestone 4

Week 7: Final Documentation and Presentation Preparation

Compile comprehensive technical documentation and user manuals.

Prepare a compelling presentation showcasing system capabilities and benefits.

Week 8: Project Wrap-up and Delivery

Rehearse presentation.

Submit deliverables and prepare for Q&A.

 Evaluation Criteria

The success of the project is measured against the following criteria:

Completion of Milestones:

Successful implementation of audio processing, transcription, segmentation, UI, visualization, and documentation.

Transcription and Segmentation Accuracy:

Quality of speech-to-text conversion.

Precision of topic boundaries relative to ground truth or natural conversation shifts.

User Experience and Documentation Quality:

Clarity and usability of the interface (Dashboard responsiveness, ease of navigation).

Well-structured, thorough project documentation and progress reports.

 Installation & Usage

Clone the Repository:

git clone [https://github.com/your-username/Automated-Podcast-Transcription.git](https://github.com/your-username/Automated-Podcast-Transcription.git)


Install Dependencies:

pip install -r requirements.txt


(Note: FFmpeg must be installed on your system)

Run the Dashboard:

streamlit run app.py
