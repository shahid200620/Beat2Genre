import os
import sys
import tempfile
import io

import joblib
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, "src")

from audio_features import extract_features

st.set_page_config(
    page_title="Beat2Genre",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "features" not in st.session_state:
    st.session_state.features = None

if "file_name" not in st.session_state:
    st.session_state.file_name = None

if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None

if "upload_bytes" not in st.session_state:
    st.session_state.upload_bytes = None

if "upload_name" not in st.session_state:
    st.session_state.upload_name = None

if "predicted_genre" not in st.session_state:
    st.session_state.predicted_genre = None

if "probability_data" not in st.session_state:
    st.session_state.probability_data = None

if "analysis_error" not in st.session_state:
    st.session_state.analysis_error = None

MODEL_PATH = "output/best_model.joblib"
SCALER_PATH = "output/scaler.joblib"
ENCODER_PATH = "output/label_encoder.joblib"
BENCHMARK_PATH = "output/benchmark_results.csv"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(ENCODER_PATH)
benchmark = pd.read_csv(BENCHMARK_PATH)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --bg-0: #0A0B0E;
    --bg-1: #14161B;
    --bg-2: #1B1E24;
    --border: rgba(255,255,255,0.08);
    --border-soft: rgba(255,255,255,0.05);
    --amber: #F2A93B;
    --amber-soft: rgba(242,169,59,0.14);
    --teal: #35D0C0;
    --teal-soft: rgba(53,208,192,0.14);
    --text-0: #F3F1EB;
    --text-1: #9BA0A8;
    --text-2: #6B7078;
    --success: #6FCF97;
    --danger: #E5646B;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--text-0);
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 12% 8%, rgba(242,169,59,0.06), transparent 32%),
        radial-gradient(circle at 88% 18%, rgba(53,208,192,0.05), transparent 30%),
        var(--bg-0);
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stToolbar"] {
    visibility: hidden;
}

[data-testid="stDecoration"] {
    display: none;
}

.block-container {
    max-width: 880px;
    padding: 2rem 1.5rem 5rem;
}

section[data-testid="stSidebar"] {
    background: var(--bg-1);
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] * {
    color: var(--text-0);
}

.mono {
    font-family: 'IBM Plex Mono', monospace;
}

/* Sidebar brand + spec sheet */
.brand {
    padding: 0.25rem 0 1.25rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.25rem;
}

.brand-mark {
    font-size: 1.6rem;
    margin-bottom: 0.4rem;
}

.brand-name {
    font-family: 'Sora', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: -0.01em;
}

.brand-tag {
    margin-top: 0.3rem;
    color: var(--text-1);
    font-size: 0.85rem;
    line-height: 1.5;
}

.spec-list {
    display: flex;
    flex-direction: column;
    gap: 0.1rem;
}

.spec-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.6rem 0;
    border-bottom: 1px solid var(--border-soft);
    font-size: 0.88rem;
}

.spec-row span:first-child {
    color: var(--text-1);
}

.spec-row span:last-child {
    color: var(--text-0);
    font-weight: 500;
}

.sidebar-note {
    margin-top: 1.5rem;
    padding-top: 1.25rem;
    border-top: 1px solid var(--border);
    color: var(--text-2);
    font-size: 0.82rem;
    line-height: 1.6;
}

/* Hero */
.hero {
    padding: 3rem 1rem 2.5rem;
    text-align: center;
}

.eq-bars {
    display: flex;
    align-items: flex-end;
    justify-content: center;
    gap: 5px;
    height: 52px;
    margin: 0 auto 1.75rem;
}

.eq-bar {
    width: 5px;
    height: 44px;
    border-radius: 3px 3px 0 0;
    background: linear-gradient(180deg, var(--amber), var(--teal));
    animation: eq-pulse 1.1s ease-in-out infinite;
    transform-origin: bottom;
}

.eq-bar:nth-child(1) { animation-delay: 0s; animation-duration: 1.1s; }
.eq-bar:nth-child(2) { animation-delay: 0.15s; animation-duration: 0.9s; }
.eq-bar:nth-child(3) { animation-delay: 0.3s; animation-duration: 1.3s; }
.eq-bar:nth-child(4) { animation-delay: 0.1s; animation-duration: 1.0s; }
.eq-bar:nth-child(5) { animation-delay: 0.25s; animation-duration: 1.15s; }
.eq-bar:nth-child(6) { animation-delay: 0.05s; animation-duration: 0.95s; }
.eq-bar:nth-child(7) { animation-delay: 0.35s; animation-duration: 1.25s; }
.eq-bar:nth-child(8) { animation-delay: 0.2s; animation-duration: 1.05s; }
.eq-bar:nth-child(9) { animation-delay: 0.4s; animation-duration: 0.85s; }

@keyframes eq-pulse {
    0%, 100% { transform: scaleY(0.28); opacity: 0.75; }
    50% { transform: scaleY(1); opacity: 1; }
}

@media (prefers-reduced-motion: reduce) {
    .eq-bar { animation: none; transform: scaleY(0.65); }
}

.hero-title {
    font-family: 'Sora', sans-serif;
    font-size: clamp(2.1rem, 4.2vw, 3.1rem);
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.15;
    color: var(--text-0);
    margin-bottom: 0.9rem;
}

.hero-subtitle {
    max-width: 480px;
    margin: 0 auto;
    color: var(--text-1);
    font-size: 1.02rem;
    line-height: 1.7;
}

/* Section headers */
.section-title {
    margin-top: 2.5rem;
    margin-bottom: 0.5rem;
    font-family: 'Sora', sans-serif;
    color: var(--text-0);
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: -0.015em;
}

.section-description {
    margin-bottom: 1.2rem;
    color: var(--text-1);
    line-height: 1.7;
    font-size: 0.95rem;
}

/* Upload */
.upload-zone {
    padding: 0.25rem;
}

div[data-testid="stFileUploaderDropzone"] {
    min-height: 148px;
    border-radius: 16px;
    border: 1px dashed rgba(242,169,59,0.35);
    background: var(--bg-1);
    transition: border-color 0.2s ease, background 0.2s ease;
}

div[data-testid="stFileUploaderDropzone"]:hover {
    border-color: rgba(242,169,59,0.7);
    background: var(--bg-2);
}

div[data-testid="stFileUploaderDropzone"] button {
    border-radius: 8px;
}

div[data-testid="stAudio"] {
    margin: 0.75rem 0 1.1rem;
}

/* Buttons */
button[kind="primary"] {
    background: linear-gradient(135deg, var(--amber), #D9901F) !important;
    color: #14161B !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    box-shadow: 0 8px 22px rgba(242,169,59,0.22);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}

button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 10px 26px rgba(242,169,59,0.32);
}

button[kind="secondary"] {
    border-radius: 10px;
}

/* Result */
.result-card {
    padding: 2.4rem 1.5rem;
    border-radius: 20px;
    text-align: center;
    background: var(--bg-1);
    border: 1px solid var(--border);
}

.result-label {
    color: var(--text-1);
    font-size: 0.82rem;
    font-weight: 500;
}

.result-genre {
    margin: 0.55rem 0;
    color: var(--amber);
    font-family: 'Sora', sans-serif;
    font-size: clamp(2.2rem, 5vw, 3.2rem);
    line-height: 1.1;
    font-weight: 800;
    letter-spacing: -0.02em;
    text-transform: capitalize;
}

.result-description {
    max-width: 380px;
    margin: 0 auto;
    color: var(--text-1);
    font-size: 0.92rem;
    line-height: 1.6;
}

/* Stat cards */
.stat-card {
    padding: 1.2rem 1rem;
    border-radius: 14px;
    text-align: center;
    background: var(--bg-1);
    border: 1px solid var(--border);
}

.stat-label {
    color: var(--text-2);
    font-size: 0.72rem;
    font-weight: 500;
}

.stat-value {
    margin-top: 0.3rem;
    color: var(--text-0);
    font-size: 1.25rem;
    font-weight: 600;
}

/* Probability rows */
.probability-row {
    padding: 0.9rem 1rem;
    margin-bottom: 0.55rem;
    border-radius: 12px;
    background: var(--bg-1);
    border: 1px solid var(--border);
}

.probability-row.top-rank {
    background: var(--amber-soft);
    border: 1px solid rgba(242,169,59,0.3);
}

.probability-header {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin-bottom: 0.5rem;
}

.probability-rank {
    width: 1.8rem;
    color: var(--text-2);
    font-size: 0.78rem;
}

.top-rank .probability-rank {
    color: var(--amber);
}

.probability-genre {
    flex: 1;
    color: var(--text-0);
    font-size: 0.95rem;
    font-weight: 600;
}

.probability-value {
    color: var(--text-1);
    font-size: 0.88rem;
}

.top-rank .probability-value {
    color: var(--amber);
}

.probability-track {
    width: 100%;
    height: 6px;
    border-radius: 999px;
    background: rgba(255,255,255,0.06);
    overflow: hidden;
}

.probability-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--teal), var(--amber));
    transition: width 0.6s ease;
}

/* Alerts / dataframes */
[data-testid="stAlert"] {
    border-radius: 12px;
}

[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
}

button:focus-visible, [tabindex]:focus-visible {
    outline: 2px solid var(--teal);
    outline-offset: 2px;
}

@media (max-width: 600px) {
    .block-container { padding: 1.25rem 1rem 3rem; }
    .hero { padding: 2rem 0.5rem 1.5rem; }
    .section-title { font-size: 1.2rem; }
}

.analysis-card {
    padding: 1.25rem;
    border-radius: 16px;
    background: var(--bg-1);
    border: 1px solid var(--border);
    margin-bottom: 1rem;
}

.analysis-card-title {
    font-family: 'Sora', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 0.4rem;
}

.analysis-card-text {
    color: var(--text-1);
    font-size: 0.84rem;
    line-height: 1.6;
}

.page-breadcrumb {
    display: inline-block;
    padding: 0.4rem 1rem;
    border-radius: 999px;
    background: var(--amber-soft);
    border: 1px solid rgba(242,169,59,0.35);
    color: var(--amber);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    white-space: nowrap;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-mark">🎧</div>
            <div class="brand-name">Beat2Genre</div>
            <div class="brand-tag">Identifies a track's genre from its
            audio signal — rhythm, energy, and tone.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-note" style="margin-top:0; border-top:none; padding-top:0;">EXPLORE</div>',
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigate",
        ["🏠 Prediction", "🎧 Audio Features", "📊 Audio Analysis", "🤖 Model Performance"],
        label_visibility="collapsed"
    )

    st.markdown(
        """
        <div class="spec-list">
            <div class="spec-row"><span>Model</span><span class="mono">Random Forest</span></div>
            <div class="spec-row"><span>Genres</span><span class="mono">10</span></div>
            <div class="spec-row"><span>Features</span><span class="mono">58</span></div>
            <div class="spec-row"><span>Dataset</span><span class="mono">GTZAN</span></div>
        </div>
        <div class="sidebar-note">
            Upload a track to get a prediction with a full
            probability breakdown across every supported genre.
        </div>
        """,
        unsafe_allow_html=True
    )

page_titles = {
    "🏠 Prediction": "Prediction",
    "🎧 Audio Features": "Audio Features",
    "📊 Audio Analysis": "Audio Analysis",
    "🤖 Model Performance": "Model Performance"
}

st.markdown(
    f'<div class="page-breadcrumb">BEAT2GENRE / {page_titles[page].upper()}</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero">
        <div class="eq-bars">
            <div class="eq-bar"></div>
            <div class="eq-bar"></div>
            <div class="eq-bar"></div>
            <div class="eq-bar"></div>
            <div class="eq-bar"></div>
            <div class="eq-bar"></div>
            <div class="eq-bar"></div>
            <div class="eq-bar"></div>
            <div class="eq-bar"></div>
        </div>
        <div class="hero-title">Every track has a signature.</div>
        <div class="hero-subtitle">
            Beat2Genre reads the rhythm, energy, and tone of a song
            and matches it to the genre it most resembles.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

if page == "🏠 Prediction":
    st.markdown(
        '<div class="section-title">Upload a track</div>',
        unsafe_allow_html=True
    )
    
    if st.session_state.upload_bytes:
        st.markdown(
            '<div class="section-description">A track is already loaded '
            'below. Remove it if you want to analyze a different file.</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="section-description">Drop in an audio file and '
            'Beat2Genre extracts its characteristics automatically — no '
            'technical values to enter by hand.</div>',
            unsafe_allow_html=True
        )
    
        st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
    
        uploaded_file = st.file_uploader(
            "Drop a track, or click to browse",
            type=["wav", "mp3", "m4a", "flac"],
            help="Supported formats: WAV, MP3, M4A and FLAC."
        )
    
        st.markdown('</div>', unsafe_allow_html=True)
    
        if uploaded_file is not None:
            st.session_state.upload_bytes = uploaded_file.getvalue()
            st.session_state.upload_name = uploaded_file.name
            st.rerun()
    
    if st.session_state.upload_bytes:
        file_name = st.session_state.upload_name
        file_extension = os.path.splitext(file_name)[1]
    
        st.markdown(
            '<div class="section-title">Your upload</div>',
            unsafe_allow_html=True
        )
    
        upload_info_col, upload_remove_col = st.columns([4, 1])
    
        with upload_info_col:
            st.caption(f"Selected file: {file_name}")
    
        with upload_remove_col:
            remove_file = st.button(
                "Remove file",
                use_container_width=True
            )
    
        if remove_file:
            st.session_state.upload_bytes = None
            st.session_state.upload_name = None
            st.session_state.features = None
            st.session_state.file_name = None
            st.session_state.audio_bytes = None
            st.session_state.predicted_genre = None
            st.session_state.probability_data = None
            st.session_state.analysis_error = None
            st.rerun()
    
        st.audio(st.session_state.upload_bytes)
    
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_extension
        ) as temp_file:
            temp_file.write(st.session_state.upload_bytes)
            audio_path = temp_file.name
    
        analyze_track = st.button(
            "Analyze my track",
            type="primary",
            use_container_width=True
        )
    
        if analyze_track:
            try:
                with st.spinner("Listening to the track and extracting features..."):
                    features = extract_features(audio_path)
    
                    X = pd.DataFrame([features])
                    X_scaled = scaler.transform(X)
                    X_scaled = pd.DataFrame(
                        X_scaled,
                        columns=X.columns
                    )
    
                    prediction = model.predict(X_scaled)[0]
                    probabilities = model.predict_proba(X_scaled)[0]
    
                    predicted_genre = label_encoder.inverse_transform(
                        [prediction]
                    )[0]
    
                    genres = label_encoder.inverse_transform(
                        model.classes_
                    )
    
                    probability_data = pd.DataFrame({
                        "Genre": genres,
                        "Probability": probabilities
                    }).sort_values(
                        "Probability",
                        ascending=False
                    )
    
                st.session_state.features = features
                st.session_state.file_name = file_name
                st.session_state.audio_bytes = st.session_state.upload_bytes
                st.session_state.predicted_genre = predicted_genre
                st.session_state.probability_data = probability_data
                st.session_state.analysis_error = None
    
            except Exception as error:
                st.session_state.analysis_error = str(error)
    
            finally:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
    
            st.rerun()
    
        if st.session_state.analysis_error:
            st.error(
                "Beat2Genre couldn't read this file. Try a different "
                "audio file."
            )
    
            st.caption(f"Details: {st.session_state.analysis_error}")
    
        has_result = (
            st.session_state.predicted_genre is not None
            and st.session_state.file_name == file_name
        )
    
        if has_result:
            features = st.session_state.features
            predicted_genre = st.session_state.predicted_genre
            probability_data = st.session_state.probability_data
    
            top_probability = probability_data.iloc[0]["Probability"]
    
            st.markdown(
                '<div class="section-title">Result</div>',
                unsafe_allow_html=True
            )
    
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Most likely genre</div>
                    <div class="result-genre">{predicted_genre}</div>
                    <div class="result-description">
                        The model found this genre to be the strongest match
                        for the audio characteristics of your track.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    
            st.write("")
    
            stat1, stat2, stat3 = st.columns(3)
    
            with stat1:
                st.markdown(
                    f"""
                    <div class="stat-card">
                        <div class="stat-label">Top probability</div>
                        <div class="stat-value mono">{top_probability:.1%}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    
            with stat2:
                st.markdown(
                    """
                    <div class="stat-card">
                        <div class="stat-label">Features analyzed</div>
                        <div class="stat-value mono">58</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    
            with stat3:
                st.markdown(
                    """
                    <div class="stat-card">
                        <div class="stat-label">Classifier</div>
                        <div class="stat-value">Random Forest</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    
            st.success(
                f"Beat2Genre classified this track as "
                f"{predicted_genre.title()}."
            )
    
            st.markdown(
                '<div class="section-title">Genre breakdown</div>',
                unsafe_allow_html=True
            )
    
            st.markdown(
                '<div class="section-description">How the track scores '
                'against all ten supported genres. Higher means a '
                'stronger match to that genre\'s audio pattern.</div>',
                unsafe_allow_html=True
            )
    
            probability_values = probability_data["Probability"].tolist()
            probability_genres = probability_data["Genre"].tolist()
    
            for position, (genre, probability) in enumerate(
                zip(probability_genres, probability_values),
                start=1
            ):
                percentage = probability * 100
                rank_class = "top-rank" if position == 1 else ""
    
                st.markdown(
                    f"""
                    <div class="probability-row {rank_class}">
                        <div class="probability-header">
                            <span class="probability-rank mono">{position:02d}</span>
                            <span class="probability-genre">{genre.title()}</span>
                            <span class="probability-value mono">{percentage:.1f}%</span>
                        </div>
                        <div class="probability-track">
                            <div class="probability-fill"
                                 style="width:{percentage}%;">
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    
            second_genre = probability_data.iloc[1]["Genre"]
            second_probability = probability_data.iloc[1]["Probability"]
    
            difference = top_probability - second_probability
    
            if difference >= 0.20:
                interpretation = (
                    f"The model has a clear preference for "
                    f"{predicted_genre.title()} over the other genres."
                )
            elif difference >= 0.08:
                interpretation = (
                    f"{predicted_genre.title()} is the strongest match, "
                    f"although some characteristics are shared with "
                    f"{second_genre.title()}."
                )
            else:
                interpretation = (
                    f"The prediction is relatively close between "
                    f"{predicted_genre.title()} and {second_genre.title()}, "
                    f"so the track contains characteristics associated with "
                    f"multiple genres."
                )
    
            st.info(interpretation)
    
            st.markdown(
                '<div class="section-title">Audio profile</div>',
                unsafe_allow_html=True
            )
    
            st.markdown(
                '<div class="section-description">The main measurements '
                'extracted from your track — the numerical signals the '
                'classifier used, rather than manually entered values.</div>',
                unsafe_allow_html=True
            )
    
            profile_columns = st.columns(3)
    
            profile_data = [
                ("Tempo", f"{features['tempo']:.1f} BPM"),
                ("RMS energy", f"{features['rms_mean']:.4f}"),
                ("Spectral centroid", f"{features['spectral_centroid_mean']:.1f} Hz"),
                ("Spectral bandwidth", f"{features['spectral_bandwidth_mean']:.1f} Hz"),
                ("Zero crossing rate", f"{features['zero_crossing_rate_mean']:.4f}"),
                ("Chroma", f"{features['chroma_stft_mean']:.4f}")
            ]
    
            for index, (label, value) in enumerate(profile_data):
                with profile_columns[index % 3]:
                    st.markdown(
                        f"""
                        <div class="stat-card">
                            <div class="stat-label">{label}</div>
                            <div class="stat-value mono">{value}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        elif not st.session_state.analysis_error:
            st.info(
                "Click \"Analyze my track\" to see the genre prediction "
                "for this file."
            )

if page == "🎧 Audio Features":
    st.markdown(
        '<div class="section-title">🎧 Audio Features</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">A complete breakdown of the 58 numerical characteristics extracted directly from your uploaded audio.</div>',
        unsafe_allow_html=True
    )

    if st.session_state.features is None:
        st.info("Upload and analyze a track from the Prediction page first.")
    else:
        features = st.session_state.features

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Analyzed track</div>
                <div class="result-genre" style="font-size:1.8rem;">
                    {st.session_state.file_name}
                </div>
                <div class="result-description">
                    58 audio characteristics extracted for classification
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        basic_features = [
            "length",
            "tempo"
        ]

        energy_features = [
            "rms_mean",
            "rms_var"
        ]

        spectral_features = [
            "spectral_centroid_mean",
            "spectral_centroid_var",
            "spectral_bandwidth_mean",
            "spectral_bandwidth_var",
            "rolloff_mean",
            "rolloff_var",
            "zero_crossing_rate_mean",
            "zero_crossing_rate_var"
        ]

        harmonic_features = [
            "chroma_stft_mean",
            "chroma_stft_var",
            "harmony_mean",
            "harmony_var",
            "perceptr_mean",
            "perceptr_var"
        ]

        mfcc_features = [
            key for key in features
            if key.startswith("mfcc")
        ]

        categories = [
            ("🎵 Basic & Rhythm", basic_features),
            ("⚡ Energy", energy_features),
            ("🌈 Spectral", spectral_features),
            ("🎼 Harmonic", harmonic_features),
            ("🧠 MFCC", mfcc_features)
        ]

        for category_name, category_features in categories:
            st.markdown(
                f'<div class="section-title">{category_name}</div>',
                unsafe_allow_html=True
            )

            rows = []

            for feature_name in category_features:
                if feature_name in features:
                    rows.append({
                        "Feature": feature_name.replace("_", " ").title(),
                        "Technical Name": feature_name,
                        "Value": f"{float(features[feature_name]):.6f}"
                    })

            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True
            )

if page == "📊 Audio Analysis":
    st.markdown(
        '<div class="section-title">📊 Audio Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">Visual analysis of the acoustic characteristics extracted from your uploaded track.</div>',
        unsafe_allow_html=True
    )

    if st.session_state.features is None:
        st.info("Upload and analyze a track from the Prediction page first.")
    else:
        features = st.session_state.features

        if st.session_state.audio_bytes:
            audio_data, sample_rate = librosa.load(
                io.BytesIO(st.session_state.audio_bytes),
                sr=None,
                mono=True
            )

            duration = len(audio_data) / sample_rate
            time_axis = np.linspace(0, duration, len(audio_data))

            st.markdown(
                '<div class="section-title">〰️ Waveform</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                """
                <div class="analysis-card">
                    <div class="analysis-card-title">Audio Signal</div>
                    <div class="analysis-card-text">
                        The waveform shows how the amplitude of the uploaded audio
                        changes over time. Peaks represent stronger signal activity,
                        while lower regions represent quieter portions.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            fig, ax = plt.subplots(figsize=(11, 4))

            ax.plot(time_axis, audio_data)

            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Amplitude")
            ax.set_title("Waveform of Uploaded Audio")
            ax.grid(alpha=0.15)

            fig.tight_layout()

            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Analyzed track</div>
                <div class="result-genre" style="font-size:1.8rem;">
                    {st.session_state.file_name}
                </div>
                <div class="result-description">
                    Visual representation of the extracted audio characteristics
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-title">🌈 Spectral Profile</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="analysis-card">
                <div class="analysis-card-title">What are spectral features?</div>
                <div class="analysis-card-text">
                    These measurements describe how energy is distributed across the
                    frequency spectrum of the audio. They help the model distinguish
                    different tonal and textural characteristics between genres.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        spectral_names = [
            "Spectral Centroid",
            "Spectral Bandwidth",
            "Spectral Rolloff",
            "Zero Crossing Rate"
        ]

        spectral_values = [
            features["spectral_centroid_mean"],
            features["spectral_bandwidth_mean"],
            features["rolloff_mean"],
            features["zero_crossing_rate_mean"]
        ]

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.barh(spectral_names[::-1], spectral_values[::-1])

        ax.set_xlabel("Feature Value")
        ax.set_title("Spectral Characteristics")
        ax.grid(axis="x", alpha=0.15)

        fig.tight_layout()

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown(
            '<div class="section-title">⚡ Energy & Rhythm</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="analysis-card">
                <div class="analysis-card-title">What do these measurements tell us?</div>
                <div class="analysis-card-text">
                    Tempo represents the estimated beats per minute, while RMS energy
                    describes the overall strength of the audio signal. Chroma captures
                    the distribution of musical pitch classes.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        energy_col1, energy_col2, energy_col3 = st.columns(3)

        with energy_col1:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">Tempo</div>
                    <div class="stat-value mono">{features["tempo"]:.1f} BPM</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with energy_col2:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">RMS Energy</div>
                    <div class="stat-value mono">{features["rms_mean"]:.5f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with energy_col3:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">Chroma</div>
                    <div class="stat-value mono">{features["chroma_stft_mean"]:.5f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown(
            '<div class="section-title">🧠 MFCC Profile</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="analysis-card">
                <div class="analysis-card-title">Why MFCCs matter</div>
                <div class="analysis-card-text">
                    Mel-Frequency Cepstral Coefficients capture important characteristics
                    of the audio spectrum. Beat2Genre uses 20 MFCC coefficients, with
                    both mean and variance contributing to the feature representation.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        mfcc_names = []
        mfcc_values = []

        for index in range(1, 21):
            name = f"mfcc{index}_mean"
            mfcc_names.append(f"MFCC {index}")
            mfcc_values.append(features[name])

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.bar(mfcc_names, mfcc_values)

        ax.set_xlabel("MFCC Coefficient")
        ax.set_ylabel("Mean Value")
        ax.set_title("20 MFCC Coefficients")
        ax.grid(axis="y", alpha=0.15)

        plt.xticks(rotation=45)

        fig.tight_layout()

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

if page == "🤖 Model Performance":
    st.markdown(
        '<div class="section-title">🤖 Model Performance</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">Benchmark results from the classification models evaluated during Beat2Genre development.</div>',
        unsafe_allow_html=True
    )

    best_row = benchmark.loc[benchmark["f1_score"].idxmax()]

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">Selected classifier</div>
            <div class="result-genre" style="font-size:2rem;">
                {best_row["model_name"]}
            </div>
            <div class="result-description">
                Selected because it achieved the highest macro F1-score
                across the evaluated models.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    model_col1, model_col2, model_col3 = st.columns(3)

    with model_col1:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Accuracy</div>
                <div class="stat-value mono">{best_row["accuracy"]:.1%}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with model_col2:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Macro F1</div>
                <div class="stat-value mono">{best_row["f1_score"]:.1%}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with model_col3:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Models Tested</div>
                <div class="stat-value mono">{len(benchmark)}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="section-title">📊 Accuracy Comparison</div>',
        unsafe_allow_html=True
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    values = benchmark["accuracy"] * 100

    ax.bar(
        benchmark["model_name"],
        values
    )

    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Classification Accuracy")
    ax.grid(axis="y", alpha=0.15)

    for index, value in enumerate(values):
        ax.text(
            index,
            value + 1,
            f"{value:.1f}%",
            ha="center",
            fontsize=9
        )

    plt.xticks(rotation=15)

    fig.tight_layout()

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown(
        '<div class="section-title">🎯 Macro F1 Comparison</div>',
        unsafe_allow_html=True
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    values = benchmark["f1_score"] * 100

    ax.bar(
        benchmark["model_name"],
        values
    )

    ax.set_ylabel("Macro F1-score (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Macro F1-score by Model")
    ax.grid(axis="y", alpha=0.15)

    for index, value in enumerate(values):
        ax.text(
            index,
            value + 1,
            f"{value:.1f}%",
            ha="center",
            fontsize=9
        )

    plt.xticks(rotation=15)

    fig.tight_layout()

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown(
        '<div class="section-title">📈 Complete Benchmark</div>',
        unsafe_allow_html=True
    )

    benchmark_display = benchmark.copy()

    benchmark_display["accuracy"] = (
        benchmark_display["accuracy"] * 100
    ).round(2)

    benchmark_display["precision"] = (
        benchmark_display["precision"] * 100
    ).round(2)

    benchmark_display["recall"] = (
        benchmark_display["recall"] * 100
    ).round(2)

    benchmark_display["f1_score"] = (
        benchmark_display["f1_score"] * 100
    ).round(2)

    benchmark_display.columns = [
        "Model",
        "Accuracy (%)",
        "Precision (%)",
        "Recall (%)",
        "Macro F1 (%)"
    ]

    st.dataframe(
        benchmark_display,
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        '<div class="section-title">💡 Model Selection</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="analysis-card">
            <div class="analysis-card-title">Why Random Forest?</div>
            <div class="analysis-card-text">
                Random Forest achieved the highest macro F1-score among the
                evaluated models. Macro F1 gives equal importance to each
                genre, making it a suitable metric for comparing performance
                across the ten music classes.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )