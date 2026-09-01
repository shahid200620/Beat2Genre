import os
import sys
import tempfile

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
    background: rgba(10,11,14,0.85);
    backdrop-filter: blur(12px);
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

st.markdown(
    '<div class="section-title">Upload a track</div>',
    unsafe_allow_html=True
)

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

if uploaded_file:
    file_name = uploaded_file.name
    file_extension = os.path.splitext(file_name)[1]

    st.markdown(
        '<div class="section-title">Your upload</div>',
        unsafe_allow_html=True
    )

    st.caption(f"Selected file: {file_name}")

    st.audio(uploaded_file)

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=file_extension
    ) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
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

            st.markdown(
                '<div class="section-title">📊 Track Analytics</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="section-description">A visual look at how the '
                'track compares across genres and the audio characteristics '
                'used during classification.</div>',
                unsafe_allow_html=True
            )

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                fig, ax = plt.subplots(figsize=(7, 5))

                chart_data = probability_data.sort_values(
                    "Probability",
                    ascending=True
                )

                ax.barh(
                    chart_data["Genre"].str.title(),
                    chart_data["Probability"] * 100
                )

                ax.set_xlabel("Model confidence (%)")
                ax.set_title("Genre Probability")
                ax.grid(axis="x", alpha=0.15)

                for index, value in enumerate(
                    chart_data["Probability"] * 100
                ):
                    ax.text(
                        value + 0.5,
                        index,
                        f"{value:.1f}%",
                        va="center",
                        fontsize=9
                    )

                ax.set_xlim(0, max(100, chart_data["Probability"].max() * 100 + 10))
                fig.tight_layout()

                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            with chart_col2:
                feature_names = [
                    "Tempo",
                    "RMS Energy",
                    "Spectral Centroid",
                    "Spectral Bandwidth",
                    "Zero Crossing Rate",
                    "Chroma"
                ]

                feature_values = [
                    features["tempo"],
                    features["rms_mean"],
                    features["spectral_centroid_mean"],
                    features["spectral_bandwidth_mean"],
                    features["zero_crossing_rate_mean"],
                    features["chroma_stft_mean"]
                ]

                normalized_values = []

                for value in feature_values:
                    maximum = max(abs(value), 1)
                    normalized_values.append(
                        abs(value) / maximum
                    )

                fig, ax = plt.subplots(figsize=(7, 5))

                ax.barh(
                    feature_names[::-1],
                    normalized_values[::-1]
                )

                ax.set_xlabel("Relative signal level")
                ax.set_title("Audio Feature Profile")
                ax.grid(axis="x", alpha=0.15)

                fig.tight_layout()

                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            st.markdown(
                '<div class="section-title">🏆 Model Benchmark</div>',
                unsafe_allow_html=True
            )

            benchmark_display = benchmark.copy()

            benchmark_display["model_name"] = (
                benchmark_display["model_name"]
                .str.replace("Logistic Regression", "Logistic Regression")
            )

            benchmark_col1, benchmark_col2 = st.columns(2)

            with benchmark_col1:
                fig, ax = plt.subplots(figsize=(7, 5))

                ax.bar(
                    benchmark_display["model_name"],
                    benchmark_display["accuracy"] * 100
                )

                ax.set_ylabel("Accuracy (%)")
                ax.set_title("Model Accuracy")
                ax.set_ylim(0, 100)
                ax.grid(axis="y", alpha=0.15)

                for index, value in enumerate(
                    benchmark_display["accuracy"] * 100
                ):
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

            with benchmark_col2:
                fig, ax = plt.subplots(figsize=(7, 5))

                ax.bar(
                    benchmark_display["model_name"],
                    benchmark_display["f1_score"] * 100
                )

                ax.set_ylabel("Macro F1-score (%)")
                ax.set_title("Model F1-score")
                ax.set_ylim(0, 100)
                ax.grid(axis="y", alpha=0.15)

                for index, value in enumerate(
                    benchmark_display["f1_score"] * 100
                ):
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
                '<div class="section-title">📈 Benchmark Overview</div>',
                unsafe_allow_html=True
            )

            benchmark_chart = benchmark_display.set_index(
                "model_name"
            )[["accuracy", "precision", "recall", "f1_score"]].copy()

            benchmark_chart = benchmark_chart * 100

            fig, ax = plt.subplots(figsize=(11, 5))

            benchmark_chart.plot(
                kind="bar",
                ax=ax
            )

            ax.set_ylabel("Score (%)")
            ax.set_xlabel("Model")
            ax.set_title("Classification Model Performance")
            ax.set_ylim(0, 100)
            ax.grid(axis="y", alpha=0.15)
            ax.legend(
                title="Metric",
                loc="lower right"
            )

            plt.xticks(rotation=15)
            fig.tight_layout()

            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            st.markdown(
                '<div class="section-title">📋 Performance Summary</div>',
                unsafe_allow_html=True
            )

            performance_table = benchmark_display.copy()

            performance_table["accuracy"] = (
                performance_table["accuracy"] * 100
            ).round(2)

            performance_table["precision"] = (
                performance_table["precision"] * 100
            ).round(2)

            performance_table["recall"] = (
                performance_table["recall"] * 100
            ).round(2)

            performance_table["f1_score"] = (
                performance_table["f1_score"] * 100
            ).round(2)

            performance_table.columns = [
                "Model",
                "Accuracy (%)",
                "Precision (%)",
                "Recall (%)",
                "F1-score (%)"
            ]

            st.dataframe(
                performance_table,
                use_container_width=True,
                hide_index=True
            )

        except Exception as error:
            st.error(
                "Beat2Genre couldn't read this file. Try a different "
                "audio file."
            )

            st.caption(f"Details: {error}")

        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)