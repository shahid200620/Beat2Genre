# 🎧 Beat2Genre — Music Genre Classification

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Beat2Genre-FF4B4B?style=for-the-badge\&logo=streamlit\&logoColor=white)](https://beat2genre-song-prediction.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue?style=for-the-badge\&logo=python\&logoColor=white)](https://www.python.org/)
[![Scikit--learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange?style=for-the-badge)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?style=for-the-badge\&logo=streamlit\&logoColor=white)](https://streamlit.io/)

## 🎵 Project Overview

**Beat2Genre** is a machine learning application that predicts the genre of a music track from its audio characteristics.

Instead of asking users to manually enter technical audio values, the application allows them to upload an audio file and analyzes the track automatically. The system extracts 58 numerical audio features, processes them using the same preprocessing pipeline used during training, and passes them to a trained Random Forest classifier.

The application currently recognizes **10 music genres**:

**Blues · Classical · Country · Disco · Hip-Hop · Jazz · Metal · Pop · Reggae · Rock**

The project was built as a practical machine learning workflow covering data exploration, preprocessing, classification, model benchmarking, evaluation, serialization, and deployment.

### 🎧 Try the Live Application

**[Launch Beat2Genre →](https://beat2genre-song-prediction.streamlit.app/)**

Upload a supported audio file, select **Analyze My Track**, and see the predicted genre along with probability estimates and visual analytics.

---

## 🎯 Project Objective

The goal of this project is to build and evaluate a reproducible classification pipeline capable of distinguishing music genres using pre-extracted audio features.

The project focuses on three important areas of applied machine learning:

* **Classification Modeling** — training multiple standard classification algorithms.
* **Model Benchmarking** — comparing models using consistent evaluation metrics.
* **Model Serialization** — saving the selected model so it can be reused without retraining.

The final system extends the original training workflow into a simple interactive application where a user can provide a real audio track and receive a genre prediction.

---

## 📊 Dataset

The project uses the **GTZAN Music Genre Classification** dataset.

The modeling dataset used for this project is the `features_30_sec.csv` file, where each row represents a 30-second music track and contains extracted audio characteristics.

The dataset used for model development contains:

* **1,000 audio samples**
* **10 music genres**
* **100 samples per genre**
* **58 numerical audio features**
* **1 target label**

The feature set contains characteristics such as:

* Chroma STFT
* RMS energy
* Spectral centroid
* Spectral bandwidth
* Spectral rolloff
* Zero-crossing rate
* Harmony
* Perceptual features
* Tempo
* 20 MFCC measurements and their variances

The raw dataset is intentionally excluded from version control through `.gitignore`.

---

## 🔬 Machine Learning Workflow

Beat2Genre follows a straightforward and reproducible machine learning pipeline:

```text
GTZAN Feature Dataset
        │
        ▼
Data Exploration
        │
        ▼
Feature Selection
        │
        ▼
Label Encoding
        │
        ▼
Train / Test Split
        │
        ▼
StandardScaler
        │
        ▼
┌──────────────────────────────┐
│      Model Benchmarking      │
│                              │
│ Logistic Regression          │
│ Support Vector Machine       │
│ Random Forest                │
└──────────────────────────────┘
        │
        ▼
Model Evaluation
        │
        ▼
Macro F1-score Comparison
        │
        ▼
Random Forest Selected
        │
        ▼
Model Serialization
        │
        ▼
Interactive Audio Prediction
```

The scaler is fitted only on the training data before being applied to the test data, helping avoid information from the test set influencing preprocessing.

---

## 🧠 Models Evaluated

Three classification models were trained using the same processed training and testing data:

1. **Logistic Regression**
2. **Support Vector Machine (SVM)**
3. **Random Forest Classifier**

The models were evaluated using:

* Accuracy
* Macro Precision
* Macro Recall
* Macro F1-score

Macro F1-score was used as the primary selection metric because it gives equal importance to each genre.

---

## 🏆 Results

The final benchmark produced the following results:

| Model               |   Accuracy |  Precision |     Recall | Macro F1-score |
| ------------------- | ---------: | ---------: | ---------: | -------------: |
| Logistic Regression |     74.00% |     75.37% |     74.00% |         74.01% |
| SVM                 |     73.50% |     74.09% |     73.50% |         73.40% |
| **Random Forest**   | **78.00%** | **78.40%** | **78.00%** |     **77.80%** |

### 🥇 Selected Model: Random Forest

Random Forest achieved the highest macro F1-score at **77.80%**, making it the selected model for the final application.

The trained model was serialized using `joblib` and stored as:

```text
output/best_model.joblib
```

The saved model was also reloaded and tested independently to verify that it continued to produce the expected predictions.

---

## 🎧 Real Audio Prediction

One of the main improvements in Beat2Genre is the ability to work with an actual audio file instead of requiring users to manually enter feature values.

The prediction process is:

```text
Audio Upload
     ↓
Audio Feature Extraction
     ↓
58 Features
     ↓
Saved StandardScaler
     ↓
Saved Random Forest Model
     ↓
Genre Prediction
     ↓
Probability Breakdown
```

The application supports common audio formats including:

* WAV
* MP3
* M4A
* FLAC

The interface also presents the extracted audio profile and visualizes the model's probability estimates across the supported genres.

---

## 📈 Visual Analytics

Beat2Genre includes several visual representations to make the model output easier to understand.

### Genre Probability

The application displays the model's estimated probability for each supported genre, allowing users to see not only the top prediction but also the alternative genres considered by the classifier.

### Audio Feature Profile

Important characteristics extracted from the uploaded track are displayed through dedicated statistics and visualizations, including:

* Tempo
* RMS energy
* Spectral centroid
* Spectral bandwidth
* Zero-crossing rate
* Chroma

### Model Benchmark

The application also presents comparisons between the three trained models using:

* Accuracy
* Precision
* Recall
* F1-score

This makes the final model selection transparent rather than simply assuming that one algorithm is better.

---

## 🖥️ Live Application

Beat2Genre is deployed using **Streamlit Community Cloud**.

### Live Demo

**https://beat2genre-song-prediction.streamlit.app/**

The application provides:

* Modern interactive interface
* Audio upload
* Audio preview
* One-click genre analysis
* Genre prediction
* Probability breakdown
* Audio feature statistics
* Model benchmark charts
* Performance comparison
* Supported genre information

---

## 📁 Project Structure

```text
Beat2Genre/
│
├── app.py
│
├── data/
│   └── features_30_sec.csv
│
├── notebooks/
│   └── eda.ipynb
│
├── output/
│   ├── benchmark_results.csv
│   ├── benchmark_comparison.png
│   ├── best_model.joblib
│   ├── best_model_metrics.json
│   ├── label_encoder.joblib
│   ├── scaler.joblib
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   └── y_test.csv
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── audio_features.py
│   ├── predict_audio.py
│   ├── validate_features.py
│   └── verify_project.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

The raw CSV dataset is excluded from Git tracking through `.gitignore` to keep the repository lightweight.

---

## ⚙️ Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/shahid200620/Beat2Genre.git
cd Beat2Genre
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment on Windows

```cmd
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🧪 Reproducing the Machine Learning Pipeline

The preprocessing workflow can be executed with:

```bash
python src/preprocess.py
```

The training and benchmarking workflow can then be executed with:

```bash
python src/train.py
```

The training script produces:

* `benchmark_results.csv`
* `benchmark_comparison.png`
* `best_model_metrics.json`
* `best_model.joblib`

The project also includes validation utilities for checking saved models and comparing extracted audio features.

---

## 💾 Model Persistence

The final Random Forest classifier is stored using `joblib`.

The project also saves:

```text
best_model.joblib
scaler.joblib
label_encoder.joblib
```

This allows the application to reuse the trained machine learning pipeline without retraining the classifier every time a user uploads a track.

---

## 🛡️ Reproducibility

Several decisions were made to keep the workflow reproducible:

* Fixed `random_state` during train-test splitting.
* Stratified train-test split.
* Standardized numerical features.
* Scaler fitted only on training data.
* Consistent feature ordering.
* Saved label encoder.
* Saved preprocessing scaler.
* Serialized final classifier.
* Benchmark results stored as a CSV.
* Final model metrics stored as JSON.

---

## 📌 Project Limitations

Beat2Genre is a learning-focused machine learning project and should be viewed accordingly.

The model is trained on the available GTZAN-derived feature dataset, so its performance depends on how closely a new audio track resembles the characteristics represented in the training data.

Music genres can also overlap significantly. A track may contain characteristics associated with several genres, so the probability breakdown can sometimes be more informative than the top prediction alone.

The current application focuses on genre classification rather than identifying individual artists, songs, or subgenres.

---

## 🔭 Future Improvements

Possible future improvements include:

* Expanding the training dataset.
* Adding more diverse audio samples.
* Experimenting with additional classification algorithms.
* Improving feature engineering.
* Performing systematic hyperparameter tuning.
* Adding confusion-matrix visualization.
* Evaluating cross-validation performance.
* Supporting longer or shorter audio clips.
* Adding more detailed audio analysis.

---

## 🛠️ Technologies Used

| Technology       | Purpose                            |
| ---------------- | ---------------------------------- |
| Python           | Core programming language          |
| Pandas           | Data processing                    |
| NumPy            | Numerical computation              |
| Scikit-learn     | Machine learning                   |
| Librosa          | Audio feature extraction           |
| Joblib           | Model serialization                |
| Matplotlib       | Data visualization                 |
| Seaborn          | Visualization support              |
| Jupyter Notebook | Exploratory data analysis          |
| Streamlit        | Interactive web application        |
| Git & GitHub     | Source control and project hosting |

---

## 📚 Learning Outcomes

Through this project, I worked through a complete small-scale machine learning workflow:

* Understanding a structured audio feature dataset.
* Performing exploratory data analysis.
* Preparing numerical features.
* Encoding categorical target labels.
* Avoiding preprocessing data leakage.
* Training multiple classification algorithms.
* Comparing models using consistent metrics.
* Selecting a model based on macro F1-score.
* Saving trained models for later use.
* Extracting features directly from real audio.
* Building an interactive machine learning application.
* Visualizing model predictions and benchmark results.
* Managing the project using Git and GitHub.
* Deploying the final application for public use.

---

## 👨‍💻 Project

**Beat2Genre — Music Genre Classification**

Built as a practical machine learning project focused on turning a standard classification workflow into a usable audio prediction application.

### Links

🎧 **Live Application:**
https://beat2genre-song-prediction.streamlit.app/

💻 **Source Code:**
https://github.com/shahid200620/Beat2Genre

---

## ⭐ Final Note

Beat2Genre started as a model benchmarking exercise and evolved into a complete, interactive application.

The main idea is simple:

> **Upload a track, let the model listen to its characteristics, and see which genre it thinks fits best.**

The project is intentionally built around understandable machine learning techniques so that each stage — from preprocessing to model selection and prediction — can be followed and reproduced.
