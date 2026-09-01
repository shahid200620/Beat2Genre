\# Beat2Genre



Beat2Genre is a machine learning project that classifies music into different genres using extracted audio features from the GTZAN music genre dataset.



The project explores the dataset, prepares the audio features for machine learning, trains multiple classification models, compares their performance using consistent evaluation metrics, and saves the best-performing model for future predictions.



\## Project Overview



Music contains many measurable characteristics such as tempo, spectral properties, chroma information, and Mel-Frequency Cepstral Coefficients (MFCCs). These characteristics can be used to identify patterns associated with different music genres.



In this project, pre-extracted audio features are used to classify tracks into 10 genres:



\- Blues

\- Classical

\- Country

\- Disco

\- Hip-Hop

\- Jazz

\- Metal

\- Pop

\- Reggae

\- Rock



Three classification algorithms were benchmarked:



\- Logistic Regression

\- Support Vector Machine

\- Random Forest



The models were evaluated using accuracy, macro-averaged precision, macro-averaged recall, and macro-averaged F1-score. Macro F1-score was used as the primary metric for selecting the final model.



\## Dataset



The project uses the GTZAN music genre feature dataset.



The dataset contains 1,000 music tracks represented by 60 columns, including audio features and the target genre label.



There are 10 genres with 100 tracks per genre, giving a balanced target distribution.



The project uses the pre-extracted `features\_30\_sec.csv` file rather than processing the original audio files.



\## Project Structure



```text

Beat2Genrei/

│

├── data/

│   └── features\_30\_sec.csv

│

├── notebooks/

│   └── eda.ipynb

│

├── output/

│   ├── benchmark\_results.csv

│   ├── benchmark\_comparison.png

│   ├── best\_model\_metrics.json

│   ├── best\_model.joblib

│   ├── scaler.joblib

│   └── label\_encoder.joblib

│

├── src/

│   ├── preprocess.py

│   ├── train.py

│   └── verify\_project.py

│

├── .gitignore

├── README.md

└── requirements.txt

