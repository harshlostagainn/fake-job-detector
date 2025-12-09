# 🕵️‍♂️ Fake Job Posting Detector (AI + NLP)

This project detects whether a job posting is **REAL** or **FAKE / FRAUDULENT** using
Natural Language Processing and Machine Learning.

## ✨ Features

- Takes raw job description text as input
- Cleans and processes text using NLP
- Uses TF-IDF features + Logistic Regression
- Trained on the **Kaggle Real/Fake Job Posting Prediction** dataset
- Simple Streamlit web app interface

## 🧠 Tech Stack

- Python 3
- Pandas, NumPy
- Scikit-learn (TF-IDF, Logistic Regression)
- Streamlit (Web App)
- Joblib (Model saving)
- Kaggle dataset: `fake_job_postings.csv`

## 📂 Project Structure

```text
fake-job-detector/
├── app_streamlit.py       # Streamlit web app
├── train_model.py         # Model training script
├── nlp_utils.py           # Text cleaning utilities
├── data/
│   └── fake_job_postings.csv
├── Models/
│   ├── fake_job_model.pkl
│   └── tfidf_vectorizer.pkl
├── requirements.txt
└── .gitignore
