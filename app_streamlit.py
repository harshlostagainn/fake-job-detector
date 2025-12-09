import streamlit as st
import joblib
import numpy as np
from pathlib import Path

from nlp_utils import clean_text

MODEL_PATH = Path("Models/fake_job_model.pkl")
VECTORIZER_PATH = Path("Models/tfidf_vectorizer.pkl")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

st.set_page_config(page_title="Fake Job Detector", page_icon="🕵️‍♂️", layout="centered")

st.title("🕵️‍♂️ Fake Job Posting Detector")
st.write(
    "Paste any job description below and this AI model will tell you how likely it is "
    "to be **REAL** or **FAKE**."
)

job_text = st.text_area("📄 Enter Job Description", height=250)

# threshold jisme hum bolenge ki fake hai
FAKE_THRESHOLD = 0.60

if st.button("🔍 Detect"):
    if job_text.strip() == "":
        st.warning("Please enter some job description text.")
    else:
        cleaned = clean_text(job_text)
        vec = vectorizer.transform([cleaned])

        # probabilities nikalte hain
        proba = model.predict_proba(vec)[0]
        real_prob = float(proba[0])   # class 0 = REAL
        fake_prob = float(proba[1])   # class 1 = FAKE

        if fake_prob >= FAKE_THRESHOLD:
            st.error(f"🚨 This job posting looks **FAKE / FRAUDULENT** "
                     f"(fake score: {fake_prob:.2f})")
        else:
            st.success(f"✅ This job posting looks **LIKELY REAL / GENUINE** "
                       f"(real score: {real_prob:.2f})")

        st.markdown("#### 📊 Probability breakdown")
        st.write(
            {
                "Real / Genuine probability": round(real_prob, 3),
                "Fake / Fraud probability": round(fake_prob, 3),
            }
        )

st.markdown("---")
st.caption("Built by Harsh | AI + NLP Project")
