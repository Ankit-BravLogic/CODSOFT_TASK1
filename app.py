
from pathlib import Path

import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "genre_classifier.pkl"

st.set_page_config(
    page_title="Movie Genre Classifier",
    page_icon="🎬",
    layout="centered",
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("🎬 Movie Genre Classification")
st.write(
    "Discover a movie's likely genre using machine learning "
    "and Natural Language Processing."
)

st.info(
    "Enter a movie plot or synopsis below. "
    "The model predicts one genre from 27 categories."
)

description = st.text_area(
    "Movie description",
    placeholder=(
        "Example: A detective investigates a mysterious "
        "crime in a small town..."
    ),
    height=180,
)

if st.button("Predict Genre", type="primary", use_container_width=True):
    if not description.strip():
        st.warning("Please enter a movie description first.")
    else:
        with st.spinner("Analyzing the movie description..."):
            prediction = model.predict([description.strip()])[0]

        st.success(f"Predicted Genre: {prediction.title()}")
        st.caption(
            "This is a machine learning prediction, "
            "not a guaranteed or multi-genre classification."
        )

st.divider()
st.caption("CodSoft Machine Learning Internship | Project 3")
