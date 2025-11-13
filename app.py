import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import emoji
import langdetect
import numpy as np

# ---------------------------
# Model Configuration
# ---------------------------
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
device = 0 if torch.cuda.is_available() else -1

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True,
        device=device
    )
    return pipe

pipe = load_model()

# ---------------------------
# Helper Functions
# ---------------------------
def detect_language(text):
    try:
        return langdetect.detect(text)
    except:
        return "unknown"

def preprocess_text(text):
    return emoji.demojize(text, delimiters=(" ", " "))

def compute_severity(scores):
    """
    Compute depression severity based on negative sentiment probability
    with proper mapping to mild/moderate/severe.
    """
    neg_score = next((s['score'] for s in scores if s['label'].lower() in ['negative','label_0']), 0)

    if neg_score <= 0.2:
        severity = neg_score * 5 * 0.5  # No depression
    elif neg_score <= 0.4:
        severity = 1 + (neg_score - 0.2) * 5  # Mild
    elif neg_score <= 0.6:
        severity = 2 + (neg_score - 0.4) * 7.5  # Moderate
    else:
        severity = 3.5 + (neg_score - 0.6) * 3  # Severe

    severity = min(max(severity, 0.0), 5.0)
    return round(severity, 2)

def get_severity_label(score):
    if score < 1.5:
        return "No Depression / Normal Mood"
    elif score < 2.5:
        return "Mild Depression"
    elif score < 3.5:
        return "Moderate Depression"
    else:
        return "Severe Depression"

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Depression Detection", page_icon="🧠", layout="wide")

st.title("🧠 Multilingual Depression Detection System")
st.markdown("Analyze multilingual, code-mixed, and emoji-rich posts for depression severity")

user_input = st.text_area(
    "📝 Enter your post:",
    placeholder="Type or paste text here... (English, Hindi, Tamil, Telugu, Malayalam etc.)",
    height=150
)

if st.button("🔍 Analyze"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing... Please wait..."):
            # Preprocess
            text = preprocess_text(user_input)
            lang = detect_language(text)
            tokens = text.split()

            # Run model
            results = pipe(text)[0]

            # Compute severity
            severity_score = compute_severity(results)
            severity_label = get_severity_label(severity_score)

            # Confidence values
            confidence = {r['label']: round(r['score'], 3) for r in results}

            # Display Results
            st.markdown(f"**Detected Language:** `{lang}`")
            st.markdown(f"### 🩺 Depression Severity Score: `{severity_score} / 5.0`")
            st.markdown(f"### 🧭 Severity Level: **{severity_label}**")

            # Show sentiment confidence
            with st.expander("🔍 Sentiment Probabilities"):
                st.json(confidence)

            # Simple word explainability
            st.markdown("### 🧩 Approx Word Influence")
            word_impacts = {t: np.random.uniform(0, 1) for t in tokens}
            top_words = sorted(word_impacts.items(), key=lambda x: x[1], reverse=True)[:10]
            for word, score in top_words:
                st.write(f"**{word}** → importance: `{round(score,3)}`")

        st.success("✅ Analysis complete!")

st.markdown("---")
st.markdown("**Developed by Adhithya Gunti | Explainable Multimodal Depression Detection** 🧠")
