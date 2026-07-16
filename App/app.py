"""
Yassir Customer Feedback Classifier
------------------------------------
Streamlit app: paste a review (French, English, or Algerian Arabic/darija) and get:
  - detected language
  - sentiment (positive / negative) with confidence
  - detected aspect(s): delivery, food, service, quality, app, pricing

Architecture:
  - French / English text -> TF-IDF + Logistic Regression (best score on this dataset, instant load)
  - Arabic-script / darija text -> DziriBERT, fine-tuned on Algerian dialect (loaded from HF Hub)
"""

import re
import joblib
import numpy as np
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langdetect import detect, LangDetectException

# ============================================================
# CONFIG -- update this if you used a different HF repo name
# ============================================================
ARABIC_MODEL_REPO = "cabluemgeek/yassir-sentiment-dziribert"
LABELS = ["negative", "positive"]  # index order must match le.classes_ from training

ARABIC_PATTERN = re.compile(r"[\u0600-\u06FF]")

ASPECTS = ["delivery", "food", "service", "quality", "app", "pricing"]

ARABIC_TO_ENGLISH = {
    "توصيل": "delivery", "التوصيل": "delivery",
    "طعام": "food", "الطعام": "food", "الأكل": "food",
    "خدمة": "service", "الخدمة": "service",
    "جودة": "quality", "الجودة": "quality",
    "تطبيق": "app", "الطتبيق": "app",
    "سعر": "pricing", "أسعار": "pricing", "السعر": "pricing", "الأسعار": "pricing",
}

# Added for the app: your original notebook only matched literal English aspect words
# or the Arabic dict above, so French reviews (e.g. "la livraison") always fell back to
# "general". This closes that gap.
FRENCH_TO_ENGLISH = {
    "livraison": "delivery", "livreur": "delivery", "livré": "delivery",
    "livrée": "delivery", "retard": "delivery",
    "nourriture": "food", "plat": "food", "repas": "food", "aliment": "food", "goût": "food",
    "service": "service", "accueil": "service", "personnel": "service",
    "qualité": "quality", "fraîcheur": "quality", "frais": "quality", "propre": "quality",
    "application": "app", "appli": "app", "bug": "app", "interface": "app",
    "prix": "pricing", "cher": "pricing", "tarif": "pricing", "coût": "pricing", "coute": "pricing",
}


# ============================================================
# TEXT PROCESSING -- mirrors the notebook's clean_text / aspect logic
# ============================================================
def clean_text(text):
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-ZÀ-ÿ\u0600-\u06FF\s?!]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_for_aspects(text):
    text = text.lower()
    for mapping in (ARABIC_TO_ENGLISH, FRENCH_TO_ENGLISH):
        for word, english_word in mapping.items():
            if word in text:
                text += f" {english_word}"
    return text


def extract_aspects(text):
    normalized = normalize_for_aspects(text)
    found = [a for a in ASPECTS if a in normalized]
    return found if found else ["general"]


def detect_language_label(raw_text):
    if ARABIC_PATTERN.search(raw_text):
        return "Arabic / Darija"
    try:
        lang = detect(raw_text)
    except LangDetectException:
        return "Unknown"
    return {"fr": "French", "en": "English"}.get(lang, lang.upper())


# ============================================================
# MODEL LOADING (cached -- runs once per session)
# ============================================================
@st.cache_resource(show_spinner="Loading TF-IDF model...")
def load_tfidf_model():
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    clf = joblib.load("tfidf_logreg.pkl")
    return vectorizer, clf


@st.cache_resource(show_spinner="Loading DziriBERT (Arabic dialect model)...")
def load_dziribert_model():
    tokenizer = AutoTokenizer.from_pretrained(ARABIC_MODEL_REPO)
    model = AutoModelForSequenceClassification.from_pretrained(ARABIC_MODEL_REPO)
    model.eval()
    return tokenizer, model


def predict_tfidf(text, vectorizer, clf):
    X = vectorizer.transform([text])
    proba = clf.predict_proba(X)[0]
    pred_idx = int(np.argmax(proba))
    return LABELS[pred_idx], float(proba[pred_idx])


def predict_dziribert(text, tokenizer, model):
    inputs = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    proba = torch.softmax(logits, dim=1)[0].numpy()
    pred_idx = int(np.argmax(proba))
    return LABELS[pred_idx], float(proba[pred_idx])


def classify_review(raw_text):
    cleaned = clean_text(raw_text)
    language = detect_language_label(raw_text)
    aspects_found = extract_aspects(cleaned)

    if ARABIC_PATTERN.search(raw_text):
        tokenizer, model = load_dziribert_model()
        sentiment, confidence = predict_dziribert(cleaned, tokenizer, model)
        model_used = "DziriBERT"
    else:
        vectorizer, clf = load_tfidf_model()
        sentiment, confidence = predict_tfidf(cleaned, vectorizer, clf)
        model_used = "TF-IDF + Logistic Regression"

    return {
        "language": language,
        "sentiment": sentiment,
        "confidence": confidence,
        "aspects": aspects_found,
        "model_used": model_used,
    }


# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="Yassir Review Classifier", page_icon="🛵", layout="centered")

st.title("🛵 Yassir Customer Feedback Classifier")
st.caption("Write a review in French, English, or Algerian Arabic/darija -- get instant sentiment + aspect detection.")

EXAMPLES = {
    "French": "La livraison était très en retard et la nourriture était froide.",
    "English": "The app crashed twice but the food quality was excellent!",
    "Arabic / Darija": "الأكل كان طيبًا بزاف ولكن السعر كان مرتفعًا",
}

with st.expander("Try an example"):
    cols = st.columns(3)
    for col, (lang, example) in zip(cols, EXAMPLES.items()):
        if col.button(lang, use_container_width=True):
            st.session_state["review_text"] = example

review_text = st.text_area(
    "Your review",
    value=st.session_state.get("review_text", ""),
    height=120,
    placeholder="Type or paste a customer review here...",
)

if st.button("Classify", type="primary", use_container_width=True):
    if not review_text.strip():
        st.warning("Please enter a review first.")
    else:
        with st.spinner("Analyzing..."):
            result = classify_review(review_text)

        st.divider()

        col1, col2, col3 = st.columns(3)
        col1.metric("Detected language", result["language"])

        sentiment_display = "🟢 Positive" if result["sentiment"] == "positive" else "🔴 Negative"
        col2.metric("Sentiment", sentiment_display)
        col3.metric("Confidence", f"{result['confidence']*100:.1f}%")

        st.subheader("Detected aspect(s)")
        st.write(" ".join(f"`{a}`" for a in result["aspects"]))

        st.caption(f"Model used: {result['model_used']}")

st.divider()
st.caption(
    "Sentiment models: TF-IDF + Logistic Regression (French/English) and DziriBERT (Algerian dialect), "
    "trained on Yassir app customer reviews. Aspect detection is keyword-based across delivery, food, "
    "service, quality, app, and pricing."
)
