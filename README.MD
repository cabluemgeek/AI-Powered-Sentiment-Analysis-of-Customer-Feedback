# AI-Powered Sentiment Analysis of Customer Feedback

Multilingual sentiment and aspect classification on customer reviews from the Yassir app — French, English, and Algerian Arabic (darija) — with a live interactive dashboard and a deployed classifier app.

🔗 **Live app**: [ai-powered-sentiment-analysis-of-customer-feedback-baouusrlueu.streamlit.app](https://ai-powered-sentiment-analysis-of-customer-feedback-baouusrlueu.streamlit.app/)
📊 **Interactive dashboard**: [cabluemgeek.github.io/AI-Powered-Sentiment-Analysis-of-Customer-Feedback](https://cabluemgeek.github.io/AI-Powered-Sentiment-Analysis-of-Customer-Feedback/interactive-dashboard.html)
🤗 **Arabic model on Hugging Face**: [cabluem/yassir-sentiment-dziribert](https://huggingface.co/cabluem/yassir-sentiment-dziribert)

---

## Overview

Yassir app reviews come in French, English, and Algerian Arabic, often mixed within the same dataset. This project:

- Classifies each review as **positive** or **negative**
- Detects which **aspect** it concerns: delivery, food, service, quality, app, or pricing
- Routes each review to the model best suited to its language automatically
- Surfaces the results in a **restaurant leaderboard** built from a second, unlabeled review dataset

## Problem

Classify customer feedback collected from the Yassir app into sentiment categories while identifying the underlying topic of complaint or praise, using text written in three different languages/scripts without a translation step.

## Approach

### Data pipeline
Cleaning, language detection, duplicate/consistency checks, feature engineering (review length, punctuation counts, token length vs. model max-length), and bilingual/RTL-aware word clouds.

### Model comparison

Five approaches were trained and evaluated on the labeled dataset (`data_reviews.csv`):

| Model | Scope | F1 (macro) |
|---|---|---|
| TF-IDF + Logistic Regression | Full dataset | **0.981** |
| mBERT | Full dataset | 0.971 |
| XLM-RoBERTa | Full dataset | 0.962 |
| DziriBERT | Arabic-script subset | 1.000* |
| MARBERT | Arabic-script subset | 1.000* |

\* Near-perfect scores on the Arabic subset should be read with caution — the validation split there is small (~74 rows), so these numbers are less statistically robust than the full-dataset results.

**Key finding**: the classical TF-IDF + Logistic Regression baseline outperformed every transformer on French/English text, while being orders of magnitude smaller and faster to load. This directly shaped the deployment architecture below.

### Aspect detection

Keyword-based, with dictionaries in French, English, and Arabic (including Algerian dialect terms) mapping to six aspect categories: delivery, food, service, quality, app, pricing.

### Deployed architecture

To stay within free-tier hosting memory limits while keeping the best accuracy per language:

- **French / English reviews** → TF-IDF + Logistic Regression (best score, ships directly in the repo, loads instantly)
- **Arabic-script / darija reviews** → DziriBERT, a transformer pretrained on ~1M Algerian-dialect tweets, hosted on Hugging Face Hub and loaded at runtime

## Repository structure

```
.
├── .devcontainer/
│   └── devcontainer.json
├── App/
│   ├── app.py                  # Streamlit classifier app
│   ├── requirements.txt
│   ├── tfidf_vectorizer.pkl
│   └── tfidf_logreg.pkl
├── world clouds/
│   ├── arabic word cloud - negative.png         
│   ├── arabic word cloud - positive.png
│   ├── latin word cloud - negative.png
│   └── latin word cloud - positive.png
├── AI_Powered_Sentiment_Analysis_of_Customer_Feedback.ipynb   # full pipeline: EDA, modeling, evaluation
├── README.md
├── interactive-dashboard.html  # toggle-based EDA dashboard (Chart.js, static)
└── requirements.txt
```

## Data

- `data_reviews.csv` — labeled reviews (Sentiment, Rating, Language, Aspect_Focus), used for training and evaluation
- `restaurant_reviews_v2.csv` — unlabeled reviews tied to anonymized restaurant IDs, used for the leaderboard/insights
- `test_food.csv` — held-out test set for the original Kaggle competition submission

## Results

- **723 positive / 413 negative** reviews in the labeled set
- Reviews split roughly evenly across languages: **34.5% French, 33.2% English, 32.3% Arabic**
- Food Quality (339) and Delivery Speed (312) are the most frequently mentioned aspects; Burger Quality and Food Temperature are near-empty long-tail categories
- A small number of mid-rating (3-star) reviews were flagged as label-inconsistent during EDA (rated neutrally but labeled negative)

## Tech stack

Python · Transformers (mBERT, DziriBERT, XLM-RoBERTa, MARBERT) · Scikit-learn · Streamlit · Hugging Face Hub · Chart.js · Plotly

## Running locally

**Notebook**: open in Google Colab (GPU runtime recommended — training 4 transformer models on CPU is slow and memory-heavy).

**App**:
```bash
cd App
pip install -r requirements.txt
streamlit run app.py
```

## Limitations & next steps

- Arabic-subset model scores (DziriBERT/MARBERT) need validation on a larger, held-out sample before being trusted at face value
- Aspect detection is keyword-based rather than learned — works well for explicit mentions but misses implicit/paraphrased references
- No "neutral" class exists in the actual labeled data despite the original brief mentioning three sentiment categories
- Restaurant IDs are anonymized tokens with no way to cross-reference real-world restaurant names

## Author

Kamelia Zenati — [portfolio](https://cabluemgeek.github.io) · [GitHub](https://github.com/cabluemgeek)