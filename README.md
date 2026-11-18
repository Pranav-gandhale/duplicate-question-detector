# Duplicate Question Detector (Project P32)

A Streamlit application that searches a small forum Q&A knowledge base for
semantically similar questions, then synthesizes a cited answer from existing
threads. It includes PII redaction and content moderation, an embedding-space
plot, FAQ clustering, and a downloadable PNG infographic.

## Features

- Sentence-BERT (`all-MiniLM-L6-v2`) embeddings with cosine similarity.
- A scikit-learn TF-IDF fallback, plus a built-in fallback when optional ML
  dependencies are not installed. The fallback keeps the demo and evaluation
  usable offline, but does not provide the semantic quality of Sentence-BERT.
- OpenAI answer merging and FAQ summaries when `OPENAI_API_KEY` is configured;
  deterministic, extractive fallbacks otherwise.
- Moderation and redaction for email addresses, phone numbers, SSNs, common API
  keys/bearer tokens, IPv4/IPv6 addresses, and a small profanity list. Profanity is blocked;
  sensitive values are replaced before indexing or search.
- Per-session question ingestion; submitted questions and answers do not enter
  the shared base index or another user's session.
- PCA / Plotly visualization and K-Means theme summaries.
- A 20-query Precision@3 evaluation that writes `evaluation_results.csv`.

## Setup

Use Python 3.10 or newer. From this directory:

```powershell
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Set an OpenAI key only if you want generated (rather than fallback) answer
merging and FAQ summaries. When configured, sanitized questions and retrieved
forum content are sent to OpenAI for those features:

```powershell
$env:OPENAI_API_KEY = "your-key"
# Optional:
$env:OPENAI_MODEL = "gpt-4o-mini"
```

The Sentence-BERT model is downloaded by `sentence-transformers` on first use.
If the model cannot be loaded, the app logs the issue and uses TF-IDF where
available. Do not put credentials or unredacted private data in the forum JSON.

## Run evaluation

```powershell
python evaluate.py
```

The script prints each query's Precision@3, the overall mean, and writes
`evaluation_results.csv` in the project directory. Precision@3 is the count of
retrieved relevant thread IDs divided by three.

## Launch the application

```powershell
streamlit run app.py
```

The app has tabs for search and answer synthesis, the interactive embedding
map, moderated per-session ingestion, and FAQ infographic generation. The FAQ
tab saves `faq_infographic.png` in the project directory and offers a download.

## Project layout

```text
app.py
evaluate.py
requirements.txt
data/forum_qa.json
src/
  moderation.py
  retriever.py
  merger.py
  visualizer.py
```
