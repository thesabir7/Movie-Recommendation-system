
# 🎬 S Movie Recommendation System

A clean, deployable Movie Recommender built with **Streamlit**, supporting:
- **Item-based Collaborative Filtering**
- **Content-based** (genres via TF-IDF)
- **Hybrid** (weighted blend)

## ✅ Features
- Search a movie by typing the title
- Pick algorithm: CF / Content / Hybrid
- Adjustable number of recommendations
- Works directly with **MovieLens** (`ml-latest-small`) format: `movies.csv`, `ratings.csv`

## 📦 Project Structure
```text
streamlit-movie-recommender/
├── app.py
├── requirements.txt
└── README.md
```


## ☁️ Deploy on Streamlit Community Cloud
1. Push this folder to a **GitHub** repo.
2. Go to https://share.streamlit.io/ → New app → Select your repo/branch → `app.py`.
3. In the app sidebar, ensure paths are `movies.csv` and `ratings.csv` (if you added them to the repo).
   - If dataset is large, consider keeping only a subset or provide a public file URL and add your own downloader.

## 🧠 How it Works
- **CF (item-item)**: builds a user–movie rating matrix, normalizes by user mean, and computes item–item cosine similarity.
- **Content**: TF-IDF over `genres` text; cosine similarity between movies.
- **Hybrid**: Min–Max normalize both similarity scores and compute `alpha * CF + (1 - alpha) * Content`.

## 🔮 Ideas to Extend
- Add **poster images** (TMDB API) and external links.
- Include **user-based CF** or **matrix factorization (SVD)**.
- Enrich content with **tag genome** or plot summaries.
- Persist **user sessions** and let users rate movies during runtime.
