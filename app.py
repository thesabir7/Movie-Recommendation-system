
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

@st.cache_data
def load_data(movies_path: str, ratings_path: str):
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    # Basic cleaning
    movies['title'] = movies['title'].astype(str)
    movies['genres'] = movies['genres'].astype(str).replace("(no genres listed)", "")
    return movies, ratings

@st.cache_resource
def build_item_item_sim(ratings: pd.DataFrame):
    # Pivot to user x movie matrix
    user_movie = ratings.pivot_table(index='userId', columns='movieId', values='rating')
    # Normalize by subtracting user mean to reduce user bias
    user_means = user_movie.mean(axis=1)
    norm = user_movie.sub(user_means, axis=0)
    norm = norm.fillna(0.0)

    # Item-item cosine similarity
    sim = cosine_similarity(norm.T)
    sim_df = pd.DataFrame(sim, index=norm.columns, columns=norm.columns)
    return sim_df

@st.cache_resource
def build_content_sim(movies: pd.DataFrame):
    # Use TF-IDF over genres string (e.g., "Action|Adventure|Sci-Fi" -> "Action Adventure Sci-Fi")
    corpus = movies['genres'].str.replace("|", " ", regex=False).fillna("")
    tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    X = tfidf.fit_transform(corpus)
    sim = cosine_similarity(X)
    sim_df = pd.DataFrame(sim, index=movies['movieId'], columns=movies['movieId'])
    return sim_df

def get_similar_movies(movie_id: int, sim_df: pd.DataFrame, n: int = 10):
    if movie_id not in sim_df.index:
        return pd.Index([]), pd.Series(dtype=float)
    scores = sim_df.loc[movie_id].drop(labels=[movie_id], errors='ignore')
    top = scores.sort_values(ascending=False).head(n)
    return top.index, top

def hybrid_scores(movie_id: int, sim_cf: pd.DataFrame, sim_content: pd.DataFrame, alpha: float = 0.5, n: int = 10):
    idx_cf, s_cf = get_similar_movies(movie_id, sim_cf, n=1000)  # broader candidate pool
    idx_ct, s_ct = get_similar_movies(movie_id, sim_content, n=1000)
    cand = pd.Index(idx_cf).union(idx_ct)

    # Align and combine
    s_cf_all = s_cf.reindex(cand).fillna(0.0)
    s_ct_all = s_ct.reindex(cand).fillna(0.0)
    # Normalize each similarity list to [0,1] before combining
    scaler = MinMaxScaler()
    s_cf_norm = pd.Series(scaler.fit_transform(s_cf_all.to_numpy().reshape(-1,1)).flatten(), index=cand)
    s_ct_norm = pd.Series(scaler.fit_transform(s_ct_all.to_numpy().reshape(-1,1)).flatten(), index=cand)
    combo = alpha * s_cf_norm + (1 - alpha) * s_ct_norm
    top = combo.sort_values(ascending=False).head(n)
    return top.index, top

def display_recommendations(movie_ids, movies_df: pd.DataFrame, scores=None):
    df = movies_df[movies_df['movieId'].isin(movie_ids)].copy()
    # Keep input order of movie_ids
    order = {mid:i for i, mid in enumerate(movie_ids)}
    df['rank'] = df['movieId'].map(order)
    if scores is not None:
        df['score'] = df['movieId'].map(scores.to_dict()).round(4)
    df = df.sort_values('rank').drop(columns=['rank'])
    cols = ['title', 'genres']
    if 'score' in df.columns:
        cols = ['score'] + cols
    st.dataframe(df[cols], use_container_width=True, hide_index=True)

st.title("🎬 Movie Recommendation System")
st.write("Suggests movies using **Collaborative Filtering**, **Content-Based**, or a **Hybrid** of both.")

with st.sidebar:
    st.header("⚙️ Settings")
    movies_path = st.text_input("Path to movies.csv", value="movies.csv")
    ratings_path = st.text_input("Path to ratings.csv", value="ratings.csv")
    algo = st.selectbox("Algorithm", ["Item-based CF", "Content-based (Genres)", "Hybrid"])
    n_recs = st.slider("Number of recommendations", 5, 30, 10)
    alpha = st.slider("Hybrid weight (CF vs Content)", 0.0, 1.0, 0.6, 0.05, help="Only for Hybrid: 1.0 gives full CF; 0.0 gives full Content.")

# Load data
try:
    movies, ratings = load_data(movies_path, ratings_path)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# Build similarities
try:
    sim_cf = build_item_item_sim(ratings)
    sim_content = build_content_sim(movies)
except Exception as e:
    st.error(f"Failed to build similarity matrices: {e}")
    st.stop()

# Prepare a neat label (title (year) • movieId)
def title_with_year(row):
    # Many MovieLens titles already contain (Year); just pass through.
    t = row['title']
    return f"{t} • id={row['movieId']}"

movies['label'] = movies.apply(title_with_year, axis=1)

# Search & select
st.subheader("🔎 Pick a movie you like")
# Text search for convenience
query = st.text_input("Type to search title:", "")
if query.strip():
    options = movies[movies['title'].str.contains(query, case=False, na=False)].sort_values('title')['label'].tolist()
    if options:
        selected = st.selectbox("Matching titles:", options, index=0, key="match_select")
    else:
        st.warning("No matching titles. Clear the search or try another query.")
        selected = None
else:
    selected = st.selectbox("Or choose from the list:", movies.sort_values('title')['label'].tolist(), index=0, key="full_select")

run = st.button("🔁 Recommend")

if run and selected:
    # Extract movieId
    try:
        movie_id = int(selected.split("id=")[-1])
    except Exception:
        st.error("Could not parse movieId from selection.")
        st.stop()

    if algo == "Item-based CF":
        idx, scores = get_similar_movies(movie_id, sim_cf, n=n_recs)
    elif algo == "Content-based (Genres)":
        idx, scores = get_similar_movies(movie_id, sim_content, n=n_recs)
    else:
        idx, scores = hybrid_scores(movie_id, sim_cf, sim_content, alpha=alpha, n=n_recs)

    if len(idx) == 0:
        st.info("No recommendations found for this title. Try another movie.")
    else:
        st.success(f"Top {len(idx)} recommendations ({algo}):")
        display_recommendations(idx, movies, scores)

st.markdown("---")
#st.markdown("**Data Compatibility:** Works out-of-the-box with the MovieLens dataset (e.g., `ml-latest-small`) which provides `movies.csv` and `ratings.csv`. Place both files next to `app.py`.")
