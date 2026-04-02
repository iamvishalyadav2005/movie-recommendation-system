import streamlit as st
import numpy as np
import pandas as pd
import requests
import re

API_KEY = st.secrets["API_KEY"]

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# 🔥 CLEAN TITLE
# =========================
def clean_title(title):
    title = re.sub(r"\(\d{4}\)", "", title)
    title = re.sub(r"[^a-zA-Z0-9 ]", " ", title)
    return title.strip()

# =========================
# 🎬 FETCH POSTER (FAST + CLEAN)
# =========================
@st.cache_data
def fetch_poster(movie_title):
    try:
        cleaned_title = clean_title(movie_title)
        url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={cleaned_title}"
        data = requests.get(url).json()

        if data.get("results"):
            for movie in data["results"]:
                if movie.get("poster_path"):
                    return "https://image.tmdb.org/t/p/w342" + movie["poster_path"]
    except:
        pass

    return "https://dummyimage.com/300x450/1c1c1c/ffffff&text=No+Poster"

# =========================
# 🎥 DISPLAY (NETFLIX STYLE)
# =========================
def display_movies(recs):
    num_cols = 5

    for i in range(0, len(recs), num_cols):
        cols = st.columns(num_cols)

        for j, col in enumerate(cols):
            if i + j < len(recs):
                row = recs.iloc[i + j]

                with col:
                    poster_url = fetch_poster(row['title'])

                    st.image(poster_url, use_container_width=True)

                    st.markdown(
                        f"<div class='movie-title'>{row['title']}</div>",
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        f"<div class='movie-rating'>⭐ {row['avg_rating']:.1f}</div>",
                        unsafe_allow_html=True
                    )

# =========================
# 📊 DATA LOADING
# =========================
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")

    movies['genres'] = movies['genres'].fillna('(no genres listed)')

    movie_stats = ratings.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
    movie_stats.columns = ['movieId', 'avg_rating', 'rating_count']

    movies_full = movies.merge(movie_stats, on='movieId', how='left')
    movies_full[['avg_rating', 'rating_count']] = movies_full[['avg_rating', 'rating_count']].fillna(0)

    movies_full['popularity'] = movies_full['rating_count']
    pop = movies_full['popularity'].values.astype(float)
    movies_full['popularity_norm'] = (pop - pop.min()) / (pop.max() - pop.min() + 1e-8)

    return movies_full, ratings

# =========================
# 🧠 CONTENT MODEL
# =========================
@st.cache_resource
def build_content_similarity(movies_full):
    genres_clean = movies_full['genres'].str.replace('|', ' ', regex=False)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(genres_clean)
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

# =========================
# 🤝 COLLAB FILTERING
# =========================
@st.cache_resource
def build_cf_similarity(movies_full, ratings):
    user_item = ratings.pivot_table(index='movieId', columns='userId', values='rating').fillna(0.0)
    user_item_aligned = user_item.reindex(movies_full['movieId']).fillna(0.0)
    return cosine_similarity(user_item_aligned.values, user_item_aligned.values)

# =========================
# 🔧 HELPERS
# =========================
def get_title_index_map(movies_full):
    unique_titles = movies_full.reset_index().drop_duplicates(subset=['title'], keep='first')
    return pd.Series(unique_titles['index'].values, index=unique_titles['title'])

def get_top_popular(movies_full, n=10, min_ratings=50):
    df = movies_full[movies_full['rating_count'] >= min_ratings]
    df = df.sort_values(by=['avg_rating', 'rating_count'], ascending=[False, False])
    return df.head(n)[['title', 'genres', 'avg_rating', 'rating_count']]

def recommend_from_similarity(movies_full, sim_matrix, title_to_index, title, n=10):
    if title not in title_to_index:
        return None, "Movie not found"

    idx = int(title_to_index[title])
    sim_scores = list(enumerate(sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+21]

    movie_indices = [i[0] for i in sim_scores]
    recs = movies_full.iloc[movie_indices].drop_duplicates(subset=['title']).head(n)

    return recs[['title', 'genres', 'avg_rating', 'rating_count']], None

def recommend_hybrid(movies_full, content_sim, cf_sim, title_to_index, title, n=10):
    if title not in title_to_index:
        return None, "Movie not found"

    idx = int(title_to_index[title])

    scores = (
        0.6 * content_sim[idx] +
        0.3 * cf_sim[idx] +
        0.1 * movies_full['popularity_norm'].values
    )

    scores[idx] = -1
    indices = np.argsort(scores)[::-1][:n+20]

    recs = movies_full.iloc[indices].drop_duplicates(subset=['title']).head(n)

    return recs[['title', 'genres', 'avg_rating', 'rating_count']], None

# =========================
# 🚀 MAIN APP
# =========================
def main():

    st.set_page_config(page_title="Movie Recommender", layout="wide")

    # 🎨 PROFESSIONAL UI
    st.markdown("""
<style>

/* App Background */
.stApp {
    background: linear-gradient(to bottom, #141414, #000000);
    color: white;
}

/* Header Styling */
.netflix-title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #E50914;
    margin-bottom: 0;
}

.netflix-subtitle {
    text-align: center;
    color: #bbbbbb;
    margin-top: 0;
    margin-bottom: 20px;
}

/* Movie Card */
.movie-card {
    text-align: center;
}

/* Movie Title */
.movie-title {
    font-size: 14px;
    font-weight: 600;
    margin-top: 6px;
}

/* Rating */
.movie-rating {
    font-size: 12px;
    color: #bbbbbb;
}

/* Poster Styling */
.stImage img {
    border-radius: 12px;
    transition: all 0.3s ease;
    border: 2px solid transparent;
}

/* Hover Effect (Netflix Style) */
.stImage img:hover {
    transform: scale(1.1);
    border: 2px solid #E50914;
    box-shadow: 0px 4px 20px rgba(229, 9, 20, 0.6);
}

/* Remove spacing */
div[data-testid="column"] {
    padding: 4px !important;
}

</style>
""", unsafe_allow_html=True)

    st.markdown("<div class='netflix-title'>Smart Movie Recommender</div>", unsafe_allow_html=True)
    st.markdown("<div class='netflix-subtitle'>AI-powered personalized movie suggestions 🎬</div>", unsafe_allow_html=True)

    with st.spinner("Loading..."):
        movies_full, ratings = load_data()
        content_sim = build_content_similarity(movies_full)
        cf_sim = build_cf_similarity(movies_full, ratings)
        title_to_index = get_title_index_map(movies_full)

    st.sidebar.title("⚙️ Settings")

    mode = st.sidebar.selectbox("Recommendation Type",
        ["Popularity-based", "Content-based", "Collaborative Filtering", "Hybrid"])

    n_recs = st.sidebar.slider("Number of Movies", 5, 20, 10)

    if mode == "Popularity-based":
        st.markdown("## 🔥 Popular Movies")
        recs = get_top_popular(movies_full, n=n_recs)
        display_movies(recs)

    else:
        movie_input = st.text_input("Search a movie")

        if movie_input:
            matches = [t for t in movies_full['title'] if movie_input.lower() in t.lower()]

            if matches:
                selected = st.selectbox("Select movie", matches)

                if st.button("Recommend"):
                    if mode == "Content-based":
                        recs, _ = recommend_from_similarity(movies_full, content_sim, title_to_index, selected, n_recs)
                    elif mode == "Collaborative Filtering":
                        recs, _ = recommend_from_similarity(movies_full, cf_sim, title_to_index, selected, n_recs)
                    else:
                        recs, _ = recommend_hybrid(movies_full, content_sim, cf_sim, title_to_index, selected, n_recs)

                    st.markdown(f"## 🎯 Recommendations for {selected}")
                    display_movies(recs)

            else:
                st.warning("No movie found")

if __name__ == "__main__":
    main()
