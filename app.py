import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="🎬 Movie Recommender System",
    page_icon="🎥",
    layout="wide"
)

# --------------------------------------------------
# Load data
# --------------------------------------------------

def load_data():
    movies = pd.read_csv("data/movies.csv")
    ratings = pd.read_csv("data/ratings.csv")
    return movies, ratings

movies, ratings = load_data()

# --------------------------------------------------
# Load XGBoost model
# --------------------------------------------------
@st.cache_resource
def load_xgb():
    return joblib.load("xgb_hybrid_model.pkl")

xgb_model = load_xgb()

# --------------------------------------------------
# Helper: extract year from title
# --------------------------------------------------
def extract_year(title):
    match = re.search(r"\((\d{4})\)", title)
    return match.group(1) if match else ""

def trailer_link(title):
    year = extract_year(title)
    query = f"{title} official trailer {year}"
    return f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"

# --------------------------------------------------
# Content-based model
# --------------------------------------------------
@st.cache_data
def build_content_model(movies):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["genres"].fillna(""))
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(movies.index, index=movies["title"]).drop_duplicates()
    return cosine_sim, indices

cosine_sim, indices = build_content_model(movies)

# --------------------------------------------------
# Feature functions
# --------------------------------------------------
def user_based_score(movie_idx, liked_indices):
    return np.max(cosine_sim[liked_indices, movie_idx])

def item_based_score(movie_idx, liked_indices):
    return np.max(cosine_sim[movie_idx, liked_indices])

def popularity_score(movie_id):
    count = ratings[ratings["movieId"] == movie_id].shape[0]
    return np.log1p(count)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("⚙️ Controls")

num_recs = st.sidebar.slider("Number of recommendations", 3, 15, 5)
show_trailer = st.sidebar.checkbox("🎬 Show trailer links", value=False)

st.sidebar.divider()

search_title = st.sidebar.text_input("🔎 Search a movie title")
if search_title:
    results = movies[movies["title"].str.contains(search_title, case=False)]
    if results.empty:
        st.sidebar.error("❌ Movie not found")
    else:
        st.sidebar.success("✅ Movie found")
        st.sidebar.dataframe(results[["title"]].head(5), height=200)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    """
    <h1 style='text-align:center;'>🎬 Movie Recommender System</h1>
    <p style='text-align:center; font-size:18px;'>
    Content-Based • Popularity • XGBoost Hybrid
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# --------------------------------------------------
# Tabs
# --------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    ["🎞 Similar Movies", "🔥 Popular Movies", "🤖 Hybrid (XGBoost)"]
)

# --------------------------------------------------
# TAB 1: Content-based
# --------------------------------------------------
with tab1:
    st.subheader("🎞 Similar Movies")
    st.caption("Select **one movie** you like. Recommendations are based on **genre similarity**.")

    movie_title = st.selectbox("Choose a movie", sorted(movies["title"].unique()))

    if st.button("Recommend similar movies"):
        idx = indices[movie_title]
        scores = list(enumerate(cosine_sim[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        rec_indices = [
            i[0] for i in scores
            if movies.iloc[i[0]]["title"] != movie_title
        ][:num_recs]

        for i in rec_indices:
            title = movies.iloc[i]["title"]
            st.markdown(f"**🎬 {title}**")
            st.caption(movies.iloc[i]["genres"])
            if show_trailer:
                st.markdown(f"[▶️ Watch trailer]({trailer_link(title)})")
            st.divider()

# --------------------------------------------------
# TAB 2: Popularity-based (WITH GENRE + SORT OPTION)
# --------------------------------------------------
with tab2:
    st.subheader("🔥 Popular Movies")
    st.caption(
        "Choose one or more genres, then decide how to rank movies: "
        "**Most Rated** or **Newly Released**."
    )

    # ---- Extract all genres ----
    all_genres = sorted(
        set(g for sub in movies["genres"].dropna().str.split("|") for g in sub)
    )

    selected_genres = st.multiselect(
        "🎭 Select genres (1–3 recommended)",
        all_genres
    )

    sort_option = st.radio(
        "📊 Rank movies by",
        ["Most Rated", "Newly Released"],
        horizontal=True
    )

    # ---- Filter by genre ----
    filtered_movies = movies.copy()

    if selected_genres:
        filtered_movies = filtered_movies[
            filtered_movies["genres"].apply(
                lambda x: any(g in x for g in selected_genres) if isinstance(x, str) else False
            )
        ]

    # ---- Popularity (rating count) ----
    rating_counts = ratings.groupby("movieId")["rating"].count()

    filtered_movies = filtered_movies.merge(
        rating_counts, on="movieId", how="left"
    ).rename(columns={"rating": "rating_count"})

    filtered_movies["rating_count"] = filtered_movies["rating_count"].fillna(0)

    # ---- Extract year from title ----
    filtered_movies["year"] = (
        filtered_movies["title"]
        .str.extract(r"\((\d{4})\)")
        .astype(float)
    )

    # ---- Sorting ----
    if sort_option == "Most Rated":
        filtered_movies = filtered_movies.sort_values(
            "rating_count", ascending=False
        )
    else:  # Newly Released
        filtered_movies = filtered_movies.sort_values(
            "year", ascending=False
        )

    # ---- Display ----
    top_movies = filtered_movies.head(num_recs)

    if top_movies.empty:
        st.warning("No movies found for the selected filters.")
    else:
        for _, row in top_movies.iterrows():
            title = row["title"]
            year = int(row["year"]) if not np.isnan(row["year"]) else ""

            st.markdown(f"**🔥 {title}**")
            st.caption(row["genres"])

            if show_trailer:
                st.markdown(
                    f"[▶️ Watch trailer](https://www.youtube.com/results?search_query={title.replace(' ', '+')}+official+trailer)"

                )

            st.divider()



# --------------------------------------------------
# TAB 3: Hybrid XGBoost
# --------------------------------------------------
with tab3:
    st.subheader("🤖 Hybrid Recommendations (XGBoost)")
    st.caption(
        "Select **3–5 favorite movies**. "
        "The model ranks unseen movies using similarity and popularity."
    )
    st.info("⚠️ This XGBoost hybrid recommender is **experimental**. "
            "It is currently being improved with better feature engineering "
            "and ranking-based learning for more stable recommendations."
    )        

    liked_movies = st.multiselect(
        "Select up to 5 movies you like",
        sorted(movies["title"].unique()),
        max_selections=5
    )

    if st.button("Get XGBoost recommendations"):
        if not liked_movies:
            st.warning("Please select at least one movie.")
        else:
            liked_indices = [indices[m] for m in liked_movies]
            candidates = movies[~movies["title"].isin(liked_movies)]

            rows = []
            for _, row in candidates.iterrows():
                rows.append([
                    row["movieId"],
                    user_based_score(row.name, liked_indices),
                    item_based_score(row.name, liked_indices),
                    popularity_score(row["movieId"])
                ])

            feature_df = pd.DataFrame(
                rows,
                columns=[
                    "movieId",
                    "user_based_score",
                    "item_based_score",
                    "popularity_score"
                ]
            )

            X = feature_df[
                ["user_based_score", "item_based_score", "popularity_score"]
            ]

            feature_df["score"] = xgb_model.predict_proba(X)[:, 1]

            top = (
                feature_df.sort_values("score", ascending=False)
                .head(num_recs)
                .merge(movies, on="movieId")
            )

            for i, row in top.iterrows():
                title = row["title"]
                st.markdown(f"**🤖 {title}**")
                st.caption(row["genres"])

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("👍", key=f"like_{i}"):
                        st.success("Thank you for your participation!")

                with col2:
                    if st.button("👎", key=f"dislike_{i}"):
                        st.success("Thank you for your participation!")

                if show_trailer:
                    st.markdown(
                        f"[▶️ Watch trailer](https://www.youtube.com/results?search_query={title.replace(' ', '+')}+official+trailer)"
                    )

                st.divider()

# --------------------------------------------------
# Footer (NAME ADDED)
# --------------------------------------------------
st.markdown(
    """
    <hr>
    <p style='text-align:center; font-size:14px;'>
    Built with ❤️ by <b>Moein Gazestani</b><br>
    Streamlit • Scikit-learn • XGBoost
    </p>
    """,
    unsafe_allow_html=True
)
