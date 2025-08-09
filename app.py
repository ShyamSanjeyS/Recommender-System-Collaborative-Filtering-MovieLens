# movie_recommender_streamlit.py
# Streamlit web app for User-based and Item-based Collaborative Filtering
# Place ratings.csv in the same folder as this file and run:
#    pip install streamlit pandas numpy scikit-learn
#    streamlit run movie_recommender_streamlit.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Movie Recommender", layout="wide")

@st.cache_data
def load_ratings(path="ratings.csv"):
    df = pd.read_csv(path)
    return df

@st.cache_data
def train_data_split(ratings_df, test_size=0.3, random_state=42):
    train, test = train_test_split(ratings_df, test_size=test_size, random_state=random_state)
    return train.reset_index(drop=True), test.reset_index(drop=True)

@st.cache_data
def build_user_item_matrix(df):
    return df.pivot(index='userId', columns='movieId', values='rating').fillna(0)

@st.cache_data
def build_movie_user_matrix(df):
    return df.pivot(index='movieId', columns='userId', values='rating').fillna(0)

@st.cache_data
def user_similarity_matrix(user_item_matrix):
    sim = cosine_similarity(user_item_matrix)
    sim = np.nan_to_num(sim)
    return pd.DataFrame(sim, index=user_item_matrix.index, columns=user_item_matrix.index)

@st.cache_data
def item_similarity_matrix(movie_user_matrix):
    sim = cosine_similarity(movie_user_matrix)
    sim = np.nan_to_num(sim)
    # movie_user_matrix.index contains movieIds
    return pd.DataFrame(sim, index=movie_user_matrix.index, columns=movie_user_matrix.index)

# Predict using user-based CF
def predict_user_based(user_item, user_sim):
    # user_sim: DataFrame indexed by userId
    # user_item: DataFrame indexed by userId columns movieId
    # prediction = (S * R) / sum(abs(S))
    sim = user_sim.values
    ratings = user_item.values
    with np.errstate(divide='ignore', invalid='ignore'):
        pred = sim.dot(ratings) / (np.abs(sim).sum(axis=1)[:, np.newaxis] + 1e-8)
    pred_df = pd.DataFrame(pred, index=user_item.index, columns=user_item.columns)
    return pred_df

# Predict using item-based CF
def predict_item_based(user_item, item_sim):
    # item_sim: DataFrame indexed by movieId
    # user_item: DataFrame indexed by userId columns movieId
    sim = item_sim.values
    ratings = user_item.values
    with np.errstate(divide='ignore', invalid='ignore'):
        pred = ratings.dot(sim) / (np.abs(sim).sum(axis=1)[np.newaxis, :] + 1e-8)
    pred_df = pd.DataFrame(pred, index=user_item.index, columns=user_item.columns)
    return pred_df

# Helper: get recommendations for a user
def get_top_n_recommendations(predictions_df, train_df, user_id, n=5):
    # predictions_df: DataFrame indexed by userId columns movieId
    if user_id not in predictions_df.index:
        return pd.DataFrame(columns=['movieId','score'])
    user_pred = predictions_df.loc[user_id].copy()
    # remove movies already rated in train
    rated = train_df[train_df['userId'] == user_id]['movieId'].unique()
    user_pred.loc[rated] = -np.inf
    top_n = user_pred.sort_values(ascending=False).head(n)
    return top_n.reset_index().rename(columns={user_id: 'score', 'movieId': 'movieId'})

# Evaluation on test set: compute RMSE and MAE between predicted and actual for the rated pairs
def evaluate(predictions_df, test_df):
    # Build test matrix aligned with predictions_df
    test_matrix = test_df.pivot(index='userId', columns='movieId', values='rating')
    # Iterate over non-null entries in test_matrix
    squared_errors = []
    abs_errors = []
    count = 0
    for u in test_matrix.index:
        if u not in predictions_df.index:
            continue
        for m in test_matrix.columns:
            true = test_matrix.at[u, m]
            if pd.isna(true):
                continue
            if m not in predictions_df.columns:
                continue
            pred = predictions_df.at[u, m]
            if pd.isna(pred):
                continue
            squared_errors.append((pred - true) ** 2)
            abs_errors.append(abs(pred - true))
            count += 1
    if count == 0:
        return None, None
    rmse = np.sqrt(np.mean(squared_errors))
    mae = np.mean(abs_errors)
    return rmse, mae

# ---------------------
# Streamlit UI
# ---------------------
st.title("🎬 Movie Recommender — User-based & Item-based CF")

ratings = load_ratings()
st.sidebar.markdown("### Data info")
st.sidebar.write(f"Total ratings: {len(ratings)}")
st.sidebar.write(f"Unique users: {ratings['userId'].nunique()}")
st.sidebar.write(f"Unique movies: {ratings['movieId'].nunique()}")

# Train-test split controls
st.sidebar.markdown("---")
test_size = st.sidebar.slider("Test set fraction", 0.05, 0.5, 0.30, 0.05)
random_state = st.sidebar.number_input("Random seed", min_value=0, value=42)

train_df, test_df = train_data_split(ratings, test_size=float(test_size), random_state=int(random_state))

# Build matrices
user_item_train = build_user_item_matrix(train_df)
movie_user_train = build_movie_user_matrix(train_df)

# Similarities
user_sim = user_similarity_matrix(user_item_train)
item_sim = item_similarity_matrix(movie_user_train)

# Predictions
user_pred = predict_user_based(user_item_train, user_sim)
item_pred = predict_item_based(user_item_train, item_sim)

# Select user and system
st.sidebar.markdown("---")
system = st.sidebar.selectbox("Recommendation system", ["User-based", "Item-based"]) 
user_list = sorted(ratings['userId'].unique())
selected_user = st.sidebar.selectbox("Pick a userId", user_list, index=0)
num_rec = st.sidebar.slider("Number of recommendations", 1, 20, 5)

col1, col2 = st.columns([2,3])
with col1:
    st.header("Top recommendations")
    if system == "User-based":
        recs = get_top_n_recommendations(user_pred, train_df, selected_user, n=num_rec)
    else:
        recs = get_top_n_recommendations(item_pred, train_df, selected_user, n=num_rec)

    if recs.empty:
        st.write("No recommendations available for this user (user might be new or has rated everything).")
    else:
        # join with movie titles if available
        if 'title' in ratings.columns:
            movies = ratings[['movieId','title']].drop_duplicates().set_index('movieId')
            recs['title'] = recs['movieId'].map(movies['title'])
        recs = recs.rename(columns={selected_user: 'score'}).reset_index(drop=True)
        st.dataframe(recs.rename(columns={0:'movieId'}), use_container_width=True)

with col2:
    st.header("System performance on test set")
    st.write("The app evaluates RMSE and MAE on the current train/test split (predictions only for users and movies present in training).")
    if st.button("Run evaluation"):
        with st.spinner('Computing evaluation...'):
            if system == "User-based":
                rmse, mae = evaluate(user_pred, test_df)
            else:
                rmse, mae = evaluate(item_pred, test_df)
        if rmse is None:
            st.write("Not enough overlap between training and test data to evaluate.")
        else:
            st.metric("RMSE", f"{rmse:.4f}")
            st.metric("MAE", f"{mae:.4f}")

# Show a few sample ratings and allow exploring a movie
st.markdown("---")
st.subheader("Explore dataset samples")
st.write(train_df.sample(n=min(10, len(train_df))))

st.markdown("---")
st.caption("Helpful notes: This app is a simple collaborative filtering demo. For production, use sparse matrix optimizations, kNN, regularization, and handle cold-start users/movies.")
