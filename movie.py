# Recommender System: Collaborative filtering on MovieLens dataset
# Includes:
# - User-based (user-user) collaborative filtering
# - Item-based (item-item) collaborative filtering
# - Evaluation (RMSE and MAE)
# Dataset: ratings.csv (ensure it's in the same directory)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Load ratings CSV file
ratings = pd.read_csv("ratings.csv")

# View basic info
print(ratings.head())
print(ratings.tail())
print(ratings.shape)
print(ratings.info())  # No nulls

# --- Train-Test Split ---
X_train, X_test = train_test_split(ratings, test_size=0.30, random_state=42)
print(X_train.shape, X_test.shape)

# --- User-based Collaborative Filtering ---

# Create User-Item matrix
user_data = X_train.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Create dummy_train (0 = already rated, 1 = not rated)
dummy_train = X_train.copy()
dummy_train['rating'] = dummy_train['rating'].apply(lambda x: 0 if x > 0 else 1)
dummy_train = dummy_train.pivot(index='userId', columns='movieId', values='rating').fillna(1)

# Create dummy_test (1 = already rated, 0 = not rated)
dummy_test = X_test.copy()
dummy_test['rating'] = dummy_test['rating'].apply(lambda x: 1 if x > 0 else 0)
dummy_test = dummy_test.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# --- User Similarity Matrix (Cosine Similarity) ---
user_similarity = cosine_similarity(user_data)
user_similarity[np.isnan(user_similarity)] = 0
print(user_similarity.shape)

# Predict ratings (dot product)
user_predicted_ratings = np.dot(user_similarity, user_data)

# Filter only unrated movies for each user
user_final_ratings = np.multiply(user_predicted_ratings, dummy_train)

# Top 5 movie recommendations for User 42
print("Top 5 User-based recommendations for user 42:")
print(user_final_ratings.iloc[42].sort_values(ascending=False).head(5))

# --- Item-based Collaborative Filtering ---

# Movie-User matrix
movie_features = X_train.pivot(index='movieId', columns='userId', values='rating').fillna(0)

# Item Similarity Matrix (Cosine)
item_similarity = cosine_similarity(movie_features)
item_similarity[np.isnan(item_similarity)] = 0
print(item_similarity.shape)

# Predict ratings using item similarity
item_predicted_ratings = np.dot(movie_features.T, item_similarity)

# Filter only unrated movies
item_final_ratings = np.multiply(item_predicted_ratings, dummy_train)

# Top 5 movie recommendations for User 42
print("Top 5 Item-based recommendations for user 42:")
print(item_final_ratings.iloc[42].sort_values(ascending=False).head(5))

# --- Evaluation (User-based) ---

# Recreate test user matrix
test_user_features = X_test.pivot(index='userId', columns='movieId', values='rating').fillna(0)
test_user_similarity = cosine_similarity(test_user_features)
test_user_similarity[np.isnan(test_user_similarity)] = 0

# Predict user ratings on test data
user_predicted_ratings_test = np.dot(test_user_similarity, test_user_features)

# Multiply with dummy_test to only keep predictions on rated movies
test_user_final_rating = np.multiply(user_predicted_ratings_test, dummy_test)

# Normalize between (0.5, 5)
X = test_user_final_rating.copy()
X = X[X > 0]
scaler = MinMaxScaler(feature_range=(0.5, 5))
scaler.fit(X)
pred = scaler.transform(X)

# RMSE and MAE (User-based)
total_non_nan = np.count_nonzero(~np.isnan(pred))
test = X_test.pivot(index='userId', columns='movieId', values='rating')

rmse = np.sqrt(((test - pred) ** 2).sum().sum() / total_non_nan)
mae = np.abs(pred - test).sum().sum() / total_non_nan

print("User-based CF Evaluation:")
print("RMSE:", rmse)
print("MAE:", mae)

# --- Evaluation (Item-based) ---

test_item_features = X_test.pivot(index='movieId', columns='userId', values='rating').fillna(0)
test_item_similarity = cosine_similarity(test_item_features)
test_item_similarity[np.isnan(test_item_similarity)] = 0

item_predicted_ratings_test = np.dot(test_item_features.T, test_item_similarity)
test_item_final_rating = np.multiply(item_predicted_ratings_test, dummy_test)

# Normalize between (0.5, 5)
X = test_item_final_rating.copy()
X = X[X > 0]
scaler = MinMaxScaler(feature_range=(0.5, 5))
scaler.fit(X)
pred = scaler.transform(X)

# RMSE and MAE (Item-based)
total_non_nan = np.count_nonzero(~np.isnan(pred))
rmse = np.sqrt(((test - pred) ** 2).sum().sum() / total_non_nan)
mae = np.abs(pred - test).sum().sum() / total_non_nan

print("Item-based CF Evaluation:")
print("RMSE:", rmse)
print("MAE:", mae)
