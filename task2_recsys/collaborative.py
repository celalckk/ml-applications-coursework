import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- Load data ---
print("Loading MovieLens dataset...")
movies = pd.read_csv("task2_recsys/data/ml-latest-small/movies.csv")
ratings = pd.read_csv("task2_recsys/data/ml-latest-small/ratings.csv")

print(f"Movies : {len(movies)}")
print(f"Ratings: {len(ratings)}")
print(f"Users  : {ratings['userId'].nunique()}\n")

# --- Build user-item matrix ---
print("Building user-item rating matrix...")
user_item_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0)

print(f"Matrix shape: {user_item_matrix.shape} (users x movies)\n")


def user_based_recommendations(user_id, top_n=10, k_similar_users=20):
    """
    Find k users most similar to the target user,
    then recommend movies they liked that the target user hasn't seen.
    """
    if user_id not in user_item_matrix.index:
        return f"User {user_id} not found."

    # Compute similarity between this user and all others
    user_vector = user_item_matrix.loc[user_id].values.reshape(1, -1)
    similarities = cosine_similarity(user_vector, user_item_matrix.values)[0]

    # Get top-k most similar users (excluding self)
    sim_series = pd.Series(similarities, index=user_item_matrix.index)
    sim_series = sim_series.drop(user_id)
    top_users = sim_series.sort_values(ascending=False).head(k_similar_users)

    # Movies the target user has already rated
    seen_movies = set(ratings[ratings["userId"] == user_id]["movieId"])

    # Aggregate ratings from similar users
    scores = {}
    for other_user, sim in top_users.items():
        if sim <= 0:
            continue
        other_ratings = ratings[
            (ratings["userId"] == other_user) & (~ratings["movieId"].isin(seen_movies))
        ]
        for _, row in other_ratings.iterrows():
            scores.setdefault(row["movieId"], []).append(row["rating"] * sim)

    # Average weighted scores
    ranked = sorted(
        [(mid, sum(s) / len(s)) for mid, s in scores.items()],
        key=lambda x: x[1],
        reverse=True
    )[:top_n]
    result = pd.DataFrame([
        {
            "movieId": mid,
            "title": movies[movies["movieId"] == mid]["title"].values[0],
            "score": round(score, 3)
        }
        for mid, score in ranked
    ])
    return result
def item_based_recommendations(movie_title, top_n=10, k_similar_items=20):
    """
    For a given movie, find k most similar movies (based on user rating patterns)
    and recommend them.
    """
    matches = movies[movies["title"].str.contains(movie_title, case=False, na=False, regex=False)]
    if matches.empty:
        return f"Movie '{movie_title}' not found."

    movie_id = matches.iloc[0]["movieId"]
    if movie_id not in user_item_matrix.columns:
        return f"Movie '{matches.iloc[0]['title']}' has no ratings."

    item_matrix = user_item_matrix.T
    movie_vector = item_matrix.loc[movie_id].values.reshape(1, -1)
    similarities = cosine_similarity(movie_vector, item_matrix.values)[0]

    sim_series = pd.Series(similarities, index=item_matrix.index)
    sim_series = sim_series.drop(movie_id)
    top_items = sim_series.sort_values(ascending=False).head(top_n)

    result = pd.DataFrame([
        {
            "movieId": mid,
            "title": movies[movies["movieId"] == mid]["title"].values[0],
            "similarity": round(sim, 4)
        }
        for mid, sim in top_items.items()
    ])
    return matches.iloc[0]["title"], result
print("=" * 70)
print("COLLABORATIVE FILTERING RESULTS")
print("=" * 70)

# Test User-Based
print("\n[USER-BASED] Recommendations for user 1:")
print("-" * 70)
recs = user_based_recommendations(user_id=1, top_n=5)
if isinstance(recs, pd.DataFrame):
    for _, row in recs.iterrows():
        print(f"  {row['title'][:60]:<62s} | score={row['score']}")
else:
    print(recs)
print("\n[ITEM-BASED] Because you watched 'Toy Story':")
print("-" * 70)
source, recs = item_based_recommendations("Toy Story", top_n=5)
print(f"Source: {source}")
if isinstance(recs, pd.DataFrame):
    for _, row in recs.iterrows():
        print(f"  {row['title'][:60]:<62s} | sim={row['similarity']}")

print("\n" + "=" * 70)
print("Collaborative filtering demo complete!")
print("=" * 70)