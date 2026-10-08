
import pandas as pd

# --- Load data ---
print("Loading MovieLens dataset...")
movies = pd.read_csv("task2_recsys/data/ml-latest-small/movies.csv")
ratings = pd.read_csv("task2_recsys/data/ml-latest-small/ratings.csv")

print(f"Movies : {len(movies)}")
print(f"Ratings: {len(ratings)}\n")

# Merge for convenience
df = ratings.merge(movies, on="movieId")
def get_popular_movies(top_n=10, min_ratings=50):
    """Most rated movies with high average rating (weighted)."""
    stats = df.groupby(["movieId", "title"]).agg(
        avg_rating=("rating", "mean"),
        num_ratings=("rating", "count")
    ).reset_index()
    
    stats = stats[stats["num_ratings"] >= min_ratings]

    # Score = avg_rating weighted by log of num_ratings (simple heuristic)
    stats["score"] = stats["avg_rating"] * (stats["num_ratings"] ** 0.5)
    stats = stats.sort_values("score", ascending=False).head(top_n)
    return stats[["title", "avg_rating", "num_ratings", "score"]]

def also_bought_with(movie_title, top_n=10, min_rating=4.0):
    """
    Find movies that were rated highly by the same users who liked the given movie.
    """
    matches = movies[movies["title"].str.contains(movie_title, case=False, na=False, regex=False)]
    if matches.empty:
        return f"Movie '{movie_title}' not found."

    target_id = matches.iloc[0]["movieId"]

    # Find users who liked the target movie (rating >= 4.0)
    target_lovers = df[(df["movieId"] == target_id) & (df["rating"] >= min_rating)]["userId"].unique()

    # Among those users, find other highly-rated movies
    others = df[
        (df["userId"].isin(target_lovers)) &
        (df["movieId"] != target_id) &
        (df["rating"] >= min_rating)
    ]

    # Count and average
    counts = others.groupby(["movieId", "title"]).agg(
        count=("rating", "count"),
        avg=("rating", "mean")
    ).reset_index().sort_values("count", ascending=False).head(top_n)

    return matches.iloc[0]["title"], counts

def recently_viewed(user_id, top_n=10):
    """
    Return the N most recent movies a user rated/interacted with.
    Timestamps in MovieLens simulate the viewing history.
    """
    user_ratings = df[df["userId"] == user_id].sort_values("timestamp", ascending=False)
    if user_ratings.empty:
        return f"User {user_id} has no ratings."

    result = user_ratings.head(top_n)[["title", "rating", "timestamp"]]
    result["timestamp"] = pd.to_datetime(result["timestamp"], unit="s").dt.strftime("%Y-%m-%d")
    return result

def same_genre(movie_title, top_n=10, min_ratings=20):
    """Recommend top-rated movies in the same primary genre."""
    matches = movies[movies["title"].str.contains(movie_title, case=False, na=False, regex=False)]
    if matches.empty:
        return f"Movie '{movie_title}' not found."

    target_genre = matches.iloc[0]["genres"].split("|")[0]

    # Filter movies with this genre
    same = movies[movies["genres"].str.contains(target_genre, na=False)].copy()

    # Get their ratings
    merged = df[df["movieId"].isin(same["movieId"])].groupby(["movieId", "title"]).agg(
        avg=("rating", "mean"),
        count=("rating", "count")
    ).reset_index()
    merged = merged[merged["count"] >= min_ratings].sort_values("avg", ascending=False).head(top_n)

    return target_genre, merged

print("=" * 70)
print("HEURISTIC (RULE-BASED) RECOMMENDATIONS")
print("=" * 70)

print("\n[1] POPULAR MOVIES (weighted by rating count):")
print("-" * 70)
pop = get_popular_movies(top_n=5)
for _, row in pop.iterrows():
    print(f"  {row['title'][:55]:<57s} | avg={row['avg_rating']:.2f} | n={int(row['num_ratings'])}")

print("\n[2] ALSO BOUGHT WITH 'Toy Story':")
print("-" * 70)
source, recs = also_bought_with("Toy Story", top_n=5)
print(f"  Source: {source}")
for _, row in recs.iterrows():
    print(f"  {row['title'][:55]:<57s} | co-count={int(row['count'])} | avg={row['avg']:.2f}")

print("\n[3] RECENTLY VIEWED by user 1:")
print("-" * 70)
rec = recently_viewed(user_id=1, top_n=5)
if isinstance(rec, pd.DataFrame):
    for _, row in rec.iterrows():
        print(f"  {row['title'][:55]:<57s} | rating={row['rating']} | date={row['timestamp']}")
else:
    print(rec)

print("\n[4] SAME GENRE as 'Toy Story':")
print("-" * 70)
genre, recs = same_genre("Toy Story", top_n=5)
print(f"  Genre: {genre}")
for _, row in recs.iterrows():
    print(f"  {row['title'][:55]:<57s} | avg={row['avg']:.2f} | n={int(row['count'])}")

print("\n" + "=" * 70)
print("Heuristic recommendations demo complete!")
print("=" * 70)