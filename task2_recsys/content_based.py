import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
print("Loading MovieLens dataset...")
movies = pd.read_csv("task2_recsys/data/ml-latest-small/movies.csv")
ratings = pd.read_csv("task2_recsys/data/ml-latest-small/ratings.csv")

print(f"Movies : {len(movies)}")
print(f"Ratings: {len(ratings)}")
print(f"Users  : {ratings['userId'].nunique()}\n")
movies["genres_clean"] = movies["genres"].str.replace("|", " ", regex=False)
movies["genres_clean"] = movies["genres_clean"].fillna("")
print("Building TF-IDF matrix on genres...")
tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
tfidf_matrix = tfidf.fit_transform(movies["genres_clean"])
print("Computing cosine similarity...")
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print(f"Similarity matrix shape: {cosine_sim.shape}\n")


def get_content_based_recommendations(title, top_n=10):
    """Given a movie title, return top N most similar movies by genre."""
    matches = movies[movies["title"].str.contains(title, case=False, na=False, regex=False)]
    if matches.empty:
        return f"Movie '{title}' not found."

    idx = matches.index[0]
    movie_title = movies.loc[idx, "title"]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]

    movie_indices = [i[0] for i in sim_scores]
    movie_scores = [i[1] for i in sim_scores]

    result = pd.DataFrame({
        "movieId": movies["movieId"].iloc[movie_indices].values,
        "title": movies["title"].iloc[movie_indices].values,
        "genres": movies["genres"].iloc[movie_indices].values,
        "similarity": [round(s, 4) for s in movie_scores],
    })
    return movie_title, result


# --- Test ---
print("=" * 70)
print("CONTENT-BASED RECOMMENDATIONS")
print("=" * 70)

test_movies = ["Toy Story", "Matrix", "Godfather"]

for movie in test_movies:
    source, recs = get_content_based_recommendations(movie, top_n=5)
    print(f"\nBecause you watched: {source}")
    print("-" * 70)
    for _, row in recs.iterrows():
        print(f"  {row['title'][:50]:<52s} | {row['genres'][:40]:<42s} | sim={row['similarity']}")

print("\n" + "=" * 70)
print("Content-based filtering demo complete!")
print("=" * 70)