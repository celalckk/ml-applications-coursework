# Task 2: Recommender System

A recommender system for a movie streaming service (e-commerce / media analogy), built with classic RecSys approaches plus heuristic rules.

# Implemented Approaches

| # | Type | File | Description |
|---|------|------|-------------|
| 1 | Content-Based | `content_based.py` | TF-IDF on genres + cosine similarity |
| 2 | Collaborative (User) | `collaborative.py` | User-user cosine similarity, weighted ratings |
| 3 | Collaborative (Item) | `collaborative.py` | Item-item cosine similarity |
| 4 | Heuristic: Popularity | `heuristics.py` | Top-rated + most-rated movies (weighted) |
| 5 | Heuristic: Also Bought | `heuristics.py` | Co-occurrence among users who liked a movie |
| 6 | Heuristic: Recently Viewed | `heuristics.py` | User's most recent interactions |
| 7 | Heuristic: Same Genre | `heuristics.py` | Top-rated in same primary genre |

**Dataset:** MovieLens `ml-latest-small` — 100,000 ratings, 9,000 movies, 600 users.

---

# Part I: Approach Details

## 1. Content-Based Filtering

**How it works:** Each movie is represented as a TF-IDF vector over its genres. Cosine similarity is computed between all movie pairs. Given a movie the user liked, we recommend the top-N most similar movies.

**Advantages:**
- Works with **no user history** (only item metadata).
- Explainable: "because you watched X, here are similar movies".
- No cold-start for new users.

**Limitations:**
- Only uses genres (weak signal) — quality would improve with plot summaries, cast, director.
- Cannot discover cross-genre preferences.

## 2. Collaborative Filtering

**How it works:**
- **User-Based:** Find k users most similar to the target user (by rating patterns), recommend movies they liked that the target user has not seen.
- **Item-Based:** For a given movie, find the k most similar movies (by user rating patterns) and recommend them.

**Advantages:**
- Captures **latent preferences** that are not visible from genres.
- Improves over time as more ratings are collected.

**Limitations:**
- **Cold start** for new users/items.
- **Sparsity:** User-item matrix is >99% empty.
- **Scalability:** O(users × items) similarity computation.

## 3. Heuristic Approaches (Baseline)

Used as fallback when the ML model has no signal (new user, no data):

- **Popularity:** Best weighted score of avg_rating × √(num_ratings).
- **Also Bought With:** Co-occurrence counting among users who liked a target movie.
- **Recently Viewed:** Chronological list of the user's last interactions.
- **Same Genre:** Top-rated movies in the same primary genre.

**Advantages:**
- Zero training required.
- Always available (no cold start).
- Easy to explain to business stakeholders.

**Limitations:**
- No personalization.
- Can create a "popularity bubble" (same items shown to everyone).

---

# Part II: Problems, Solutions and Discussions

## a) Core RecSys Problems

### Problem 1: Cold Start

**What it is:** New users (no history) or new items (no ratings) cannot be recommended/recommended-to.

**Solution:** Short onboarding survey — ask the user to rate 5–10 popular movies at signup.

**Problem with this solution:** Increased early user churn (users abandon long signup flows).

**Discussion:** Best practice is **hybrid onboarding** — show a "skip" button, use implicit signals (clicked trailers, browsing time) in parallel. Netflix and Spotify use this strategy: mandatory signup steps are minimized, while implicit signals immediately feed the model.

### Problem 2: Ethical Issues

**What it is:** Recommendation systems can create **filter bubbles**, amplify **polarizing content**, or push **addictive** usage patterns. Example: "People also bought this" can recommend harmful or extremist content.

**Solution:** Add **diversity** and **serendipity** constraints to the ranking. Filter out categories flagged by moderation. Introduce "content you might not agree with" slots.

**Problem with this solution:** Diversity can reduce click-through rate (CTR) in the short term, hurting business KPIs.

**Discussion:** Best practice is to treat **fairness and diversity as long-term business metrics**, not just short-term CTR. Track them explicitly in A/B tests, with separate dashboards. Regulatory pressure (EU DSA, AI Act) also pushes platforms to make these metrics visible.

### Problem 3: Popularity Bias

**What it is:** The system over-recommends already-popular items, making it hard for new or niche content to surface.

**Solution:** Apply **inverse popularity weighting** during ranking, or cap the number of times a single item can be recommended per user per day.

**Problem with this solution:** Popular items usually have high CTR, so the loss may reduce short-term revenue.

**Discussion:** Use a **multi-objective ranker** — business KPIs plus content-diversity constraints. Amazon and YouTube use such multi-objective setups.

## b) Production Problems

### Problem: Slow Response Time

**What it is:** Real-time similarity computation (e.g., user-user cosine on the full matrix) is too slow for online serving.

**Solution:** Pre-compute user and item embeddings offline, then use **Approximate Nearest Neighbor (ANN)** search (e.g., FAISS, ScaNN) at request time.

**Problem with this solution:** Requires an offline pipeline, more infrastructure, and periodic re-training.

**Discussion:** Two-stage architecture is the industry standard:
1. **Candidate generation** (fast, ~100s of items) — using ANN.
2. **Ranking** (slow but precise) — a heavier model that scores the candidates.

### Problem: Training-Serving Skew

**What it is:** Features computed differently in training vs. serving pipelines, degrading real-world performance.

**Solution:** Use a **feature store** (Feast, Tecton) shared between offline and online pipelines.

**Problem with this solution:** Additional infrastructure cost and engineering effort.

**Discussion:** For small products, the same code should be reused for both pipelines until the team grows. Only introduce a feature store when the friction is proven.

---

# Part III: Evaluation and Monitoring

## ML Metrics

- **RMSE / MAE:** Accuracy of rating predictions (Collaborative Filtering).
- **Precision@K / Recall@K:** Fraction of relevant items in the top-K recommendations.
- **MAP@K (Mean Average Precision):** Ranking quality, standard in RecSys.
- **NDCG@K (Normalized Discounted Cumulative Gain):** Position-aware ranking metric.
- **Coverage:** Fraction of the catalog recommended at least once.
- **Diversity / Novelty:** How varied / non-obvious the recommendations are.

## Business Metrics

- **Click-Through Rate (CTR):** Clicks / impressions on recommendations.
- **Conversion Rate:** Purchases / clicks.
- **Session Time:** Average user time on the platform.
- **Retention (Day-7, Day-30):** Users returning after a period.
- **Revenue per User (ARPU):** Direct business impact.
- **Churn Rate:** Users lost per period.

## Infrastructure Metrics

- **Latency (p50, p95, p99):** Time to serve a recommendation.
- **Throughput (QPS):** Requests per second handled.
- **GPU / CPU Utilization:** Cost and capacity planning.
- **Cache Hit Rate:** Efficiency of pre-computed results.
- **Error Rate:** Failed requests.

## How to Know the Model is Working

- **Offline:** RMSE / Precision@K above baseline, measured on a held-out test set.
- **Online:** A/B test against the previous algorithm; monitor CTR and session time.
- **Long-term:** Track diversity / fairness dashboards to catch filter bubbles.
- **Infrastructure:** Latency within SLA, error rate close to zero.

## Possible Improvements to the Existing Model

1. **Add more item features** (plot, cast, director, tags) to the content-based model.
2. **Use matrix factorization (SVD / ALS)** instead of raw cosine similarity for collaborative filtering.
3. **Neural RecSys** (e.g., Neural Collaborative Filtering, two-tower models).
4. **Implicit feedback** (views, clicks) in addition to explicit ratings.
5. **Real-time updates** of user embeddings based on recent interactions.
6. **Multi-armed bandit** for exploration / exploitation balance.
7. **Two-stage architecture:** candidate generation (ANN) + ranking (GBDT / NN).
8. **Business-aware ranking:** incorporate margin, availability, promotions into the score.