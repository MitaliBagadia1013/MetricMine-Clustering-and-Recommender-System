# Data Mining Coursework: Clustering and Recommender Systems

Coursework implementing core data mining algorithms from scratch (no `sklearn.cluster` or `sklearn.neighbors` shortcuts for the core logic), covering K-means clustering and collaborative filtering recommender systems.

## Task 1: K-means with different distance metrics

K-means implemented from scratch and compared across three distance metrics:

- Euclidean distance
- Cosine distance
- Jaccard distance

Compares SSE (sum of squared errors), clustering accuracy (via majority-vote label assignment), and iteration count under three different stopping conditions: centroids no longer changing, SSE increasing, and a fixed iteration cap.

## Task 2: Recommender systems

Three recommendation approaches implemented and evaluated via k-fold cross-validation on a movie ratings dataset:

- **User-based collaborative filtering** — recommends based on similar users' ratings
- **Item-based collaborative filtering** — recommends based on similar items
- **Probabilistic Matrix Factorization (PMF)** — learns latent user/item factors

Also compares similarity metrics (Cosine, MSD, Pearson) and the effect of neighborhood size (K) on prediction accuracy (MAE, RMSE).

## Running it

Requires `pandas`, `numpy`, `scikit-learn` (for evaluation metrics only), and `matplotlib`. The K-means task expects a `kmeans_data.zip` file (data + labels) in the working directory; the recommender task expects a `ratings_small.csv` file (MovieLens-style user/item/rating data).

## Note

This is coursework, not a production system — implemented to demonstrate understanding of the underlying algorithms rather than to be a reusable library.
