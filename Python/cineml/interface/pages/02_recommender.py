"""
Page 2 — Personalisation Engine (M2)
Interactive recommendation explorer: enter a user ID, compare ALS vs Two-Tower,
inspect item embeddings, and view offline evaluation metrics.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _bootstrap import ROOT, COMPONENTS  # noqa: E402

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_loader import load_als, load_movies, load_ratings, load_two_tower

st.title("Personalisation Engine")
st.caption("Module 2 — ALS & Two-Tower neural recommender")
st.divider()

# ── Load assets ────────────────────────────────────────────────────────────────
with st.spinner("Loading models…"):
    als      = load_als()
    tt_model = load_two_tower()
    movies   = load_movies()
    ratings  = load_ratings()

_title_map: dict[int, str] = {}
if movies is not None:
    _title_map = dict(zip(movies["movieId"], movies.get("title_clean", movies["title"])))


def _title(mid: int) -> str:
    return _title_map.get(mid, f"Movie #{mid}")


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_recs, tab_similar, tab_eval, tab_explore = st.tabs(
    ["🎯 Get Recommendations", "🔗 Similar Items", "📈 Evaluation", "🔭 Embedding Explorer"]
)

# ── Tab 1: Recommendations ────────────────────────────────────────────────────
with tab_recs:
    st.subheader("Get recommendations for a user")

    col_left, col_right = st.columns([1, 2])
    with col_left:
        if ratings is not None:
            uid_min, uid_max = int(ratings["userId"].min()), int(ratings["userId"].max())
        else:
            uid_min, uid_max = 1, 162_000

        user_id = st.number_input("User ID", min_value=uid_min, max_value=uid_max,
                                   value=uid_min + 100, step=1)
        n_recs  = st.slider("How many recommendations", 5, 30, 10)
        model_choice = st.radio("Model", ["ALS", "Two-Tower", "Side-by-side"],
                                 horizontal=True)
        run = st.button("Get recommendations ↗", type="primary", use_container_width=True)

    with col_right:
        if run:
            # Show what this user has already rated
            if ratings is not None:
                user_history = ratings[ratings["userId"] == user_id].nlargest(5, "rating")
                if not user_history.empty:
                    st.caption("User's top-rated movies (from training set)")
                    history_df = user_history[["movieId", "rating"]].copy()
                    history_df["title"] = history_df["movieId"].map(_title)
                    st.dataframe(
                        history_df[["title", "rating"]].rename(columns={"title": "Movie"}),
                        hide_index=True, use_container_width=True, height=180,
                    )

    if run:
        st.divider()

        def _get_als_recs(uid, n):
            if als is None:
                return []
            try:
                return als.recommend(uid, n)
            except KeyError:
                return []

        def _make_rec_df(recs):
            if not recs:
                return pd.DataFrame(columns=["title", "score"])
            df = pd.DataFrame(recs, columns=["movie_id", "score"])
            df["title"] = df["movie_id"].map(_title)
            df["score"] = df["score"].round(4)
            return df[["title", "score"]]

        if model_choice in ("ALS", "Side-by-side"):
            als_recs = _get_als_recs(user_id, n_recs)
            als_df   = _make_rec_df(als_recs)

        if model_choice == "Two-Tower":
            st.info("Two-Tower inference requires the trained model. "
                    "Run `python module2_recommender/train.py --model two-tower`")

        if model_choice == "ALS":
            if als_df.empty:
                st.warning(f"User {user_id} not in ALS training set (cold start). "
                            "Try a different user ID.")
            else:
                st.subheader(f"ALS — top {n_recs} for user {user_id}")
                c1, c2 = st.columns([1, 1])
                with c1:
                    st.dataframe(als_df, hide_index=True, use_container_width=True)
                with c2:
                    fig = px.bar(als_df.head(10), x="score", y="title",
                                 orientation="h", title="Score distribution",
                                 color="score",
                                 color_continuous_scale="Purples")
                    fig.update_layout(yaxis={"autorange": "reversed"},
                                      coloraxis_showscale=False,
                                      height=320, margin=dict(t=36,b=0,l=0,r=0))
                    st.plotly_chart(fig, use_container_width=True)

        elif model_choice == "Side-by-side":
            c1, c2 = st.columns(2)
            with c1:
                st.caption("ALS (matrix factorisation)")
                if als_df.empty:
                    st.warning("User not in ALS training set.")
                else:
                    st.dataframe(als_df, hide_index=True, use_container_width=True)
            with c2:
                st.caption("Two-Tower (neural)")
                st.info("Train the Two-Tower model to see comparison results here.")


# ── Tab 2: Similar items ──────────────────────────────────────────────────────
with tab_similar:
    st.subheader("Find similar movies")

    movie_options = list(_title_map.items()) if _title_map else [(1, "Movie #1")]
    movie_options_display = {f"{t} (#{mid})": mid for mid, t in movie_options[:2000]}

    selected = st.selectbox("Select a movie", list(movie_options_display.keys()),
                             index=min(42, len(movie_options_display) - 1))
    n_sim = st.slider("How many similar items", 5, 20, 10, key="sim_slider")
    sim_btn = st.button("Find similar ↗", type="primary")

    if sim_btn:
        movie_id = movie_options_display[selected]
        if als is None:
            st.info("Train the ALS model first: `python module2_recommender/train.py --model als`")
        else:
            try:
                sims = als.similar_items(movie_id, n_sim)
                sim_df = pd.DataFrame(sims, columns=["movie_id", "similarity"])
                sim_df["title"] = sim_df["movie_id"].map(_title)
                sim_df["similarity"] = sim_df["similarity"].round(4)

                c1, c2 = st.columns([1, 1])
                with c1:
                    st.dataframe(sim_df[["title", "similarity"]],
                                 hide_index=True, use_container_width=True)
                with c2:
                    fig = go.Figure(go.Bar(
                        x=sim_df["similarity"], y=sim_df["title"],
                        orientation="h", marker_color="#10b981",
                    ))
                    fig.update_layout(yaxis={"autorange": "reversed"},
                                      title=f"Similar to: {selected.split(' (#')[0]}",
                                      height=320, margin=dict(t=36,b=0,l=0,r=0))
                    st.plotly_chart(fig, use_container_width=True)
            except KeyError:
                st.warning("Movie not found in ALS model. Try another title.")


# ── Tab 3: Evaluation metrics ─────────────────────────────────────────────────
with tab_eval:
    st.subheader("Offline evaluation")

    if ratings is not None and als is not None:
        with st.spinner("Running evaluation on held-out test set (may take 30s)…"):
            from module2_recommender.evaluation import evaluate_recommender, print_report

            split = int(len(ratings) * 0.9)
            train_df = ratings.iloc[:split]
            test_df  = ratings.iloc[split:].sample(min(2000, len(ratings) - split), random_state=42)

            results = evaluate_recommender(
                recommend_fn=lambda uid, k: [m for m, _ in als.recommend(uid, k)],
                test_df=test_df,
                train_df=train_df,
                k=10,
            )

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("NDCG@10",    f"{results['ndcg@10']:.4f}")
        col2.metric("MAP@10",     f"{results['map@10']:.4f}")
        col3.metric("Hit@10",     f"{results['hit@10']:.4f}")
        col4.metric("Novelty",    f"{results['novelty']:.2f}")
        col5.metric("Coverage",   f"{results['catalogue_coverage']:.2%}")

        # Radar chart
        categories = ["NDCG@10", "MAP@10", "Hit@10", "Novelty (norm)", "Coverage"]
        vals = [
            results["ndcg@10"],
            results["map@10"],
            results["hit@10"],
            min(results["novelty"] / 20, 1.0),
            results["catalogue_coverage"],
        ]
        fig = go.Figure(go.Scatterpolar(
            r=vals + [vals[0]], theta=categories + [categories[0]],
            fill="toself", fillcolor="rgba(99,102,241,0.2)",
            line_color="#6366f1",
        ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                           title="ALS model radar", height=350)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Train the ALS model and ensure ratings data is available to run evaluation.")

    with st.expander("Metric definitions"):
        st.markdown("""
| Metric | Description | Interpretation |
|--------|-------------|----------------|
| NDCG@10 | Normalised discounted cumulative gain at 10 | Quality of ranking; penalises relevant items ranked lower |
| MAP@10 | Mean average precision at 10 | Precision weighted by rank position |
| Hit@10 | % of users with ≥1 relevant item in top 10 | Recall proxy |
| Novelty | Mean self-information of recommended items | Higher = more surprising / less popular items |
| Coverage | % of catalogue ever recommended | Diversity health — low coverage means filter bubble |
""")


# ── Tab 4: Embedding explorer ─────────────────────────────────────────────────
with tab_explore:
    st.subheader("Item embedding space (ALS)")
    st.caption("Projects 128-dim item embeddings to 2D via PCA to show genre clustering.")

    if als is not None and movies is not None:
        try:
            from sklearn.decomposition import PCA

            item_factors = als.model.item_factors  # (n_items, d)
            item_ids_enc = list(als._item_dec.keys())
            item_ids_orig = [als._item_dec[i] for i in item_ids_enc]

            # Subsample for speed
            n_show = min(2000, len(item_ids_enc))
            rng = np.random.default_rng(42)
            idx  = rng.choice(len(item_ids_enc), size=n_show, replace=False)
            factors_sample = item_factors[idx]

            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(factors_sample)

            ids_sample = [item_ids_orig[i] for i in idx]
            title_sample = [_title(mid) for mid in ids_sample]

            genre_map = {}
            if "genres" in movies.columns:
                for _, row in movies.iterrows():
                    g = row["genres"]
                    genre_map[row["movieId"]] = (g[0] if isinstance(g, list) and g else "Unknown")

            genre_sample = [genre_map.get(mid, "Unknown") for mid in ids_sample]

            embed_df = pd.DataFrame({
                "x": coords[:, 0], "y": coords[:, 1],
                "title": title_sample, "genre": genre_sample,
            })

            fig = px.scatter(embed_df, x="x", y="y", color="genre",
                              hover_name="title", opacity=0.65,
                              title=f"Item embedding space ({n_show} movies, PCA 2D)",
                              height=500)
            fig.update_traces(marker=dict(size=4))
            fig.update_layout(legend=dict(title="Genre"), margin=dict(t=40,b=0,l=0,r=0))
            st.plotly_chart(fig, use_container_width=True)

            var = pca.explained_variance_ratio_
            st.caption(f"PCA explains {var[0]:.1%} + {var[1]:.1%} = {sum(var):.1%} of variance. "
                       "Clusters indicate ALS learned genre-coherent embeddings.")
        except Exception as e:
            st.warning(f"Could not project embeddings: {e}")
    else:
        st.info("Train the ALS model to visualise its item embedding space.")
