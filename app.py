import streamlit as st
import pandas as pd
import joblib
import difflib

# --- Load Artifacts ---
@st.cache_data
def load_artifacts():
    try:
        df = pd.read_pickle("processed_new_df.pkl")
        feature_matrix = joblib.load("weighted_features.joblib")
        model = joblib.load("knn_model_final.joblib")
        return df, feature_matrix, model

    except FileNotFoundError as e:
        st.error(f"Could not find file: {e}")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None, None


df, feature_matrix, model = load_artifacts()

# --- Recommendation Function ---
def get_recommendations(game_name, df_source, matrix_source, model_source, platform='Any', top_n=10):
    matches = df_source[df_source['name'].str.lower() == game_name.lower()]

    if not matches.empty:
        game_idx = matches.index[0]
        matched_game = matches.iloc[0]['name']
    else:
        all_game_names = df_source['name'].tolist()
        closest_matches = difflib.get_close_matches(game_name, all_game_names, n=1, cutoff=0.6)
        if not closest_matches:
            return f"No match found for '{game_name}'.", None
        matched_game = closest_matches[0]
        game_idx = df_source[df_source['name'] == matched_game].index[0]

    game_vector = matrix_source[game_idx]

    distances, indices = model_source.kneighbors(game_vector, n_neighbors=50)

    similar_indices = indices.flatten()
    similar_indices = similar_indices[similar_indices != game_idx]

    recommendations = df_source.iloc[similar_indices].copy()

    if platform != 'Any':
        platform_col = f'parent_platforms_{platform}'
        if platform_col in df_source.columns:
            recommendations = recommendations[recommendations[platform_col] == 1]

    recommendations = recommendations.head(top_n)

    if recommendations.empty:
        return f"No games found like '{matched_game}' on {platform}.", None

    def get_labels(row, prefix):
        cols = [c for c in df_source.columns if c.startswith(prefix)]
        return ' | '.join([c.replace(prefix, '') for c in cols if row[c] == 1])

    recommendations['Platforms'] = recommendations.apply(
        lambda x: get_labels(x, 'parent_platforms_'), axis=1
    )
    recommendations['Genres'] = recommendations.apply(
        lambda x: get_labels(x, 'genres_'), axis=1
    )

    return matched_game, recommendations[['name', 'rating', 'Platforms', 'Genres']]

# --- UI ---
st.set_page_config(layout="wide", page_title="Game Recommender")
st.title("🎮 Advanced Game Recommender")

if df is not None and model is not None:

    st.sidebar.header("Configuration")

    platform_cols = sorted(
        c.replace('parent_platforms_', '')
        for c in df.columns if c.startswith('parent_platforms_')
    )

    selected_platform = st.sidebar.selectbox(
        "Filter by Platform (Optional)",
        ['Any'] + platform_cols
    )

    # --- Platform-aware game list ---
    if selected_platform == 'Any':
        game_list = sorted(df['name'].unique())
    else:
        platform_col = f'parent_platforms_{selected_platform}'
        game_list = sorted(df[df[platform_col] == 1]['name'].unique())

    st.write("### Find your next favorite game")

    game_input = st.selectbox("Select a game you like:", game_list)

    if st.button("Recommend", type="primary"):

        with st.spinner("Analyzing gameplay patterns and similarity..."):
            title, recs = get_recommendations(
                game_input,
                df,
                feature_matrix,
                model,
                platform=selected_platform,
                top_n=10
            )

        if recs is None:
            st.error(title)

        else:
            st.success(f"Recommendations based on '{title}'")

            # --- Selected Game Display ---
            selected_game = df[df['name'] == title].iloc[0]

            st.markdown("## 🎯 Selected Game")

            col1, col2 = st.columns([1, 2])

            with col1:
                if pd.notna(selected_game.get('background_image')):
                    st.image(
                        selected_game['background_image'],
                        use_container_width=True
                    )

            with col2:
                st.markdown(f"### {selected_game['name']}")

                if pd.notna(selected_game.get('description')):
                    with st.expander("See full description"):
                        st.write(selected_game['description'])

            st.divider()

            # --- Recommendations Display ---
            st.markdown("## 🔥 Recommended Games")

            for _, row in recs.iterrows():

                rec_full = df[df['name'] == row['name']].iloc[0]

                col1, col2 = st.columns([1, 3])

                with col1:
                    if pd.notna(rec_full.get('background_image')):
                        st.image(
                            rec_full['background_image'],
                            use_container_width=True
                        )

                with col2:
                    st.markdown(f"### {row['name']} ⭐ {row['rating']:.2f}")

                    if pd.notna(rec_full.get('description')):
                        st.write(rec_full['description'][:300] + "...")
                        with st.expander("See more"):
                            st.write(rec_full['description'])

                    st.caption(f"Platforms: {row['Platforms']}")
                    st.caption(f"Genres: {row['Genres']}")

                st.divider()

else:
    st.warning("Missing required files. Please upload model artifacts.")
