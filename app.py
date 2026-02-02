import streamlit as st
import pandas as pd
import joblib
import difflib

# --- Load Artifacts ---
@st.cache_data
def load_artifacts():
    try:
        # Load the Dataframe
        df = pd.read_pickle("processed_df.pkl")
        
        # Load the Weighted Matrix (Numbers + Text + Categories)
        feature_matrix = joblib.load("weighted_features.joblib")
        
        # Load the Model
        model = joblib.load("knn_model_final.joblib")
        
        return df, feature_matrix, model
    except FileNotFoundError as e:
        # This will show a clear error if your files are missing
        st.error(f"Could not find file: {e}. Please ensure 'processed_df.pkl', 'weighted_features.joblib', and 'knn_model_final.joblib' are in the same folder as app.py.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None, None

df, feature_matrix, model = load_artifacts()

# --- Helper Functions ---
def get_recommendations(game_name, df_source, matrix_source, model_source, platform='Any', top_n=10):
    # 1. Find Game Index (Exact -> Fuzzy)
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

    # 2. Get the Feature Vector directly from the sparse matrix
    game_vector = matrix_source[game_idx]
    
    # 3. Find Neighbors (Fetch 50 to have enough buffer for platform filtering)
    distances, indices = model_source.kneighbors(game_vector, n_neighbors=50)
    
    # 4. Filter Results
    similar_indices = indices.flatten()
    similar_indices = similar_indices[similar_indices != game_idx] # Remove self
    
    recommendations = df_source.iloc[similar_indices].copy()

    # Apply Platform Filter
    if platform != 'Any':
        platform_col = f'parent_platforms_{platform}'
        if platform_col in df_source.columns:
            recommendations = recommendations[recommendations[platform_col] == 1]
    
    # Limit to top N
    recommendations = recommendations.head(top_n)

    if recommendations.empty:
        return f"No games found like '{matched_game}' on {platform}.", None

    # 5. Format Output
    def get_labels(row, prefix):
        cols = [c for c in df_source.columns if c.startswith(prefix)]
        return ' | '.join([c.replace(prefix, '') for c in cols if row[c] == 1])

    recommendations['Platforms'] = recommendations.apply(lambda x: get_labels(x, 'parent_platforms_'), axis=1)
    recommendations['Genres'] = recommendations.apply(lambda x: get_labels(x, 'genres_'), axis=1)

    return matched_game, recommendations[['name', 'rating', 'Platforms', 'Genres']]

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Game Recommender")
st.title("🎮 Advanced Game Recommender")

if df is not None and model is not None:
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Platform Selector
    platform_cols = [c.replace('parent_platforms_', '') for c in df.columns if 'parent_platforms_' in c]
    platform_cols.sort()
    selected_platform = st.sidebar.selectbox("Filter by Platform (Optional)", ['Any'] + platform_cols)

    # Main Input
    st.write("### Find your next favorite game")
    game_list = sorted(df['name'].unique())
    game_input = st.selectbox("Select a game you like:", game_list)

    if st.button("Recommend", type="primary"):
        with st.spinner(f"Analyzing game mechanics, genres, and descriptions..."):
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
            st.dataframe(
                recs,
                hide_index=True,
                column_config={
                    "rating": st.column_config.NumberColumn(format="%.2f ⭐"),
                }
            )
else:
    st.warning("Please upload the 'processed_df.pkl', 'weighted_features.joblib', and 'knn_model_final.joblib' files.")
