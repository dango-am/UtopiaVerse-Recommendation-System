import streamlit as st
import pandas as pd
import joblib
import difflib
import sklearn

# --- Load Artifacts (runs only once using caching) ---
import streamlit as st
import pandas as pd
import joblib

@st.cache_data
def load_artifacts():
    try:
        df = pd.read_pickle("processed_new_df.pkl")
        scaled_features = pd.read_csv("scaled_features.csv")
        weighted_features = joblib.load("weighted_features.joblib")

        # Force index alignment
        df = df.reset_index(drop=True)
        scaled_features = scaled_features.reset_index(drop=True)

        assert len(df) == len(scaled_features), "Mismatch between df and features!"

        return df, scaled_features, weighted_features

    except FileNotFoundError as e:
        st.error(f"Missing file: {e}")
        return None, None, None

df, features, model = load_artifacts()


# --- Helper Functions ---
def get_recommendations(game_name, df_source, feature_source, model_source, platform='Any', top_n=10):
    """
    Finds a game and returns top_n recommendations.
    This logic now perfectly matches the notebook's process.
    """
    all_game_names = df_source['name'].tolist()
    closest_matches = difflib.get_close_matches(game_name, all_game_names, n=1, cutoff=0.6)

    if not closest_matches:
        return f"No close match found for '{game_name}'.", None

    matched_game = closest_matches[0]
    
    try:
        # Get the original index of the matched game
        game_idx = df_source[df_source['name'] == matched_game].index[0]
    except IndexError:
        return f"Error finding '{matched_game}' in the dataframe.", None

    # Retrieve the pre-scaled feature vector for the game
    feature_vector = feature_source.loc[game_idx].values.reshape(1, -1)
    
    # Query for more neighbors than needed to allow for filtering
    query_neighbors = min(len(df_source), 100) 
    distances, indices = model_source.kneighbors(feature_vector, n_neighbors=query_neighbors)
    
    # Get original indices of recommendations (excluding the game itself)
    rec_indices = [df_source.index[i] for i in indices[0] if df_source.index[i] != game_idx]

    # Apply platform filtering AFTER getting recommendations
    if platform != 'Any':
        platform_col = f'parent_platforms_{platform}'
        if platform_col in df_source.columns:
            final_rec_indices = [idx for idx in rec_indices if df_source.loc[idx, platform_col] == 1]
            recommendations = df_source.loc[final_rec_indices].head(top_n).copy()
        else:
            return f"Platform '{platform}' not found.", None
    else:
        recommendations = df_source.loc[rec_indices].head(top_n).copy()
        
    if recommendations.empty:
        return f"No recommendations found for '{matched_game}' on platform '{platform}'.", None

    # --- Format for Display ---
    platform_cols = [c for c in df_source.columns if 'parent_platforms_' in c]
    genre_cols = [c for c in df_source.columns if 'genres_' in c]
    
    recommendations['Platforms'] = recommendations[platform_cols].apply(
        lambda row: ' | '.join([c.replace('parent_platforms_', '') for c in platform_cols if row[c] == 1]), axis=1
    )
    recommendations['Genres'] = recommendations[genre_cols].apply(
        lambda row: ' | '.join([c.replace('genres_', '') for c in genre_cols if row[c] == 1]), axis=1
    )
    
    return matched_game, recommendations[['name', 'rating', 'Platforms', 'Genres']]

# --- Streamlit User Interface ---
st.set_page_config(layout="wide")
st.title("🎮 Game Recommender System")

if df is not None:
    st.sidebar.header("Filter Options")
    rec_type = st.sidebar.radio("Filter by:", ('Game Only', 'Game + Platform'))

    platform_choice = 'Any'
    if rec_type == 'Game + Platform':
        platform_list = sorted([col.replace('parent_platforms_', '') for col in df.columns if 'parent_platforms_' in col and col != 'parent_platforms_'])
        platform_choice = st.sidebar.selectbox("Choose a platform:", options=platform_list)

    st.write("Enter a game you like to get recommendations.")
    game_list = sorted(df['name'].unique())
    game_name_input = st.selectbox("Choose or type a game:", options=game_list)

    if st.button("Get Recommendations", type="primary"):
        if game_name_input:
            with st.spinner('Finding similar games...'):
                title, recs = get_recommendations(game_name_input, df, features, model, platform=platform_choice)
            
            if recs is None:
                st.warning(title)
            else:
                st.success(f"Recommendations for '{title}' (Platform: {platform_choice})")
                st.dataframe(recs.reset_index(drop=True))
        else:
            st.warning("Please select a game.")
