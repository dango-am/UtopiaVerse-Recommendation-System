import streamlit as st
import pandas as pd
import joblib
import difflib
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="GameVerse - Ultimate Game Discovery",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Optimized Gaming CSS Styling ---
st.markdown("""
<style>
    /* Main Background with Animated Gradient */
    .stApp {
        background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #1a1a2e);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Epic Main Header */
    .main-header {
        font-size: 5rem;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4, #FFEAA7, #FF6B6B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        background-size: 300% 300%;
        animation: rainbow 3s ease infinite;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 900;
        text-shadow: 0 0 50px rgba(255, 107, 107, 0.7);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    @keyframes rainbow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Gaming-themed Sidebar */
    .css-1d391kg, .css-1lcbmhc {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%) !important;
        border-right: 2px solid #4ECDC4;
    }
    
    /* Spacious Game Card */
    .game-card {
        background: linear-gradient(135deg, rgba(30, 30, 60, 0.95) 0%, rgba(10, 10, 40, 0.95) 100%);
        border-radius: 25px;
        padding: 30px 25px;
        margin: 20px 0;
        box-shadow: 
            0 20px 60px rgba(0, 0, 0, 0.5),
            0 0 40px rgba(255, 107, 107, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        border: 2px solid rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(15px);
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
        min-height: 450px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100%;
    }
    
    .game-card:hover {
        transform: translateY(-8px);
        box-shadow: 
            0 25px 70px rgba(0, 0, 0, 0.6),
            0 0 50px rgba(255, 107, 107, 0.4);
    }
    
    .game-card-header {
        color: #fff;
        font-size: 1.8rem;
        font-weight: 900;
        margin-bottom: 15px;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        text-align: center;
        line-height: 1.2;
        min-height: 60px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .game-card-content {
        color: #e0e0e0;
        font-size: 1.1rem;
        line-height: 1.6;
        flex-grow: 1;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    
    /* Compact Image Container */
    .game-image-container {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.6);
        border: 2px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
        background: linear-gradient(45deg, #1a1a2e, #16213e);
        padding: 10px;
        height: 200px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 15px;
    }
    
    .game-image-container:hover {
        transform: scale(1.05);
        box-shadow: 0 20px 50px rgba(255, 107, 107, 0.4);
        border: 2px solid rgba(255, 107, 107, 0.4);
    }
    
    /* Compact Platform Tags */
    .platform-tag {
        display: inline-block;
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 8px 12px;
        border-radius: 15px;
        margin: 5px 8px 5px 0;
        font-size: 0.9rem;
        font-weight: 600;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .platform-tag:hover {
        transform: translateY(-2px) scale(1.03);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    /* Compact Genre Tags */
    .genre-tag {
        display: inline-block;
        background: linear-gradient(45deg, #FF6B6B, #ee5a24);
        color: white;
        padding: 8px 12px;
        border-radius: 15px;
        margin: 5px 8px 5px 0;
        font-size: 0.9rem;
        font-weight: 600;
        box-shadow: 0 6px 20px rgba(255, 107, 107, 0.4);
        transition: all 0.3s ease;
    }
    
    .genre-tag:hover {
        transform: translateY(-2px) scale(1.03);
        box-shadow: 0 8px 25px rgba(255, 107, 107, 0.6);
    }
    
    /* Compact Rating Badge */
    .rating-badge {
        background: linear-gradient(45deg, #FFD700, #FFA500, #FF8C00);
        color: #000;
        padding: 12px 20px;
        border-radius: 25px;
        font-weight: 800;
        display: inline-block;
        box-shadow: 0 8px 25px rgba(255, 215, 0, 0.6);
        font-size: 1.2rem;
        margin: 15px auto;
        text-align: center;
        min-width: 100px;
    }
    
    /* Compact Release Date */
    .release-date {
        background: linear-gradient(45deg, #4ECDC4, #45B7D1);
        color: white;
        padding: 8px 15px;
        border-radius: 15px;
        font-weight: 600;
        display: inline-block;
        margin: 10px 0;
        font-size: 1rem;
        box-shadow: 0 5px 15px rgba(78, 205, 196, 0.4);
    }
    
    /* Massive Recommendation Header */
    .recommendation-header {
        background: linear-gradient(45deg, #FF6B6B, #ee5a24, #4ECDC4, #45B7D1);
        padding: 30px;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 30px 0;
        box-shadow: 0 15px 40px rgba(255, 107, 107, 0.4);
        border: 2px solid rgba(255, 255, 255, 0.2);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    /* Epic Button */
    .stButton button {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4);
        color: white;
        border: none;
        padding: 20px 40px;
        border-radius: 50px;
        font-size: 1.2rem;
        font-weight: 800;
        transition: all 0.3s ease;
        box-shadow: 0 10px 30px rgba(255, 107, 107, 0.5);
        position: relative;
        overflow: hidden;
    }
    
    .stButton button:hover {
        transform: translateY(-5px) scale(1.05);
        box-shadow: 0 15px 40px rgba(255, 107, 107, 0.7);
    }
    
    /* Giant Stats Cards */
    .stat-card {
        background: linear-gradient(135deg, rgba(30, 30, 60, 0.9) 0%, rgba(10, 10, 40, 0.9) 100%);
        border-radius: 20px;
        padding: 25px;
        margin: 15px 0;
        border: 2px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
        text-align: center;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    /* Huge Section Headers */
    .section-header {
        color: #4ECDC4;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 1.5rem;
        text-shadow: 0 0 20px rgba(78, 205, 196, 0.5);
        text-align: center;
    }
    
    /* Sub Header */
    .sub-header {
        color: #FF6B6B;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 1rem 0 0.8rem 0;
        text-align: center;
    }
    
    /* Info Text */
    .info-text {
        color: #e0e0e0;
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 2rem;
        line-height: 1.6;
    }
    
    /* Tags Container */
    .tags-container {
        margin: 15px 0;
        text-align: center;
        padding: 10px;
    }
    
    /* Content Section */
    .content-section {
        margin: 15px 0;
        padding: 15px;
    }
    
    /* Game Grid Container */
    .game-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
        gap: 25px;
        margin: 30px 0;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a2e;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        border-radius: 5px;
        border: 2px solid #1a1a2e;
    }
    
    /* Epic Footer */
    .footer {
        background: linear-gradient(90deg, #1a1a2e, #16213e, #0f3460, #1a1a2e);
        padding: 30px;
        border-radius: 20px;
        margin-top: 40px;
        text-align: center;
        color: #e0e0e0;
        border: 2px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.4);
    }
    
    /* Responsive Grid Adjustments */
    @media (max-width: 768px) {
        .game-grid {
            grid-template-columns: 1fr;
            gap: 20px;
        }
    }
</style>
""", unsafe_allow_html=True)

# --- Load Data and Model ---
@st.cache_data
def load_data():
    """Loads the pre-trained model and processed game data."""
    try:
        df = pd.read_pickle("processed_df.pkl")
        model = joblib.load("knn_model.joblib")
        features = df.drop(columns=['name','background_image','released'])
        scaler = StandardScaler()
        scaled_features_values = scaler.fit_transform(features)
        scaled_features = pd.DataFrame(
            scaled_features_values,
            index=features.index,
            columns=features.columns
        )
        return df, model, scaled_features
    except FileNotFoundError:
        st.error("Model or data files not found. Please ensure 'processed_df.pkl' and 'knn_model.joblib' are in the same directory.")
        return None, None, None

df, model, features = load_data()

# --- CORRECTED Helper Functions ---
def get_recommendations(game_name, df_source, model_source, feature_source, platform='Any', top_n=10):
    """Finds a game and returns top_n recommendations, with an optional platform filter."""
    try:
        # Create a working copy of the dataframe to avoid modifying the original
        current_df = df_source.copy()

        # If platform filter is specified, check if the platform exists and filter the dataframe
        if platform != 'Any':
            platform_col_name = f'parent_platforms_{platform}'
            if platform_col_name in current_df.columns:
                # Create a filtered dataframe for this platform
                platform_filtered_df = current_df[current_df[platform_col_name] == 1]
                if platform_filtered_df.empty:
                    return f"No games found for the platform '{platform}'.", None
                # Use the filtered dataframe for game matching
                working_df = platform_filtered_df
            else:
                return f"Platform '{platform}' not found.", None
        else:
            # If no platform filter, use the entire dataframe
            working_df = current_df

        # Find the closest matching game name in the working dataframe
        all_game_names = working_df['name'].tolist()
        closest_matches = difflib.get_close_matches(game_name, all_game_names, n=1, cutoff=0.6)
        if not closest_matches:
            return f"No close match found for '{game_name}' on the selected platform. Please try another title.", None

        matched_game = closest_matches[0]
        
        # Get the index of the matched game in the working dataframe
        game_idx_list = working_df[working_df['name'] == matched_game].index
        if len(game_idx_list) == 0:
            return f"Error finding '{matched_game}' in the dataframe.", None
        game_idx = game_idx_list[0]

        # Verify that the game index exists in the feature source
        if game_idx not in feature_source.index:
            return f"Game index {game_idx} not found in feature matrix.", None
            
        # Get the feature vector for the matched game
        feature_vector = feature_source.loc[game_idx].values.reshape(1, -1)
        
        # Calculate the number of neighbors to request (should not exceed available games)
        n_neighbors_to_request = min(top_n + 1, len(feature_source))
        
        # Get recommendations using the model
        distances, indices = model_source.kneighbors(feature_vector, n_neighbors=n_neighbors_to_request)
        
        # Get the recommendation indices (excluding the input game itself)
        rec_indices_from_model = indices[0][1:]
        
        # Convert these indices to the actual dataframe indices
        # The KNN model returns indices relative to the feature matrix, which should match the original dataframe
        rec_indices = [idx for idx in rec_indices_from_model if idx < len(df_source)]
        
        if not rec_indices:
            return f"No valid recommendations found for '{matched_game}'.", None

        # Get the recommended games from the original dataframe using the indices
        if platform != 'Any':
            # For platform-specific recommendations, we need to filter the results
            # First get all recommendations from the original dataframe
            all_recommendations_df = df_source.iloc[rec_indices]
            # Then filter by platform
            final_recs_df = all_recommendations_df[all_recommendations_df[platform_col_name] == 1].head(top_n)
        else:
            # For all platforms, just take the top_n recommendations
            final_recs_df = df_source.iloc[rec_indices].head(top_n)

        # Check if we have any recommendations left after filtering
        if final_recs_df.empty:
            return f"No platform-compatible recommendations found for '{matched_game}' on {platform}.", None

        # Create a copy of the final recommendations for display
        recommendations = final_recs_df.copy()

        # Extract platform and genre columns
        platform_cols = [col for col in df_source.columns if 'parent_platforms_' in col]
        genre_cols = [col for col in df_source.columns if 'genres_' in col]

        # Create display columns for platforms and genres
        recommendations['Platforms'] = recommendations[platform_cols].apply(
            lambda row: ' | '.join([col.replace('parent_platforms_', '') for col in platform_cols if row[col] == 1]),
            axis=1
        )

        recommendations['Genres'] = recommendations[genre_cols].apply(
            lambda row: ' | '.join([col.replace('genres_', '') for col in genre_cols if row[col] == 1]),
            axis=1
        )

        # Select which columns to display in the final output
        display_columns = ['name', 'rating', 'Platforms', 'Genres']
        if 'background_image' in recommendations.columns:
            display_columns.append('background_image')
        if 'released' in recommendations.columns:
            display_columns.append('released')

        return matched_game, recommendations[display_columns]
        
    except Exception as e:
        # Return a generic error message to avoid exposing internal details
        return f"An unexpected error occurred while generating recommendations. Please try again with a different game or platform.", None

# --- Streamlit UI ---
st.markdown('<div class="main-header"> UtopiaVerse </div>', unsafe_allow_html=True)
st.markdown('<div class="info-text">🌟 Machine Learning That Knows What You\'ll Love 🌟</div>', unsafe_allow_html=True)
st.markdown('<div class="info-text">Discover, play, and conquer your next gaming masterpiece awaits.</div>', unsafe_allow_html=True)

if df is not None:
    # --- Sidebar ---
    with st.sidebar:
        st.markdown("### 🎯 DISCOVERY SETTINGS")
        st.markdown("---")

        rec_type = st.radio(
            "**Choose your discovery mode:**",
            ('🎮 ALL PLATFORMS', '🎯 SPECIFIC PLATFORM'),
            help="Select whether to filter by platform"
        )

        platform_choice = 'Any'
        if rec_type == '🎯 SPECIFIC PLATFORM':
            platform_list = [col.replace('parent_platforms_', '') for col in df.columns if 'parent_platforms_' in col]
            platform_choice = st.selectbox("**SELECT YOUR PLATFORM:**", options=['Any'] + sorted(platform_list))

        st.markdown("---")
        st.markdown(" ⚙️ HOW IT WORKS")
        st.markdown("""
        <div style='color: #e0e0e0; font-size: 1rem;'>
        🤖 <strong>Machine Learning powered Analysis:</strong><br>
        • Game genres & features<br>
        • User ratings & reviews<br>  
        • Platform compatibility<br>
        • Release timeline<br>
        <br>
        🧠 <strong>Smart Matching:</strong><br>
        • Finds similar gameplay styles<br>
        • Matches your preferred genres<br>
        • Considers platform availability<br>
        </div>
        """, unsafe_allow_html=True)

    # --- Main Content ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="section-header">🚀 Let our Model Guide You to Your Next Epic Adventure</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-text">Select a game you love, and we\'ll reveal amazing titles you\'re destined to enjoy!</div>', unsafe_allow_html=True)

        game_list = sorted(df['name'].unique())
        game_name_input = st.selectbox(
            "**🎪 CHOOSE YOUR FAVORITE GAME:**",
            options=game_list,
            help="Start your gaming journey by selecting a title you enjoy"
        )

    with col2:
        st.markdown('<div class="section-header">📊 UTOPIAVERSE  STATS</div>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="stat-card">', unsafe_allow_html=True)
            total_games = len(df)
            st.metric("🎪 TOTAL GAMES", f"{total_games:,}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            if 'rating' in df.columns:
                st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                avg_rating = df['rating'].mean()
                st.metric("⭐ AVERAGE RATING", f"{avg_rating:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)

    # --- Recommendation Button ---
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 LAUNCH ULTIMATE GAME DISCOVERY", use_container_width=True):
            if game_name_input:
                with st.spinner('🔮 Scanning the entire gaming universe for perfect matches...'):
                    result_title, result_df = get_recommendations(game_name_input, df, model, features, platform=platform_choice)

                if result_df is None:
                    st.error(f"❌ {result_title}")
                else:
                    st.markdown(f"""
                    <div class="recommendation-header">
                        <h2 style="font-size: 2.5rem; margin: 0;">🎉 EPIC DISCOVERIES UNLOCKED! 🎉</h2>
                        <h3 style="font-size: 2rem; margin: 15px 0;">Games similar to <span style="color: #FFD700;">{result_title}</span></h3>
                        <p style="font-size: 1.2rem; margin: 0;">Platform: {platform_choice} | 🔍 Found {len(result_df)} amazing recommendations</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # --- IMPROVED GRID LAYOUT with Better Space Usage ---
                    st.markdown('<div class="game-grid">', unsafe_allow_html=True)
                    
                    for idx, row in result_df.iterrows():
                        rating_display = f"{row['rating']:.2f}" if pd.notna(row['rating']) else 'N/A'
                        release_display = f"<div class='release-date'>📅 {row['released']}</div>" if 'released' in row and pd.notna(row['released']) else ""
                        
                        st.markdown(f"""
                        <div class="game-card">
                            <div class="game-image-container">
                                {"<img src='" + row['background_image'] + "' style='width: 100%; height: 100%; object-fit: cover; border-radius: 12px;' alt='Game Art' />" 
                                 if 'background_image' in row and pd.notna(row['background_image']) and row['background_image'] != '' 
                                 else "<div style='width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; background: linear-gradient(45deg, #667eea, #764ba2); color: white; font-size: 1.2rem; font-weight: bold; border-radius: 12px;'>EPIC GAME ART</div>"}
                            </div>
                            <div class="game-card-header">{row['name']}</div>
                            <div class="game-card-content">
                                <div style="text-align: center; margin: 15px 0;">
                                    <div class="rating-badge">⭐ {rating_display}</div>
                                </div>
                                <div style="text-align: center; margin: 10px 0;">
                                    {release_display}
                                </div>
                                <div class="content-section">
                                    <div class="sub-header">🎯 PLATFORMS</div>
                                    <div class="tags-container">
                                        {"".join([f'<span class="platform-tag">{platform}</span>' for platform in row['Platforms'].split(" | ")])}
                                    </div>
                                </div>
                                <div class="content-section">
                                    <div class="sub-header">🎪 GENRES</div>
                                    <div class="tags-container">
                                        {"".join([f'<span class="genre-tag">{genre}</span>' for genre in row['Genres'].split(" | ")])}
                                    </div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("⚠️ Please select a game to begin your ultimate discovery journey!")

    # --- Epic Footer ---
    st.markdown("""
    <div class="footer">
        <h2 style="font-size: 2rem; margin: 0;">🌟 POWERED BY CUTTING-EDGE AI TECHNOLOGY 🌟</h2>
        <p style="font-size: 1.2rem; margin: 15px 0;">Machine Learning • K-Nearest Neighbors Algorithm • Advanced Game Analytics</p>
        <p style="font-size: 1.1rem; margin: 0;">🎮 BUILT FOR TRUE GAMING ENTHUSIASTS 🎮</p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.error("🚨 UNABLE TO LOAD GAME DATA! Please ensure the gaming universe files are properly configured!")
