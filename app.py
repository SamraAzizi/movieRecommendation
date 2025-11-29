import streamlit as st
import pandas as pd
import numpy as np

# Try to import sklearn with fallback
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.warning("scikit-learn not available. Using simple genre-based recommendations.")

@st.cache_data
def load_data():
    """Load and prepare movie data"""
    try:
        movies = pd.read_csv('tmdb_5000_movies.csv')
        
        # Basic data preparation
        movies['overview'] = movies['overview'].fillna('')
        movies['genres'] = movies['genres'].fillna('[]')
        
        # Extract genre names safely
        def get_genre_names(genre_str):
            try:
                if pd.isna(genre_str):
                    return ''
                genres = eval(genre_str) if isinstance(genre_str, str) else genre_str
                if isinstance(genres, list):
                    return ' '.join([g['name'] for g in genres if 'name' in g])
                return ''
            except:
                return ''
        
        movies['genre_names'] = movies['genres'].apply(get_genre_names)
        movies['combined_features'] = movies['overview'] + ' ' + movies['genre_names']
        
        return movies
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def get_simple_recommendations(movies, selected_movie, num_recommendations=10):
    """Simple recommendation based on genre matching"""
    try:
        # Get genres of selected movie
        selected_genres = movies[movies['title'] == selected_movie]['genre_names'].iloc[0]
        selected_genre_set = set(selected_genres.split())
        
        recommendations = []
        for idx, row in movies.iterrows():
            if row['title'] != selected_movie:
                movie_genres = set(row['genre_names'].split())
                common_genres = selected_genre_set.intersection(movie_genres)
                if common_genres:
                    recommendations.append((row['title'], len(common_genres)))
        
        # Sort by number of common genres
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return [movie[0] for movie in recommendations[:num_recommendations]]
    
    except Exception as e:
        st.error(f"Error in simple recommendations: {e}")
        return []

def get_ml_recommendations(movies, selected_movie, num_recommendations=10):
    """ML-based recommendations using TF-IDF"""
    try:
        # Create TF-IDF matrix
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(movies['combined_features'])
        
        # Compute cosine similarity
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        
        # Get recommendations
        indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
        idx = indices[selected_movie]
        
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:num_recommendations+1]
        movie_indices = [i[0] for i in sim_scores]
        
        return movies['title'].iloc[movie_indices].tolist()
    
    except Exception as e:
        st.error(f"Error in ML recommendations: {e}")
        return []

def main():
    st.set_page_config(
        page_title="Movie Recommendation System",
        page_icon="🎬",
        layout="wide"
    )
    
    st.title("🎬 Movie Recommendation System")
    st.markdown("Discover movies you'll love based on your favorites!")
    
    # Load data
    with st.spinner('Loading movie database...'):
        movies = load_data()
    
    if movies is None:
        st.error("Failed to load movie data. Please check if the data files are available.")
        return
    
    st.success(f"✅ Loaded {len(movies)} movies successfully!")
    
    # Movie selection
    st.sidebar.header("🔍 Find Similar Movies")
    movie_list = movies['title'].sort_values().tolist()
    selected_movie = st.sidebar.selectbox("Select a movie you like:", movie_list)
    
    num_recommendations = st.sidebar.slider("Number of recommendations:", 5, 20, 10)
    
    # Recommendation method
    if SKLEARN_AVAILABLE:
        method = st.sidebar.radio("Recommendation method:", 
                                ["Advanced (ML)", "Simple (Genre-based)"])
    else:
        method = "Simple (Genre-based)"
        st.sidebar.info("Using simple genre-based recommendations")
    
    if st.sidebar.button("🎯 Get Recommendations", type="primary"):
        with st.spinner('Finding similar movies...'):
            if method == "Advanced (ML)" and SKLEARN_AVAILABLE:
                recommendations = get_ml_recommendations(movies, selected_movie, num_recommendations)
                method_used = "ML-based content filtering"
            else:
                recommendations = get_simple_recommendations(movies, selected_movie, num_recommendations)
                method_used = "genre matching"
            
            if recommendations:
                st.header(f"🎭 Movies Similar to **{selected_movie}**")
                st.info(f"Using {method_used}")
                
                # Display recommendations
                cols = st.columns(2)
                for i, movie in enumerate(recommendations):
                    with cols[i % 2]:
                        st.write(f"**{i+1}. {movie}**")
            else:
                st.warning("No recommendations found. Please try another movie.")
    
    # Dataset info
    with st.expander("📊 Dataset Information"):
        st.write(f"**Total movies:** {len(movies)}")
        st.write("**Sample movies:**")
        st.dataframe(movies[['title', 'genre_names']].head(10).rename(
            columns={'title': 'Movie Title', 'genre_names': 'Genres'}
        ), use_container_width=True)

if __name__ == "__main__":
    main()