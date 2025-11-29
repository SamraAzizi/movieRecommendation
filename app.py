import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

# Custom function to safely convert string to list
def safe_literal_eval(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return []

# Custom function to extract director name
def get_director(x):
    crew = safe_literal_eval(x)
    if not isinstance(crew, list):
        return ""
    for i in crew:
        if i.get('job') == 'Director':
            return i.get('name', '')
    return ""

# Custom function to get top elements
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x if 'name' in i]
        return names[:3] if len(names) > 3 else names
    return []

# Custom function to clean data
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

# Load and process data
@st.cache_data
def load_data():
    try:
        # Load datasets
        movies = pd.read_csv('tmdb_5000_movies.csv')
        credits = pd.read_csv('tmdb_5000_credits.csv')
        
        st.success("✅ Dataset files found!")
        
        # Merge datasets
        credits.rename(columns={'movie_id': 'id'}, inplace=True)
        movies = movies.merge(credits, on='id')
        
        # Features for recommendation
        features = ['cast', 'crew', 'keywords', 'genres']
        
        for feature in features:
            movies[feature] = movies[feature].apply(safe_literal_eval)
        
        # Extract director
        movies['director'] = movies['crew'].apply(get_director)
        
        # Extract features
        movies['cast'] = movies['cast'].apply(get_list)
        movies['keywords'] = movies['keywords'].apply(get_list)
        movies['genres'] = movies['genres'].apply(get_list)
        
        # Clean data
        features_to_clean = ['cast', 'keywords', 'director', 'genres']
        for feature in features_to_clean:
            movies[feature] = movies[feature].apply(clean_data)
        
        # Create combined features
        def create_soup(x):
            return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])
        
        movies['soup'] = movies.apply(create_soup, axis=1)
        
        return movies
    
    except FileNotFoundError:
        st.error("❌ Dataset files not found. Please ensure both files are in the same directory.")
        return None
    except Exception as e:
        st.error(f"❌ Error processing data: {e}")
        return None

# Recommendation function
def get_recommendations(title, movies, cosine_sim):
    try:
        # Reset index and create mapping
        indices = pd.Series(movies.index, index=movies['title_x']).drop_duplicates()
        
        # Get the index of the movie
        idx = indices[title]
        
        # Get similarity scores
        sim_scores = list(enumerate(cosine_sim[idx]))
        
        # Sort movies based on similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top 10 similar movies
        sim_scores = sim_scores[1:11]
        
        # Get movie indices
        movie_indices = [i[0] for i in sim_scores]
        
        # Return top 10 similar movies
        return movies['title_x'].iloc[movie_indices]
    
    except KeyError:
        st.error("❌ Movie not found in database. Please select another movie.")
        return pd.Series([])
    except Exception as e:
        st.error(f"❌ Error generating recommendations: {e}")
        return pd.Series([])

# Main app
def main():
    st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")
    
    st.title("🎬 Movie Recommendation System")
    st.markdown("Discover movies similar to your favorites based on cast, crew, genres, and keywords!")
    
    # Load data
    with st.spinner('Loading movie database...'):
        movies = load_data()
    
    if movies is None:
        st.stop()
    
    st.success(f"✅ Successfully loaded {len(movies)} movies!")
    
    # Create recommendation engine
    with st.spinner('Building recommendation engine... This may take a moment...'):
        count_vectorizer = CountVectorizer(stop_words='english', max_features=5000)
        count_matrix = count_vectorizer.fit_transform(movies['soup'])
        cosine_sim = cosine_similarity(count_matrix, count_matrix)
    
    st.success("✅ Recommendation engine ready!")
    
    # Sidebar for movie selection
    st.sidebar.header("🎥 Find Similar Movies")
    movie_list = movies['title_x'].sort_values().tolist()
    selected_movie = st.sidebar.selectbox("Select a movie you like:", movie_list)
    
    # Number of recommendations
    num_recommendations = st.sidebar.slider("Number of recommendations:", 5, 20, 10)
    
    if st.sidebar.button("🎯 Get Recommendations", type="primary"):
        with st.spinner(f'Finding movies similar to "{selected_movie}"...'):
            recommendations = get_recommendations(selected_movie, movies, cosine_sim)
            
            if not recommendations.empty:
                st.header(f"🎭 Movies Similar to **{selected_movie}**")
                
                # Display recommendations in columns
                cols = st.columns(2)
                for i, movie in enumerate(recommendations.head(num_recommendations)):
                    with cols[i % 2]:
                        st.write(f"**{i+1}. {movie}**")
                
                # Show movie details
                with st.expander("ℹ️ About the selected movie"):
                    selected_movie_data = movies[movies['title_x'] == selected_movie].iloc[0]
                    st.write(f"**Director:** {selected_movie_data['director'].title()}")
                    st.write(f"**Genres:** {', '.join([g.title() for g in selected_movie_data['genres']])}")
                    if pd.notna(selected_movie_data['overview']):
                        st.write(f"**Overview:** {selected_movie_data['overview']}")
            else:
                st.warning("No recommendations found. Please try another movie.")
    
    # Dataset info
    with st.expander("📊 Dataset Information"):
        st.write(f"**Total movies in database:** {len(movies)}")
        st.write("**Sample of available movies:**")
        st.dataframe(movies[['title_x', 'genres']].head(10), use_container_width=True)

if __name__ == "__main__":
    main()