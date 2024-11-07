import streamlit as st
import pandas as pd
import os
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

# Configure Streamlit page
st.set_page_config(
    page_title="Netflix Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# Load data function with caching
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('./data/netflix_titles.csv')
        return df.fillna('')
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

class MovieRecommender:
    def __init__(self, df):
        self.df = df
    
    def get_content_based_recommendations(self, title):
        movie = self.df[self.df['title'].str.lower() == title.lower()]
        
        if len(movie) == 0:
            similar_titles = self.df[self.df['title'].str.lower().str.contains(title.lower(), na=False)]
            if len(similar_titles) > 0:
                return similar_titles.head(5)
            return None
        
        genre = movie['listed_in'].iloc[0]
        similar_movies = self.df[
            (self.df['listed_in'].str.contains(genre, na=False)) & 
            (self.df['title'] != movie['title'].iloc[0])
        ].head(5)
        
        return similar_movies

    def get_personalized_recommendation(self, preferences):
        keywords = preferences.lower().split()
        matched_movies = self.df[
            self.df['description'].str.lower().apply(
                lambda x: any(keyword in x for keyword in keywords)
            ) |
            self.df['listed_in'].str.lower().apply(
                lambda x: any(keyword in x for keyword in keywords)
            )
        ].sample(n=5)
        
        return matched_movies

def main():
    st.title("🎬 Netflix Movie Recommender")
    
    # Load data
    df = load_data()
    
    if df is None:
        st.error("Failed to load the movie database. Please check if the data file exists in the correct location.")
        return
    
    # Initialize recommender
    recommender = MovieRecommender(df)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Movie Search", "🎨 Personalized", "🔍 Explore"])
    
    # Tab 1: Movie-based Recommendations
    with tab1:
        st.header("Find Similar Movies")
        movie_title = st.text_input("Enter a movie title:")
        
        if movie_title:
            with st.spinner('Finding recommendations...'):
                recommendations = recommender.get_content_based_recommendations(movie_title)
                
                if recommendations is None:
                    st.warning("Movie not found in the database. Please try another title.")
                elif len(recommendations) == 5 and 'title' not in movie_title.lower():
                    st.info("Did you mean one of these movies?")
                    for _, movie in recommendations.iterrows():
                        st.write(f"- {movie['title']} ({movie['release_year']})")
                else:
                    st.success("Here are your recommendations:")
                    cols = st.columns(5)
                    for idx, (_, movie) in enumerate(recommendations.iterrows()):
                        with cols[idx]:
                            st.subheader(movie['title'])
                            st.caption(f"Year: {movie['release_year']}")
                            st.write(f"**Genre:** {movie['listed_in']}")
                            with st.expander("More Info"):
                                st.write(movie['description'])
                                if movie['director']:
                                    st.write(f"**Director:** {movie['director']}")
                                if movie['cast']:
                                    st.write(f"**Cast:** {movie['cast']}")
    
    # Tab 2: Personalized Recommendations
    with tab2:
        st.header("Get Personalized Recommendations")
        preferences = st.text_area(
            "What kind of movies do you like?",
            placeholder="Example: action movies with strong female leads"
        )
        
        if preferences:
            with st.spinner('Finding personalized recommendations...'):
                recommendations = recommender.get_personalized_recommendation(preferences)
                st.success("Here are your personalized recommendations:")
                
                for _, movie in recommendations.iterrows():
                    with st.expander(f"{movie['title']} ({movie['release_year']})"):
                        st.write(f"**Genre:** {movie['listed_in']}")
                        st.write(f"**Description:** {movie['description']}")
                        if movie['director']:
                            st.write(f"**Director:** {movie['director']}")
                        if movie['cast']:
                            st.write(f"**Cast:** {movie['cast']}")
    
    # Tab 3: Explore Movies
    with tab3:
        st.header("Explore Movies")
        col1, col2 = st.columns([2,1])
        with col1:
            num_movies = st.slider("Number of random movies to show:", 5, 20, 10)
        with col2:
            if st.button("🎲 Show Random Movies"):
                sample_movies = df.sample(n=num_movies)
                for _, movie in sample_movies.iterrows():
                    with st.expander(f"{movie['title']} ({movie['release_year']})"):
                        st.write(f"**Genre:** {movie['listed_in']}")
                        st.write(f"**Description:** {movie['description']}")
                        if movie['director']:
                            st.write(f"**Director:** {movie['director']}")
                        if movie['cast']:
                            st.write(f"**Cast:** {movie['cast']}")

    # Footer
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit")

if __name__ == "__main__":
    main()