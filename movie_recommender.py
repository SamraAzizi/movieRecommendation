import pandas as pd
import numpy as np
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
import warnings
import os
warnings.filterwarnings('ignore')

class MovieRecommender:
    def __init__(self, data_folder_path):
        # Load the movie recommendations dataset from the folder
        self.load_data(data_folder_path)
        
    def load_data(self, folder_path):
        """Load and process data from the recommendations folder"""
        try:
            # Assuming you have a main CSV file with movie data
            self.df = pd.read_csv(os.path.join(folder_path, 'netflix_titles.csv'))
            
            # Clean the data
            self.df = self.df.fillna('')
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
    
    def get_content_based_recommendations(self, title):
        print("Searching for movie...")  # Immediate feedback
        
        # Find the movie in our dataset
        movie = self.df[self.df['title'].str.lower() == title.lower()]
        
        if len(movie) == 0:
            # Show similar titles to help user
            similar_titles = self.df[self.df['title'].str.lower().str.contains(title.lower(), na=False)]
            if len(similar_titles) > 0:
                suggestion_message = "\nMovie not found. Did you mean one of these?\n"
                for _, row in similar_titles.head(5).iterrows():
                    suggestion_message += f"- {row['title']} ({row['release_year']})\n"
                return suggestion_message
            else:
                return "Movie not found in the database. Please try another title."
        
        print("Finding similar movies...")  # Progress feedback
        
        # Simplified recommendation approach for faster results
        genre = movie['listed_in'].iloc[0]
        similar_movies = self.df[
            (self.df['listed_in'].str.contains(genre, na=False)) & 
            (self.df['title'] != movie['title'].iloc[0])
        ].head(5)
        
        # Format the response
        response = f"\nRecommendations for {movie['title'].iloc[0]}:\n\n"
        count = 1
        for _, row in similar_movies.iterrows():
            response += f"{count}. {row['title']} ({row['release_year']})\n"
            response += f"   Genre: {row['listed_in']}\n"
            response += f"   Why: Similar {genre} content\n\n"
            count += 1
        
        return response

    def get_personalized_recommendation(self, preferences):
        print("Finding movies matching your preferences...")  # Progress feedback
        
        # Simple keyword matching for faster results
        keywords = preferences.lower().split()
        matched_movies = self.df[
            self.df['description'].str.lower().apply(
                lambda x: any(keyword in x for keyword in keywords)
            ) |
            self.df['listed_in'].str.lower().apply(
                lambda x: any(keyword in x for keyword in keywords)
            )
        ].sample(n=5)
        
        # Format the response
        response = "\nRecommended movies based on your preferences:\n\n"
        count = 1
        for _, row in matched_movies.iterrows():
            response += f"{count}. {row['title']} ({row['release_year']})\n"
            response += f"   Genre: {row['listed_in']}\n"
            response += f"   Description: {row['description'][:100]}...\n\n"
            count += 1
        
        return response

    def get_ollama_recommendations(self, title):
        # Initialize Ollama
        llm = Ollama(model="llama2")
        
        # Find the movie
        movie = self.df[self.df['title'].str.lower() == title.lower()]
        
        if len(movie) == 0:
            return "Movie not found in the database."
        
        # Create a prompt for the LLM
        prompt_template = """
        Based on this movie:
        Title: {title}
        Description: {description}
        Genre: {listed_in}
        Director: {director}
        Cast: {cast}

        And these similar movies:
        {similar_movies}

        Explain why these movies are similar to {title} and how they might appeal to someone who enjoyed it.
        Format your response as:
        1. Movie Title - Detailed explanation of similarity
        2. Movie Title - Detailed explanation of similarity
        etc.
        """
        
        # Get similar movies based on genre
        genre = movie['listed_in'].iloc[0]
        similar_movies = self.df[
            (self.df['listed_in'].str.contains(genre, na=False)) & 
            (self.df['title'] != movie['title'].iloc[0])
        ].head(5)
        
        # Format similar movies for prompt
        similar_movies_text = "\n".join([
            f"- {row['title']} ({row['release_year']}) - {row['description'][:100]}..."
            for _, row in similar_movies.iterrows()
        ])
        
        # Prepare movie details
        movie_details = {
            "title": movie['title'].iloc[0],
            "description": movie['description'].iloc[0],
            "listed_in": movie['listed_in'].iloc[0],
            "director": movie['director'].iloc[0],
            "cast": movie['cast'].iloc[0],
            "similar_movies": similar_movies_text
        }
        
        # Create and format prompt
        prompt = PromptTemplate(
            input_variables=["title", "description", "listed_in", "director", "cast", "similar_movies"],
            template=prompt_template
        )
        formatted_prompt = prompt.format(**movie_details)
        
        # Get LLM response
        response = llm.invoke(formatted_prompt)
        
        return response