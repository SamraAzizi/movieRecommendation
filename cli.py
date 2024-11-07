from movie_recommender import MovieRecommender

def main():
    # If your netflix_titles.csv is in a folder named 'data'
    data_folder = "./data"  # Just the folder path
    recommender = MovieRecommender(data_folder)
    
    while True:
        print("\n=== Netflix Movie Recommender ===")
        print("1. Get recommendations based on a movie")
        print("2. Get personalized recommendations")
        print("3. Show some popular movies")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == "1":
            movie_title = input("Enter a movie title: ")
            recommendations = recommender.get_content_based_recommendations(movie_title)
            print("\nRecommendations:")
            print(recommendations)
            
        elif choice == "2":
            print("\nTell us your preferences!")
            print("Example: I like action movies with strong female leads, set in modern times")
            preferences = input("Your preferences: ")
            recommendations = recommender.get_personalized_recommendation(preferences)
            print("\nRecommendations:")
            print(recommendations)
            
        elif choice == "3":
            # Show some popular movies from the dataset
            print("\nSome movies in our database:")
            sample_movies = recommender.df.sample(n=10)
            for _, movie in sample_movies.iterrows():
                print(f"- {movie['title']} ({movie['release_year']}) - {movie['listed_in']}")
            
        elif choice == "4":
            print("Thank you for using Netflix Movie Recommender!")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 