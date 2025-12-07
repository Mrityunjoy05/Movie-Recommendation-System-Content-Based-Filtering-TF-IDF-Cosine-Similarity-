# main.py
import pandas as pd
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure pandas display
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Import your pipeline functions
from data_loader import load_data
from preprocessing import preprocess
from train_recommendation import train_recommender
from recommendation_engine import MovieRecommender

if __name__ == "__main__":

    print("Starting Movie Recommendation Pipeline...\n")

    # Step 1: Load and merge raw data
    print("Step 1: Loading raw datasets...")
    dataset = load_data()  # Assumes load_data() handles paths internally
    print(f"Loaded merged dataset: {dataset.shape}")
    print("\nFirst 5 rows of merged data:")
    print(dataset.head())

    # Step 2: Preprocess and save processed data
    print("\nStep 2: Preprocessing data...")
    main_df = preprocess()  
    print(f"Preprocessed data: {main_df.shape}")
    print("\nFirst 5 rows of processed data:")
    print(main_df.head())

    # Step 3: Train recommender (uses saved processed_movies.csv)
    print("\nStep 3: Training recommender model...")
    train_recommender()  # Assumes it reads from 'data/processed_movies.csv'

    # Step 4:
    movie = MovieRecommender()
    movie.recommend('The Dark Knight Rises')

    print("\nPipeline completed successfully!")