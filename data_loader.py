import pandas as pd  # For data manipulation and analysis

# Function to load a CSV file into a pandas DataFrame
def load_csv(file_path):
    """
    Loads a CSV file into a pandas DataFrame.
    If an error occurs (e.g., file not found), it prints the error and returns None.

    Parameters:
    file_path (str): The file path of the CSV file.

    Returns:
    pd.DataFrame or None: Returns the loaded DataFrame if successful, otherwise None.
    """
    try:
        return pd.read_csv(file_path)  # Load the CSV file
    except Exception as e:
        print(f"Error: {e}")  # Print error message if loading fails
        return None  # Return None in case of failure
    

path_credits = r"C:\Users\souvi\Desktop\Recommended System\Files\Recommended System Files\tmdb_5000_credits.csv"
credits = load_csv(file_path=path_credits)  # Load df

path_movies = r"C:\Users\souvi\Desktop\Recommended System\Files\Recommended System Files\tmdb_5000_movies.csv"
movies = load_csv(file_path=path_movies)  # Load df

# def load_data(credits = credits , movies = movies ):

#     dataset = pd.merge(credits , movies ,on = 'title' )

#     return dataset

def load_data(credits_df = credits , movies_df = movies):  # Removed globals; require args
    """
    Merges credits and movies DataFrames on 'title'.

    Parameters:
    credits_df (pd.DataFrame): Loaded credits DataFrame.
    movies_df (pd.DataFrame): Loaded movies DataFrame.

    Returns:
    pd.DataFrame: Merged dataset.
    """
    if credits is None or movies is None:
        raise ValueError("Failed to load one or both DataFrames. Check file paths and CSV format.")
    if 'title' not in credits.columns or 'title' not in movies.columns:
        raise KeyError("'title' column missing in one or both DataFrames.")
    
    dataset = pd.merge(credits_df, movies_df, on='title')
    return dataset
