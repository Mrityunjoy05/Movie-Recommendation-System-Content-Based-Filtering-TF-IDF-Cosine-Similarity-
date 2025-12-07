# preprocessing.py
import pandas as pd
import os
import ast

def preprocess(path_credits=r'Files\Recommended System Files\tmdb_5000_credits.csv', path_movies=r'Files\Recommended System Files\tmdb_5000_movies.csv'):
    # 📥 Load the credits and movies datasets
    credits = pd.read_csv(path_credits)
    movies = pd.read_csv(path_movies)
    print(f"✅ Loaded credits data with shape: {credits.shape}")
    print(f"✅ Loaded movies data with shape: {movies.shape}")

    # 🔗 Merge datasets on 'title'
    dataset = pd.merge(credits, movies, on='title')
    print(f"🔗 Merged dataset with shape: {dataset.shape}")

    # 📋 Create a copy for analysis
    df = dataset.copy()
    print("✅ Created copy for processing")

    # 📊 Select key columns for analysis
    key_columns = ['movie_id', 'title', 'cast', 'crew', 'genres', 'keywords', 'overview']
    df = df.loc[:, key_columns]
    print(f"📊 Selected key columns: {key_columns}")

    # 🧹 Remove duplicate rows based on 'title'
    duplicate_count = df.duplicated(subset='title').sum()
    if duplicate_count > 0:
        df.drop_duplicates(subset='title', inplace=True)
        print(f"🧹 Dropped duplicate titles: {duplicate_count}")
    else:
        print("✅ No duplicate titles found")

    # 🧹 Handle missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        df.dropna(inplace=True)
        print(f"🧹 Dropped rows with missing values: {missing_count}")
    else:
        print("✅ No missing values found")

    # 🔧 Define helper function to convert list of dicts to names
    def converter(lst):
        result = []
        for i in ast.literal_eval(lst):
            result.append(i['name'])
        return result

    # 🔧 Define helper function to extract top 3 cast members
    def cast_converter(cast_str):
        result = []
        try:
            cast_list = ast.literal_eval(cast_str)
            for i, actor in enumerate(cast_list):
                if i < 3:  # Get first 3 only
                    result.append(actor['name'])
                else:
                    break
        except (ValueError, SyntaxError, KeyError):
            # Handle parsing errors gracefully
            return []
        return result

    # 🔧 Define helper function to extract director
    def director_converter(lst):
        result = []
        lst = ast.literal_eval(lst)
        for i in lst:
            if i['job'] == 'Director':
                result.append(i['name'])
                break
        return result

    # 🔧 Define helper function to split overview into words
    def overview_converter(text):
        lst = text.split()
        return lst

    # 🔧 Define helper function to remove spaces from list items
    def space_remover(lst):
        result = []
        for i in lst:
            result.append(i.replace(' ', ''))
        return result

    # 🎭 Apply cast converter
    df['cast'] = df['cast'].apply(cast_converter)
    print("🎭 Converted cast to top 3 names")

    # 👥 Apply crew converter (director only)
    df['crew'] = df['crew'].apply(director_converter)
    print("👥 Extracted director from crew")

    # 📚 Apply genres converter
    df['genres'] = df['genres'].apply(converter)
    print("📚 Converted genres to names")

    # 🏷️ Apply keywords converter
    df['keywords'] = df['keywords'].apply(converter)
    print("🏷️ Converted keywords to names")

    # 📝 Apply overview converter
    df['overview'] = df['overview'].apply(overview_converter)
    print("📝 Split overview into words")

    # 🧹 Apply space remover to relevant columns
    df['cast'] = df['cast'].apply(space_remover)
    df['crew'] = df['crew'].apply(space_remover)
    df['genres'] = df['genres'].apply(space_remover)
    df['keywords'] = df['keywords'].apply(space_remover)
    print("🧹 Removed spaces from categorical features")

    # 🏷️ Create tags by combining cast, crew, genres, keywords, and overview
    df['tags'] = df['cast'] + df['crew'] + df['genres'] + df['keywords'] + df['overview']
    print("🏷️ Combined features into tags")

    # 📋 Select final columns for main dataframe
    main_df = df[['movie_id', 'title', 'tags']]
    print("📋 Created main dataframe")

    # 🔤 Join tags into space-separated string
    main_df['tags'] = main_df['tags'].apply(lambda x: ' '.join(x))
    print("🔤 Joined tags into strings")

    # 🔡 Lowercase the tags
    main_df['tags'] = main_df['tags'].apply(lambda x: x.lower())
    print("🔡 Lowercased tags for consistency")

    # 💾 Save the processed dataframe
    os.makedirs('data', exist_ok=True)
    main_df.to_csv('data/processed_movies.csv', index=False)
    print("💾 Saved processed dataframe to 'data/processed_movies.csv'")

    return main_df
