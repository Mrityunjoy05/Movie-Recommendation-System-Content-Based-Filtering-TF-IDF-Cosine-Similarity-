# train_recommendation.py
import pandas as pd
import os
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def train_recommender(path=r'data\processed_movies.csv'):
    # 📥 Load the processed movies dataframe
    main_df = pd.read_csv(path)
    print(f"✅ Loaded processed data with shape: {main_df.shape}")

    # 📚 Download required NLTK resources (idempotent, runs only if needed)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)
    print("📚 NLTK resources ready")

    # 🛠️ Initialize NLTK tools
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    print("🛠️ Initialized stopwords and lemmatizer")

    # 🔧 Define text preprocessing function
    def preprocess(text):
        # 1. Lowercase the text
        text = text.lower()
        # 2. Tokenize into words
        tokens = word_tokenize(text)
        # 3. Lemmatize, remove stopwords, and keep only alphanumeric tokens
        clean_tokens = [
            lemmatizer.lemmatize(w)
            for w in tokens
            if w.isalnum() and w not in stop_words
        ]
        # 4. Rejoin into a cleaned string
        return ' '.join(clean_tokens)

    # 🧹 Apply preprocessing to tags
    main_df['modified_tags'] = main_df['tags'].apply(preprocess)
    print("🧹 Applied text preprocessing to tags")

    # 🔢 TF-IDF Vectorization
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(main_df['modified_tags'])
    print(f"🔢 Created TF-IDF matrix with shape: {tfidf_matrix.shape}")

    # 📐 Compute cosine similarity matrix
    similarity = cosine_similarity(tfidf_matrix)
    print(f"📐 Computed similarity matrix with shape: {similarity.shape}")

    # 💾 Define model saving function
    def savemodels(model, path):
        with open(path, 'wb') as f:
            pickle.dump(model, f)

    # 📁 Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # 💾 Save processed main_df and similarity matrix
    savemodels(main_df, 'models/main_df.pkl')
    savemodels(similarity, 'models/similarity.pkl')
    print("💾 Saved main_df and similarity models")
    print("🎉 Training complete! Load pickles in recommendation_engine.py to generate recs.")
    