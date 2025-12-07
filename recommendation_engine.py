# recommendation_engine.py (or evolution.py)
import pickle
import pandas as pd

class MovieRecommender:
    def __init__(self, main_df_path='models/main_df.pkl', similarity_path='models/similarity.pkl'):
        with open(main_df_path, 'rb') as f:
            self.main_df = pickle.load(f)
        with open(similarity_path, 'rb') as f:
            self.similarity = pickle.load(f)
        print("✅ Loaded recommender engine")

    def recommend(self, title, top_n=5):
        if title not in self.main_df['title'].values:
            print(f"❌ Movie '{title}' not found in dataset.")
            return
        index = self.main_df[self.main_df['title'] == title].index[0]
        distances = self.similarity[index]
        similar_items = sorted(
            list(enumerate(distances)),
            key=lambda x: x[1],
            reverse=True
        )[1:top_n + 1]
        recs = [self.main_df.iloc[i[0]]['title'] for i in similar_items]
        print(f"Recommendations for '{title}':")
        for rec in recs:
            print(f"  - {rec}")
        return recs  # Return list for programmatic use

