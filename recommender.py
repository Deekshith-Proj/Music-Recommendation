import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class MusicRecommender:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        self.features = None
        self.feature_matrix = None
        self.song_titles = None

    def load_and_prepare_data(self):
        print("Loading dataset...")
        self.df = pd.read_csv(self.csv_path)

        # Drop duplicates and missing values
        self.df.drop_duplicates(subset=["track_name", "artist_name"], inplace=True)
        self.df.dropna(inplace=True)

        # Select features for similarity
        self.features = [
            "danceability",
            "energy",
            "loudness",
            "acousticness",
            "instrumentalness",
            "valence"
        ]

        self.feature_matrix = self.df[self.features].values
        self.song_titles = self.df["track_name"] + " - " + self.df["artist_name"]

        # Normalize features
        scaler = StandardScaler()
        self.feature_matrix = scaler.fit_transform(self.feature_matrix)
        print("Data prepared successfully.")

    def recommend(self, song_name, top_n=5):
        if self.df is None:
            raise ValueError("Data not loaded. Run load_and_prepare_data() first.")

        if song_name not in self.df["track_name"].values:
            print(f"'{song_name}' not found in the dataset.")
            return

        idx = self.df[self.df["track_name"] == song_name].index[0]
        song_vector = self.feature_matrix[idx].reshape(1, -1)

        # Compute cosine similarities
        similarities = cosine_similarity(song_vector, self.feature_matrix)[0]

        # Get top N similar songs
        similar_indices = similarities.argsort()[::-1][1 : top_n + 1]
        recommendations = self.song_titles.iloc[similar_indices]

        print(f"\nRecommendations similar to '{song_name}':\n")
        for i, title in enumerate(recommendations, 1):
            print(f"{i}. {title}")


def main():
    csv_file = "Data/tcc_ceds_music.csv"
    recommender = MusicRecommender(csv_file)
    recommender.load_and_prepare_data()

    # Change this to any song in your dataset
    input_song = input("\nEnter a song name to get recommendations: ").strip()
    recommender.recommend(input_song)


if __name__ == "__main__":
    main()
