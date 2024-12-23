import blosc
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Set

class MusicRecommendationSystem:
    def __init__(self, data: pd.DataFrame, k_values: List[int] = [5, 10], similarity_threshold: float = 0.3):
        self.data = data
        self.k_values = k_values
        self.similarity_threshold = similarity_threshold
        self.tfidf = TfidfVectorizer(analyzer='word', stop_words='english')
        self.tfidf_matrix = None
        self.similarity_matrix = None
    
    def prepare_data(self):
        """Prepares TF-IDF matrix and similarity matrix"""
        self.tfidf_matrix = self.tfidf.fit_transform(self.data['text'])
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
        
    def get_recommendations(self, song_idx: int, k: int) -> List[int]:
        """Gets top k recommendations for a song"""
        if self.similarity_matrix is None:
            self.prepare_data()
            
        song_similarities = self.similarity_matrix[song_idx]
        similarity_scores = list(enumerate(song_similarities))
        sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        recommendations = [idx for idx, score in sorted_scores if idx != song_idx][:k]
        return recommendations
    
    def create_similar_songs_mapping(self) -> Dict[int, Set[int]]:
        """
        Creates a mapping of songs to their similar songs based on similarity threshold
        Returns dictionary with key=song index and value=set of similar song indices
        """
        if self.similarity_matrix is None:
            self.prepare_data()
            
        similar_songs = {}
        for i in range(len(self.data)):
            similar = set()
            similarities = self.similarity_matrix[i]
            for j, sim in enumerate(similarities):
                if i != j and sim >= self.similarity_threshold:
                    similar.add(j)
            similar_songs[i] = similar
        return similar_songs
# with open("similarity.pkl", "wb") as f:
#     f.write(compressed_pickle)