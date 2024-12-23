import pandas as pd
import numpy as np
import nltk
import pickle
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
from pathlib import Path

class SongRecommender:
    """A class to process song data and make recommendations based on lyrics similarity."""
    
    def __init__(self, csv_path: str, sample_size: int = None):
        """
        Initialize the SongRecommender with data preprocessing capabilities.
        
        Args:
            csv_path: Path to the CSV file containing song data
            sample_size: Optional size of random sample to use
        """
        self.stemmer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer(analyzer='word', stop_words='english')
        self.similarity_matrix = None
        self.df = None
        self.matrix = None
        self._load_and_process_data(csv_path, sample_size)
        
    def _load_and_process_data(self, csv_path: str, sample_size: int = None) -> None:
        """
        Load and preprocess the song dataset.
        
        Args:
            csv_path: Path to the CSV file
            sample_size: Optional size of random sample to use
        """
        # Load and sample data
        self.df = pd.read_csv(csv_path)
        if sample_size:
            self.df = self.df.sample(sample_size)
        
        # Clean and preprocess
        self.df = (self.df
                  .drop('link', axis=1)
                  .drop_duplicates()
                  .reset_index(drop=True))
        
        # Process text
        self.df['text'] = (self.df['text']
                          .str.lower()
                          .replace(r'^\w\s', ' ', regex=True)
                          .replace(r'^\r\n', ' ', regex=True)
                          .apply(self._tokenize_and_stem))
        
        # Create similarity matrix
        self._create_similarity_matrix()
    
    def _tokenize_and_stem(self, text: str) -> str:
        """
        Tokenize and stem the input text.
        
        Args:
            text: Input text to process
            
        Returns:
            Processed text with stemmed words
        """
        try:
            tokens = nltk.word_tokenize(text)
            stemmed = [self.stemmer.stem(word) for word in tokens]
            return " ".join(stemmed)
        except LookupError:
            nltk.download('punkt')
            return self._tokenize_and_stem(text)
    
    def _create_similarity_matrix(self) -> None:
        """Create TF-IDF matrix and calculate cosine similarity."""
        self.matrix = self.vectorizer.fit_transform(self.df['text'])
        self.similarity_matrix = cosine_similarity(self.matrix)
    
    def get_recommendations(self, song_name: str, similarity_threshold: float = 0.4, 
                          num_recommendations: int = 20) -> List[Tuple[str, float]]:
        """
        Get song recommendations based on lyrics similarity.
        
        Args:
            song_name: Name of the song to base recommendations on
            similarity_threshold: Minimum similarity score to include in recommendations
            num_recommendations: Maximum number of recommendations to return
            
        Returns:
            List of tuples containing (song_name, similarity_score)
        """
        try:
            # Find the index of the input song
            idx = self.df[self.df['song'] == song_name].index[0]
            
            # Get similarity scores and sort them
            distances = list(enumerate(self.similarity_matrix[idx]))
            distances.sort(key=lambda x: x[1], reverse=True)
            
            # Filter and format recommendations
            recommendations = []
            for song_idx, similarity_score in distances[1:num_recommendations + 1]:
                if similarity_score > similarity_threshold:
                    recommendations.append(
                        (self.df.iloc[song_idx]['song'], similarity_score)
                    )
            
            return recommendations
        
        except IndexError:
            raise ValueError(f"Song '{song_name}' not found in the dataset")
    
    def save_model(self, directory: str = "models") -> None:
        """
        Save the model artifacts to disk.
        
        Args:
            directory: Directory to save the model files
        """
        # Create directory if it doesn't exist
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Save similarity matrix and dataframe
        with open(f"{directory}/similarity.pkl", 'wb') as f:
            pickle.dump(self.similarity_matrix, f)
        
        with open(f"{directory}/df.pkl", 'wb') as f:
            pickle.dump(self.df, f)
            
    @classmethod
    def load_model(cls, directory: str = "models") -> 'SongRecommender':
        """
        Load a previously saved model.
        
        Args:
            directory: Directory containing the model files
            
        Returns:
            Initialized SongRecommender instance
        """
        recommender = cls.__new__(cls)
        
        # Load similarity matrix and dataframe
        with open(f"{directory}/similarity.pkl", 'rb') as f:
            recommender.similarity_matrix = pickle.load(f)
            
        with open(f"{directory}/df.pkl", 'rb') as f:
            recommender.df = pickle.load(f)
            
        recommender.stemmer = WordNetLemmatizer()
        recommender.vectorizer = TfidfVectorizer(analyzer='word', stop_words='english')
        
        return recommender

# Example usage:
if __name__ == "__main__":
    # Initialize and train the recommender
    recommender = SongRecommender("spotify_millsongdata.csv", sample_size=25000)
    
    # Get recommendations
    recommendations = recommender.get_recommendations("Someone Like You")
    
    # Print recommendations
    print("\nRecommended songs:")
    for song, score in recommendations:
        print(f"- {song} (similarity: {score:.2f})")
    
    # Save the model
    recommender.save_model()