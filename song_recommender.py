import pandas as pd
import numpy as np
import nltk
import pickle
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET

class SongRecommender:
    """A class to process song data and make recommendations based on lyrics similarity."""
    
    def __init__(self, csv_path: str, sample_size: int = None):
        """
        Initialize the SongRecommender with data preprocessing capabilities.
        
        Args:
            csv_path: Path to the CSV file containing song data
            sample_size: Optional size of random sample to use
        """
        self.stemmer = PorterStemmer()
        self.vectorizer = TfidfVectorizer(analyzer='word', stop_words='english')
        self.similarity_matrix = None
        self.df = None
        self.matrix = None
        
        # Initialize Spotify client
        self.sp = spotipy.Spotify(
            client_credentials_manager=SpotifyClientCredentials(
                client_id=SPOTIFY_CLIENT_ID,
                client_secret=SPOTIFY_CLIENT_SECRET
            )
        )
        
        self._load_and_process_data(csv_path, sample_size)

        # Auto-save after initialization
        self.save_model()
        
    def _load_and_process_data(self, csv_path: str, sample_size: int = None) -> None:
        """
        Load and preprocess the song dataset.
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
        
        # Create a combined column for display
        self.df['display_name'] = self.df['song'] + ' - ' + self.df['artist']
        
        # Process text
        self.df['text'] = (self.df['text']
                          .str.lower()
                          .replace(r'^\w\s', ' ', regex=True)
                          .replace(r'^\r\n', ' ', regex=True)
                          .apply(self._tokenize_and_stem))
        
        # Create similarity matrix
        self._create_similarity_matrix()
        
        # Add Spotify track IDs (this might take some time for large datasets)
        self.df['spotify_id'] = None
        self.df['spotify_url'] = None
        self.df['album_cover_url'] = None
        
    def _tokenize_and_stem(self, text: str) -> str:
        """Tokenize and stem the input text."""
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
    
    def get_spotify_track_info(self, song_name: str, artist_name: str) -> Optional[Dict]:
        """Get Spotify track information including ID, URL, and album cover."""
        try:
            # Search for the track
            query = f"track:{song_name} artist:{artist_name}"
            results = self.sp.search(q=query, type='track', limit=1)
            
            if results['tracks']['items']:
                track = results['tracks']['items'][0]
                return {
                    'id': track['id'],
                    'url': track['external_urls']['spotify'],
                    'album_cover_url': track['album']['images'][0]['url'] if track['album']['images'] else None
                }
            return None
        except Exception as e:
            print(f"Error getting Spotify info for {song_name}: {e}")
            return None
    
    def evaluate_recommendations(self, test_songs: List[str], threshold: float = 0.3) -> Dict[str, float]:
        """
        Evaluate the recommendation system using precision and recall metrics.
        
        Args:
            test_songs: List of song display names to evaluate
            threshold: Similarity threshold for considering a recommendation relevant
            
        Returns:
            Dictionary containing precision and recall scores
        """
        precisions = []
        recalls = []
        
        for song in test_songs:
            try:
                # Get recommendations for the song
                recommendations = self.get_recommendations(
                    song,
                    similarity_threshold=threshold,
                    num_recommendations=10
                )
                
                # Get actual relevant songs (those with similarity > threshold)
                song_idx = self.df[self.df['display_name'] == song].index[0]
                actual_relevant = self.similarity_matrix[song_idx] > threshold
                
                # Get predicted relevant songs (the recommendations)
                rec_indices = [self.df[
                    (self.df['song'] == rec[0]) & 
                    (self.df['artist'] == rec[1])
                ].index[0] for rec in recommendations]
                predicted_relevant = np.zeros(len(self.df))
                predicted_relevant[rec_indices] = 1
                
                # Calculate metrics
                precision = precision_score(actual_relevant, predicted_relevant)
                recall = recall_score(actual_relevant, predicted_relevant)
                
                precisions.append(precision)
                recalls.append(recall)
                
            except Exception as e:
                print(f"Error evaluating {song}: {e}")
                continue
        
        # Calculate average metrics
        avg_precision = np.mean(precisions) if precisions else 0
        avg_recall = np.mean(recalls) if recalls else 0
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        
        return {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1_score': f1_score
        }


    def get_recommendations(self, display_name: str, similarity_threshold: float = 0.4, 
                          num_recommendations: int = 20) -> List[Tuple[str, str, float, Optional[str], Optional[str], Optional[str]]]:
        """Get song recommendations based on lyrics similarity."""
        try:
            # Find the index of the input song
            song_idx = self.df[self.df['display_name'] == display_name].index[0]
            song_row = self.df.iloc[song_idx]
            
            # Get Spotify info for the selected song if not already cached
            if pd.isna(song_row['spotify_id']):
                song_name, artist_name = display_name.split(" - ", 1)
                spotify_info = self.get_spotify_track_info(song_name, artist_name)
                if spotify_info:
                    self.df.loc[song_idx, 'spotify_id'] = spotify_info['id']
                    self.df.loc[song_idx, 'spotify_url'] = spotify_info['url']
                    self.df.loc[song_idx, 'album_cover_url'] = spotify_info['album_cover_url']
            
            # Get similarity scores and sort them
            distances = list(enumerate(self.similarity_matrix[song_idx]))
            distances.sort(key=lambda x: x[1], reverse=True)
            
            # Filter and format recommendations
            recommendations = []
            for idx, similarity_score in distances[1:num_recommendations + 1]:
                if similarity_score > similarity_threshold:
                    row = self.df.iloc[idx]
                    
                    # Get Spotify info if not already cached
                    if pd.isna(row['spotify_id']):
                        spotify_info = self.get_spotify_track_info(row['song'], row['artist'])
                        if spotify_info:
                            self.df.loc[idx, 'spotify_id'] = spotify_info['id']
                            self.df.loc[idx, 'spotify_url'] = spotify_info['url']
                            self.df.loc[idx, 'album_cover_url'] = spotify_info['album_cover_url']
                    
                    recommendations.append(
                        (row['song'],
                         row['artist'],
                         similarity_score,
                         row['spotify_id'],
                         row['spotify_url'],
                         row['album_cover_url'])
                    )
            
            return recommendations
        
        except IndexError:
            raise ValueError(f"Song '{display_name}' not found in the dataset")
    
    def save_model(self, directory: str = "models") -> None:
        """Save the model artifacts to disk."""
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
            print(f"Saving model to {directory}...")
            with open(f"{directory}/similarity.pkl", 'wb') as f:
                pickle.dump(self.similarity_matrix, f)
            with open(f"{directory}/df.pkl", 'wb') as f:
                pickle.dump(self.df, f)
            print("Model saved successfully!")
            
        except Exception as e:
            print(f"Error saving model: {e}")
            raise e
            
    @classmethod
    def load_model(cls, directory: str = "models") -> 'SongRecommender':
        """Load a previously saved model."""
        recommender = cls.__new__(cls)
        
        try:
            with open(f"{directory}/similarity.pkl", 'rb') as f:
                recommender.similarity_matrix = pickle.load(f)
            with open(f"{directory}/df.pkl", 'rb') as f:
                recommender.df = pickle.load(f)
                
            recommender.stemmer = PorterStemmer()
            recommender.vectorizer = TfidfVectorizer(analyzer='word', stop_words='english')
            
            # Initialize Spotify client
            recommender.sp = spotipy.Spotify(
                client_credentials_manager=SpotifyClientCredentials(
                    client_id=SPOTIFY_CLIENT_ID,
                    client_secret=SPOTIFY_CLIENT_SECRET
                )
            )
            
            return recommender
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e