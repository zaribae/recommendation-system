import pandas as pd
import numpy as np
import nltk
import pickle
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict
from pathlib import Path
from sklearn.model_selection import train_test_split

class SongRecommender:
    """A class to process song data and make recommendations based on lyrics similarity."""
    
    def __init__(self, csv_path: str = None, sample_size: int = None, test_size: float = 0.2):
        """
        Initialize the SongRecommender with data preprocessing capabilities.
        
        Args:
            csv_path: Path to the CSV file containing song data (optional if loading saved model)
            sample_size: Optional size of random sample to use
            test_size: Fraction of data to use for testing (default: 0.2)
        """
        self.stemmer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer(analyzer='word', stop_words='english')
        self.similarity_matrix = None
        self.df = None
        self.matrix = None
        
        if csv_path:
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
            stemmed = [self.stemmer.lemmatize(word) for word in tokens]
            return " ".join(stemmed)
        except LookupError:
            # nltk.download('punkt')
            nltk.download('wordnet')
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
            for song_idx, similarity_score in distances[1:]:
                if similarity_score > similarity_threshold:
                    recommendations.append(
                        (self.df.iloc[song_idx]['song'], similarity_score)
                    )
                    if len(recommendations) >= num_recommendations:
                        break
            
            return recommendations
        
        except IndexError:
            raise ValueError(f"Song '{song_name}' not found in the dataset")

    def evaluate_recommendations(self, song_title: str, artist_name: str = None, similarity_threshold: float = 0.4, 
                      num_recommendations: int = 20) -> Dict:
        """
        Evaluate the recommendation system for a specific song based on recommendations from get_recommendations.
        The evaluation compares the recommendations against the top N most similar songs from the similarity matrix.
        
        Args:
            song_title: Title of the song to evaluate
            artist_name: Name of the artist (optional, helps disambiguate songs with same title)
            similarity_threshold: Minimum similarity score to include in recommendations
            num_recommendations: Number of recommendations to evaluate
            
        Returns:
            Dictionary containing evaluation metrics and detailed recommendation analysis
        """
        try:
            # Find the index of the input song
            if artist_name:
                song_mask = (self.df['song'] == song_title) & (self.df['artist'] == artist_name)
                if not any(song_mask):
                    raise ValueError(f"Song '{song_title}' by {artist_name} not found in the dataset")
                song_idx = self.df[song_mask].index[0]
            else:
                song_mask = self.df['song'] == song_title
                if not any(song_mask):
                    raise ValueError(f"Song '{song_title}' not found in the dataset")
                if sum(song_mask) > 1:
                    artists = self.df[song_mask]['artist'].unique()
                    suggestion = "\nPlease specify the artist name. Available artists for this song:"
                    for artist in artists:
                        suggestion += f"\n- {artist}"
                    raise ValueError(f"Multiple songs found with title '{song_title}'. {suggestion}")
                song_idx = self.df[song_mask].index[0]
            
            # Get recommendations using get_recommendations
            recommendations = self.get_recommendations(
                song_title, 
                similarity_threshold, 
                num_recommendations
            )

            print(recommendations)
            
            if not recommendations:
                print(f"\n⚠️ No recommendations found for '{song_title}'{f' by {artist_name}' if artist_name else ''} with current threshold")
                return {
                    'error': f"No recommendations found for '{song_title}'{f' by {artist_name}' if artist_name else ''} with current threshold",
                    'recommendations': []
                }

            # Get ground truth: top N most similar songs from similarity matrix
            all_similarities = [(idx, score) for idx, score in 
                            enumerate(self.similarity_matrix[song_idx])
                            if idx != song_idx]
            all_similarities.sort(key=lambda x: x[1], reverse=True)
            ground_truth = all_similarities[:num_recommendations]
            
            # Get indices of recommended songs
            recommended_indices = [
                self.df[self.df['song'] == song].index[0] 
                for song, _ in recommendations
            ]
            
            # Calculate metrics
            ground_truth_indices = [idx for idx, _ in ground_truth]
            
            true_positives = len(set(recommended_indices) & set(ground_truth_indices))
            false_positives = len(set(recommended_indices) - set(ground_truth_indices))
            false_negatives = len(set(ground_truth_indices) - set(recommended_indices))
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            

            
            # Prepare detailed results for recommended songs
            recommended_songs = [
                {
                    'song': song,
                    'artist': self.df.iloc[self.df[self.df['song'] == song].index[0]]['artist'],
                    'similarity': score,
                    'rank': idx + 1,
                    'in_ground_truth': self.df[self.df['song'] == song].index[0] in ground_truth_indices
                }
                for idx, (song, score) in enumerate(recommendations)
            ]
            
            # Prepare ground truth details
            ground_truth_songs = [
                {
                    'song': self.df.iloc[idx]['song'],
                    'artist': self.df.iloc[idx]['artist'],
                    'similarity': score,
                    'rank': idx + 1,
                    'was_recommended': idx in recommended_indices
                }
                for idx, (idx, score) in enumerate(ground_truth)
            ]
            
            results = {
                'input_song': {
                    'title': song_title,
                    'artist': self.df.iloc[song_idx]['artist']
                },
                'metrics': {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score
                },
                'statistics': {
                    'num_recommendations': len(recommendations),
                    'num_ground_truth': len(ground_truth),
                    'num_true_positives': true_positives,
                    'mean_similarity': np.mean([score for _, score in recommendations]),
                    'max_similarity': max([score for _, score in recommendations]),
                    'min_similarity': min([score for _, score in recommendations])
                },
                'recommended_songs': recommended_songs,
                'ground_truth_songs': ground_truth_songs,
                'threshold_used': similarity_threshold
            }
            
            print(f"True Positives", true_positives)
            print(f"False Positives", false_positives)
            print(f"False Negatives", false_negatives)
            print(f"Recommended Indices", len(recommended_indices))
            print(f"Ground Truth Indices", len(ground_truth_indices))
            
            # Print detailed evaluation results
            print("\n" + "="*60)
            print(f"📊 Evaluation Results for '{results['input_song']['title']}'")
            print(f"🎤 Artist: {results['input_song']['artist']}")
            print("="*60)
            
            print("\n📈 Metrics:")
            print(f"Precision: {results['metrics']['precision']:.3f}")
            print(f"Recall: {results['metrics']['recall']:.3f}")
            print(f"F1 Score: {results['metrics']['f1_score']:.3f}")
            
            print("\n📊 Statistics:")
            stats = results['statistics']
            print(f"Number of recommendations: {stats['num_recommendations']}")
            print(f"Number of ground truth songs: {stats['num_ground_truth']}")
            print(f"Number of correct recommendations: {stats['num_true_positives']}")
            print(f"Mean similarity score: {stats['mean_similarity']:.3f}")
            print(f"Maximum similarity: {stats['max_similarity']:.3f}")
            print(f"Minimum similarity: {stats['min_similarity']:.3f}")
            print(f"Similarity threshold used: {results['threshold_used']}")
            
            print("\n🎵 Recommended Songs:")
            print("-"*60)
            print(f"{'Rank':^6}| {'Song':^25}| {'Artist':^15}| {'Similarity':^10}| {'In Ground Truth':^12}")
            print("-"*60)
            
            for song in results['recommended_songs']:
                check = "✓" if song['in_ground_truth'] else "✗"
                print(f"{song['rank']:^6}| {song['song'][:23]:25}| {song['artist'][:13]:15}| "
                    f"{song['similarity']:^10.3f}| {check:^12}")
            
            print("\n📋 Ground Truth Songs (Top Most Similar):")
            print("-"*60)
            print(f"{'Rank':^6}| {'Song':^25}| {'Artist':^15}| {'Similarity':^10}| {'Recommended':^12}")
            print("-"*60)
            
            for song in results['ground_truth_songs']:
                check = "✓" if song['was_recommended'] else "✗"
                print(f"{song['rank']:^6}| {song['song'][:23]:25}| {song['artist'][:13]:15}| "
                    f"{song['similarity']:^10.3f}| {check:^12}")
            
            print("-"*60)
            
            return results
                
        except ValueError as e:
            print(f"\n❌ Error: {str(e)}")
            raise

    def save_model(self, directory: str = "models") -> None:
        """Save the model artifacts to disk."""
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'similarity_matrix': self.similarity_matrix,
            'df': self.df,
            'matrix': self.matrix,
            'vectorizer': self.vectorizer
        }
        
        with open(f"{directory}/model_data.pkl", 'wb') as f:
            pickle.dump(model_data, f)

        with open(f"{directory}/similarity.pkl", 'wb') as f:
            pickle.dump(self.similarity_matrix, f)

        with open(f"{directory}/df.pkl", 'wb') as f:
            pickle.dump(self.df, f)
            
    @classmethod
    def load_model(cls, directory: str = "models") -> 'SongRecommender':
        """Load a previously saved model."""
        recommender = cls()
        
        with open(f"{directory}/model_data.pkl", 'rb') as f:
            model_data = pickle.load(f)
            
        recommender.similarity_matrix = model_data['similarity_matrix']
        recommender.df = model_data['df']
        recommender.matrix = model_data['matrix']
        recommender.vectorizer = model_data['vectorizer']
        recommender.stemmer = WordNetLemmatizer()
        
        return recommender

# Example usage:
if __name__ == "__main__":
    # Option 1: Initialize and train new recommender
    recommender = SongRecommender(csv_path="spotify_millsongdata.csv", sample_size=5000)
    # recommender.save_model()
    
    # Option 2: Load existing model and evaluate
    recommender = SongRecommender.load_model()

    # Example 1: With just song title (will raise error if multiple matches found)
    # try:
    #     evaluation = recommender.evaluate_recommendations(
    #         song_title="Respect",
    #         num_recommendations=5,
    #         similarity_threshold=0.5
    #     )
    # except ValueError as e:
    #     print(e)

    # Example 2: With song title and artist name
    evaluation = recommender.evaluate_recommendations(
        song_title="Bad Reputation",
        artist_name="Foo Fighters",
        similarity_threshold=0.4,
        num_recommendations=5
    )

    # Example 3: Evaluating multiple songs by same artist
    # songs = [
    #     ("Hello", "Adele"),
    #     ("Someone Like You", "Adele"),
    #     ("Rolling in the Deep", "Adele")
    # ]

    # for song_title, artist_name in songs:
    #     try:
    #         print(f"\nEvaluating: {song_title} by {artist_name}")
    #         evaluation = recommender.evaluate_recommendations(
    #             song_title=song_title,
    #             artist_name=artist_name,
    #             similarity_threshold=0.3,
    #             num_recommendations=5
    #         )
    #     except ValueError as e:
    #         print(f"Error: {e}")