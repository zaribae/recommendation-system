import pickle
import streamlit as st
import mmap
import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from typing import Tuple, List, Dict
import logging
import numpy as np
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    CLIENT_ID = "70a9fb89662f4dac8d07321b259eaad7"
    CLIENT_SECRET = "4d6710460d764fbbb8d8753dc094d131"
    DEFAULT_IMAGE = "https://i.postimg.cc/0QNxYz4V/social.png"
    NUM_RECOMMENDATIONS = 5
    SIMILARITY_THRESHOLD = 0.4
    

class SpotifyClient:
    def __init__(self):
        try:
            client_credentials_manager = SpotifyClientCredentials(
                client_id=Config.CLIENT_ID,
                client_secret=Config.CLIENT_SECRET
            )
            self.sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
        except Exception as e:
            logger.error(f"Failed to initialize Spotify client: {e}")
            st.error("Failed to connect to Spotify API. Please check your credentials.")
            raise

    def get_song_details(self, song_name: str, artist_name: str) -> Dict:
        """Fetch song details including album cover URL and Spotify URL."""
        try:
            search_query = f"track:{song_name} artist:{artist_name}"
            results = self.sp.search(q=search_query, type="track")

            if results and results["tracks"]["items"]:
                track = results["tracks"]["items"][0]
                return {
                    'album_cover_url': track["album"]["images"][0]["url"],
                    'spotify_url': track["external_urls"]["spotify"]
                }
            
            logger.warning(f"No results found for {song_name} by {artist_name}")
            return {
                'album_cover_url': Config.DEFAULT_IMAGE,
                'spotify_url': None
            }

        except Exception as e:
            logger.error(f"Error fetching song details: {e}")
            return {
                'album_cover_url': Config.DEFAULT_IMAGE,
                'spotify_url': None
            }

class MusicRecommender:
    def __init__(self, music_data_path: str, similarity_matrix_path: str):
        """Initialize the recommender with data and similarity matrix."""
        self.load_data(music_data_path, similarity_matrix_path)
        self.spotify_client = SpotifyClient()
        self._initialize_train_test_split()

    def load_data(self, music_data_path: str, similarity_matrix_path: str) -> None:
        """Load music data and similarity matrix."""
        try:
            with open(music_data_path, 'rb') as f:
                self.music_df = pickle.load(f)
            
            with open(similarity_matrix_path, 'rb') as f:
                self.similarity = pickle.load(f)
                
            logger.info("Successfully loaded music data and similarity matrix")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            st.error("Failed to load recommendation data. Please check your data files.")
            raise

    def _initialize_train_test_split(self, test_size: float = 0.2):
        """Initialize train-test split for evaluation."""
        self.train_indices, self.test_indices = train_test_split(
            self.music_df.index,
            test_size=test_size,
            random_state=42
        )

    def get_recommendations(self, song: str) -> Tuple[List[Dict[str, str]], List[Dict], List[float]]:
        """Get music recommendations for a given song."""
        try:
            index = self.music_df[self.music_df['song'] == song].index[0]
            selected_artist = self.music_df[self.music_df['song'] == song]['artist'].iloc[0]
            
            distances = sorted(
                list(enumerate(self.similarity[index])), 
                reverse=True, 
                key=lambda x: x[1]
            )[1:Config.NUM_RECOMMENDATIONS + 1]

            recommended_songs = []
            song_details = []
            similarity_scores = []

            for i in distances:
                song_data = self.music_df.iloc[i[0]]
                recommended_songs.append({
                    'song': song_data.song,
                    'artist': song_data.artist
                })
                
                details = self.spotify_client.get_song_details(song_data.song, song_data.artist)
                song_details.append(details)
                similarity_scores.append(i[1])

            return recommended_songs, song_details, similarity_scores

        except IndexError:
            logger.error(f"Song '{song}' not found in database")
            st.error(f"Song '{song}' not found in our database")
            return [], [], []
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            st.error("An error occurred while generating recommendations")
            return [], [], []

    def evaluate_recommendations(self, num_samples: int = 100) -> Dict[str, float]:
        """
        Evaluate the recommendation system using precision and recall metrics.
        
        Args:
            num_samples: Number of test songs to evaluate
            
        Returns:
            Dictionary containing evaluation metrics
        """
        try:
            precisions = []
            recalls = []
            
            # Sample test indices to evaluate
            eval_indices = np.random.choice(self.test_indices, 
                                          size=min(num_samples, len(self.test_indices)), 
                                          replace=False)
            
            for test_idx in eval_indices:
                test_song = self.music_df.iloc[test_idx]['song']
                recommendations, _, similarity_scores = self.get_recommendations(test_song)
                
                if not recommendations:
                    continue
                
                # Get recommended song indices
                recommended_indices = [
                    self.music_df[
                        (self.music_df['song'] == rec['song']) & 
                        (self.music_df['artist'] == rec['artist'])
                    ].index[0]
                    for rec in recommendations
                ]
                
                # Get actual similar songs
                actual_similar = [
                    idx for idx, score in enumerate(self.similarity[test_idx])
                    if score > Config.SIMILARITY_THRESHOLD and idx != test_idx
                ]
                
                # Calculate metrics
                true_positives = len(set(recommended_indices) & set(actual_similar))
                precision = true_positives / len(recommendations) if recommendations else 0
                recall = true_positives / len(actual_similar) if actual_similar else 0
                
                precisions.append(precision)
                recalls.append(recall)
            
            # Calculate average metrics
            avg_precision = np.mean(precisions) if precisions else 0
            avg_recall = np.mean(recalls) if recalls else 0
            f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
            
            return {
                'precision': avg_precision,
                'recall': avg_recall,
                'f1_score': f1_score,
                'num_evaluated': len(precisions)
            }
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return {
                'precision': 0,
                'recall': 0,
                'f1_score': 0,
                'num_evaluated': 0
            }

    def get_song_display_text(self, song: str) -> Dict[str, str]:
        """Get song and artist information."""
        try:
            artist = self.music_df[self.music_df['song'] == song]['artist'].iloc[0]
            return {'song': song, 'artist': artist}
        except:
            return {'song': song, 'artist': 'Unknown Artist'}

def main():
    st.set_page_config(
        page_title="Music Recommender System",
        page_icon="🎵",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    st.header('🎵 Music Recommender System')

    try:
        recommender = MusicRecommender('./models/df.pkl', './models/similarity.pkl')
        
        # Create tabs for recommendation
        # tab1 = st.tabs(["Get Recommendations"])
        
        # with tab1:
        
        # Recommendations tab
        songs_with_artists = [f"{song} - {artist}" for song, artist in 
                            zip(recommender.music_df['song'].values, 
                                recommender.music_df['artist'].values)]
        
        selected_song_with_artist = st.selectbox(
            "Type or select a song from the dropdown",
            songs_with_artists
        )

        if selected_song_with_artist:
            selected_song = selected_song_with_artist.split(" - ")[0]
            song_info = recommender.get_song_display_text(selected_song)
            st.write(f"Selected: {song_info['song']} by {song_info['artist']}")

            if st.button('Show Recommendations'):
                with st.spinner('Generating recommendations...'):
                    recommended_songs, song_details, similarity_scores = recommender.get_recommendations(selected_song)

                    if recommended_songs and song_details:
                        cols = st.columns(Config.NUM_RECOMMENDATIONS)
                        
                        for idx, (col, song_info, details, score) in enumerate(
                            zip(cols, recommended_songs, song_details, similarity_scores)):
                            with col:
                                st.markdown(f"**{song_info['song']}**")
                                st.markdown(f"*by {song_info['artist']}*")
                                st.image(details['album_cover_url'], use_column_width=True)
                                st.markdown(f"Similarity: **{score:.1%}**")
                                if details['spotify_url']:
                                    st.markdown(f"[Listen on Spotify]({details['spotify_url']})")
        
        # with tab2:
        #     Evaluation tab
        #     st.subheader("System Evaluation")
        #     st.write("""
        #     This section evaluates the recommendation system's performance using standard metrics:
        #     - **Precision**: Proportion of recommended songs that are truly similar
        #     - **Recall**: Proportion of similar songs that were recommended
        #     - **F1 Score**: Harmonic mean of precision and recall
        #     """)
            
        #     num_samples = st.slider("Number of songs to evaluate", 
        #                           min_value=10, 
        #                           max_value=500, 
        #                           value=100, 
        #                           step=10)
            
        #     if st.button("Run Evaluation"):
        #         with st.spinner("Evaluating system performance..."):
        #             metrics = recommender.evaluate_recommendations(num_samples)
                    
        #             # Create three columns for metrics
        #             col1, col2, col3 = st.columns(3)
                    
        #             with col1:
        #                 st.metric("Precision", f"{metrics['precision']:.2%}")
        #             with col2:
        #                 st.metric("Recall", f"{metrics['recall']:.2%}")
        #             with col3:
        #                 st.metric("F1 Score", f"{metrics['f1_score']:.2%}")
                    
        #             st.info(f"Evaluation completed on {metrics['num_evaluated']} songs")

    except Exception as e:
        logger.error(f"Application error: {e.args}")
        st.error("An error occurred while starting the application. Please check the logs.")

if __name__ == "__main__":
    main()