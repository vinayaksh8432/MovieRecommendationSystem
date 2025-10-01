"""
Recommendation engine for the Movie Recommendation System.
"""

import pandas as pd
import numpy as np
from scipy import spatial
import operator
from typing import List, Tuple, Optional
import pickle
import os


class MovieRecommendationEngine:
    """Class to handle movie similarity calculations and recommendations."""
    
    def __init__(self, movies_df: pd.DataFrame):
        """
        Initialize the recommendation engine.
        
        Args:
            movies_df: Processed movies dataframe
        """
        self.movies_df = movies_df.copy()
        self.similarity_cache = {}
        self.cache_file = 'similarity_cache.pkl'
        self.load_cache()
    
    def load_cache(self):
        """Load similarity cache from file if it exists."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    self.similarity_cache = pickle.load(f)
                print(f"Loaded {len(self.similarity_cache)} cached similarity calculations")
            except Exception as e:
                print(f"Error loading cache: {e}")
                self.similarity_cache = {}
    
    def save_cache(self):
        """Save similarity cache to file."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.similarity_cache, f)
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def calculate_similarity(self, movie_id1: int, movie_id2: int) -> float:
        """
        Calculate similarity between two movies using cosine similarity.
        
        Args:
            movie_id1: Index of first movie
            movie_id2: Index of second movie
            
        Returns:
            Similarity score (lower values indicate higher similarity)
        """
        # Check cache first
        cache_key = tuple(sorted([movie_id1, movie_id2]))
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        try:
            movie_a = self.movies_df.iloc[movie_id1]
            movie_b = self.movies_df.iloc[movie_id2]
            
            # Calculate genre similarity
            genres_distance = spatial.distance.cosine(
                movie_a['genres_bin'], movie_b['genres_bin']
            )
            
            # Calculate cast similarity
            cast_distance = spatial.distance.cosine(
                movie_a['cast_bin'], movie_b['cast_bin']
            )
            
            # Calculate director similarity
            director_distance = spatial.distance.cosine(
                movie_a['director_bin'], movie_b['director_bin']
            )
            
            # Calculate keywords similarity (fixed bug from original code)
            keywords_distance = spatial.distance.cosine(
                movie_a['keywords_bin'], movie_b['keywords_bin']
            )
            
            # Combine similarities with weights
            total_distance = (
                genres_distance * 0.3 +
                cast_distance * 0.25 +
                director_distance * 0.2 +
                keywords_distance * 0.25
            )
            
            # Cache the result
            self.similarity_cache[cache_key] = total_distance
            
            return total_distance
            
        except Exception as e:
            print(f"Error calculating similarity between {movie_id1} and {movie_id2}: {e}")
            return float('inf')
    
    def find_movie_by_title(self, title: str, exact_match: bool = False) -> Optional[pd.DataFrame]:
        """
        Find a movie by title.
        
        Args:
            title: Movie title to search for
            exact_match: Whether to require exact match or allow partial
            
        Returns:
            DataFrame row of the found movie or None
        """
        try:
            if exact_match:
                matches = self.movies_df[
                    self.movies_df['original_title'].str.lower() == title.lower()
                ]
            else:
                matches = self.movies_df[
                    self.movies_df['original_title'].str.contains(title, case=False, na=False)
                ]
            
            if len(matches) == 0:
                return None
            
            # Return the first match
            return matches.iloc[0].to_frame().T
            
        except Exception as e:
            print(f"Error finding movie '{title}': {e}")
            return None
    
    def get_movie_suggestions(self, partial_title: str, max_suggestions: int = 10) -> List[str]:
        """
        Get movie title suggestions based on partial input.
        
        Args:
            partial_title: Partial movie title
            max_suggestions: Maximum number of suggestions to return
            
        Returns:
            List of suggested movie titles
        """
        try:
            if len(partial_title) < 2:
                return []
            
            matches = self.movies_df[
                self.movies_df['original_title'].str.contains(
                    partial_title, case=False, na=False
                )
            ]['original_title'].head(max_suggestions).tolist()
            
            return matches
            
        except Exception as e:
            print(f"Error getting suggestions for '{partial_title}': {e}")
            return []
    
    def get_neighbors(self, base_movie: pd.DataFrame, k: int = 10) -> List[Tuple[int, float]]:
        """
        Find K nearest neighbors for a given movie.
        
        Args:
            base_movie: DataFrame containing the base movie
            k: Number of neighbors to find
            
        Returns:
            List of tuples (movie_index, distance)
        """
        try:
            base_movie_id = base_movie['new_id'].values[0]
            distances = []
            
            for index, movie in self.movies_df.iterrows():
                if movie['new_id'] != base_movie_id:
                    distance = self.calculate_similarity(base_movie_id, movie['new_id'])
                    distances.append((movie['new_id'], distance))
            
            # Sort by distance (ascending - smaller distance means more similar)
            distances.sort(key=operator.itemgetter(1))
            
            # Return top K neighbors
            return distances[:k]
            
        except Exception as e:
            print(f"Error finding neighbors: {e}")
            return []
    
    def predict_rating(self, movie_title: str, k: int = 10) -> dict:
        """
        Predict rating for a movie and get recommendations.
        
        Args:
            movie_title: Title of the movie to analyze
            k: Number of neighbors to consider
            
        Returns:
            Dictionary containing prediction results and recommendations
        """
        try:
            # Find the movie
            target_movie = self.find_movie_by_title(movie_title)
            if target_movie is None:
                return {
                    'error': f"Movie '{movie_title}' not found",
                    'suggestions': self.get_movie_suggestions(movie_title)
                }
            
            print(f"Selected Movie: {target_movie['original_title'].values[0]}")
            
            # Get neighbors
            neighbors = self.get_neighbors(target_movie, k)
            if not neighbors:
                return {'error': 'Could not find similar movies'}
            
            # Calculate recommendations
            recommendations = []
            total_rating = 0
            
            for neighbor_id, distance in neighbors:
                movie_info = self.movies_df[self.movies_df['new_id'] == neighbor_id].iloc[0]
                
                recommendation = {
                    'title': movie_info['original_title'],
                    'genres': ', '.join(movie_info['genres']) if isinstance(movie_info['genres'], list) else str(movie_info['genres']),
                    'rating': movie_info['vote_average'],
                    'similarity_score': 1 - distance,  # Convert distance to similarity
                    'director': movie_info['director']
                }
                
                recommendations.append(recommendation)
                total_rating += movie_info['vote_average']
            
            # Calculate predicted rating
            predicted_rating = total_rating / k
            actual_rating = target_movie['vote_average'].values[0]
            
            return {
                'target_movie': {
                    'title': target_movie['original_title'].values[0],
                    'genres': ', '.join(target_movie['genres'].values[0]) if isinstance(target_movie['genres'].values[0], list) else str(target_movie['genres'].values[0]),
                    'actual_rating': actual_rating,
                    'director': target_movie['director'].values[0]
                },
                'predicted_rating': predicted_rating,
                'recommendations': recommendations,
                'prediction_accuracy': abs(predicted_rating - actual_rating)
            }
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return {'error': f'Error during prediction: {str(e)}'}
    
    def get_top_rated_movies(self, n: int = 20) -> pd.DataFrame:
        """
        Get top N rated movies.
        
        Args:
            n: Number of top movies to return
            
        Returns:
            DataFrame with top rated movies
        """
        try:
            return self.movies_df.nlargest(n, 'vote_average')[
                ['original_title', 'vote_average', 'director', 'genres']
            ]
        except Exception as e:
            print(f"Error getting top rated movies: {e}")
            return pd.DataFrame()
    
    def get_genre_statistics(self) -> dict:
        """
        Get statistics about genres in the dataset.
        
        Returns:
            Dictionary with genre statistics
        """
        try:
            # Flatten all genres
            all_genres = []
            for genres_list in self.movies_df['genres']:
                if isinstance(genres_list, list):
                    all_genres.extend(genres_list)
            
            genre_counts = pd.Series(all_genres).value_counts()
            
            return {
                'total_unique_genres': len(genre_counts),
                'most_common_genres': genre_counts.head(10).to_dict(),
                'genre_distribution': genre_counts.to_dict()
            }
            
        except Exception as e:
            print(f"Error getting genre statistics: {e}")
            return {}
    
    def __del__(self):
        """Save cache when object is destroyed."""
        self.save_cache()