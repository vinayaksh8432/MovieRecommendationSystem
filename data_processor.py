"""
Data preprocessing module for the Movie Recommendation System.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
import json
import os

from utils import (
    parse_json_column, clean_string_column, create_binary_encoding,
    get_unique_items, safe_string_conversion, extract_director
)


class MovieDataProcessor:
    """Class to handle movie data preprocessing and feature engineering."""
    
    def __init__(self, movies_path: str = 'Datasets/tmdb_5000_movies.csv',
                 credits_path: str = 'Datasets/tmdb_5000_credits.csv'):
        """
        Initialize the data processor.
        
        Args:
            movies_path: Path to movies CSV file
            credits_path: Path to credits CSV file
        """
        self.movies_path = movies_path
        self.credits_path = credits_path
        self.movies_df = None
        self.processed_movies = None
        
        # Feature lists for binary encoding
        self.genre_list = []
        self.cast_list = []
        self.director_list = []
        self.keywords_list = []
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load movies and credits data from CSV files.
        
        Returns:
            Tuple of (movies_df, credits_df)
            
        Raises:
            FileNotFoundError: If CSV files are not found
        """
        if not os.path.exists(self.movies_path):
            raise FileNotFoundError(f"Movies file not found: {self.movies_path}")
        if not os.path.exists(self.credits_path):
            raise FileNotFoundError(f"Credits file not found: {self.credits_path}")
        
        try:
            movies_df = pd.read_csv(self.movies_path)
            credits_df = pd.read_csv(self.credits_path)
            
            print(f"Loaded {len(movies_df)} movies and {len(credits_df)} credits records")
            return movies_df, credits_df
            
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def process_json_columns(self, movies_df: pd.DataFrame, credits_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process JSON columns in both dataframes.
        
        Args:
            movies_df: Movies dataframe
            credits_df: Credits dataframe
            
        Returns:
            Tuple of processed dataframes
        """
        print("Processing JSON columns...")
        
        # Process movies JSON columns
        movies_df['genres'] = parse_json_column(movies_df['genres'])
        movies_df['keywords'] = parse_json_column(movies_df['keywords'])
        movies_df['production_companies'] = parse_json_column(movies_df['production_companies'])
        
        # Process credits JSON columns
        credits_df['cast'] = parse_json_column(credits_df['cast'])
        
        # Extract director from crew
        credits_df['crew'] = credits_df['crew'].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )
        credits_df['director'] = credits_df['crew'].apply(extract_director)
        credits_df.drop('crew', axis=1, inplace=True)
        
        return movies_df, credits_df
    
    def merge_dataframes(self, movies_df: pd.DataFrame, credits_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge movies and credits dataframes.
        
        Args:
            movies_df: Movies dataframe
            credits_df: Credits dataframe
            
        Returns:
            Merged dataframe
        """
        print("Merging dataframes...")
        
        merged_df = movies_df.merge(credits_df, left_on='id', right_on='movie_id', how='left')
        
        # Select relevant columns
        columns_to_keep = ['id', 'original_title', 'genres', 'cast', 'vote_average', 'director', 'keywords']
        merged_df = merged_df[columns_to_keep]
        
        return merged_df
    
    def process_genres(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process genres column and create binary encoding.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with processed genres
        """
        print("Processing genres...")
        
        # Convert list to comma-separated string and back to list for consistency
        df['genres'] = df['genres'].apply(lambda x: ','.join(x) if isinstance(x, list) else '')
        df['genres'] = df['genres'].str.split(',')
        
        # Get unique genres and create binary encoding
        self.genre_list = get_unique_items(df['genres'])
        df['genres_bin'] = create_binary_encoding(df['genres'], self.genre_list)
        
        return df
    
    def process_cast(self, df: pd.DataFrame, max_cast: int = 4) -> pd.DataFrame:
        """
        Process cast column and create binary encoding.
        
        Args:
            df: Input dataframe
            max_cast: Maximum number of cast members to consider
            
        Returns:
            Dataframe with processed cast
        """
        print("Processing cast...")
        
        # Clean and limit cast members
        df['cast'] = clean_string_column(df['cast'], max_items=max_cast)
        
        # Get unique cast members and create binary encoding
        self.cast_list = get_unique_items(df['cast'])
        df['cast_bin'] = create_binary_encoding(df['cast'], self.cast_list)
        
        return df
    
    def process_directors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process directors column and create binary encoding.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with processed directors
        """
        print("Processing directors...")
        
        # Handle None values in director column
        df['director'] = df['director'].apply(safe_string_conversion)
        
        # Get unique directors and create binary encoding
        self.director_list = get_unique_items(df['director'].apply(lambda x: [x] if x else []))
        df['director_bin'] = df['director'].apply(
            lambda x: [1 if director == x else 0 for director in self.director_list]
        )
        
        return df
    
    def process_keywords(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process keywords column and create binary encoding.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with processed keywords
        """
        print("Processing keywords...")
        
        # Clean keywords column
        df['keywords'] = clean_string_column(df['keywords'])
        
        # Get unique keywords and create binary encoding
        self.keywords_list = get_unique_items(df['keywords'])
        df['keywords_bin'] = create_binary_encoding(df['keywords'], self.keywords_list)
        
        return df
    
    def filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out movies with missing or invalid data.
        
        Args:
            df: Input dataframe
            
        Returns:
            Filtered dataframe
        """
        print("Filtering data...")
        
        # Remove movies with zero ratings or missing directors
        initial_count = len(df)
        df = df[df['vote_average'] != 0]
        df = df[df['director'] != '']
        
        print(f"Filtered out {initial_count - len(df)} movies with missing data")
        return df
    
    def add_new_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add new sequential ID column.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with new_id column
        """
        df['new_id'] = range(len(df))
        return df
    
    def select_final_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select final columns for the processed dataset.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with selected columns
        """
        final_columns = [
            'original_title', 'genres', 'vote_average', 'genres_bin',
            'cast_bin', 'new_id', 'director', 'director_bin', 'keywords_bin'
        ]
        return df[final_columns]
    
    def process_all(self) -> pd.DataFrame:
        """
        Run the complete data processing pipeline.
        
        Returns:
            Processed movies dataframe
        """
        print("Starting data processing pipeline...")
        
        try:
            # Load data
            movies_df, credits_df = self.load_data()
            
            # Process JSON columns
            movies_df, credits_df = self.process_json_columns(movies_df, credits_df)
            
            # Merge dataframes
            merged_df = self.merge_dataframes(movies_df, credits_df)
            
            # Process each feature type
            merged_df = self.process_genres(merged_df)
            merged_df = self.process_cast(merged_df)
            merged_df = self.process_directors(merged_df)
            merged_df = self.process_keywords(merged_df)
            
            # Filter and finalize
            merged_df = self.filter_data(merged_df)
            merged_df = self.add_new_id(merged_df)
            merged_df = self.select_final_columns(merged_df)
            
            self.processed_movies = merged_df
            print(f"Data processing complete! Final dataset has {len(merged_df)} movies")
            
            return merged_df
            
        except Exception as e:
            print(f"Error in data processing: {str(e)}")
            raise
    
    def get_feature_lists(self) -> dict:
        """
        Get all feature lists for reference.
        
        Returns:
            Dictionary containing all feature lists
        """
        return {
            'genres': self.genre_list,
            'cast': self.cast_list,
            'directors': self.director_list,
            'keywords': self.keywords_list
        }