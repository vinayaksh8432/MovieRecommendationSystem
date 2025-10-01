"""
Utility functions for the Movie Recommendation System.
"""

import json
import pandas as pd
from typing import List, Any, Optional


def parse_json_column(data: pd.Series, key: str = 'name') -> pd.Series:
    """
    Parse JSON column and extract specified key values.
    
    Args:
        data: Pandas Series containing JSON strings
        key: Key to extract from JSON objects
        
    Returns:
        Pandas Series with extracted values as lists
    """
    def extract_names(json_str):
        try:
            if pd.isna(json_str):
                return []
            
            json_data = json.loads(json_str) if isinstance(json_str, str) else json_str
            if not isinstance(json_data, list):
                return []
                
            return [item.get(key, '') for item in json_data if isinstance(item, dict)]
        except (json.JSONDecodeError, TypeError):
            return []
    
    return data.apply(extract_names)


def clean_string_column(data: pd.Series, max_items: Optional[int] = None) -> pd.Series:
    """
    Clean and process string columns containing list-like data.
    
    Args:
        data: Pandas Series containing string representations of lists
        max_items: Maximum number of items to keep (None for all)
        
    Returns:
        Cleaned Pandas Series
    """
    # Remove brackets and quotes, split by comma
    cleaned = (data.astype(str)
               .str.strip('[]')
               .str.replace(' ', '', regex=False)
               .str.replace("'", '', regex=False)
               .str.replace('"', '', regex=False)
               .str.split(','))
    
    # Limit number of items if specified
    if max_items:
        cleaned = cleaned.apply(lambda x: x[:max_items] if isinstance(x, list) else x)
    
    # Sort items for consistency
    cleaned = cleaned.apply(lambda x: sorted(x) if isinstance(x, list) else x)
    
    return cleaned


def create_binary_encoding(data: pd.Series, unique_items: List[str]) -> pd.Series:
    """
    Create binary encoding for categorical data.
    
    Args:
        data: Pandas Series containing lists of items
        unique_items: List of all unique items to encode
        
    Returns:
        Pandas Series with binary encoded lists
    """
    def encode_items(item_list):
        if not isinstance(item_list, list):
            return [0] * len(unique_items)
        return [1 if item in item_list else 0 for item in unique_items]
    
    return data.apply(encode_items)


def get_unique_items(data: pd.Series) -> List[str]:
    """
    Extract all unique items from a series of lists.
    
    Args:
        data: Pandas Series containing lists
        
    Returns:
        List of unique items
    """
    unique_items = set()
    for item_list in data:
        if isinstance(item_list, list):
            unique_items.update(item_list)
    
    # Remove empty strings
    unique_items.discard('')
    return sorted(list(unique_items))


def safe_string_conversion(value: Any) -> str:
    """
    Safely convert value to string, handling None values.
    
    Args:
        value: Value to convert
        
    Returns:
        String representation
    """
    return '' if value is None else str(value)


def extract_director(crew_data: List[dict]) -> Optional[str]:
    """
    Extract director name from crew data.
    
    Args:
        crew_data: List of crew member dictionaries
        
    Returns:
        Director name or None if not found
    """
    if not isinstance(crew_data, list):
        return None
        
    for person in crew_data:
        if isinstance(person, dict) and person.get('job') == 'Director':
            return person.get('name')
    
    return None