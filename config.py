"""
Configuration settings for the Movie Recommendation System.
"""

import os
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class DataConfig:
    """Data-related configuration."""
    movies_file: str = 'Datasets/tmdb_5000_movies.csv'
    credits_file: str = 'Datasets/tmdb_5000_credits.csv'
    cache_file: str = 'similarity_cache.pkl'
    processed_data_cache: str = 'processed_movies.pkl'


@dataclass
class RecommendationConfig:
    """Recommendation algorithm configuration."""
    default_neighbors: int = 10
    max_cast_members: int = 4
    similarity_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.similarity_weights is None:
            self.similarity_weights = {
                'genres': 0.3,
                'cast': 0.25,
                'director': 0.2,
                'keywords': 0.25
            }


@dataclass
class UIConfig:
    """User interface configuration."""
    window_width: int = 900
    window_height: int = 700
    min_width: int = 800
    min_height: int = 600
    default_theme: str = "system"  # "light", "dark", or "system"
    color_theme: str = "blue"
    max_autocomplete_suggestions: int = 5
    default_top_movies_count: int = 20


@dataclass
class AppConfig:
    """Main application configuration."""
    debug_mode: bool = False
    enable_caching: bool = True
    auto_save_cache: bool = True
    log_level: str = "INFO"
    
    # Sub-configurations
    data: DataConfig = None
    recommendation: RecommendationConfig = None
    ui: UIConfig = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.recommendation is None:
            self.recommendation = RecommendationConfig()
        if self.ui is None:
            self.ui = UIConfig()


# Create default configuration instance
config = AppConfig()


def load_config_from_file(file_path: str = 'config.ini') -> AppConfig:
    """
    Load configuration from file.
    
    Args:
        file_path: Path to configuration file
        
    Returns:
        AppConfig instance
    """
    if not os.path.exists(file_path):
        print(f"Config file {file_path} not found, using defaults")
        return config
    
    try:
        import configparser
        
        parser = configparser.ConfigParser()
        parser.read(file_path)
        
        # Create new config with values from file
        new_config = AppConfig()
        
        # Data configuration
        if 'data' in parser:
            data_section = parser['data']
            new_config.data.movies_file = data_section.get('movies_file', new_config.data.movies_file)
            new_config.data.credits_file = data_section.get('credits_file', new_config.data.credits_file)
            new_config.data.cache_file = data_section.get('cache_file', new_config.data.cache_file)
        
        # Recommendation configuration
        if 'recommendation' in parser:
            rec_section = parser['recommendation']
            new_config.recommendation.default_neighbors = rec_section.getint('default_neighbors', new_config.recommendation.default_neighbors)
            new_config.recommendation.max_cast_members = rec_section.getint('max_cast_members', new_config.recommendation.max_cast_members)
        
        # UI configuration
        if 'ui' in parser:
            ui_section = parser['ui']
            new_config.ui.window_width = ui_section.getint('window_width', new_config.ui.window_width)
            new_config.ui.window_height = ui_section.getint('window_height', new_config.ui.window_height)
            new_config.ui.default_theme = ui_section.get('default_theme', new_config.ui.default_theme)
        
        # App configuration
        if 'app' in parser:
            app_section = parser['app']
            new_config.debug_mode = app_section.getboolean('debug_mode', new_config.debug_mode)
            new_config.enable_caching = app_section.getboolean('enable_caching', new_config.enable_caching)
        
        return new_config
        
    except Exception as e:
        print(f"Error loading config file: {e}")
        return config


def save_config_to_file(config_obj: AppConfig, file_path: str = 'config.ini'):
    """
    Save configuration to file.
    
    Args:
        config_obj: Configuration object to save
        file_path: Path to save configuration file
    """
    try:
        import configparser
        
        parser = configparser.ConfigParser()
        
        # Data section
        parser['data'] = {
            'movies_file': config_obj.data.movies_file,
            'credits_file': config_obj.data.credits_file,
            'cache_file': config_obj.data.cache_file
        }
        
        # Recommendation section
        parser['recommendation'] = {
            'default_neighbors': str(config_obj.recommendation.default_neighbors),
            'max_cast_members': str(config_obj.recommendation.max_cast_members)
        }
        
        # UI section
        parser['ui'] = {
            'window_width': str(config_obj.ui.window_width),
            'window_height': str(config_obj.ui.window_height),
            'default_theme': config_obj.ui.default_theme
        }
        
        # App section
        parser['app'] = {
            'debug_mode': str(config_obj.debug_mode),
            'enable_caching': str(config_obj.enable_caching)
        }
        
        with open(file_path, 'w') as f:
            parser.write(f)
        
        print(f"Configuration saved to {file_path}")
        
    except Exception as e:
        print(f"Error saving config file: {e}")


# Create sample config file if it doesn't exist
def create_sample_config():
    """Create a sample configuration file."""
    sample_path = 'config_sample.ini'
    if not os.path.exists(sample_path):
        save_config_to_file(config, sample_path)
        print(f"Sample configuration created at {sample_path}")


if __name__ == "__main__":
    create_sample_config()