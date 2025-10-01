import sys
import os
import warnings
import pickle
from typing import Optional

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import custom modules
from data_processor import MovieDataProcessor
from recommendation_engine import MovieRecommendationEngine
from gui import MovieRecommendationGUI
from config import load_config_from_file, config


class MovieRecommendationApp:
    """Main application class that orchestrates all components."""
    
    def __init__(self):
        """Initialize the application."""
        self.config = load_config_from_file()
        self.processor = None
        self.engine = None
        self.gui = None
        self.movies_df = None
    
    def check_dependencies(self) -> bool:
        """
        Check if all required dependencies are installed.
        
        Returns:
            True if all dependencies are available, False otherwise
        """
        required_packages = [
            'pandas', 'numpy', 'scipy', 'matplotlib', 
            'seaborn', 'customtkinter'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"Missing required packages: {', '.join(missing_packages)}")
            print("Please install them using: pip install " + ' '.join(missing_packages))
            return False
        
        return True
    
    def check_data_files(self) -> bool:
        """
        Check if required data files exist.
        
        Returns:
            True if data files exist, False otherwise
        """
        movies_path = self.config.data.movies_file
        credits_path = self.config.data.credits_file
        
        if not os.path.exists(movies_path):
            print(f"Error: Movies data file not found at '{movies_path}'")
            print("Please ensure the TMDB movies dataset is in the correct location.")
            return False
        
        if not os.path.exists(credits_path):
            print(f"Error: Credits data file not found at '{credits_path}'")
            print("Please ensure the TMDB credits dataset is in the correct location.")
            return False
        
        return True
    
    def load_or_process_data(self) -> bool:
        """
        Load processed data from cache or process raw data.
        
        Returns:
            True if data loading/processing succeeded, False otherwise
        """
        cache_path = self.config.data.processed_data_cache
        
        # Try to load from cache first
        if self.config.enable_caching and os.path.exists(cache_path):
            try:
                print("Loading processed data from cache...")
                with open(cache_path, 'rb') as f:
                    self.movies_df = pickle.load(f)
                print(f"Loaded {len(self.movies_df)} movies from cache")
                return True
            except Exception as e:
                print(f"Error loading cache: {e}")
                print("Falling back to data processing...")
        
        # Process data from scratch
        try:
            print("Starting data processing...")
            self.processor = MovieDataProcessor(
                movies_path=self.config.data.movies_file,
                credits_path=self.config.data.credits_file
            )
            
            self.movies_df = self.processor.process_all()
            
            # Save to cache if enabled
            if self.config.enable_caching:
                try:
                    with open(cache_path, 'wb') as f:
                        pickle.dump(self.movies_df, f)
                    print(f"Processed data cached to '{cache_path}'")
                except Exception as e:
                    print(f"Warning: Could not save cache: {e}")
            
            return True
            
        except Exception as e:
            print(f"Error processing data: {e}")
            return False
    
    def initialize_engine(self) -> bool:
        """
        Initialize the recommendation engine.
        
        Returns:
            True if initialization succeeded, False otherwise
        """
        try:
            print("Initializing recommendation engine...")
            self.engine = MovieRecommendationEngine(self.movies_df)
            print("Recommendation engine ready!")
            return True
        except Exception as e:
            print(f"Error initializing recommendation engine: {e}")
            return False
    
    def initialize_gui(self) -> bool:
        """
        Initialize the graphical user interface.
        
        Returns:
            True if initialization succeeded, False otherwise
        """
        try:
            print("Initializing GUI...")
            self.gui = MovieRecommendationGUI(self.engine)
            
            # Configure window size from config
            window_size = f"{self.config.ui.window_width}x{self.config.ui.window_height}"
            self.gui.root.geometry(window_size)
            self.gui.root.minsize(self.config.ui.min_width, self.config.ui.min_height)
            
            print("GUI ready!")
            return True
        except Exception as e:
            print(f"Error initializing GUI: {e}")
            return False
    
    def run_cli_mode(self):
        """Run the application in command-line mode."""
        print("\n" + "="*60)
        print("Movie Recommendation System - CLI Mode")
        print("="*60)
        
        while True:
            try:
                print("\nOptions:")
                print("1. Get movie recommendations")
                print("2. View top rated movies")
                print("3. Search movies")
                print("4. Show statistics")
                print("5. Exit")
                
                choice = input("\nEnter your choice (1-5): ").strip()
                
                if choice == '1':
                    self.cli_get_recommendations()
                elif choice == '2':
                    self.cli_show_top_movies()
                elif choice == '3':
                    self.cli_search_movies()
                elif choice == '4':
                    self.cli_show_statistics()
                elif choice == '5':
                    print("Thank you for using Movie Recommendation System!")
                    break
                else:
                    print("Invalid choice. Please try again.")
                    
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def cli_get_recommendations(self):
        """Get recommendations in CLI mode."""
        movie_title = input("Enter movie title: ").strip()
        if not movie_title:
            print("Please enter a valid movie title.")
            return
        
        print("Processing...")
        results = self.engine.predict_rating(movie_title)
        
        if 'error' in results:
            print(f"\n❌ {results['error']}")
            if 'suggestions' in results and results['suggestions']:
                print("\nDid you mean one of these?")
                for i, suggestion in enumerate(results['suggestions'][:5], 1):
                    print(f"{i}. {suggestion}")
            return
        
        # Display results
        target = results['target_movie']
        print(f"\n🎬 {target['title']}")
        print(f"Director: {target['director']} | Rating: {target['actual_rating']:.1f}/10")
        print(f"Genres: {target['genres']}")
        
        print(f"\nPredicted Rating: {results['predicted_rating']:.1f}/10")
        print(f"Prediction Accuracy: ±{results['prediction_accuracy']:.1f}")
        
        print("\n🌟 Recommended Movies:")
        print("-" * 80)
        for i, rec in enumerate(results['recommendations'], 1):
            print(f"{i:2d}. {rec['title']}")
            print(f"    Director: {rec['director']} | Rating: {rec['rating']:.1f}/10")
            print(f"    Similarity: {rec['similarity_score']*100:.1f}%")
            print()
    
    def cli_show_top_movies(self):
        """Show top movies in CLI mode."""
        try:
            n = int(input("How many top movies to show? (default: 20): ") or "20")
            top_movies = self.engine.get_top_rated_movies(n)
            
            print(f"\n🏆 Top {n} Rated Movies:")
            print("-" * 80)
            
            for i, (_, movie) in enumerate(top_movies.iterrows(), 1):
                print(f"{i:2d}. {movie['original_title']} ({movie['vote_average']:.1f}/10)")
                print(f"    Director: {movie['director']}")
                genres = movie['genres'] if isinstance(movie['genres'], list) else [movie['genres']]
                print(f"    Genres: {', '.join(genres)}")
                print()
                
        except ValueError:
            print("Please enter a valid number.")
    
    def cli_search_movies(self):
        """Search movies in CLI mode."""
        query = input("Enter search term: ").strip()
        if not query:
            print("Please enter a search term.")
            return
        
        suggestions = self.engine.get_movie_suggestions(query, max_suggestions=20)
        
        if suggestions:
            print(f"\n🔍 Movies matching '{query}':")
            print("-" * 50)
            for i, movie in enumerate(suggestions, 1):
                print(f"{i:2d}. {movie}")
        else:
            print(f"No movies found matching '{query}'")
    
    def cli_show_statistics(self):
        """Show dataset statistics in CLI mode."""
        stats = self.engine.get_genre_statistics()
        
        print(f"\n📊 Dataset Statistics:")
        print("-" * 40)
        print(f"Total movies: {len(self.movies_df)}")
        print(f"Unique genres: {stats.get('total_unique_genres', 'N/A')}")
        
        if 'most_common_genres' in stats:
            print("\nTop 10 Genres:")
            for genre, count in list(stats['most_common_genres'].items())[:10]:
                print(f"  {genre}: {count} movies")
        
        # Rating statistics
        ratings = self.movies_df['vote_average']
        print(f"\nRating Statistics:")
        print(f"  Average rating: {ratings.mean():.1f}")
        print(f"  Highest rating: {ratings.max():.1f}")
        print(f"  Lowest rating: {ratings.min():.1f}")
    
    def run(self, cli_mode: bool = False):
        """
        Run the application.
        
        Args:
            cli_mode: If True, run in command-line mode instead of GUI
        """
        print("🎬 Movie Recommendation System v1.0")
        print("=====================================\n")
        
        # Check dependencies
        if not self.check_dependencies():
            return False
        
        # Check data files
        if not self.check_data_files():
            return False
        
        # Load/process data
        if not self.load_or_process_data():
            return False
        
        # Initialize engine
        if not self.initialize_engine():
            return False
        
        if cli_mode:
            # Run in CLI mode
            self.run_cli_mode()
        else:
            # Initialize and run GUI
            if not self.initialize_gui():
                return False
            
            print("Starting GUI application...")
            self.gui.run()
        
        return True


def main():
    """Main entry point for the application."""
    # Check command line arguments
    cli_mode = '--cli' in sys.argv or '--terminal' in sys.argv
    
    try:
        app = MovieRecommendationApp()
        success = app.run(cli_mode=cli_mode)
        
        if not success:
            print("\nApplication failed to start. Please check the error messages above.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nApplication interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nCritical error: {e}")
        if config.debug_mode:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()