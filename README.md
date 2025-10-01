# Movie Recommendation System 🎬

An advanced movie recommendation system built with Python that uses content-based filtering and cosine similarity to suggest movies based on user preferences. The system features a modern GUI, data caching, and comprehensive error handling.

## Features ✨

-   **Content-Based Filtering**: Uses movie genres, cast, directors, and keywords for recommendations
-   **Modern GUI**: Built with CustomTkinter for a sleek, responsive interface
-   **CLI Mode**: Command-line interface for terminal users
-   **Data Caching**: Intelligent caching system to improve performance
-   **Auto-complete Search**: Smart search suggestions as you type
-   **Statistical Analysis**: Genre distribution and rating analysis charts
-   **Top Movies**: Browse highest-rated movies in the dataset
-   **Error Handling**: Comprehensive error handling and user feedback

## Dataset 📊

This system uses the TMDb 5000 Movie Dataset:

-   **tmdb_5000_movies.csv**: Movie information including genres, keywords, and ratings
-   **tmdb_5000_credits.csv**: Cast and crew information

Download the dataset from [Kaggle](https://www.kaggle.com/tmdb/tmdb-movie-metadata) and place it in the `Datasets/` folder.

## Installation 🛠️

### Prerequisites

-   Python 3.7 or higher
-   pip package manager

### Setup

1. **Clone or download this repository**

    ```bash
    git clone <repository-url>
    cd MovieRecommendationSystem
    ```

2. **Install required packages**

    ```bash
    pip install pandas numpy scipy matplotlib seaborn customtkinter
    ```

3. **Download the dataset**

    - Download the TMDb dataset from Kaggle
    - Create a `Datasets` folder in the project directory
    - Place `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` in the `Datasets` folder

4. **Run the application**

    ```bash
    # GUI Mode (default)
    python main.py

    # CLI Mode
    python main.py --cli
    ```

## Project Structure 📁

```
MovieRecommendationSystem/
├── main.py                 # Main application entry point
├── data_processor.py       # Data preprocessing and cleaning
├── recommendation_engine.py # Core recommendation algorithms
├── gui.py                  # Graphical user interface
├── utils.py               # Utility functions
├── config.py              # Configuration management
├── README.md              # Project documentation
├── Datasets/              # Dataset files (user provided)
│   ├── tmdb_5000_movies.csv
│   └── tmdb_5000_credits.csv
└── Model/                 # Jupyter notebook for experimentation
    └── Model.ipynb
```

## Usage 🚀

### GUI Mode

1. **Search for Movies**: Type a movie title in the search box
2. **Get Recommendations**: Click "Get Recommendations" to see similar movies
3. **View Statistics**: Check out genre distribution and rating charts
4. **Browse Top Movies**: Explore the highest-rated movies in the dataset

### CLI Mode

Run with `python main.py --cli` to access:

-   Movie recommendations
-   Top-rated movies
-   Movie search
-   Dataset statistics

### Key Features Explained

#### Recommendation Algorithm

The system uses a weighted combination of:

-   **Genres** (30%): Movie categories and themes
-   **Cast** (25%): Main actors (top 4 per movie)
-   **Director** (20%): Film director
-   **Keywords** (25%): Movie-specific keywords and tags

#### Similarity Calculation

Uses cosine similarity to find movies with similar characteristics. Lower distances indicate higher similarity.

## Configuration ⚙️

The system supports configuration through `config.py`:

```python
# Example configuration
config = AppConfig(
    debug_mode=False,
    enable_caching=True,
    data=DataConfig(
        movies_file='Datasets/tmdb_5000_movies.csv',
        credits_file='Datasets/tmdb_5000_credits.csv'
    ),
    recommendation=RecommendationConfig(
        default_neighbors=10,
        max_cast_members=4
    ),
    ui=UIConfig(
        window_width=900,
        window_height=700,
        default_theme="system"
    )
)
```

## Performance Optimizations 🚀

-   **Data Caching**: Processed data is cached to avoid reprocessing
-   **Similarity Caching**: Similarity calculations are cached for faster subsequent queries
-   **Binary Encoding**: Efficient binary representation of categorical features
-   **Optimized Data Structures**: Pandas DataFrames for efficient data manipulation

## Troubleshooting 🔧

### Common Issues

1. **Missing Dataset Files**

    ```
    Error: Movies data file not found at 'Datasets/tmdb_5000_movies.csv'
    ```

    **Solution**: Download the TMDb dataset and place it in the `Datasets/` folder

2. **Missing Dependencies**

    ```
    Missing required packages: customtkinter
    ```

    **Solution**: Install missing packages with pip

3. **GUI Not Starting**

    - Ensure you have a display available (for GUI mode)
    - Try CLI mode: `python main.py --cli`

4. **Memory Issues**
    - Enable caching in config to reduce memory usage
    - Use CLI mode for lower memory footprint

### Performance Tips

-   **First Run**: Initial data processing takes time; subsequent runs are faster due to caching
-   **Memory**: System requires ~1-2GB RAM for full dataset processing
-   **Storage**: Cache files require additional ~100-200MB disk space

## Technical Details 🔬

### Data Preprocessing Pipeline

1. **JSON Parsing**: Extract information from nested JSON fields
2. **Feature Engineering**: Create binary encodings for categorical features
3. **Data Cleaning**: Remove movies with missing ratings or directors
4. **Feature Selection**: Select relevant columns for recommendation

### Recommendation Engine

1. **Feature Vectorization**: Convert movie features to binary vectors
2. **Similarity Calculation**: Compute cosine similarity between movies
3. **Neighbor Finding**: Identify K most similar movies
4. **Rating Prediction**: Weighted average of neighbor ratings

### GUI Architecture

-   **CustomTkinter**: Modern, themed GUI components
-   **Threading**: Non-blocking operations for better UX
-   **Matplotlib Integration**: Embedded charts and visualizations
-   **Auto-complete**: Real-time search suggestions

## Contributing 🤝

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Future Enhancements 🔮

-   [ ] **Collaborative Filtering**: Add user-based recommendations
-   [ ] **Deep Learning**: Implement neural network-based recommendations
-   [ ] **Web Interface**: Create a web-based GUI
-   [ ] **API Integration**: Connect to live movie databases
-   [ ] **User Profiles**: Add user preference learning
-   [ ] **Movie Posters**: Display movie images and posters

## License 📄

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments 🙏

-   **TMDb**: For providing the movie dataset
-   **Python Community**: For the excellent libraries used
-   **Contributors**: Everyone who has contributed to this project

## Support 💬

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Review the configuration options
3. Try both GUI and CLI modes
4. Open an issue with detailed error messages

---

**Built with ❤️ for movie enthusiasts and data science learners**

The Movie Recommendation System is a machine learning-based Python project that suggests movies to users based on their preferences. It uses the K-Nearest Neighbors (KNN) algorithm to recommend movies similar to the ones a user likes, factoring in movie genres and user ratings.

## Features

-   Recommends movies based on similar genres and user preferences.
-   Uses the KNN algorithm for providing accurate movie suggestions.
-   Works with a dataset of movies and ratings to make intelligent recommendations.
-   Easy-to-use interface with customizable recommendation criteria.

## Installation

#### Prerequisites

Make sure you have Python 3.x and `pip` installed on your machine. You also need the following libraries:

```bash
pip install numpy pandas scikit-learn
```

## Datasets

The recommendation system uses a dataset that contains:

-   Movie information (title, genres, etc.)
-   User ratings for various movies

You can either use your own dataset or download a public dataset like `MovieLens`.

## How it Works

-   The system reads the dataset containing movie titles, genres, and ratings.
-   The `K-Nearest Neighbors (KNN)` algorithm is applied to find movies that are similar to the user’s preferences based on genres and ratings.
-   The system then recommends movies that are closest to the ones the user has rated highly.

## Run Locally

Clone the project

```bash
git clone https://github.com/vinayaksh8432/MovieRecommendationSystem.git
```

Run the Application

```bash
python -u main.py
```

## Contributing

Contributions are always welcome!

Feel free to open issues or pull requests. Any suggestions or improvements are welcome!
