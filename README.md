# Movie Recommendation System

The Movie Recommendation System is a machine learning-based Python project that suggests movies to users based on their preferences. It uses the K-Nearest Neighbors (KNN) algorithm to recommend movies similar to the ones a user likes, factoring in movie genres and user ratings.


## Features

- Recommends movies based on similar genres and user preferences.
- Uses the KNN algorithm for providing accurate movie suggestions.
- Works with a dataset of movies and ratings to make intelligent recommendations.
- Easy-to-use interface with customizable recommendation criteria.


## Installation

#### Prerequisites
Make sure you have Python 3.x and `pip` installed on your machine. You also need the following libraries:

```bash
pip install numpy pandas scikit-learn
```



    
## Datasets
The recommendation system uses a dataset that contains:
- Movie information (title, genres, etc.)
- User ratings for various movies

You can either use your own dataset or download a public dataset like `MovieLens`.
## How it Works
- The system reads the dataset containing movie titles, genres, and ratings.
- The `K-Nearest Neighbors (KNN)` algorithm is applied to find movies that are similar to the user’s preferences based on genres and ratings.
- The system then recommends movies that are closest to the ones the user has rated highly.
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
