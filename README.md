# Movie Recommender System

A clean and user-friendly movie recommendation system built with Streamlit that displays movie posters from OMDB API.

## Features

- Hybrid recommendation system combining user-based collaborative filtering and genre content-based filtering
- Movie poster display using OMDB API
- Customizable number of recommendations
- Clean, minimalistic interface with movie details and IMDb links

## Installation

1. Clone this repository or download the files
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:

```bash
streamlit run movie_recommender_app.py
```

2. Open your browser and go to http://localhost:8501
3. Search for a movie title in the search box
4. Enter the number of recommendations you want to see
5. Browse the recommended movies with their posters and details

## How It Works

This system uses a hybrid approach for movie recommendations:

1. **User-Based Collaborative Filtering**: Uses Pearson correlation to find similar users based on their rating patterns
2. **Content-Based Filtering**: Uses movie genres to find similar movies
3. **Hybrid Approach**: Combines both methods to provide accurate recommendations

The system follows these steps:
- Creates a user-item matrix from the ratings data
- Calculates mean user ratings
- Performs mean-centered normalization
- Computes user similarity using Pearson correlation
- Calculates genre similarity using cosine similarity
- Makes predictions by combining both methods

## Dataset

The system uses the MovieLens dataset with the following files:
- `u.data`: Contains user ratings for movies
- `u.item`: Contains movie information including titles, genres, and IMDb URLs

## Developer

Developed by Ahsan 