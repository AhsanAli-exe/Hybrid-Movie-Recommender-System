import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from sklearn.metrics.pairwise import cosine_similarity

# Load data functions
@st.cache_data
def load_data1():
    """
    Loading the MovieLens dataset:
    - u.data: contains user ratings for movies
    - u.item: contains movie information including title, genre, etc.
    """
    ratingsDf = pd.read_csv(
        'u.data',
        sep='\t',
        names=['user_id','item_id','rating','timestamp']
    )
    ratingsDf = ratingsDf.drop('timestamp', axis=1)

    # Load movie data
    moviesDf = pd.read_csv(
        'u.item',
        sep='|',
        header=None,
        encoding='latin-1',
        names=['movie_id','title','release_date','video_release_date','IMDB_URL','unknown', 
               'Action','Adventure','Animation','Children','Comedy','Crime','Documentary', 
               'Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance', 
               'Sci-Fi','Thriller','War','Western']
    )
    moviesDf = moviesDf.drop(['video_release_date'], axis=1)
    
    return ratingsDf,moviesDf

def load_data():
    ratingsDf = pd.read_csv("ratings.csv",encoding='latin-1')
    moviesDf = pd.read_csv("movies.csv",encoding='latin-1')
    
    #extracting release date 
    moviesDf['release_date'] = moviesDf['title'].str.extract(r'\((\d{4})\)')
    
    # adding IMDB URL column if missing
    if 'IMDB_URL' not in moviesDf.columns:
        moviesDf['IMDB_URL'] = 'https://www.imdb.com/search/title/?title=' + moviesDf['title'].str.replace(r' \(\d{4}\)', '', regex=True)
    
    genre_list = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary',
                 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
                 'Sci-Fi', 'Thriller', 'War', 'Western']
    
    # first adding unknown genre column
    moviesDf['unknown'] = 0
    for genre in genre_list:
        moviesDf[genre] = moviesDf['genres'].str.contains(genre).astype(int)
    
    return ratingsDf, moviesDf

# Function to get movie poster from OMDB API
def get_movie_poster(movieTitle,apiKey="e49fdf78"):
    """Get movie poster from OMDB API using the movie title"""
    try:
        #cleaning the movie title by removing the year(if present)
        cleanTitle = re.sub(r'\(\d{4}\)','',movieTitle).strip()
        baseUrl = "http://www.omdbapi.com/"
        params = {"apikey": apiKey,"t": cleanTitle}
        
        #make the API call
        response = requests.get(baseUrl,params=params)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("Response") == "True" and "Poster" in data:
                posterUrl = data["Poster"]
                if posterUrl != "N/A":
                    return posterUrl,data
        
        #if no poster found, return None
        return None,None
    
    except Exception as e:
        print(f"Error fetching poster: {e}")
        return None,None


# this function creates a user-item matrix from the ratings data
def create_user_item_matrix(ratingsDf):
    """
    Example:
    If ratings_df contains:
    user_id | item_id | rating
    -------------------------
    1       | 101     | 5
    1       | 102     | 3
    2       | 101     | 4
    2       | 103     | 2
    
    The resulting user-item matrix will be:
       item_id | 101 | 102 | 103
    user_id    |     |     |
    -------------------------
    1          | 5   | 3   | NaN
    2          | 4   | NaN | 2
    """
    userItemMatrix = pd.pivot_table(
        ratingsDf, 
        values='rating', 
        index='user_id', 
        columns='item_id'
    )
    return userItemMatrix

#mean user ratings
def calculate_mean_user_ratings(userItemMatrix):
    """
    Example:
    If user_item_matrix is:
       | Movie1 | Movie2 | Movie3 | Movie4
    ---|--------|--------|--------|-------
    U1 | 5      | 3      | 4      | NaN
    U2 | 2      | 4      | 1      | 5
    
    The calculated means would be:
    U1: (5 + 3 + 4) / 3 = 4.0
    U2: (2 + 4 + 1 + 5) / 4 = 3.0
    
    Returns: series with user_id as index and mean rating as value
    """
    userMeans = {}
    for user in userItemMatrix.index:
        userRatings = userItemMatrix.loc[user,:].dropna()
        ratingsSum = sum(userRatings)
        if len(userRatings) > 0:
            userMeans[user] = ratingsSum/len(userRatings)
        else:
            userMeans[user] = 0
    return pd.Series(userMeans)

# Calculate mean-centered ratings
def calculate_mean_centered_ratings(userItemMatrix,meanUserRatings):
    """
    Example:
    If user_item_matrix is:
       | Movie1 | Movie2 | Movie3
    ---|--------|--------|--------
    U1 | 5      | 3      | 4
    U2 | 2      | 4      | 3
    
    And mean_user_ratings is:
    U1: 4.0
    U2: 3.0
    
    The mean-centered ratings would be:
       | Movie1 | Movie2 | Movie3
    ---|--------|--------|--------
    U1 | 1.0    | -1.0   | 0.0
    U2 | -1.0   | 1.0    | 0.0
    
    This helps identify which items a user likes more or less than their average.
    """
    ratingsDiff = userItemMatrix.copy()
    for user in ratingsDiff.index:
        userRatings = ratingsDiff.loc[user,:]
        userMean = meanUserRatings[user]
        for item in userRatings.index:
            if not pd.isna(userRatings[item]):
                ratingsDiff.loc[user,item] = userRatings[item] - userMean
    return ratingsDiff

# Calculate user similarity matrix using Pearson correlation
def calculate_user_similarity(ratingsDiff):
    """
    Calculate similarity between users using Pearson correlation on mean-centered ratings.
    
    Example:
    If ratings_diff (mean-centered ratings) is:
       | Movie1 | Movie2 | Movie3
    ---|--------|--------|--------
    U1 | 1.0    | -1.0   | 0.0
    U2 | -1.0   | 1.0    | 0.0
    
    Pearson correlation measures how linearly related the rating patterns are.
    
    The resulting similarity matrix would be:
       | U1  | U2
    ---|-----|-----
    U1 | 1.0 | -1.0
    U2 | -1.0| 1.0
    
    Where:
    - 1.0 means perfect positive correlation (same rating pattern)
    - -1.0 means perfect negative correlation (opposite rating pattern)
    - 0.0 means no correlation
    
    This allows us to find users with similar taste (high positive correlation).
    """
    ratingsDiffFilled = ratingsDiff.fillna(0)
    userSimilarity = ratingsDiffFilled.T.corr(method='pearson')
    userSimilarity = userSimilarity.fillna(0)
    return userSimilarity

# Calculate genre similarity matrix
def calculate_genre_similarity(moviesDf):
    """
    Calculate similarity between movies based on their genres using cosine similarity.
    
    Example:
    If we have 3 movies with genre information:
    Movie1: [1, 0, 1, 0] (Genre1, Genre3)
    Movie2: [0, 1, 1, 0] (Genre2, Genre3)
    Movie3: [1, 0, 0, 1] (Genre1, Genre4)
    
    Cosine similarity between movies would be:
    - Movie1 & Movie2: 0.5 (they share 1 genre out of 3 unique genres)
    - Movie1 & Movie3: 0.5 (they share 1 genre out of 3 unique genres)
    - Movie2 & Movie3: 0.0 (they don't share any genres)
    
    Higher values indicate more genre similarity.
    """
    genreCols = moviesDf.columns[5:]
    movieGenres = moviesDf[['movie_id'] + list(genreCols)]
    genreMatrix = movieGenres.iloc[:,1:].values
    genreSim = cosine_similarity(genreMatrix)
    genreSimilarity = pd.DataFrame(
        genreSim, 
        index=movieGenres['movie_id'].values,
        columns=movieGenres['movie_id'].values
    )
    return genreSimilarity,genreCols 