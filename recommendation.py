import streamlit as st # For @st.cache_data
import pandas as pd
import numpy as np
import re 
from preparation import (
    create_user_item_matrix,
    calculate_mean_user_ratings,
    calculate_mean_centered_ratings,
    calculate_user_similarity,
    calculate_genre_similarity
)

# preparing the recommendation system
@st.cache_data # Caching the entire preparation is a good idea
def prepare_recommendation_system(ratingsDf,moviesDf):
    """Prepare all necessary components for the recommendation system"""
    userItemMatrix = create_user_item_matrix(ratingsDf)
    meanUserRatings = calculate_mean_user_ratings(userItemMatrix)
    ratingsDiff = calculate_mean_centered_ratings(userItemMatrix,meanUserRatings)
    userSimilarity = calculate_user_similarity(ratingsDiff)
    genreSimilarity,genreCols = calculate_genre_similarity(moviesDf)
    return userItemMatrix,userSimilarity,genreSimilarity,meanUserRatings,genreCols

#predict rating for a movie using user-based collaborative filtering
def predict_rating_user_based_cf(targetMovieId,itemId,userItemMatrix,userSimilarity,meanUserRatings):
    """
    Predict the rating for a movie using user-based collaborative filtering.
    
    The prediction formula is(taught in course):
    r̂(u,i) = r̄(u) + [ Σ(sim(u,v) * (r(v,i) - r̄(v))) ] / [ Σ|sim(u,v)| ]
    
    Where:
    - r̂(u,i) is the predicted rating for user u on item i
    - r̄(u) is the mean rating for user u
    - sim(u,v) is the similarity between users u and v
    - r(v,i) is the rating given by user v to item i
    - r̄(v) is the mean rating for user v
    
    Example:
    If we want to predict a rating for Movie3 for User1, we would:
    1. Find users who rated Movie3
    2. Calculate the weighted average of their ratings, where weights are
       the similarities between User1 and those users
    3. Adjust the result by User1's mean rating
    """
    
    usersWhoRatedTarget = userItemMatrix[targetMovieId].dropna().index
    commonUsers = userItemMatrix[itemId].dropna().index.intersection(usersWhoRatedTarget)
    if len(commonUsers) == 0:
        return None
    
    sims = userSimilarity.loc[commonUsers,commonUsers].values
    ratings = userItemMatrix.loc[commonUsers,itemId].values
    means = meanUserRatings[commonUsers].values
    
    try:
        numerator = np.sum((ratings - means) * sims)
        denominator = np.sum(np.abs(sims))
        if denominator != 0:
            predictedRating = meanUserRatings[commonUsers[0]] + (numerator / denominator)
            return predictedRating
        return None
    except Exception as e:
        print(f"Error in predict_rating_user_based_cf: {e}") 
        return None

# Get movie recommendations
def get_movie_recommendations(movieTitle,userItemMatrix,userSimilarity,genreSimilarity, 
                             meanUserRatings,moviesDf,genreCols,nRecommendations=5,alpha=0.7):
    """
    Get movie recommendations using a hybrid approach combining:
    1. User-based collaborative filtering (70% weight by default)
    2. Content-based filtering using genres (30% weight by default)
    
    The hybrid score formula is:
    score = α * CF_score + (1-α) * content_score
    
    Where:
    - α is the weight parameter (default 0.7)
    - CF_score is the collaborative filtering predicted rating
    - content_score is the genre similarity score (scaled to 1-5 range)
    """
    escapedMovieTitle = re.escape(movieTitle)
    searchPattern = f"(?i)(^{escapedMovieTitle}|.*\\b{escapedMovieTitle})"
    
    movieMatches = moviesDf[moviesDf['title'].str.contains(searchPattern,regex=True,na=False)]

    if movieMatches.empty:
        movieMatches = moviesDf[moviesDf['title'].str.contains(movieTitle,case=False,regex=False,na=False)]
        if movieMatches.empty:
            return None,None,None
    
    movieId = movieMatches.iloc[0]['movie_id']
    originalMovie = moviesDf[moviesDf['movie_id'] == movieId].iloc[0]
    
    targetMovieGenres = []
    for genreColName in genreCols:
        # Check if the column exists in originalMovie before accessing
        if genreColName in originalMovie and originalMovie[genreColName] == 1:
            targetMovieGenres.append(genreColName)
            
    predictedRatings = {}
    cfScores = {}
    genreScoresOutput = {}
    
    for itemId in userItemMatrix.columns:
        if itemId == movieId:
            continue
        
        predictedRatingCf = predict_rating_user_based_cf(
            movieId,itemId,userItemMatrix,userSimilarity,meanUserRatings
        )
        
        if predictedRatingCf is None:
            continue
        
        # ensureing movieId and itemId exist in genreSimilarity index
        if movieId not in genreSimilarity.index or itemId not in genreSimilarity.columns:
            # handling case where movie or item might not be in genre similarity matrix (e.g. new item)
            # for now, skip or assign default similarity
            genreSim = 0
        else:
            genreSim = genreSimilarity.loc[movieId,itemId]
            
        scaledGenreScore = 5 * genreSim 
        combinedRating = alpha * predictedRatingCf + (1 - alpha) * scaledGenreScore
        
        predictedRatings[itemId] = combinedRating
        cfScores[itemId] = predictedRatingCf
        genreScoresOutput[itemId] = scaledGenreScore
    
    if not predictedRatings:
        return None,None,None
        
    predictedRatingsSeries = pd.Series(predictedRatings)
    predictedRatingsSeries = predictedRatingsSeries.sort_values(ascending=False)
    
    if predictedRatingsSeries.empty:
        return None,None,None
    
    # Geting item IDs from the index of the series
    topItemIds = predictedRatingsSeries.head(nRecommendations).index
    # Using these IDs to filter moviesDf and to map scores
    recommendedMoviesDf = moviesDf[moviesDf['movie_id'].isin(topItemIds)].copy()
    
    # map scores using the topItemIds
    # ensure the topItems Series used for mapping is correctly constructed if needed
    # predictedRatingsSeries already contains the similarity_score for mapping by movie_id
    recommendedMoviesDf['similarity_score'] = recommendedMoviesDf['movie_id'].map(predictedRatingsSeries)
    recommendedMoviesDf['cf_score'] = recommendedMoviesDf['movie_id'].map(cfScores)
    recommendedMoviesDf['genre_score'] = recommendedMoviesDf['movie_id'].map(genreScoresOutput)
    
    # Sort final Df by similarity score
    recommendedMoviesDf = recommendedMoviesDf.sort_values(by='similarity_score',ascending=False)
    return recommendedMoviesDf,originalMovie,targetMovieGenres

# Get movie genres
def get_movie_genres(movieRow,genreCols):
    genres = [] 
    for colName in genreCols:
        if colName in movieRow and movieRow[colName] == 1:
            genres.append(colName)
    return genres 