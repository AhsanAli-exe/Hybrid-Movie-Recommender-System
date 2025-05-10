import streamlit as st
from preparation import load_data,get_movie_poster
from recommendation import (
    prepare_recommendation_system,
    get_movie_recommendations,
    get_movie_genres
)
import pandas as pd
st.set_page_config(
    page_title="Movie Recommender System",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
.white-container {
    background-color: white;
    padding: 20px;
    border-radius: 10px;
    color: black;
    margin-bottom: 20px;
}
.movie-title {
    font-size: 29px;
    font-weight: bold;
    margin-bottom: 10px;
    color: #000080;
}
.movie-info {
    font-size: 21px;
    margin-bottom: 5px;
    color: #000080;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.title("_Movie_ _Recommendation_ :blue[System] :sunglasses:")
st.subheader("Developed by: :blue[Nerdy Freaks]")
#loading data
ratingsDf,moviesDf = load_data()
with st.spinner("Preparing recommendation system..."):
    (userItemMatrix,userSimilarity,genreSimilarity,
        meanUserRatings,genreCols) = prepare_recommendation_system(ratingsDf,moviesDf)

st.sidebar.markdown("### About")
st.sidebar.info(
    "This movie recommender system uses a hybrid approach with user-based collaborative filtering "
    "and content-based filtering using movie genres. The system suggests movies based on user "
    "ratings and genre similarities."
)

col1,col2 = st.columns([3,1])
with col1:
    movieSearch = st.text_input("Search for a movie:",placeholder="Enter a movie title...")
with col2:
    nRecommendations = st.number_input("Number of recommendations:",min_value=1,max_value=10,value=5)
if movieSearch:
    with st.spinner("Finding recommendations..."):
        (recommendedMoviesDf,originalMovie,
            targetGenres) = get_movie_recommendations(
            movieSearch,
            userItemMatrix,
            userSimilarity,
            genreSimilarity,
            meanUserRatings,
            moviesDf,
            genreCols,
            nRecommendations=nRecommendations
        )
    
    if recommendedMoviesDf is None:
        st.error(f"Movie '{movieSearch}' not found. Please try another movie.")
    else:
        st.markdown("### Your Selected Movie")
        # Get movie poster and data
        posterUrl,movieData = get_movie_poster(originalMovie['title'] if isinstance(originalMovie,pd.Series) else originalMovie.get('title'))
        html_content = f"""
        <div class="white-container">
            <div style="display: flex; align-items: start;">
                <div style="flex: 1; margin-right: 20px;">
                    <img src="{posterUrl if posterUrl else 'minimalist-movie-poster.jpg'}" style="width: 200px;" />
                </div>
                <div style="flex: 3;">
                    <div class="movie-title">{originalMovie['title']}</div>
                    <div class="movie-info"><strong>Release Date:</strong> {originalMovie['release_date']}</div>
                    <div class="movie-info"><strong>Genres:</strong> {', '.join(targetGenres)}</div>
                    <div class="movie-info"><a href="{originalMovie['IMDB_URL']}" target="_blank" style="color: #000080;">View on IMDb</a></div>
                </div>
            </div>
        </div>
        """
        
        st.markdown(html_content, unsafe_allow_html=True)
        st.markdown("""
        <style>
        .fallback-container { display: none; }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown("### Recommended Movies")
        num_display_cols = 3 
        displayCols = st.columns(num_display_cols)
        
        for i,(_,movieRow) in enumerate(recommendedMoviesDf.iterrows()):
            # movieRow is a Pandas Series representing the movie
            movieGenres = get_movie_genres(movieRow,genreCols)
            posterUrlRec,_ = get_movie_poster(movieRow['title'])
            with displayCols[i % num_display_cols]:
                if posterUrlRec:
                    st.image(posterUrlRec,width=200)
                else:
                    st.image("minimalist-movie-poster.jpg",width=200)
                st.markdown(f"**{movieRow['title']}**")
                st.markdown(f"**Genres:** {', '.join(movieGenres)}")
                # st.markdown(f"[View on IMDb]({movieRow['IMDB_URL']})")
                if 'similarity_score' in movieRow:
                    st.markdown(f"**Score:** {movieRow['similarity_score']:.2f}")
                st.markdown("---")

