import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# Title
st.title("Movie Recommendation System")

# Step 1: Use session state to persist genre selection
if "genre_selected" not in st.session_state:
    st.session_state.genre_selected = False

# Step 2: Genre selection form
with st.form(key='genre_form'):
    st.write("What type of movie do you want to watch today?")
    genre = st.selectbox("Please select a genre:", [
        'Drama', 'Comedy', 'Thriller', 'Action', 'Romance', 'Adventure', 'Crime',
        'Sci-Fi', 'Horror', 'Fantasy', 'Children', 'Animation', 'Mystery',
        'Documentary', 'War', 'Musical', 'Western', 'Film-Noir'
    ])
    genre_submit_button = st.form_submit_button(label='Submit Genre')

# Step 3: Process genre selection
if genre_submit_button:
    st.session_state.genre_selected = True
    st.session_state.selected_genre = genre

# Step 4: Only display the ratings form if a genre is selected
if st.session_state.genre_selected:
    # Load the movie data file
    movies = pd.read_csv('Sandbox/MovieLens/ml-latest-small/movies.csv')

    # Filter movies by selected genre
    movies_list = movies[movies['genres'].str.contains(st.session_state.selected_genre, case=False, na=False)]
    movies_to_rate = movies_list[['movieId', 'title']].copy()
    movies_to_rate.rename(columns={"movieId": "Movie ID", "title": "Movie Title"}, inplace=True)

    # Initialize session state for ratings
    if "ratings" not in st.session_state:
        st.session_state.ratings = {}

    # Ratings form
    with st.form(key='ratings_form'):
        st.write(f"Please rate some of these {st.session_state.selected_genre} movies:")
        st.write("Rate the movies from 1 to 10, where 1 means you don't like it at all and 10 means you love it!")
        st.write("Note: Leaving a rating at 0 means you don't want to rate it.")

        # Create two columns: one for movie titles and one for sliders
        for _, row in movies_to_rate.head(10).iterrows():
            movie_id = row["Movie ID"]
            movie_title = row["Movie Title"]
            col1, col2 = st.columns([3, 1])  # Adjust the column width ratio as needed
            with col1:
                st.text(movie_title)  # Display the movie title without a line feed
            with col2:
                # Use session state to persist slider values
                st.session_state.ratings[movie_id] = st.slider(
                    "", min_value=0, max_value=10, step=1, key=f"slider_{movie_id}"
                )

        # Submit button for the ratings form
        ratings_submit_button = st.form_submit_button(label='Submit Ratings')

    # Step 5: Process ratings submission
    if ratings_submit_button:
        # Filter out movies with a default rating of 0
        filtered_ratings = {
            movie_id: rating
            for movie_id, rating in st.session_state.ratings.items()
            if rating > 0
        }

        # Display the collected ratings
        st.write("Your Ratings:")
        st.write(filtered_ratings)
