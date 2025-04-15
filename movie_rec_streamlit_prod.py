import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from streamlit_super_slider import st_slider

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
    st.session_state.page_number = 0
    st.session_state.ratings = {}

# Step 4: Only display the ratings form if a genre is selected
if st.session_state.genre_selected:
    # Load the movie data file
    df_movies = pd.read_csv('movies.csv')

    # Filter movies by selected genre
    movies_list = df_movies[df_movies['genres'].str.contains(st.session_state.selected_genre, case=False, na=False)]
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

        # Add custom CSS to handle text wrapping and alignment
        st.markdown("""
        <style>
            /* Align slider and text vertically */
            .stSlider {
                padding-top: 0rem;
                padding-bottom: 0rem;
                margin-top: -1.5rem;
            }

            /* Adjust text positioning with wrapping */
            .movie-title {
                word-wrap: break-word;
                white-space: normal;
                line-height: 1.2;
                padding-right: 1rem;
                margin: 0;
                display: flex;
                align-items: center;
                min-height: 2.5rem;
            }

            /* Remove column padding */
            div[data-testid="column"] {
                padding: 0rem;
                margin: 0rem;
            }

            /* Adjust vertical block spacing */
            div[data-testid="stVerticalBlock"] > div {
                padding-top: 0.25rem;
                padding-bottom: 0.25rem;
            }
        </style>
        """, unsafe_allow_html=True)

        # Initialize session state for pagination and ratings
        if "page_number" not in st.session_state:
            st.session_state.page_number = 0
        if "ratings" not in st.session_state:
            st.session_state.ratings = {}

        # Constants for pagination
        MOVIES_PER_PAGE = 5
        total_movies = len(movies_to_rate)
        total_pages = -(-total_movies // MOVIES_PER_PAGE)  # Ceiling division
        start_idx = st.session_state.page_number * MOVIES_PER_PAGE
        end_idx = min(start_idx + MOVIES_PER_PAGE, total_movies)

        # Display current page info
        st.write(f"Page {st.session_state.page_number + 1} of {total_pages}")

        # Create columns for the movie list
        for _, row in movies_to_rate.iloc[start_idx:end_idx].iterrows():
            movie_id = row["Movie ID"]
            movie_title = row["Movie Title"]
            col1, col2 = st.columns([2, 3])

            with col1:
                st.markdown(f'<div class="movie-title">{movie_title}</div>', unsafe_allow_html=True)
            with col2:
                st.session_state.ratings[movie_id] = st_slider(
                    values={0: 'nr', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 
                           6: '6', 7: '7', 8: '8', 9: '9', 10: '10'},
                    min_value=0,
                    max_value=10,
                    dots=True,
                    steps=1,
                    key=f"slider_{movie_id}"
                )

        # Create pagination controls
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.form_submit_button("Previous") and st.session_state.page_number > 0:
                st.session_state.page_number -= 1
                st.rerun()
        with col2:
            if st.form_submit_button("Next") and st.session_state.page_number < total_pages - 1:
                st.session_state.page_number += 1
                st.rerun()
        with col3:
            ratings_submit_button = st.form_submit_button("Submit Ratings")

        # Submit button for the ratings form
        # ratings_submit_button = st.form_submit_button(label='Submit Ratings')

    # Step 5: Process ratings submission
    if ratings_submit_button:
        # Filter out movies with a default rating of 0
        filtered_ratings = {
            movie_id: rating / 2
            for movie_id, rating in st.session_state.ratings.items()
            if rating > 0
        }

        # Display the collected ratings
        # st.write("Your Ratings:")
        # st.write(filtered_ratings)

        # Convert filtered_ratings to the desired format
        new_user_ratings = {
            'movieId': list(filtered_ratings.keys()),
            'rating': list(filtered_ratings.values())
        }

        # Display the collected ratings
        # st.write("Your Ratings:")
        # st.write(new_user_ratings)
        # print("Your Ratings:")
        # print(new_user_ratings)

# Inference and recommendation

        # Define model class identical to the one used in training
        class RecSysModel(nn.Module):
            def __init__(self, n_users, n_movies, n_factors):
                super().__init__()
                self.user_embed = nn.Embedding(n_users, n_factors)
                self.movie_embed = nn.Embedding(n_movies, n_factors)
                self.out = nn.Linear(n_factors * 2, 1)
        
            def forward(self, users, movies):
                user_embeds = self.user_embed(users)
                movie_embeds = self.movie_embed(movies)
                output = torch.cat([user_embeds, movie_embeds], dim=1)
                output = self.out(output)
                return output
     
        # set device to cuda if available, otherwise use cpu
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
        ###################
        # Load the MovieLens movies dataset
        # df_movies = pd.read_csv('movies.csv')
        ###################
      
        # Load the model and label encoders for users and movies
      
        # Load the model and label encoders from the trained model .pth file
        checkpoint = torch.load('models/movie_rec_sys_r1.pth', weights_only=False)
     
        # Extract the label encoders
        lbl_user_loaded = checkpoint['lbl_user']
        lbl_movie_loaded = checkpoint['lbl_movie']
        user_embed = checkpoint['user_embed']
        movie_embed = checkpoint['movie_embed']
        n_factors = checkpoint['n_factors']
        
        # Define the model architecture
        n_users = len(lbl_user_loaded.classes_)
        n_movies = len(lbl_movie_loaded.classes_)
        
        model_loaded = RecSysModel(n_users, n_movies, n_factors)
        
        # Load the model's state dictionary
        model_loaded.load_state_dict(checkpoint['model_state_dict'])
        
        # Set the model to evaluation mode
        model_loaded.eval()
        
        # Recommend the top K movies for a given user
        # Assuming the model is already defined and trained
        # model = RecSysModel(n_users, n_movies, n_factors)
        
        def recommend_movies_for_new_user(model, user_ids, movie_ids, top_k=10):
            # Get all movie ids
            movie_ids = torch.LongTensor(range(len(lbl_movie_loaded.classes_)))
        
            # Use a placeholder user id (e.g., 0)
            user_id = 0
            user_ids = torch.LongTensor([user_id] * len(lbl_movie_loaded.classes_))
        
            # Get predictions for all movies
            with torch.no_grad():
                all_predictions = model(user_ids, movie_ids)
        
            # Ensure top_k does not exceed the number of available movies
            top_k = min(top_k, len(all_predictions))
        
            # Get top k movie predictions
            top_k_predictions, top_k_indices = torch.topk(all_predictions, top_k, dim=0, largest=True, sorted=True)
        
            return top_k_predictions, top_k_indices
        
        # Define ratings for a new user and generate recommendations
        
        # Define ratings for a new user to be used for recommendations
        
        # new_user_ratings = {
        #     'movieId': [4896, 5816, 8368, 40815, 54001, 69844, 318], # Harry Potter Movies
        #     'rating': [5, 5, 5, 5, 5, 5, 0]
        # }
        
        # Convert the new user ratings to a DataFrame
        df_new_user_ratings = pd.DataFrame(new_user_ratings)
        
        # Encode the movie IDs using the loaded label encoder
        df_new_user_ratings['movieId'] = lbl_movie_loaded.transform(df_new_user_ratings['movieId'])
        
        # Encode the user ID using the loaded label encoder
        user_id = 0  # Placeholder user ID
        
        # Add unseen user_id to lbl_user_loaded
        if user_id not in lbl_user_loaded.classes_:
            lbl_user_loaded.classes_ = np.append(lbl_user_loaded.classes_, user_id)
        
        df_new_user_ratings['userId'] = lbl_user_loaded.transform([user_id] * len(df_new_user_ratings))
        
        # Convert the DataFrame to tensors
        user_tensor = torch.LongTensor(df_new_user_ratings['userId'].values)
        movie_tensor = torch.LongTensor(df_new_user_ratings['movieId'].values)
        
        # Convert ratings to tensor
        ratings_tensor = torch.FloatTensor(df_new_user_ratings['rating'].values)
        
        # Get recommendations for the new user by passing the user and movie tensors to the model
        top_k_predictions, top_k_indices = recommend_movies_for_new_user(model_loaded, user_tensor, movie_tensor, top_k=100)
        
        # Convert top_k_indices to movie IDs
        top_k_movie_ids = lbl_movie_loaded.inverse_transform(top_k_indices.numpy())
        
        # Ensure the top_k_movie_ids and top_k_predictions are 1D tensors
        top_k_movie_ids = top_k_movie_ids.flatten() if len(top_k_movie_ids.shape) > 1 else top_k_movie_ids
        top_k_predictions = top_k_predictions.flatten() if len(top_k_predictions.shape) > 1 else top_k_predictions
        
        # Convert top_k_movie_ids to DataFrame
        df_recommendations = pd.DataFrame({
            'movieId': top_k_movie_ids,
            'predicted_rating': top_k_predictions.numpy()
        })
        
        # Merge with movie titles
        df_recommendations = df_recommendations.merge(df_movies, on='movieId', how='left')

        # Filter out movies that the user has already rated
        # Convert encoded movie IDs back to original IDs
        rated_movie_ids = lbl_movie_loaded.inverse_transform(df_new_user_ratings['movieId'].values)

        # Create boolean mask of movies NOT in rated_movie_ids
        mask = ~df_recommendations['movieId'].isin(rated_movie_ids)
        # Apply mask to keep only unrated movies
        df_recommendations = df_recommendations[mask]
        df_recommendations = df_recommendations.sort_values(by='predicted_rating', ascending=False)

        # Convert predicted ratings to 1 decimal place
        df_recommendations['predicted_rating'] = df_recommendations['predicted_rating'].astype(float)
        df_recommendations['predicted_rating'] = df_recommendations['predicted_rating'].round(1)

        # Filter out movies that don't match the selected genre
        df_recommendations = df_recommendations[df_recommendations['genres'].str.contains(st.session_state.selected_genre, case=False, na=False)]
        
        # Display the top K recommended movies
        
        # Show all rows and columns
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.float_format', lambda x: '%.1f' % x) # Set float format to 1 decimal places
        
        # Display the top-k recommended movies
        # st.write("Top 10 Recommended Movies:")
        # st.write(df_recommendations[['title', 'predicted_rating']].head(10).to_string(index=False))
        # st.dataframe(df_recommendations[['title', 'predicted_rating']].head(10), use_container_width=True)

        # Add custom CSS for table styling
        st.markdown("""
        <style>
            /* Center the column headers */
            [data-testid="stDataFrame"] th {
                text-align: center !important;
            }

            /* Center the predicted rating values */
            [data-testid="stDataFrame"] td:nth-child(2) {
                text-align: center !important;
            }

            /* Add some padding and borders */
            [data-testid="stDataFrame"] table {
                width: 100%;
                border-collapse: collapse;
            }

            [data-testid="stDataFrame"] td, [data-testid="stDataFrame"] th {
                padding: 8px;
                border: 1px solid #ddd;
            }
        </style>
        """, unsafe_allow_html=True)

        # Display the dataframe with custom column names and styling
        st.write("Top 10 Recommended Movies:")
        df_display = df_recommendations[['title', 'predicted_rating']].head(10).copy()
        df_display.columns = ['Movie Title', 'Rating']  # Rename columns for display
        st.dataframe(df_display, use_container_width=True, hide_index=True)
