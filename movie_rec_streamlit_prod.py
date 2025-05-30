import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# super-slider did not register ratings when using slider feature
# from streamlit_super_slider import st_slider

# Title
st.title("Movie Recommendation System")

# Step 1: Use session state to persist genre selection
if "genre_selected" not in st.session_state:
    st.session_state.genre_selected = False

# Step 2: Genre selection form
with st.form(key='genre_form'):
    # Replace the two st.write statements with a single markdown
    st.markdown("""
    ### What type of movie would you like to watch?
    <div style="margin-top: -1.0em; margin-bottom: -0.0em;">Please select a genre:</div>
    """, unsafe_allow_html=True)

    genre = st.selectbox("", [  # Empty label since we're showing the text above
        'Drama', 'Comedy', 'Thriller', 'Action', 'Romance', 'Adventure', 'Crime',
        'Sci-Fi', 'Horror', 'Fantasy', 'Children', 'Animation', 'Mystery',
        'Documentary', 'War', 'Musical', 'Western', 'Film-Noir', 'All Genres'
    ])
    genre_submit_button = st.form_submit_button(label='Submit Genre')

# Step 3: Process genre selection
if genre_submit_button:
    st.session_state.genre_selected = True
    st.session_state.selected_genre = genre
    st.session_state.page_number = 0

# Step 4: Only display the ratings form if a genre is selected
if st.session_state.genre_selected:
    # Load the movie and ratings data file
    df_movies = pd.read_csv('movies.csv')
    df_ratings = pd.read_csv('ratings.csv')
    # Filter movies by selected genre
    if st.session_state.selected_genre == 'All Genres':
        movies_list = df_movies.copy()
    else:
        # Filter movies based on the selected genre
        movies_list = df_movies[df_movies['genres'].str.contains(st.session_state.selected_genre, case=False, na=False)]
    # Combine with ratings data to sort by number of ratings
    movies_list = movies_list.merge(df_ratings.groupby('movieId').size().reset_index(name='num_ratings'), on='movieId', how='left')
    # Sort by number of ratings in descending order
    movies_list = movies_list.sort_values(by='num_ratings', ascending=False)
    movies_to_rate = movies_list[['movieId', 'title']].copy()
    movies_to_rate.rename(columns={"movieId": "Movie ID", "title": "Movie Title"}, inplace=True)

    # Ratings form
    with st.form(key='ratings_form'):
        # Replace the three st.write statements with a single markdown
        st.markdown(f"""
        <div style='line-height: 1.5; margin-bottom: 1em;'>
            <h4 style='margin: 0; padding: 0; margin-bottom: 0.5em;'>Please rate some of these "{st.session_state.selected_genre}" movies:</h4>
            <p style='margin: 0; padding: 0;'>Rate the movies from 1 to 10, where 1 means you don't like it at all and 10 means you love it!</p>
            <p style='margin: 0; padding: 0;'>Note: Leaving a rating at 0 means you don't want to include that movie in the ratings.</p>
        </div>
        """, unsafe_allow_html=True)

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
                line-height: 1.0;
                padding-right: 1rem;
                margin: 0;
                display: flex;
                align-items: center;
                min-height: 1.5rem;
            }

            /* Remove column padding */
            div[data-testid="column"] {
                padding: 0rem;
                margin: 0rem;
            }

            /* Adjust vertical block spacing */
            div[data-testid="stVerticalBlock"] > div {
                padding-top: 0.05 rem;
                padding-bottom: 0.05 rem;
            }
        </style>
        """, unsafe_allow_html=True)

        # Initialize session state for pagination and ratings
        if "page_number" not in st.session_state:
            st.session_state.page_number = 0
        if "ratings" not in st.session_state:
            st.session_state.ratings = {}
        print("Session State Ratings:", st.session_state.ratings)

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
            col1, col2 = st.columns([4, 6])

            with col1:
                st.markdown(f'<div class="movie-title">{movie_title}</div>', unsafe_allow_html=True)
            with col2:
                # Create a slider for each movie
                slider_value = st.slider(
                    label="",  # hide the movie title; use movie_title to show
                    min_value=0,
                    max_value=10,
                    value=0,
                    step=1,
                    key=movie_id,
                    format="%d",
                    help=""  # "Rate the movie from 1 to 10, where 1 means you don't like it at all and 10 means you love it!  Note: Leaving a rating at 0 means you don't want to include that movie in the ratings.",
                )

                # super-slider did not register ratings when using slider feature
                # Implement a custom slider using st_slider for super-slider
                # slider_key = f"slider_{movie_id}"
                # slider_value = st_slider(
                #     values={0: 'nr', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 
                #             6: '6', 7: '7', 8: '8', 9: '9', 10: '10'},
                #     min_value=0,
                #     max_value=10,
                #     dots=True,
                #     steps=1,
                #     key=slider_key
                # )

            st.session_state.ratings[movie_id] = slider_value

        # Create pagination controls
        b_col1, b_col2, b_col3 = st.columns([2, 4, 2])
        with b_col1:
            if st.form_submit_button("Previous Page") and st.session_state.page_number > 0:
                st.session_state.page_number -= 1
                st.rerun()
        with b_col2:
            if st.form_submit_button("Next Page") and st.session_state.page_number < total_pages - 1:
                st.session_state.page_number += 1
                st.rerun()
        with b_col3:
            ratings_submit_button = st.form_submit_button("Submit Ratings")

    # Step 5: Process ratings submission
    if ratings_submit_button:
        # Filter out movies with a default rating of 0
        filtered_ratings = {
            movie_id: rating / 2
            for movie_id, rating in st.session_state.ratings.items()
            if rating > 0
        }

        # Convert filtered_ratings to the desired format
        new_user_ratings = {
            'movieId': list(filtered_ratings.keys()),
            'rating': list(filtered_ratings.values())
        }

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
        
        # Define ratings for a new user to be used for recommendations
        
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

        # If the selected genre is 'All Genres', no need to filter by genre
        if st.session_state.selected_genre == 'All Genres':
            df_recommendations = df_recommendations.sort_values(by='predicted_rating', ascending=False)
        else:
            # Filter out movies that don't match the selected genre
            df_recommendations = df_recommendations[df_recommendations['genres'].str.contains(st.session_state.selected_genre, case=False, na=False)]
        
        # Display the top K recommended movies
        
        # Show all rows and columns
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.float_format', lambda x: '%.1f' % x) # Set float format to 1 decimal places
        
        st.markdown("""
        <style>
            /* Style the table container */
            .dataframe {
                width: 100% !important;
                margin: 0 !important;
            }

            /* Style table elements */
            table {
                width: 100% !important;
                margin: 0 !important;
            }

            /* Style the headers with specific widths */
            th:first-child {
                width: 90% !important;
                font-size: 18px;
                text-align: left !important;
            }

            th:last-child {
                width: 10% !important;
                font-size: 18px;
                text-align: center !important;
            }

            /* Style the cells */
            td:first-child {
                width: 90% !important;
                padding: 8px;
                font-size: 16px;
            }

            td:last-child {
                width: 10% !important;
                text-align: center !important;
                padding: 8px;
                font-size: 16px;
            }
        </style>
        """, unsafe_allow_html=True)

        # Get the size of the df_recommendations DataFrame
        num_rows, num_cols = df_recommendations.shape
        if num_rows > 10:
            # Limit the display to the top 10 rows
            df_recommendations = df_recommendations.head(10)
            num_rows = 10
        else:
            # Display all rows if there are 10 or fewer
            pass
        # Display the dataframe with custom formatting
        st.write(f"### 🎬 Top {num_rows} Recommended Movies")
        df_display = df_recommendations[['title', 'predicted_rating']].copy()
        df_display.columns = ['Movie Title', 'Your Predicted Rating']
        df_display['Your Predicted Rating'] = df_display['Your Predicted Rating'].astype(float)
        df_display['Your Predicted Rating'] = df_display['Your Predicted Rating'].clip(upper=5.0)  # Clip ratings to a maximum of 5.0
        df_display['Your Predicted Rating'] = df_display['Your Predicted Rating'].apply(lambda x: f"{x * 2:.1f} ⭐")

        # Convert to HTML with full width table
        html_table = df_display.to_html(index=False, classes=['dataframe'])
        st.markdown(html_table, unsafe_allow_html=True)
