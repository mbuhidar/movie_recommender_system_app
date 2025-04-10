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
    df_movies = pd.read_csv('Sandbox/MovieLens/ml-latest-small/movies.csv')

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

        # Create two columns: one for movie titles and one for sliders
        for _, row in movies_to_rate.head(50).iterrows():
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
            movie_id: rating / 2
            for movie_id, rating in st.session_state.ratings.items()
            if rating > 0
        }

        # Display the collected ratings
        st.write("Your Ratings:")
        st.write(filtered_ratings)

        # Convert filtered_ratings to the desired format
        new_user_ratings = {
            'movieId': list(filtered_ratings.keys()),
            'rating': list(filtered_ratings.values())
        }

        # Display the collected ratings
        # st.write("Your Ratings:")
        # st.write(df_ratings)
        # print("Your Ratings:")
        # print(df_ratings)

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
        df_movies = pd.read_csv('Sandbox/MovieLens/ml-latest-small/movies.csv')
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
        
        new_user_ratings = {
            'movieId': [4896, 5816, 8368, 40815, 54001, 69844, 318], # Harry Potter Movies
            'rating': [5, 5, 5, 5, 5, 5, 0]
        }
        
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
        top_k_predictions, top_k_indices = recommend_movies_for_new_user(model_loaded, user_tensor, movie_tensor, top_k=10)
        
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
        
        
        # Display the top K recommended movies
        
        # Show all rows and columns
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.float_format', lambda x: '%.1f' % x) # Set float format to 1 decimal places
        
        # Display the top-k recommended movies
        st.write("Top-k recommended movies:")
        # st.write(df_recommendations[['title', 'predicted_rating']].head(10).to_string(index=False))
        st.dataframe(df_recommendations[['title', 'predicted_rating']].head(10), use_container_width=True)
