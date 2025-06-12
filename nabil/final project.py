**************
import pandas as pd
print(pd.__version__)

***************
from sklearn.preprocessing import StandardScaler
print("Scikit-learn is working!")

***************
import pandas as pd

# Step 1: Load the dataset
df = pd.read_csv("C:/Users/91948/datasetfinal.csv")  # Use your actual path

# Step 2: Explore the data
print(df.head())        # See first 5 rows
print(df.columns)       # See all column names
print(df.info())        # Info about data types and missing values
print(df.describe())    # Summary statistics of numeric columns
print(df.isnull().sum())  # Count missing values in each column

****************
import pandas as pd

dataset = pd.read_csv('datasetfinal.csv')
print(dataset.columns)

****************

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from difflib import get_close_matches

# Load dataset
dataset = pd.read_csv(r"C:\Users\91948\datasetfinal.csv")

# Define feature columns
feature_cols = [
    'popularity', 'duration_ms', 'explicit', 'danceability', 'energy', 'key',
    'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
    'liveness', 'valence', 'tempo', 'time_signature'
]

# Add genre columns
genre_cols = [col for col in dataset.columns if col.startswith('track_genre_')]
feature_cols.extend(genre_cols)

# Fill missing values
features = dataset[feature_cols].fillna(0)

# Mapping track names to index
song_titles = dataset['track_name'].str.lower()
title_to_index = {title: idx for idx, title in enumerate(song_titles)}

def recommend_songs(input_song, top_n=5):
    input_song = input_song.lower()

    if input_song in title_to_index:
        idx = title_to_index[input_song]
    else:
        matches = get_close_matches(input_song, song_titles, n=1, cutoff=0.6)
        if matches:
            print(f"\n❓ Exact song not found. Showing recommendations based on closest match: '{matches[0]}'")
            idx = title_to_index[matches[0]]
        else:
            print("❌ Song not found. Showing top popular songs.\n")
            top_popular = dataset.sort_values('popularity', ascending=False).head(top_n)
            return [(row['track_name'], row['album_name'], row['danceability'], row['energy'])
                    for _, row in top_popular.iterrows()]

    # Compute similarity
    input_vector = features.iloc[idx].values.reshape(1, -1)
    all_vectors = features.values
    sim_scores = cosine_similarity(input_vector, all_vectors)[0]
    sim_scores_indices = np.argsort(sim_scores)[::-1]

    recommendations = []
    added = set()

    for i in sim_scores_indices:
        if i == idx:
            continue
        row = dataset.iloc[i]
        track = row['track_name']
        album = row['album_name']
        dance = row['danceability']
        energy = row['energy']

        if track not in added:
            recommendations.append((track, album, dance, energy))
            added.add(track)
        if len(recommendations) >= top_n:
            break

    return recommendations

# Main loop
while True:
    song_name = input("\nEnter song name: ").strip()
    if song_name.lower() == 'exit':
        print("👋 Exiting the recommender.")
        break
    if not song_name:
        continue

    recs = recommend_songs(song_name)

    print("\n🎧 Recommended Songs:")
    for idx, (track, album, dance, energy) in enumerate(recs, 1):
        print(f"{idx}. {track} (Album: {album}) | Danceability: {dance:.2f} | Energy: {energy:.2f}")

****************************
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id='a43524b757824c9987ed1b47930aa26a',
    client_secret='ca13ef37e1b94ebaa499830d7914429e'
))

results = sp.search(q='Believer', type='track', limit=1)
print(results['tracks']['items'][0]['name'])

*****************************
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def get_similar_songs(input_features, dataset):
    features_df = dataset[['danceability', 'energy', 'tempo', ...]]  # pick relevant features

    # Convert input to DataFrame
    input_vector = pd.DataFrame([[
        input_features['danceability'],
        input_features['energy'],
        input_features['tempo'],
        ...
    ]], columns=features_df.columns)

    similarities = cosine_similarity(input_vector, features_df)
    top_indices = similarities[0].argsort()[-5:][::-1]  # Top 5

    return dataset.iloc[top_indices]

*******************************
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

# Load dataset
dataset = pd.read_csv(r"C:\Users\91948\datasetfinal.csv")

# Define feature columns
feature_cols = [
    'popularity', 'duration_ms', 'explicit', 'danceability', 'energy', 'key',
    'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
    'liveness', 'valence', 'tempo', 'time_signature'
]

# Add genre columns
genre_cols = [col for col in dataset.columns if col.startswith('track_genre_')]
feature_cols.extend(genre_cols)

# Fill missing values
features = dataset[feature_cols].fillna(0)

# Mapping track names to index
song_titles = dataset['track_name'].str.lower()
title_to_index = {title: idx for idx, title in enumerate(song_titles)}

# Recommendation function
def recommend_songs(input_song, top_n=5):
    input_song = input_song.lower()

    if input_song in title_to_index:
        idx = title_to_index[input_song]
    else:
        matches = get_close_matches(input_song, song_titles, n=1, cutoff=0.6)
        if matches:
            st.info(f"Exact song not found. Showing recommendations based on closest match: '{matches[0]}'")
            idx = title_to_index[matches[0]]
        else:
            st.warning("Song not found. Showing top popular songs instead.")
            top_popular = dataset.sort_values('popularity', ascending=False).head(top_n)
            return [(row['track_name'], row['album_name'], row['danceability'], row['energy'], row.get('track_id', None))
                    for _, row in top_popular.iterrows()]

    input_vector = features.iloc[idx].values.reshape(1, -1)
    all_vectors = features.values
    sim_scores = cosine_similarity(input_vector, all_vectors)[0]
    sim_scores_indices = np.argsort(sim_scores)[::-1]

    recommendations = []
    added = set()

    for i in sim_scores_indices:
        if i == idx:
            continue
        row = dataset.iloc[i]
        track = row['track_name']
        album = row['album_name']
        dance = row['danceability']
        energy = row['energy']
        track_id = row.get('track_id', None)

        if track not in added:
            recommendations.append((track, album, dance, energy, track_id))
            added.add(track)
        if len(recommendations) >= top_n:
            break

    return recommendations

# ------------------------------
# Streamlit Web Interface
# ------------------------------
st.set_page_config(page_title="🎵 Music Recommender", layout="centered")
st.title("🎶 Music Recommendation System")

song_input = st.text_input("Enter a song name:", "")

if st.button("Get Recommendations"):
    if not song_input.strip():
        st.error("Please enter a song name.")
    else:
        recs = recommend_songs(song_input)
        if recs:
            for idx, (track, album, dance, energy, track_id) in enumerate(recs, 1):
                st.markdown(f"### {idx}. {track}")
                st.write(f"**Album:** {album}  \n**Danceability:** {dance:.2f}  \n**Energy:** {energy:.2f}")
                if track_id:
                    url = f"https://open.spotify.com/track/{track_id}"
                    st.markdown(f"[🔗 Listen on Spotify]({url})")
                st.markdown("---")






