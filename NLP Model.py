from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

app = Flask(__name__)

# === Spotify API ===
client_credentials_manager = SpotifyClientCredentials(
    client_id='***',
    client_secret='***'
)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def get_spotify_info(track, artist):
    query = f"{track} {artist}"
    results = sp.search(q=query, type='track', limit=1)
    items = results['tracks']['items']
    if items:
        item = items[0]
        album_art = item['album']['images'][0]['url'] if item['album']['images'] else None
        return album_art
    return None

# === Load and scale dataset ===
df = pd.read_csv(r"C:\Users\poorn\OneDrive\Desktop\internship\llm\dataset.csv")
audio_features = [
    "danceability", "energy", "valence", "tempo", "acousticness",
    "instrumentalness", "liveness", "speechiness"
]
df = df.dropna(subset=audio_features)
scaler = MinMaxScaler()
df[audio_features] = scaler.fit_transform(df[audio_features])

# === Sentence Transformer ===
model = SentenceTransformer("all-MiniLM-L6-v2")
feature_prompts = {
    "danceability": "danceable song good for dancing",
    "energy": "high energy loud fast song",
    "valence": "happy positive mood uplifting",
    "tempo": "fast beat or high tempo music",
    "acousticness": "soft acoustic calm natural instruments",
    "instrumentalness": "instrumental music without vocals",
    "liveness": "live performance concert feel",
    "speechiness": "talking speech spoken words"
}
feature_embeddings = {
    feature: model.encode(desc)
    for feature, desc in feature_prompts.items()
}

def compute_feature_weights(user_query):
    query_vec = model.encode(user_query)
    weights = {}
    for feature, feature_vec in feature_embeddings.items():
        sim = cosine_similarity([query_vec], [feature_vec])[0][0]
        weights[feature] = max(sim, 0)
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    return weights

def recommend_songs_by_query(user_query, top_k=5):
    weights = compute_feature_weights(user_query)
    weighted_scores = np.zeros(len(df))
    for feature in audio_features:
        weighted_scores += df[feature].values * weights.get(feature, 0)
    top_indices = np.argsort(weighted_scores)[-top_k:][::-1]
    return build_song_data(top_indices)

def recommend_similar_by_track(track_name, top_k=5):
    row = df[df['track_name'] == track_name]
    if row.empty:
        return []
    row_features = row[audio_features].values[0]
    scores = cosine_similarity([row_features], df[audio_features])[0]
    indices = np.argsort(scores)[-top_k-1:][::-1]
    indices = [i for i in indices if df.iloc[i]['track_name'] != track_name][:top_k]
    return build_song_data(indices)

def build_song_data(indices):
    songs = []
    for i in indices:
        song = df.iloc[i]
        album_art = get_spotify_info(song['track_name'], song['artists'])
        songs.append({
            "track_name": song['track_name'],
            "artists": song['artists'],
            "track_genre": song['track_genre'],
            "album_art": album_art or "https://via.placeholder.com/100?text=No+Image"
        })
    return songs

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    query = request.form['query']
    results = recommend_songs_by_query(query)
    return render_template('index.html', query=query, results=results)

@app.route('/song/<track_name>')
def similar(track_name):
    similar_songs = recommend_similar_by_track(track_name)
    return render_template('similar.html', selected_song=track_name, results=similar_songs)

if __name__ == '__main__':
    app.run(debug=True)
