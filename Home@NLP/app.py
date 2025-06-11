from flask import Flask, request, render_template_string, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import random

app = Flask(__name__)

# === Spotify API ===
client_credentials_manager = SpotifyClientCredentials(
    client_id='46bf26a25fb74e92b4437aed65227788',
    client_secret='ba60599ca4b744a28a8df7f6b9d5f66c'
)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# === Load and scale dataset ===
df = pd.read_csv("dataset.csv")
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
feature_embeddings = {f: model.encode(desc) for f, desc in feature_prompts.items()}

def get_spotify_info(track, artist):
    query = f"{track} {artist}"
    results = sp.search(q=query, type='track', limit=1)
    items = results['tracks']['items']
    if items:
        item = items[0]
        return {
            "album_art": item['album']['images'][0]['url'] if item['album']['images'] else None,
            "uri": item['uri']
        }
    return {"album_art": None, "uri": None}

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

def recommend_songs_by_query(user_query, top_k=10):
    weights = compute_feature_weights(user_query)
    weighted_scores = np.zeros(len(df))
    for feature in audio_features:
        weighted_scores += df[feature].values * weights.get(feature, 0)
    top_indices = np.argsort(weighted_scores)[-top_k:][::-1]
    return build_song_data(top_indices)

def recommend_similar_by_track(track_name, top_k=10):
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
        info = get_spotify_info(song['track_name'], song['artists'])
        songs.append({
            "track_name": song['track_name'],
            "artists": song['artists'],
            "track_genre": song['track_genre'],
            "album_art": info["album_art"] or "https://via.placeholder.com/150?text=No+Image",
            "spotify_uri": info["uri"]
        })
    return songs

def get_random_songs(n=20):
    indices = random.sample(range(len(df)), n)
    return build_song_data(indices)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Music Recommender</title>
    <style>
        body {
            font-family: 'Segoe UI', sans-serif;
            background-color: black;
            color: white;
            padding: 0;
            margin: 0;
        }
        header {
            background-color: #111;
            color: white;
            padding: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        header h1 {
            margin: 0;
        }
        form {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 10px;
            margin: 20px;
        }
        input[type="text"] {
            width: 50%;
            padding: 10px;
            border-radius: 6px;
            border: none;
        }
        input[type="submit"], .upload-btn, .refresh-btn {
            padding: 10px 15px;
            background-color: green;
            color: white;
            font-weight: bold;
            border: none;
            border-radius: 6px;
            cursor: pointer;
        }
        .refresh-btn {
            font-size: 16px;
        }
        .songs-grid {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 20px;
            padding: 20px;
        }
        .song-card {
            background-color: #222;
            padding: 10px;
            border-radius: 12px;
            width: 200px;
            text-align: center;
            box-shadow: 0 0 10px #00ff00aa;
            cursor: pointer;
            transition: transform 0.3s;
        }
        .song-card:hover {
            transform: scale(1.05);
            box-shadow: 0 0 20px #00ff00cc;
        }
        .song-card img {
            width: 100%;
            border-radius: 12px;
        }
        .back-home {
            display: block;
            text-align: center;
            margin: 40px 0;
        }
    </style>
</head>
<body>
    <header>
        <div>Home</div>
        <h1>Music Recommendation System</h1>
    </header>
    <form method="POST" action="/recommend">
        <a class="upload-btn" href="/upload">Upload</a>
        <input type="text" name="query" placeholder="Describe your mood or song preference...">
        <input type="submit" value="Recommend">
        <a href="/" class="refresh-btn">🔄</a>
    </form>

    {% if selected_song %}
        <h2 style="text-align:center;">Now Playing: {{ selected_song }}</h2>
        {% if selected_uri %}
            <div style="display:flex;justify-content:center;">
                <iframe src="https://open.spotify.com/embed/track/{{ selected_uri.split(':')[-1] }}" width="300" height="80" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>
            </div>
        {% endif %}
    {% endif %}

    {% if query %}
        <h2 style="text-align:center;">Top Recommendations for: "{{ query }}"</h2>
    {% elif selected_song %}
        <h2 style="text-align:center;">Recommended Songs Similar to: "{{ selected_song }}"</h2>
    {% else %}
        <h2 style="text-align:center;">Featured Songs</h2>
    {% endif %}

    <div class="songs-grid">
        {% for song in results %}
        <div class="song-card" onclick="window.location.href='/song/{{ song.track_name }}';">
            <img src="{{ song.album_art }}" alt="Album Art">
            <div><strong>{{ song.track_name }}</strong></div>
            <div>{{ song.artists }}</div>
        </div>
        {% endfor %}
    </div>

    {% if request.path != '/' %}
        <div class="back-home">
            <a href="/" style="color:lightgreen;">← Back to Home</a>
        </div>
    {% endif %}
</body>
</html>
'''

@app.route('/')
def home():
    songs = get_random_songs()
    return render_template_string(HTML_TEMPLATE, results=songs)

@app.route('/recommend', methods=['POST'])
def recommend():
    query = request.form['query']
    results = recommend_songs_by_query(query)
    return render_template_string(HTML_TEMPLATE, query=query, results=results)

@app.route('/song/<track_name>')
def song(track_name):
    similar_songs = recommend_similar_by_track(track_name)
    info = get_spotify_info(track_name, df[df['track_name'] == track_name]['artists'].values[0])
    return render_template_string(HTML_TEMPLATE, selected_song=track_name, selected_uri=info['uri'], results=similar_songs)

@app.route('/upload')
def upload():
    return "<h1>Upload Page (Under Construction)</h1><a href='/'>Back to Home</a>"

if __name__ == '__main__':
    app.run(debug=True)
