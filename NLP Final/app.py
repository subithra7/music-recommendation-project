from flask import Flask, request, render_template_string
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
        info = get_spotify_info(song['track_name'], song['artists'])
        songs.append({
            "track_name": song['track_name'],
            "artists": song['artists'],
            "track_genre": song['track_genre'],
            "album_art": info["album_art"] or "https://via.placeholder.com/100?text=No+Image",
            "spotify_uri": info["uri"]
        })
    return songs

# === Templates ===
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Music Recommender</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e1e2f, #12121d);
            color: #e0e0e0;
            padding: 30px;
            margin: 0;
        }
        h1 {
            color: #00ffcc;
            margin-bottom: 25px;
            text-align: center;
            font-weight: 700;
        }
        form {
            display: flex;
            justify-content: center;
            margin-bottom: 30px;
            gap: 10px;
        }
        input[type="text"] {
            width: 60%;
            padding: 12px 15px;
            font-size: 18px;
            border-radius: 8px;
            border: none;
            outline: none;
            box-shadow: 0 0 5px #00ffcc;
            background-color: #29293d;
            color: #e0e0e0;
        }
        input[type="submit"] {
            padding: 12px 25px;
            background-color: #00ffcc;
            color: #12121d;
            font-weight: 700;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            box-shadow: 0 0 10px #00ffcc;
        }
        .recommendation {
            background-color: #2c2c3e;
            padding: 20px;
            margin: 15px auto;
            border-radius: 12px;
            max-width: 700px;
            display: flex;
            align-items: center;
            gap: 20px;
            box-shadow: 0 0 10px #00ffcc66;
            transition: transform 0.2s ease;
        }
        .recommendation:hover {
            transform: scale(1.02);
            box-shadow: 0 0 20px #00ffccaa;
        }
        .album-art {
            width: 100px;
            height: 100px;
            border-radius: 12px;
            object-fit: cover;
        }
        .song-info {
            flex-grow: 1;
        }
        .song-info strong {
            font-size: 1.2em;
            color: #00ffc2;
        }
        .song-info div {
            margin: 6px 0;
            font-size: 0.9em;
            color: #b0f0e5;
        }
        iframe {
            margin: 20px auto;
            display: block;
            border-radius: 12px;
        }
        a {
            color: #00ffcc;
            text-align: center;
            display: block;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <h1>🎵 Music Recommendation System</h1>
    <form method="POST" action="/recommend">
        <input type="text" name="query" placeholder="Describe your mood or song preference..." required>
        <input type="submit" value="Recommend Songs">
    </form>

    {% if selected_song %}
        <h2>Now Playing: "{{ selected_song }}"</h2>
        {% if selected_uri %}
            <iframe src="https://open.spotify.com/embed/track/{{ selected_uri.split(':')[-1] }}" width="300" height="80" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>
        {% endif %}
    {% endif %}

    {% if query %}
        <h2>Top Recommendations for: "{{ query }}"</h2>
    {% elif selected_song %}
        <h2>Recommended Songs Similar to "{{ selected_song }}"</h2>
    {% endif %}

    {% for song in results %}
        <div class="recommendation" onclick="window.location.href='/song/{{ song.track_name }}';">
            <img class="album-art" src="{{ song.album_art }}" alt="Album Art">
            <div class="song-info">
                <strong>{{ song.track_name }}</strong>
                <div>Artist: {{ song.artists }}</div>
                <div>Genre: {{ song.track_genre }}</div>
                {% if song.spotify_uri %}
                    <iframe src="https://open.spotify.com/embed/track/{{ song.spotify_uri.split(':')[-1] }}" width="250" height="80" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>
                {% endif %}
            </div>
        </div>
    {% endfor %}

    {% if selected_song %}
        <a href="/">← Back to Home</a>
    {% endif %}
</body>
</html>
'''

# === Routes ===

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/recommend', methods=['POST'])
def recommend():
    query = request.form['query']
    results = recommend_songs_by_query(query)
    return render_template_string(HTML_TEMPLATE, query=query, results=results)

@app.route('/song/<track_name>')
def similar(track_name):
    similar_songs = recommend_similar_by_track(track_name)
    info = get_spotify_info(track_name, df[df['track_name'] == track_name]['artists'].values[0])
    return render_template_string(
        HTML_TEMPLATE,
        selected_song=track_name,
        selected_uri=info['uri'],
        results=similar_songs
    )

if __name__ == '__main__':
    app.run(debug=True)
