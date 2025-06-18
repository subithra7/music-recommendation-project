from flask import Flask, request, render_template_string
import os, random
import pandas as pd
import numpy as np
import librosa
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# === Spotify Credentials ===
sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(
    client_id='46bf26a25fb74e92b4437aed65227788',
    client_secret='ba60599ca4b744a28a8df7f6b9d5f66c'
))

# === Load Dataset ===
df = pd.read_csv("dataset.csv")
audio_features = [
    "danceability", "energy", "valence", "tempo",
    "acousticness", "instrumentalness", "liveness", "speechiness"
]
df = df.dropna(subset=audio_features)
scaler = MinMaxScaler()
df[audio_features] = scaler.fit_transform(df[audio_features])

# === Sentence Transformer for Mood Queries ===
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

# === Utilities ===
def extract_features(path):
    y, sr = librosa.load(path)
    features = {
        'danceability': librosa.feature.rms(y=y).mean(),
        'energy': librosa.feature.spectral_centroid(y=y, sr=sr).mean(),
        'valence': librosa.feature.zero_crossing_rate(y).mean(),
        'tempo': librosa.beat.tempo(y=y, sr=sr)[0] / 250,
        'acousticness': librosa.feature.spectral_bandwidth(y=y, sr=sr).mean(),
        'instrumentalness': librosa.feature.spectral_flatness(y=y).mean(),
        'liveness': librosa.feature.spectral_rolloff(y=y, sr=sr).mean(),
        'speechiness': librosa.feature.mfcc(y=y, sr=sr).mean()
    }
    scaled = scaler.transform([list(features.values())])[0]
    return scaled

def compute_feature_weights(user_query):
    query_vec = model.encode(user_query)
    weights = {}
    for feature, fvec in feature_embeddings.items():
        sim = cosine_similarity([query_vec], [fvec])[0][0]
        weights[feature] = max(sim, 0)
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()} if total > 0 else weights

def get_spotify_info(track, artist):
    results = sp.search(q=f"{track} {artist}", type='track', limit=1)
    items = results['tracks']['items']
    if items:
        item = items[0]
        return {
            "album_art": item['album']['images'][0]['url'] if item['album']['images'] else None,
            "uri": item['uri']
        }
    return {"album_art": None, "uri": None}

def build_song_data(indices, skip=[]):
    songs = []
    for i in indices:
        song = df.iloc[i]
        if song['track_name'] in skip:
            continue
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
    return build_song_data(random.sample(range(len(df)), n))

def recommend_by_query(user_query, top_k=10):
    weights = compute_feature_weights(user_query)
    scores = np.zeros(len(df))
    for feature in audio_features:
        scores += df[feature].values * weights.get(feature, 0)
    indices = np.argsort(scores)[-top_k:][::-1]
    return build_song_data(indices)

def recommend_similar_by_track(track_name, top_k=10):
    row = df[df['track_name'].str.lower() == track_name.lower()]
    if row.empty:
        return []
    target = row[audio_features].values[0]
    scores = cosine_similarity([target], df[audio_features])[0]
    indices = np.argsort(scores)[-top_k-1:][::-1]
    indices = [i for i in indices if df.iloc[i]['track_name'].lower() != track_name.lower()]
    return build_song_data(indices)

# === HTML TEMPLATE ===
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<title>Music Recommender</title>
<style>
    body { font-family: 'Segoe UI', sans-serif; background: black; color: white; margin: 0; padding: 0; }
    header { background: #111; padding: 20px; text-align: center; font-size: 28px; font-weight: bold; }
    form { display: flex; flex-wrap: wrap; justify-content: center; gap: 10px; margin: 20px; align-items: center; }
    input[type="text"] { padding: 10px; border-radius: 6px; border: none; width: 40%; }
    input[type="submit"], .refresh-btn, .custom-file-upload {
        padding: 10px 15px; background: green; color: white; font-weight: bold;
        border: none; border-radius: 6px; cursor: pointer; text-decoration: none;
    }
    .custom-file-upload {
        display: inline-block;
        position: relative;
        overflow: hidden;
    }
    .custom-file-upload input[type="file"] {
        position: absolute;
        left: 0;
        top: 0;
        opacity: 0;
        cursor: pointer;
    }
    .songs-grid { display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; padding: 20px; }
    .song-card {
        background: #222; padding: 10px; border-radius: 12px; width: 200px;
        text-align: center; box-shadow: 0 0 10px #00ff00aa; transition: transform 0.3s; cursor: pointer;
    }
    .song-card:hover { transform: scale(1.05); box-shadow: 0 0 20px #00ff00cc; }
    .song-card img { width: 100%; border-radius: 12px; }
    .back-home { text-align: center; margin: 40px; }
</style>
</head>
<body>
<header>Music Recommendation System</header>
<form method="POST" action="/recommend" enctype="multipart/form-data">
    <label class="custom-file-upload">
        <input type="file" name="file" accept=".mp3,.wav,.ogg">
        Upload file
    </label>
    <input type="text" name="query" placeholder="Describe your mood or genre...">
    <input type="submit" value="Recommend">
    <a href="/" class="refresh-btn">🔄</a>
</form>

{% if selected_song %}
<h2 style="text-align:center;">Now Playing: {{ selected_song }}</h2>
{% if selected_uri %}
<div style="text-align:center;"><iframe src="https://open.spotify.com/embed/track/{{ selected_uri.split(':')[-1] }}" width="300" height="80" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe></div>
{% endif %}
{% elif query %}
<h2 style="text-align:center;">Top Recommendations for: "{{ query }}"</h2>
{% else %}
<h2 style="text-align:center;">Featured Songs</h2>
{% endif %}

<div class="songs-grid">
    {% for song in results %}
    <div class="song-card" onclick="window.location.href='/song/{{ song.track_name }}';">
        <img src="{{ song.album_art }}">
        <div><strong>{{ song.track_name }}</strong></div>
        <div>{{ song.artists }}</div>
    </div>
    {% endfor %}
</div>

{% if request.path != '/' %}
<div class="back-home"><a href="/" style="color:lightgreen;">← Back to Home</a></div>
{% endif %}
</body>
</html>
"""

# === Routes ===
@app.route('/')
def home():
    songs = get_random_songs()
    return render_template_string(HTML_TEMPLATE, results=songs)

@app.route('/recommend', methods=['POST'])
def recommend():
    # File Upload
    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        features = extract_features(filepath)
        scores = cosine_similarity([features], df[audio_features])[0]
        top_indices = np.argsort(scores)[-10:][::-1]
        os.remove(filepath)
        return render_template_string(HTML_TEMPLATE, selected_song="Your Uploaded Audio", results=build_song_data(top_indices))

    # Text Query
    query = request.form.get('query', '').strip()
    if not query:
        return render_template_string(HTML_TEMPLATE, results=get_random_songs())

    exact_match = df[df['track_name'].str.lower() == query.lower()]
    exact_songs = build_song_data(exact_match.index.tolist()) if not exact_match.empty else []
    similar_songs = recommend_by_query(query)
    similar_filtered = [s for s in similar_songs if s['track_name'].lower() != query.lower()]
    combined = exact_songs + similar_filtered[:10 - len(exact_songs)]
    return render_template_string(HTML_TEMPLATE, query=query, results=combined)

@app.route('/song/<track_name>')
def song(track_name):
    similar = recommend_similar_by_track(track_name)
    info = get_spotify_info(track_name, df[df['track_name'].str.lower() == track_name.lower()]['artists'].values[0])
    return render_template_string(HTML_TEMPLATE, selected_song=track_name, selected_uri=info["uri"], results=similar)

# === Run the App ===
if __name__ == '__main__':
    app.run(debug=True)
