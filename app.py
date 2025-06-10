from flask import Flask, request, render_template_string
import librosa, numpy as np, pandas as pd, os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Spotify setup
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id="your_spotify_client_ID",
    client_secret="your_spotify_secret_code"
))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load dataset
df = pd.read_csv(r'C:\Users\SUBITHRA\Downloads\music recommed\SpotifyFeatures_normalized.csv')
scaler = MinMaxScaler()
pats = scaler.fit_transform(df[['energy','acousticness','duration_ms']])
spotify_scaled = np.hstack([df[['tempo_normalized']].values, pats])

HOME_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Music Recommender</title>
  <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@500&family=Roboto&display=swap" rel="stylesheet">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body {
      margin: 0;
      padding: 0;
      background: url('https://wallpaperaccess.com/full/8416990.gif') no-repeat center center fixed;
      background-size: cover;
      font-family: 'Orbitron', sans-serif;
      color: white;
    }
    .overlay {
      background: rgba(0, 0, 0, 0.85);
      min-height: 100vh;
      display: flex;
      justify-content: center;
      align-items: center;
    }
    .container {
      background: #1c1c1c;
      padding: 40px;
      border-radius: 20px;
      box-shadow: 0 0 30px #1db954;
      max-width: 550px;
      text-align: center;
    }
    h1 {
      color: #1db954;
      margin-bottom: 20px;
    }
    .form-control {
      background-color: #2a2a2a;
      border: none;
      color: white;
    }
    .btn-custom {
      background-color: #1db954;
      color: white;
      font-weight: bold;
      border-radius: 30px;
      border: none;
      transition: 0.3s;
    }
    .btn-custom:hover {
      background-color: #1ed760;
      box-shadow: 0 0 10px #1ed760, 0 0 20px #1ed760;
    }
  </style>
</head>
<body>
  <div class="overlay">
    <div class="container">
      <h1>Music Recommender</h1>
      <p class="mb-4">Upload a song clip and discover your top 5 matches!</p>
      <form method="POST" enctype="multipart/form-data" action="/recommend">
        <input class="form-control mb-3" type="file" name="file" accept=".mp3,.wav" required>
        <button class="btn btn-custom w-100" type="submit">Get Recommendations</button>
      </form>
    </div>
  </div>
</body>
</html>
"""

RESULT_HTML = """<!doctype html><html lang="en"><head>
<meta charset="UTF-8"><title>Recommendations</title>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@500&family=Roboto&display=swap" rel="stylesheet">
<style>
  body {
    margin: 0;
    padding: 0;
    background: url('https://wallpaperaccess.com/full/8416990.gif') no-repeat center center fixed;
    background-size: cover;
    font-family: 'Orbitron', sans-serif;
    color: white;
  }
  .overlay {
    background: rgba(0, 0, 0, 0.85);
    min-height: 100vh;
    padding: 40px 20px;
    display: flex;
    justify-content: center;
    align-items: center;
  }
  .container {
    background: #1c1c1c;
    padding: 40px;
    border-radius: 20px;
    box-shadow: 0 0 30px #1db954;
    max-width: 800px;
    width: 100%;
    text-align: center;
  }
  h2 {
    color: #1db954;
    margin-bottom: 30px;
    font-size: 2em;
    text-shadow: 0 0 10px #1db954;
  }
  .song {
    background: #111;
    border-radius: 15px;
    margin: 25px 0;
    box-shadow: 0 0 20px rgba(29,185,84,0.4);
    padding: 15px;
    transition: transform 0.2s ease;
  }
  .song:hover {
    transform: scale(1.02);
    box-shadow: 0 0 25px rgba(29,185,84,0.8);
  }
  .song-header {
    display: flex;
    align-items: center;
    text-align: left;
  }
  .song img {
    width: 100px;
    height: 100px;
    object-fit: cover;
    border-radius: 10px;
    margin-right: 15px;
  }
  .details {
    flex: 1;
  }
  .details h3 {
    margin: 0;
    color: #1db954;
    font-size: 1.2em;
    text-shadow: 0 0 5px #1db954;
  }
  .details p {
    margin: 5px 0;
    color: #ccc;
  }
  iframe {
    margin-top: 10px;
    width: 100%;
    height: 80px;
    border: none;
    border-radius: 10px;
  }
  .btn {
    display: inline-block;
    margin-top: 40px;
    padding: 12px 30px;
    background: #1db954;
    color: #000;
    border: none;
    border-radius: 30px;
    font-size: 16px;
    font-weight: bold;
    cursor: pointer;
    transition: 0.3s;
    text-decoration: none;
  }
  .btn:hover {
    background: #1ed760;
    box-shadow: 0 0 15px #1ed760;
  }
</style>
</head>
<body>
  <div class="overlay">
    <div class="container">
      <h2>Top 5 Tracks</h2>
      {% for s in songs %}
      <div class="song">
        <div class="song-header">
          <img src="{{ s['image'] }}" alt="Album art">
          <div class="details">
            <h3>{{ s['name'] }}</h3>
            <p>{{ s['artist'] }}</p>
          </div>
        </div>
        <iframe src="https://open.spotify.com/embed/track/{{ s['uri'].split(':')[-1] }}" allow="encrypted-media"></iframe>
      </div>
      {% endfor %}
      <a href="/" class="btn">🎧 Try Another</a>
    </div>
  </div>
</body>
</html>"""


@app.route("/", methods=["GET"])
def home():
    return render_template_string(HOME_HTML)

@app.route("/recommend", methods=["POST"])
def recommend():
    file = request.files['file']
    path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(path)

    y, sr = librosa.load(path, duration=60)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(tempo)
    rmse = float(np.mean(librosa.feature.rms(y=y)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    dur = float(len(y)/sr*1000)

    tmin, tmax = df['tempo'].min(), df['tempo'].max()
    tnorm = (tempo - tmin)/(tmax - tmin)
    feat = pd.DataFrame([[tnorm, rmse, zcr, dur]], columns=['tempo_normalized','energy','acousticness','duration_ms'])
    features_scaled = scaler.transform(feat[['energy','acousticness','duration_ms']])
    features_in = np.hstack([feat[['tempo_normalized']].values, features_scaled])

    sims = cosine_similarity(features_in, spotify_scaled)[0]
    top5 = df.iloc[sims.argsort()[::-1][:5]].copy()

    songs = []
    for _, row in top5.iterrows():
        q = f"{row['track_name']} {row['artist_name']}"
        res = sp.search(q=q, type='track', limit=1)
        if res['tracks']['items']:
            tr = res['tracks']['items'][0]
            songs.append({
                'name': tr['name'],
                'artist': tr['artists'][0]['name'],
                'image': tr['album']['images'][0]['url'] if tr['album']['images'] else '',
                'uri': tr['uri']
            })

    return render_template_string(RESULT_HTML, songs=songs)

if __name__ == "__main__":
    app.run(debug=True)
