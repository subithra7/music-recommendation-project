from flask import Flask, request, render_template_string
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load dataset
df = pd.read_csv(r'D:\Aravind\archive\SpotifyFeatures_normalized.csv')

# Pre-scale Spotify dataset
scaler = MinMaxScaler()
spotify_scaled_other = scaler.fit_transform(df[['energy', 'acousticness', 'duration_ms']])
spotify_scaled = np.hstack([df[['tempo_normalized']].values, spotify_scaled_other])

# HTML templates
HOME_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Music Recommendation</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body {
      background: linear-gradient(to right, #e0eafc, #cfdef3);
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .container {
      max-width: 700px;
      margin-top: 80px;
      background: white;
      padding: 40px;
      border-radius: 15px;
      box-shadow: 0 12px 30px rgba(0, 0, 0, 0.1);
    }
    .btn-primary {
      background-color: #5e60ce;
      border: none;
    }
    .btn-primary:hover {
      background-color: #4ea8de;
    }
  </style>
</head>
<body>
  <div class="container text-center">
    <h1 class="mb-4">🎶 Upload a Song</h1>
    <p class="text-muted mb-4">(Only the first 60 seconds will be processed)</p>
    <form method="POST" enctype="multipart/form-data" action="/recommend">
      <div class="mb-3">
        <input class="form-control" type="file" name="file" accept=".mp3,.wav" required>
      </div>
      <button class="btn btn-primary" type="submit">Get Recommendations</button>
    </form>
  </div>
</body>
</html>
"""

RESULT_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Recommendations</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body {
      background: linear-gradient(135deg, #e0eafc, #cfdef3);
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .container {
      max-width: 700px;
      margin-top: 60px;
      background: white;
      padding: 30px;
      border-radius: 15px;
      box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    .song-link a {
      text-decoration: none;
      color: #5e60ce;
      font-weight: bold;
    }
  </style>
</head>
<body>
  <div class="container text-center">
    <h2 class="mb-4">🎧 Top 5 Recommendations</h2>
    {% for index, row in recommendations.iterrows() %}
      <div class="song-link mb-3">
        <a href="https://www.youtube.com/results?search_query={{ row['track_name'] }}+{{ row['artist_name'] }}" target="_blank">
          {{ loop.index }}. {{ row['track_name'] }} - {{ row['artist_name'] }}
        </a>
      </div>
    {% endfor %}
    <a href="/" class="btn btn-primary mt-4">Upload Another Song</a>
  </div>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HOME_HTML)

@app.route("/recommend", methods=["POST"])
def recommend():
    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Extract features from uploaded song
    y, sr = librosa.load(filepath, duration=60)
    tempo_arr, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = tempo_arr.item()
    rmse = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    duration_ms = len(y) / sr * 1000

    # Normalize tempo
    tempo_min = df['tempo'].min()
    tempo_max = df['tempo'].max()
    tempo_normalized = (tempo - tempo_min) / (tempo_max - tempo_min)

    # Build input feature vector
    input_song_features = pd.DataFrame([[
        tempo_normalized, rmse, zcr, duration_ms
    ]], columns=['tempo_normalized', 'energy', 'acousticness', 'duration_ms'])

    # Scale input features (except tempo which is already normalized)
    input_scaled_other = scaler.transform(input_song_features[['energy', 'acousticness', 'duration_ms']])
    input_scaled = np.hstack([input_song_features[['tempo_normalized']].values, input_scaled_other])

    # Similarity
    similarities = cosine_similarity(input_scaled, spotify_scaled)
    top_indices = similarities[0].argsort()[::-1][:5]
    recommendations = df.iloc[top_indices]

    return render_template_string(RESULT_HTML, recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
