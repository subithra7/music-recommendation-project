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

# Load your Spotify dataset once when server starts
df = pd.read_csv(r'D:\Aravind\archive\SpotifyFeatures_normalized.csv')

# Pre-scale the Spotify features for similarity
scaler = MinMaxScaler()
spotify_scaled_other = scaler.fit_transform(df[['energy', 'acousticness', 'duration_ms']])
spotify_scaled = np.hstack([df[['tempo_normalized']].values, spotify_scaled_other])

# Homepage HTML
HOME_HTML = """

<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
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


# Result HTML
RESULT_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>🎵 Music Recommendation</title>
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
    <h1 class="mb-4">🎶 Upload Your Song</h1>
    <form method="POST" enctype="multipart/form-data" action="/recommend">
      <div class="mb-3">
        <input class="form-control" type="file" name="file" accept=".mp3,.wav" required>
      </div>
      <button class="btn btn-primary" type="submit">Get Recommendations</button>
    </form>
  </div>
</body>
</html>

<h2 class="mt-5">🎧 Top 5 Similar Songs</h2>
<div class="table-responsive">
  <table class="table table-hover table-bordered align-middle shadow rounded-4 overflow-hidden">
    <thead class="table-dark">
      <tr>
        <th scope="col">🎵 Track Name</th>
        <th scope="col">🎤 Artist</th>
        <th scope="col">⏱️ Tempo (Normalized)</th>
        <th scope="col">⚡ Energy</th>
        <th scope="col">🌿 Acousticness</th>
        <th scope="col">🕒 Duration (ms)</th>
      </tr>
    </thead>
    <tbody>
      {% for _, row in recommendations.iterrows() %}
      <tr>
        <td>{{ row.track_name }}</td>
        <td>{{ row.artist_name }}</td>
        <td>{{ "{:.4f}".format(row.tempo_normalized) }}</td>
        <td>{{ "{:.4f}".format(row.energy) }}</td>
        <td>{{ "{:.4f}".format(row.acousticness) }}</td>
        <td>{{ (row.duration_ms) }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
</div>

"""

@app.route('/')
def home():
    return HOME_HTML

@app.route('/recommend', methods=['POST'])
def recommend():
    file = request.files['file']
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Extract features using librosa
        y, sr = librosa.load(filepath, duration=60)
        tempo_arr, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo_arr) if np.ndim(tempo_arr) == 0 else tempo_arr.item()
        rmse = float(np.mean(librosa.feature.rms(y=y)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        duration_ms = float(len(y) / sr * 1000)

        # Normalize tempo using dataset min/max
        tempo_min = df['tempo'].min()
        tempo_max = df['tempo'].max()
        tempo_normalized = (tempo - tempo_min) / (tempo_max - tempo_min)

        # Build input vector and scale
        input_song_features = pd.DataFrame([[
            tempo_normalized, rmse, zcr, duration_ms
        ]], columns=['tempo_normalized', 'energy', 'acousticness', 'duration_ms'])

        input_scaled_other = scaler.transform(input_song_features[['energy', 'acousticness', 'duration_ms']])
        input_scaled = np.hstack([
            input_song_features[['tempo_normalized']].values,
            input_scaled_other
        ])

        similarities = cosine_similarity(input_scaled, spotify_scaled)
        top_indices = similarities[0].argsort()[::-1][:5]
        recommendations = df.iloc[top_indices]

        return render_template_string(
            RESULT_HTML,
            
            tempo_normalized=f"{tempo_normalized:.4f}",
            rmse=f"{rmse:.4f}",
            zcr=f"{zcr:.4f}",
            duration_ms=f"{int(duration_ms)}",
            recommendations=recommendations
        )

    return "No file uploaded", 400

if __name__ == '__main__':
    app.run(debug=True)
