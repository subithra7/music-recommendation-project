import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# Load Spotify dataset (already has tempo_normalized)
df = pd.read_csv(r'D:\Aravind\archive\SpotifyFeatures_normalized.csv')

# Load your input song (first 60 seconds)
y, sr = librosa.load(r'D:\Aravind\Songs\Aaj_ki_raat.mp3', duration=60)

# Extract features
tempo_arr, _ = librosa.beat.beat_track(y=y, sr=sr)
tempo = tempo_arr.item()  # Extract scalar value safely
rmse = np.mean(librosa.feature.rms(y=y))
zcr = np.mean(librosa.feature.zero_crossing_rate(y))
duration_ms = len(y) / sr * 1000

# Normalize input tempo using existing min/max from your dataset
tempo_min = df['tempo'].min()
tempo_max = df['tempo'].max()
tempo_normalized = (tempo - tempo_min) / (tempo_max - tempo_min)

# Build input song feature vector
input_song_features = pd.DataFrame([[
    tempo_normalized,
    rmse,
    zcr,
    duration_ms
]], columns=['tempo_normalized', 'energy', 'acousticness', 'duration_ms'])

# Scale only the other 3 features from Spotify dataset
scaler = MinMaxScaler()
spotify_scaled_other = scaler.fit_transform(df[['energy', 'acousticness', 'duration_ms']])

# Combine with already-normalized tempo
spotify_scaled = np.hstack([
    df[['tempo_normalized']].values,
    spotify_scaled_other
])

input_scaled_other = scaler.transform(input_song_features[['energy', 'acousticness', 'duration_ms']])
input_scaled = np.hstack([
    input_song_features[['tempo_normalized']].values,
    input_scaled_other
])

# Compute cosine similarity
similarities = cosine_similarity(input_scaled, spotify_scaled)

# Get top 5 recommendations
top_indices = similarities[0].argsort()[::-1][:5]
recommendations = df.iloc[top_indices]

print("🎵 Features of Your Input Song ('Muththa Mazhai'):")
print(f"Tempo (bpm)         : {tempo:.2f}")
print(f"Tempo Normalized    : {tempo_normalized:.4f}")
print(f"Energy (RMSE)       : {rmse:.4f}")
print(f"Acousticness (ZCR)  : {zcr:.4f}")
print(f"Duration (ms)       : {duration_ms:.0f}")
print("\n🎧 Recommended Songs Based on Similar Audio Features:")
print(recommendations[['track_name', 'artist_name', 'tempo_normalized', 'energy', 'acousticness', 'duration_ms']])
