from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# === Load dataset ===
file_path = r'C:\Users\ELCOT\Desktop\MSC\Intern\music2\dataset.csv'
df = pd.read_csv(file_path)

audio_features = [
    "danceability", "energy", "valence", "tempo", "acousticness",
    "instrumentalness", "liveness", "speechiness"
]
df = df.dropna(subset=audio_features)
scaler = MinMaxScaler()
df[audio_features] = scaler.fit_transform(df[audio_features])

# === Sentence Transformer Setup ===
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

def recommend_songs(user_query, top_k=5):
    weights = compute_feature_weights(user_query)
    weighted_scores = np.zeros(len(df))
    for feature in audio_features:
        weighted_scores += df[feature].values * weights.get(feature, 0)
    top_indices = np.argsort(weighted_scores)[-top_k:][::-1]
    return df.iloc[top_indices][["track_name", "artists", "track_genre"]]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_query = request.form['query']
    results = recommend_songs(user_query)
    return render_template('index.html', query=user_query, results=results.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
