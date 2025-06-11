import streamlit as st
import pandas as pd
import requests
import base64
import re
import nltk
from nltk.corpus import words
import ftfy
from difflib import get_close_matches
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Run once to download NLTK word list
nltk.download('words')
english_vocab = set(words.words())

# === Helper Functions ===
def get_spotify_token(client_id, client_secret):
    auth_str = f"{client_id}:{client_secret}"
    b64_auth_str = base64.b64encode(auth_str.encode()).decode()
    res = requests.post(
        "https://accounts.spotify.com/api/token",
        headers={"Authorization": f"Basic {b64_auth_str}"},
        data={"grant_type": "client_credentials"}
    )
    if res.status_code == 200:
        return res.json().get("access_token")
    else:
        st.error("Failed to get Spotify token.")
        return None

def is_english_ascii(text):
    try:
        text.encode('ascii')
        return True
    except UnicodeEncodeError:
        return False

def mostly_english(text, threshold=0.6):
    tokens = re.findall(r'[a-zA-Z]+', text.lower())
    if not tokens:
        return False
    english_count = sum(1 for word in tokens if word in english_vocab)
    return (english_count / len(tokens)) >= threshold

def fix_mojibake(text):
    try:
        return text.encode('latin1').decode('utf-8')
    except:
        return ftfy.fix_text(text)

def get_spotify_track(query, token):
    headers = {"Authorization": f"Bearer {token}"}
    params = {"q": query, "type": "track", "limit": 1}
    res = requests.get("https://api.spotify.com/v1/search", headers=headers, params=params)
    if res.status_code == 200 and res.json()['tracks']['items']:
        return res.json()['tracks']['items'][0]
    return None

def show_preview(track_id):
    embed_url = f"https://open.spotify.com/embed/track/{track_id}"
    st.markdown(f"""
        <iframe src="{embed_url}" width="300" height="80" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\harin\OneDrive\Desktop\music-recommender\data\dataset_normalized_final (1).csv", encoding='latin1')
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    df['track_name'] = df['track_name'].astype(str).apply(fix_mojibake).apply(ftfy.fix_text).str.lower().str.strip()
    df = df[df['track_name'].apply(mostly_english)]
    df = df[df['track_name'].apply(is_english_ascii)]
    df = df[df['track_name'].str.len() > 0]
    return df.reset_index(drop=True)

@st.cache_data
def preprocess(df):
    scaler = StandardScaler()
    return scaler.fit_transform(df[['energy', 'tempo', 'duration_ms']])

def find_best_match(user_input, df):
    input_clean = user_input.strip().lower()
    track_names = df['track_name'].str.lower().str.strip()

    exact = df[track_names == input_clean]
    if not exact.empty:
        return exact.iloc[0]['track_name'], exact.index[0]

    substring = df[track_names.str.contains(re.escape(input_clean))]
    if not substring.empty:
        return substring.iloc[0]['track_name'], substring.index[0]

    fuzzy = get_close_matches(input_clean, track_names, n=1, cutoff=0.85)
    if fuzzy:
        idx = df[track_names == fuzzy[0]].index[0]
        return fuzzy[0], idx

    return None, None

def recommend(df, features, idx, n=10):
    model = NearestNeighbors(n_neighbors=n + 10)
    model.fit(features)
    distances, indices = model.kneighbors([features[idx]])
    seen = set()
    original = (df.iloc[idx]['track_name'], df.iloc[idx].get('album_name', ''))
    seen.add(original)

    recs = [{
        'track_name': df.iloc[idx]['track_name'].title(),
        'album_name': df.iloc[idx].get('album_name', ''),
        'duration_ms': df.iloc[idx].get('duration_ms', ''),
        'note': '🎯 Original track'
    }]

    for i in indices[0]:
        if len(recs) >= n + 1:
            break
        track = df.iloc[i]
        key = (track['track_name'], track.get('album_name', ''))
        if key not in seen:
            seen.add(key)
            recs.append({
                'track_name': track['track_name'].title(),
                'album_name': track.get('album_name', ''),
                'duration_ms': track.get('duration_ms', ''),
                'note': ''
            })
    return pd.DataFrame(recs)

# === Main App ===
st.set_page_config("🎧 Music Recommender", layout="centered")
st.title("🎵 Music Recommender")
st.markdown("Type a song to get similar recommendations and play a preview.")

CLIENT_ID = "38d8e85069404722b99cc824910a83a4"
CLIENT_SECRET = "8592a74b5e194f248d18c792b78c9184"
token = get_spotify_token(CLIENT_ID, CLIENT_SECRET)

df = load_data()
scaled = preprocess(df)

user_input = st.text_input("Enter a song:")

if user_input and token:
    match, idx = find_best_match(user_input, df)
    if match and idx is not None:
        st.success(f"✅ Match: **{match.title()}**")
        recommendations = recommend(df, scaled, idx, 10)
        st.markdown("### 🎧 Recommendations:")
        for _, row in recommendations.iterrows():
            st.markdown(f"**🎵 {row['track_name']}** — _{row['album_name']}_")
            result = get_spotify_track(row['track_name'], token)
            if result:
                show_preview(result['id'])
    else:
        st.warning("No good match found. Try another song.")

