import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import re
import nltk
from nltk.corpus import words
import ftfy
from difflib import get_close_matches

# Download once (comment after first run)
nltk.download('words')
english_vocab = set(words.words())

def is_english_ascii(text):
    try:
        text.encode('ascii')
        return True
    except UnicodeEncodeError:
        return False

def mostly_english(text, threshold=0.6):
    if isinstance(text, str):
        tokens = re.findall(r'[a-zA-Z]+', text.lower())
        if not tokens:
            return False
        english_count = sum(1 for word in tokens if word in english_vocab)
        return (english_count / len(tokens)) >= threshold
    return False

def fix_mojibake(text):
    try:
        # First, decode latin1 to bytes, then decode those bytes again as utf-8
        bytes_text = text.encode('latin1')
        decoded_once = bytes_text.decode('utf-8')
        # Sometimes double-encoded, decode again
        bytes_double = decoded_once.encode('latin1')
        decoded_twice = bytes_double.decode('utf-8')
        return decoded_twice
    except (UnicodeEncodeError, UnicodeDecodeError):
        try:
            # Fallback to the original attempt
            return text.encode('latin1').decode('utf-8')
        except UnicodeDecodeError:
            try:
                return text.encode('utf-8').decode('latin1')
            except UnicodeDecodeError:
                # If still fails, fallback to ftfy's fix_text
                return ftfy.fix_text(text)


# === Load & Prepare Data ===
@st.cache_data
def load_data():
    file_path = r"C:\Users\harin\OneDrive\Desktop\music-recommender\data\dataset_normalized_final (1).csv"
    # Read CSV with latin1 encoding to avoid decode errors initially
    df = pd.read_csv(file_path, encoding='latin1')

    # Clean column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    if 'track_name' in df.columns:
        # Fix encoding issues
        df['track_name'] = df['track_name'].astype(str).apply(fix_mojibake)
        # Normalize text and lowercase
        df['track_name'] = df['track_name'].apply(ftfy.fix_text).str.lower()
        df['track_name'] = df['track_name'].str.strip()

        # Filter only mostly English and ASCII-only track names
        df = df[df['track_name'].apply(mostly_english)]
        df = df[df['track_name'].apply(is_english_ascii)]

        # Remove empty track names (including whitespace only)
        df = df[df['track_name'].str.len() > 0]

        df = df.reset_index(drop=True)

    return df

@st.cache_data
def preprocess(df):
    feature_columns = ['energy', 'tempo', 'duration_ms']
    features = df[feature_columns]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features

def find_best_match(user_input, df):
    user_input_clean = user_input.strip().lower()
    track_names = df['track_name'].astype(str).str.strip().str.lower()

    # 1. Exact match
    exact_match = df[track_names == user_input_clean]
    if not exact_match.empty:
        idx = exact_match.index[0]
        return exact_match.loc[idx, 'track_name'], idx

    # 2. Substring match
    substring_match = df[track_names.str.contains(rf'\b{re.escape(user_input_clean)}\b', na=False)]
    if not substring_match.empty:
        idx = substring_match.index[0]
        return substring_match.loc[idx, 'track_name'], idx

    # 3. Fuzzy match with stricter cutoff
    close_matches = get_close_matches(user_input_clean, track_names, n=1, cutoff=0.85)
    if close_matches:
        match = close_matches[0]
        if user_input_clean in match or match in user_input_clean:
            idx = df[track_names == match].index[0]
            return match, idx

    return None, None

def recommend(df, scaled_features, idx, n_recommendations=10):
    model = NearestNeighbors(n_neighbors=n_recommendations + 10, metric='euclidean')
    model.fit(scaled_features)

    distances, indices = model.kneighbors([scaled_features[idx]])

    seen = set()
    original = (df.iloc[idx]['track_name'], df.iloc[idx].get('album_name', ''))
    seen.add(original)

    recommendations = [{
        'track_name': df.iloc[idx]['track_name'].title(),
        'album_name': df.iloc[idx].get('album_name', ''),
        'duration_ms': df.iloc[idx].get('duration_ms', ''),
        'note': '🔹 This is your searched song'
    }]

    for i in indices[0]:
        if len(recommendations) >= n_recommendations + 1:
            break
        track = df.iloc[i]
        track_key = (track['track_name'], track.get('album_name', ''))
        if track_key not in seen:
            seen.add(track_key)
            recommendations.append({
                'track_name': track['track_name'].title(),
                'album_name': track.get('album_name', ''),
                'duration_ms': track.get('duration_ms', ''),
                'note': ''
            })

    return pd.DataFrame(recommendations)

# === Streamlit UI ===
st.set_page_config(page_title="🎧 Music Recommender", layout="centered")
st.title("🎵 Music Recommender")
st.markdown("Find songs similar to your favorite!")

df = load_data()
scaled = preprocess(df)

user_input = st.text_input("Enter a track (e.g. 'Perfect song'):") 

if user_input:
    match_str, idx = find_best_match(user_input, df)

    if match_str and idx is not None:
        matched_track = df.loc[idx]
        st.markdown(f"**Matched phrase found:** _{match_str}_")
        st.success(f"✅ Match found: **{matched_track['track_name'].title()}**")
        st.markdown(f"#### 🎧 Songs including and similar to **{matched_track['track_name'].title()}**:")
        recommendations = recommend(df, scaled, idx, n_recommendations=10)
        st.dataframe(recommendations[['track_name', 'album_name', 'duration_ms']], use_container_width=True)

    else:
        # Fallback seeds if no match found
        fallback_seeds = ['control', 'montero', 'darkside', 'enemy']
        combined_indices = []

        for seed in fallback_seeds:
            matches = df[df['track_name'].str.lower() == seed]
            if not matches.empty:
                combined_indices.append(matches.index[0])

        if combined_indices:
            combined_recs = pd.DataFrame()
            seen_tracks = set()

            for idx in combined_indices:
                seed_track = df.iloc[idx]
                seed_dict = {
                    'track_name': seed_track['track_name'].title(),
                    'album_name': seed_track.get('album_name', ''),
                    'duration_ms': seed_track.get('duration_ms', ''),
                    'note': '🎵 Fallback seed song'
                }
                if seed_dict['track_name'].lower() not in seen_tracks:
                    combined_recs = pd.concat([combined_recs, pd.DataFrame([seed_dict])], ignore_index=True)
                    seen_tracks.add(seed_dict['track_name'].lower())

                recs = recommend(df, scaled, idx, n_recommendations=5)
                recs_filtered = recs[~recs['track_name'].str.lower().isin(seen_tracks)]
                combined_recs = pd.concat([combined_recs, recs_filtered], ignore_index=True)
                seen_tracks.update(recs_filtered['track_name'].str.lower().tolist())

            st.info(f"🎧 Showing songs similar to **{user_input.title()}** (similar vibe from: Control, Montero, Darkside, Enemy).")
            st.dataframe(combined_recs[['track_name', 'album_name', 'duration_ms']], use_container_width=True)
        else:
            st.error("⚠️ None of the fallback songs were found in your dataset.")
