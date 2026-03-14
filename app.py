import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

st.set_page_config(page_title="Anime Recommender", layout="wide")
# Load data
df = pickle.load(open('anime.pkl', 'rb'))

# Compute similarity once and cache
@st.cache_resource
def compute_similarity():
    cv = CountVectorizer(max_features=5000)
    vectors = cv.fit_transform(df['tags']).toarray()
    return cosine_similarity(vectors)

similarity = compute_similarity()

# Recommend function
def recommend(anime_name, n=10):
    if anime_name not in df['English name'].values:
        return []
    
    idx = df[df['English name'] == anime_name].index[0]
    idx = df.index.get_loc(idx)
    
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = scores[1:n+1]
    
    indices = [i[0] for i in scores]
    return df.iloc[indices][['English name', 'Score', 'Image URL']].values.tolist()

# UI
st.title("🎌 Anime Recommendation System")

anime_list = df['English name'].values
selected = st.selectbox("Select an anime you like:", anime_list)

if st.button("Recommend"):
    results = recommend(selected)
    
    if not results:
        st.error("Anime not found!")
    else:
        cols = st.columns(5)
        for i, (name, score, img_url) in enumerate(results[:5]):
            with cols[i]:
                st.image(img_url, use_column_width=True)
                st.caption(f"**{name}**")
                st.caption(f"⭐ {score}")
        
        cols2 = st.columns(5)
        for i, (name, score, img_url) in enumerate(results[5:]):
            with cols2[i]:
                st.image(img_url, use_column_width=True)
                st.caption(f"**{name}**")
                st.caption(f"⭐ {score}")