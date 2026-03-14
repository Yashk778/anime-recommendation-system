# 🎌 Anime Recommendation System

A content-based filtering recommendation system that suggests anime based on similarity in genres, studios, type, and source material.

## How It Works
- Anime features (genres, studios, type, source) are combined into tags
- Tags are vectorized using CountVectorizer
- Cosine similarity is computed between all anime
- Top 10 most similar anime are returned for any selected title

## Dataset
[MyAnimeList Dataset 2023](https://www.kaggle.com/datasets/sajidali1996/anime-dataset-2023) — 8,000+ anime with English names, scores, and poster images.

## Tech Stack
- Python
- Pandas, Scikit-learn
- Streamlit

## Live Demo
[Link to your deployed app]