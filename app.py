import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from wordcloud import WordCloud

df = pd.read_csv("avis_clients.csv")
model = joblib.load("model_sentiment.pkl")

st.title("🧠 Analyse des avis clients tunisiens")
st.markdown("Ce projet détecte automatiquement le **sentiment** d’un avis sur un produit local 🇹🇳")

user_input = st.text_area("✍️ Écris un avis client ici :")

if st.button("Analyser le sentiment"):
    if user_input:
        prediction = model.predict([user_input])[0]
        st.success(f"🔍 Sentiment détecté : **{prediction.upper()}**")
    else:
        st.warning("Veuillez entrer un avis.")

st.subheader("📊 Statistiques globales")
sentiment_counts = df["Sentiment"].value_counts()
st.bar_chart(sentiment_counts)

st.subheader("☁️ Nuage de mots (tous avis)")
text = " ".join(df["Avis"])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
st.image(wordcloud.to_array())

st.subheader("🌟 Produits les plus appréciés")
top_products = df[df["Sentiment"] == "positif"]["Produit"].value_counts().head(5)
st.write(top_products)
