import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re

# Preprocessing function
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Hapus URL
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Hapus karakter non-alfabet tetapi tetap pertahankan spasi
    text = text.lower()  # Ubah ke huruf kecil
    return text.strip()  # Hapus spasi berlebih di awal/akhir

# Streamlit application
st.title("Twitter Data Visualization")

# File upload
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # a. Display raw data
    st.subheader("Raw Data")
    st.write(df)
    st.write("The raw data consists of the original tweets fetched from Twitter.")

    # b. Clean data and display
    df['cleaned_text'] = df['text'].apply(clean_text)
    st.subheader("Cleaned Data")
    st.write(df[['text', 'cleaned_text']])
    st.write("The cleaned data removes unnecessary characters and converts text to lowercase.")

    # c. Generate word cloud
    st.subheader("Word Cloud")
    all_text = ' '.join(df['cleaned_text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Word Cloud of Tweets")
    st.pyplot(plt)
    st.write("The word cloud visualizes the most frequently occurring words in the tweets.")

    # d. Sentiment analysis
    st.subheader("Sentiment Analysis")
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df['cleaned_text'])
    # Mock sentiment analysis using Logistic Regression
    lr = LogisticRegression()
    # Assume labels are provided for training (mock data here)
    labels = [1 if i % 2 == 0 else 0 for i in range(len(df))]
    lr.fit(X, labels)
    df['predicted_sentiment'] = lr.predict(X)
    st.write(df[['cleaned_text', 'predicted_sentiment']])
    st.write("The sentiment analysis predicts whether a tweet is positive (1) or negative (0).")

    # e. Visualize sentiment distribution
st.subheader("Sentiment Distribution")

# Menghitung jumlah sentimen positif dan negatif
sentiment_counts = df['predicted_sentiment'].value_counts()
negative_count = sentiment_counts.get(0, 0)
positive_count = sentiment_counts.get(1, 0)

# Menyiapkan data untuk pie chart
labels = [f'Negative (0): {negative_count}', f'Positive (1): {positive_count}']
sizes = [negative_count, positive_count]
colors = ['#ff9999', '#66b3ff']  # Warna yang lebih lembut

# Membuat pie chart
plt.figure(figsize=(10, 6))
wedges, texts, autotexts = plt.pie(
    sizes,
    labels=labels,
    autopct='%1.1f%%',
    startangle=140,
    colors=colors,
    explode=(0.1, 0),  # Menonjolkan slice Negative
    textprops={'fontsize': 12}
)

# Menyesuaikan judul dan gaya font
plt.title('Sentiment Distribution (Negative vs Positive)', fontsize=16, fontweight='bold')
for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontsize(12)

# Equal aspect ratio untuk memastikan pie chart berbentuk lingkaran
plt.axis('equal')
plt.tight_layout()

# Menampilkan pie chart di Streamlit
st.pyplot(plt)

# Menampilkan jumlah sentimen positif dan negatif
st.write(f"Negative Sentiment: {negative_count}")
st.write(f"Positive Sentiment: {positive_count}")
