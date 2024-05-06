import re

from flask import Flask, jsonify, request
from flask_cors import CORS
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


app = Flask(__name__)
CORS(app)


# Ensure NLTK resources are available
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


# Function to clean text
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A)  # Keep only letters
    text = text.lower()  # Lowercase
    tokenized = word_tokenize(text)  # Tokenize
    return [word for word in tokenized if word not in stop_words]


# Function to vectorize articles
def article_vector(article_tokens, model, weight=1.0):
    vector = np.zeros(model.vector_size)
    num_words = 0
    for token in article_tokens:
        if token in model.wv:
            vector += model.wv[token] * weight
            num_words += 1
    if num_words > 0:
        vector /= num_words
    return vector


def find_similar_articles_by_id(base_article_id, all_articles, article_vectors, top_n=5):
    # Find the index of the base article using its ID
    base_index = next(i for i, article in enumerate(all_articles) if article.get('id') == base_article_id)

    # Compute similarities
    similarities = cosine_similarity([article_vectors[base_index]], article_vectors)[0]

    # Get indices of the articles with highest similarity scores
    similar_indices = np.argsort(-similarities)

    # Explicitly filter out the base article to ensure it's not included
    similar_indices = [i for i in similar_indices if i != base_index][1:top_n + 1]

    # Return the IDs and headlines of the similar articles
    return [all_articles[i]['id'] for i in similar_indices]


def get_ids_of_similar_articles(base, list):
    # Original news data plus the base article
    news_data = list
    base_article = base

    # Preprocess all news data including the base article
    all_news = news_data + [base_article]
    preprocessed_news = [clean_text(item['headline'] + " " + item['description'] + " " + item['teaser']) for item in all_news]

    # Train the Word2Vec model
    model = Word2Vec(sentences=preprocessed_news, vector_size=300, window=5, min_count=1, workers=4)

    # Vectorize all articles
    article_vectors = np.array([article_vector(article, model) for article in preprocessed_news])

    # Example usage
    return find_similar_articles_by_id(base_article['id'], all_news, article_vectors, top_n=5)


def find_similar_articles(article_vectors, base_indexes, top_n=5):
    # Compute similarities
    base_vectors = [article_vectors[i] for i in base_indexes]
    similarities = cosine_similarity(base_vectors, article_vectors)

    # Average the similarities across the base topics
    average_similarities = np.mean(similarities, axis=0)

    # Get indices of the articles with highest similarity scores
    similar_indices = np.argsort(-average_similarities)

    # Return the indices excluding the base indices
    return [i for i in similar_indices if i not in base_indexes][:top_n]


@app.route('/recommendations', methods=['POST'])
def recommendations():
    topics = request.json.get('topics')
    base_topic = request.json.get('base_topic')
    print(base_topic)

    return jsonify({'ids': get_ids_of_similar_articles(base_topic, topics)})


@app.route('/general_recommendations', methods=['POST'])
def general_recommendations():
    topics = request.json.get('topics')
    base_topics = request.json.get('base_topics')  # Expecting a list of base topics
    keywords = request.json.get('keywords', [])  # List of keywords representing user activity

    # Combine topics and base topics into a single list for processing
    all_news = topics + base_topics

    # Combine all article texts and add a 'document' from keywords
    all_texts = [article['headline'] + " " + article['description'] + " " + article['teaser'] for article in all_news]
    all_texts += [' '.join(keywords * 2)]  # Treat keywords as an additional document

    # Preprocess the combined texts
    preprocessed_texts = [clean_text(text) for text in all_texts]

    # Train the Word2Vec model
    model = Word2Vec(sentences=preprocessed_texts, vector_size=300, window=5, min_count=1, workers=4)

    # Vectorize all articles including the one formed from keywords
    article_vectors = np.array([article_vector(text, model) for text in preprocessed_texts])

    # Find indexes of the base articles
    base_indexes = list(range(len(topics), len(topics) + len(base_topics)))

    # Include the keywords vector in the similarity calculation if keywords were provided
    if len(keywords):
        base_indexes.append(len(article_vectors) - 1)  # Include keywords index as part of the base

    # Get aggregated recommendations
    recommendations = find_similar_articles(article_vectors, base_indexes, top_n=5)

    # Return IDs of similar articles, ensuring we don't include the keywords 'document'
    recommended_ids = [all_news[i]['id'] for i in recommendations if i < len(all_news)]

    return jsonify({'ids': recommended_ids})


@app.route('/collaborative-filter', methods=['POST'])
def collaborative_filter():
    data = request.json
    print(data)

    # Convert list of dictionaries to a DataFrame
    df = pd.DataFrame(data)

    # Count how many times each topic has been viewed
    topic_popularity = df['topic_id'].value_counts()

    # Function to recommend topics based on popularity
    def recommend_popular_topics(num_recommendations=5):
        # Get the top 'num_recommendations' topics based on their view counts
        return list(topic_popularity.nlargest(num_recommendations).index)

    # Get recommendations for a new user
    recommendations_for_new_user = recommend_popular_topics(5)
    print("Recommended topics for a new user:", recommendations_for_new_user)

    return jsonify({'ids': recommendations_for_new_user})


if __name__ == '__main__':
    app.run(debug=True, port=1000)
