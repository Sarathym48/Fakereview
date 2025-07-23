from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
import pandas as pd
import numpy as np
import pickle
import string
import unicodedata
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob

# Setup
app = Flask(__name__)
CORS(app)

# Load models
with open("randomforestregressor.pkl", "rb") as f:
    model1 = pickle.load(f)

with open("Xgbclassifier.pkl", "rb") as f:
    model2 = pickle.load(f)

# Load TF-IDF vectorizers
with open("tfidf_vectorizer1.pkl", "rb") as f:
    tfidf_vectorizer1 = pickle.load(f)

with open("tfidf_vectorizer2.pkl", "rb") as f:
    tfidf_vectorizer2 = pickle.load(f)

# Sentiment + stopwords
analyzer = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

# Memory to store latest results
latest_results = None

### Clean Unicode Text ###
def clean_text(text):
    try:
        return unicodedata.normalize('NFKD', str(text)).encode('utf-8', 'ignore').decode('utf-8')
    except:
        return str(text)

### Model 1 Preprocessing ###
def preprocess_model1(reviews):
    df = pd.DataFrame(reviews, columns=["Review"])
    df["review_length"] = df["Review"].apply(lambda x: len(str(x).split()))
    df["sentiment"] = df["Review"].apply(lambda x: analyzer.polarity_scores(str(x))["compound"])
    df["repeat_count"] = df["Review"].apply(lambda x: max(Counter(str(x).lower().split()).values()) if x else 0)
    df["exclamation_count"] = df["Review"].apply(lambda x: str(x).count("!"))
    df["first_person_count"] = df["Review"].apply(lambda x: sum(1 for word in str(x).lower().split() if word in {"i", "me", "my", "mine", "we", "us", "our", "ours"}))
    X_tfidf = tfidf_vectorizer1.transform(df["Review"]).toarray()
    X_additional = df[['review_length', 'sentiment', 'repeat_count', 'exclamation_count', 'first_person_count']].values
    return np.hstack((X_tfidf, X_additional))

### Model 2 Preprocessing ###
def preprocess_model2(reviews):
    df = pd.DataFrame(reviews, columns=["text_"])
    df["review_length"] = df["text_"].apply(lambda x: len(str(x).split()))
    df["polarity"] = df["text_"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df["subjectivity"] = df["text_"].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)
    df["stopword_ratio"] = df["text_"].apply(lambda x: len([word for word in str(x).split() if word.lower() in stop_words]) / max(len(str(x).split()), 1))
    df["punctuation_count"] = df["text_"].apply(lambda x: len([char for char in str(x) if char in string.punctuation]))
    X_tfidf = tfidf_vectorizer2.transform(df["text_"]).toarray()
    X_additional = df[['review_length', 'polarity', 'subjectivity', 'stopword_ratio', 'punctuation_count']].values
    return np.hstack((X_tfidf, X_additional))

### Contextual Score ###
def compute_contextual_score(target_review, additional_reviews):
    if not target_review or not additional_reviews:
        return 0

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(additional_reviews).toarray()
    avg_review_vector = np.mean(tfidf_matrix, axis=0).reshape(1, -1)
    target_tfidf = vectorizer.transform([target_review])
    cosine_sim = cosine_similarity(target_tfidf, avg_review_vector).flatten()[0]

    additional_sentiments = [TextBlob(r).sentiment.polarity for r in additional_reviews]
    avg_sentiment = np.mean(additional_sentiments)
    target_sentiment = TextBlob(target_review).sentiment.polarity
    sentiment_consistency = 1 - abs(target_sentiment - avg_sentiment)
    

    style_vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    style_matrix = style_vectorizer.fit_transform(additional_reviews)
    avg_style_vector = np.mean(style_matrix.toarray(), axis=0).reshape(1, -1)
    target_style_tfidf = style_vectorizer.transform([target_review])
    language_style_consistency = cosine_similarity(target_style_tfidf, avg_style_vector).flatten()[0]

    weights = {
        "cosine_similarity": 0.4,
        "sentiment_consistency": 0.5,
        "language_style_consistency": 0.1,
    }

    contextual_score = (
        weights["cosine_similarity"] * cosine_sim +
        weights["sentiment_consistency"] * sentiment_consistency +
        weights["language_style_consistency"] * language_style_consistency
    )

    if cosine_sim == 0:
        contextual_score = 0

    return contextual_score

### Reliability Grade ###
def compute_reliability_grade(avg_score):
    if avg_score >= 134:
        return "A"
    elif avg_score >= 118:
        return "B"
    elif avg_score >= 100:
        return "C"
    elif avg_score >= 78:
        return "D" 
    else:
        return "E"

### Prediction Pipeline ###
def predict_review(reviews):
    reviews = [clean_text(r) for r in reviews]

    features_model1 = preprocess_model1(reviews)
    features_model2 = preprocess_model2(reviews)

    contextual_scores = []
    for i, review in enumerate(reviews):
        additional_reviews = reviews[:i] + reviews[i+1:]
        score = compute_contextual_score(review, additional_reviews)
        contextual_scores.append(score)

    pred1 = model1.predict(features_model1)
    pred2 = model2.predict(features_model2)

    regression_score = pred1
    class_score = np.where(pred2 == 1, 20, 0)
    final_scores = regression_score + (np.array(contextual_scores) * 100) + class_score

    df_results = pd.DataFrame({
        "Review": reviews,
        "Regression Score": np.floor(regression_score).astype(int),
        "Classification Score": class_score,
        "Contextual Score": np.floor(np.array(contextual_scores) * 100).astype(int),
        "Final Score": np.floor(final_scores).astype(int)
    })

    

    df_results.to_csv('Reviews.csv', index=False, encoding='utf-8-sig')

    return df_results

### API Route to Process ###
@app.route('/process_reviews', methods=['POST'])
def process_reviews():
    global latest_results
    try:
        data = request.json
        reviews = data.get("reviews", [])
        product_name = data.get("productName", "Unknown Product")
        product_image = data.get("productImage", "")

        if not reviews:
            return jsonify({"error": "No reviews received"}), 400

        df_results = predict_review(reviews)

        avg_score = df_results["Final Score"].mean()
        grade = compute_reliability_grade(avg_score)
        top_reviews = df_results.sort_values(by="Final Score", ascending=False).head(5).to_dict(orient="records")

        # Store for dashboard route
        latest_results = {
            "productName": product_name,
            "productImage": product_image,
            "grade": grade,
            "average_score": round(avg_score, 2),
            "top_reviews": top_reviews
        }

        return jsonify({
            "message": "Reviews processed",
            "count": len(reviews),
            "productName": product_name,
            "productImage": product_image,
            "reliability_grade": grade,
            "average_score": round(avg_score, 2),
            "top_reviews": top_reviews,
            "results": df_results.to_dict(orient="records")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

### API Route to Get Latest ###
@app.route('/get_latest_reviews', methods=['GET'])
def get_latest_reviews():
    if not latest_results:
        return jsonify({"message": "No reviews analyzed yet"}), 404
    return jsonify(latest_results)

if __name__ == '__main__':
    app.run(debug=True)
