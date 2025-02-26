from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient
import datetime
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for the entire app
CORS(app)  # This will allow all origins by default

# Alternatively, if you want to limit CORS to specific origins (e.g., localhost:5500)
# CORS(app, origins=["http://127.0.0.1:5500"])

# Initialize Hugging Face GPT-2 model pipeline for text generation
generator = pipeline("text-generation", model="gpt2")

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")  # Use your MongoDB connection string
db = client["ai_system"]  # Database name
queries_collection = db["queries"]  # Collection for storing user queries


@app.route("/query", methods=["POST"])
def get_ai_response():
    data = request.get_json()
    query = data.get("query")

    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        # Generate a response using Hugging Face's GPT-2 model
        response = generator(query, max_length=100, num_return_sequences=1)
        ai_response = response[0]["generated_text"].strip()

        # Store the query in MongoDB
        queries_collection.insert_one(
            {"query_text": query, "timestamp": datetime.datetime.utcnow()}
        )

        return jsonify({"response": ai_response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/suggestions", methods=["GET"])
def get_suggestions():
    # Fetch all queries from MongoDB
    queries = list(queries_collection.find({}, {"_id": 0, "query_text": 1}))

    if len(queries) < 2:
        return jsonify({"suggestions": []})

    # Extract the list of queries
    query_history = [q["query_text"] for q in queries]

    # Create TF-IDF matrix
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(query_history)

    # Get the cosine similarity between the last query and all previous queries
    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

    # Get the indices of the most similar queries
    related_indices = np.argsort(similarities[0])[-3:][::-1]
    suggestions = [query_history[i] for i in related_indices]

    return jsonify({"suggestions": suggestions})


@app.route("/history", methods=["GET"])
def get_history():
    # Fetch all queries from MongoDB
    queries = list(
        queries_collection.find({}, {"_id": 0, "query_text": 1, "timestamp": 1})
    )
    return jsonify({"history": queries})


# Main function to start the Flask app
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
