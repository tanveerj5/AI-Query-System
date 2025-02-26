from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
from pymongo import MongoClient
import datetime
import nltk
from nltk.corpus import stopwords
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download stopwords
nltk.download("stopwords")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize Hugging Face GPT-2 model pipeline for text generation
generator = pipeline("text-generation", model="gpt2")

# Load Hugging Face GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client["ai_system"]
queries_collection = db["queries"]

# Load English stopwords
stop_words = set(stopwords.words("english"))


def clean_text(text):
    """Removes stopwords, punctuation, and converts to lowercase."""
    text = text.translate(str.maketrans("", "", string.punctuation)).lower()
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return " ".join(filtered_words)


@app.route("/query", methods=["POST"])
def get_ai_response():
    """Generates an AI response based on a cleaned user query."""
    data = request.get_json()
    query = data.get("query")

    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        # Clean the query
        cleaned_query = clean_text(query)

        # Generate response using GPT-2
        response = generator(cleaned_query, max_length=100, num_return_sequences=1)
        ai_response = response[0]["generated_text"].strip()

        # Store the original query in MongoDB
        queries_collection.insert_one(
            {"query_text": query, "timestamp": datetime.datetime.utcnow()}
        )

        return jsonify({"response": ai_response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/suggestions", methods=["GET"])
def get_suggestions():
    """Suggests related queries based on past user queries using TF-IDF similarity."""
    # Fetch all queries from MongoDB
    queries = list(queries_collection.find({}, {"_id": 0, "query_text": 1}))

    if len(queries) < 2:
        return jsonify({"suggestions": []})

    # Extract query history and clean the text
    query_history = [clean_text(q["query_text"]) for q in queries]

    # Get the last cleaned query
    last_query = query_history[-1]

    # Generate the prompt for GPT-2
    prompt = f"Suggest related queries to '{last_query}'"

    # Tokenize the input prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    # Initialize a set to store unique suggestions
    unique_suggestions = set()

    while len(unique_suggestions) < 3:
        # Generate suggestions (5 suggestions for diversity)
        outputs = model.generate(
            inputs,
            max_length=50,  # max length of generated sequence
            num_return_sequences=5,  # Generate 5 sequences
            no_repeat_ngram_size=2,  # prevent repeating n-grams
            top_k=50,  # sample from top k words
            top_p=0.9,  # nucleus sampling
            temperature=0.6,  # lower randomness for more relevant suggestions
            pad_token_id=tokenizer.eos_token_id,  # padding token
            num_beams=5,  # Using beam search with beam size of 5
            early_stopping=True,  # Stop when the best sequence is found
        )

        # Decode and clean suggestions
        suggestions = [
            tokenizer.decode(output, skip_special_tokens=True)
            .replace(prompt, "")
            .strip()
            for output in outputs
        ]

        # Add unique suggestions to the set
        unique_suggestions.update(suggestions)

        if len(unique_suggestions) >= 3:
            break

    # Convert set to list and return exactly 3 suggestions
    unique_suggestions = list(unique_suggestions)[:3]

    return jsonify({"suggestions": unique_suggestions})


@app.route("/history", methods=["GET"])
def get_history():
    """Fetches user query history from MongoDB."""
    queries = list(
        queries_collection.find({}, {"_id": 0, "query_text": 1, "timestamp": 1})
    )
    return jsonify({"history": queries})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
