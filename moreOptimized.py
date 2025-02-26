from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from transformers import pipeline
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from pymongo import MongoClient
import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for the entire app
CORS(app)  # This will allow all origins by default

# Initialize Hugging Face GPT-2 model pipeline for text generation
generator = pipeline("text-generation", model="gpt2")

# Initialize Hugging Face GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# MongoDB connection (assuming queries_collection is initialized)
client = MongoClient("mongodb://localhost:27017/")
db = client["ai_system"]
queries_collection = db["queries"]


@app.route("/query", methods=["POST"])
def get_ai_response():
    data = request.get_json()
    query = data.get("query")

    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        # Generate a response using Hugging Face's GPT-2 model
        # Generate response based on the query
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

    # Get the last query for generating suggestions
    last_query = query_history[-1]

    # Generate the prompt for GPT-2
    prompt = f"Suggest related queries to '{last_query}'"

    # Tokenize the input prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    # Initialize a set to store unique suggestions
    unique_suggestions = set()

    while len(unique_suggestions) < 3:
        # Generate suggestions (e.g., 5 suggestions to ensure diversity)
        outputs = model.generate(
            inputs,
            max_length=50,  # max length of generated sequence
            num_return_sequences=5,  # Generate 5 sequences (must be <= num_beams)
            no_repeat_ngram_size=2,  # prevent repeating n-grams
            top_k=50,  # sample from top k words
            top_p=0.95,  # nucleus sampling (top p probability)
            temperature=0.7,  # randomness of output
            pad_token_id=tokenizer.eos_token_id,  # padding token
            num_beams=5,  # Using beam search with beam size of 5
            early_stopping=True,  # Stop as soon as the best beam is found
        )

        # Decode the generated sequences and clean them
        suggestions = [
            tokenizer.decode(output, skip_special_tokens=True)
            .replace(prompt, "")
            .strip()
            for output in outputs
        ]

        # Add unique suggestions to the set (set automatically removes duplicates)
        unique_suggestions.update(suggestions)

        # Ensure we're not generating too many suggestions
        if len(unique_suggestions) >= 3:
            break

    # Convert the set back to a list and slice to get exactly 3 suggestions
    unique_suggestions = list(unique_suggestions)[:3]

    return jsonify({"suggestions": unique_suggestions})


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
