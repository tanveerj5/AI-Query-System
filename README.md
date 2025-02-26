# AI-Powered Query System 🤖

This project implements a simple AI-based system that provides text-based query responses, topic recommendations, and keeps a log of past queries. The system uses OpenAI's GPT API, Cohere, or a custom NLP model to generate responses to user queries and recommends related topics using machine learning techniques.

## Table of Contents 📑

- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Tech Stack Requirements](#tech-stack-requirements)
- [API Endpoints](#api-endpoints)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)

## Introduction 🌟

This project implements the following tasks:

1. **AI-Powered Text Processing (NLP)**: A system where users can send text-based queries to an API and get AI-generated responses that are contextually relevant.
2. **Topic-Based Query Recommendations (ML)**: A recommendation engine that analyzes past user queries, suggests related topics, and stores these suggestions.
3. **API & Backend Implementation**: A REST API that allows users to interact with the system.

## Features ✨

- **AI-based Text Response**: Get AI-generated responses for any query submitted by the user.
- **Topic Recommendations**: The system suggests three related topics based on the user’s past queries using NLP techniques like TF-IDF, cosine similarity, or a pre-trained model.
- **Query History**: Retrieves the history of user queries for logging and tracking purposes.
- **REST API**: Exposes endpoints for users to interact with the system via HTTP requests.

## Technologies Used 🛠️

- **Backend Framework**: Flask (or FastAPI) for creating the REST API.
- **NLP Libraries**: OpenAI GPT API, Cohere API, or custom NLP models for text processing.
- **Machine Learning**: TF-IDF, cosine similarity for topic recommendation.
- **Database**: SQLite, MongoDB, or a similar lightweight database for storing user queries and topic suggestions.
- **Python**: Python 3.x

## Tech Stack Requirements ⚙️

- **NLP Model**: OpenAI’s GPT API, Cohere, or a self-built NLP model (spaCy, Hugging Face, or TF-IDF).
- **Backend**: Python (Flask/Django/FastAPI) or Node.js (Express).
- **Database**: PostgreSQL, Firebase, or MongoDB.
- **Frontend (Optional Bonus)**: 
  - HTML, Bootstrap, and JavaScript for the frontend, or
  - React.js or Vue.js for building dynamic user interfaces.

## API Endpoints 🌐

### 1. POST `/query` 💬
**Description**: Accepts a user-submitted query and returns an AI-generated response.

- **Request Body**:
    ```json
    {
      "query": "How does AI work?"
    }
    ```

- **Response**:
    ```json
    {
      "response": "AI works by using algorithms to simulate human intelligence."
    }
    ```

### 2. GET `/suggestions` 🧠
**Description**: Returns three suggested topics based on past queries using machine learning techniques.

- **Response**:
    ```json
    {
      "suggestions": ["Artificial Intelligence", "Machine Learning", "Neural Networks"]
    }
    ```

### 3. GET `/history` 📜
**Description**: Retrieves the history of past user queries.

- **Response**:
    ```json
    {
      "history": [
        "How does AI work?",
        "What is machine learning?",
        "Explain neural networks."
      ]
    }
    ```

## Setup Instructions ⚙️

### Prerequisites

- Python 3.x
- `pip` (Python package manager)

### Installation 🛠️

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/ai-query-system.git
    cd ai-query-system
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Create a `.env` file and add the necessary API keys:
    - OpenAI API key
    - Cohere API key (if using Cohere for NLP)

4. Run the application:
    ```bash
    python app.py
    ```

    The server will be running at `http://127.0.0.1:5000`.

## Usage 🖥️

- **Submitting Queries**: Use a REST client (like Postman or cURL) to send a POST request to `/query` with your query.
- **Getting Suggested Topics**: Send a GET request to `/suggestions` to retrieve related topics based on user history.
- **Viewing Query History**: Send a GET request to `/history` to see past queries.

Example cURL command to submit a query:
```bash
curl -X POST http://127.0.0.1:5000/query -H "Content-Type: application/json" -d '{"query": "What is AI?"}'
