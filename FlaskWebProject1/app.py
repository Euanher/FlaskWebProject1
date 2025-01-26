from uu import Error
from flask import Flask, request, jsonify
import os
import logging
import json 
import requests
from transformers import pipeline  # type: ignore
import openai
import boto3
from botocore.exceptions import ClientError

# Flask App Initialization
app = Flask(__name__, static_url_path="/static")

# Configuration
AWS_REGION = "us-east-1"
AWS_ACCESS_KEY_ID = "test"
AWS_SECRET_ACCESS_KEY = "test"
AWS_ENDPOINT_URL = "http://localhost:4566"  # Localstack for testing
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")
llama_model_url = "http://localhost:5000"
llama_model_id = "hermes-3-llama-3.2-3b"
CONFIG_FILE_PATH = "static/data/doc_rag.json"
KENDRA_INDEX_ID = "ls-xArASidO-deKO-4515-XArI-BIVo1000d275"   

# Create the AWS Kendra client
kendra_client = boto3.client(
    'kendra',
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    endpoint_url=AWS_ENDPOINT_URL  # Optional, useful for testing with LocalStack
)
# OpenAI Setup
openai.api_key = OPENAI_API_KEY

# Initialize the text-generation model
generator = pipeline("text-generation", model="gpt2")

# Logging setup
logging.basicConfig(logging.error, level=logging.INFO)
logger = logging.getLogger(__name__)

# Global RAG data
rag_data = {}

# Load RAG data
def load_rag_data():
    global rag_data
    try:
        with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as file:
            rag_data = json.load(file)
        logger.info("RAG data loaded successfully.")
    except FileNotFoundError:
        logger.error(f"RAG config file not found at: {CONFIG_FILE_PATH}")
        rag_data = {}
    except json.JSONDecodeError as error:
        logger.error(f"Error decoding RAG JSON: {error}")
        rag_data = {}
    except Exception as error:
        logger.error(f"Unexpected error loading RAG data: {error}")
        rag_data = {}

# Load RAG data on startup
load_rag_data()

# Mock model information
models = [
    {"id": "qwen2.5-3b-instruct", "name": "Qwen 2.5 3B Instruct"},
    {"id": "gpt-3.5-turbo", "name": "GPT 3.5 Turbo"},
    {"id": "hermes-3-llama-3.2-3b", "name": "Hermes 3 LLaMA 3.2 3B"},
]

# Tokenizer functions
def tokenize_input(input_text):
    return input_text.split()

def rag_tokenizer(messages):
    tokenized_messages = []
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        tokenized_messages.append({
            "role": role,
            "tokens": tokenize_input(content),
        })
    return tokenized_messages

# API Endpoints
@app.route('/v1/models', methods=['GET'])
def get_models():
    return jsonify({"models": models}), 200

def search_kendra(query, tokenized_messages="messages"):
    try:
        query = tokenized_messages.json.get({query:"input_text"})
        # Placeholder for Kendra search logic
        return [f"Result for {query}"]
    except ClientError as e:
        logger.error(f"Error querying Kendra: {e}")
        return ["Error querying Kendra."]

# Function to determine the response type based on user input
def determine_response_type_logic(user_input):
    """
    Determine the type of response based on user input.

    Args:
        user_input (str): The input provided by the user.

    Returns:
        str: The category of the response.
    """
    user_input_lower = user_input.lower()

    # Check for keywords in the input and return the associated category
    for category, keywords in rag_data.items():
        if any(keyword in user_input_lower for keyword in keywords):
            return category

    return "default"  # Return "default" if no category matches

# Define /v1/get_response route
@app.route("/v1/get_response", methods=["POST"])
def get_response():
    try:
        # Extract user input from the request
        user_input = request.json.get("user_input", "")
        
        if not user_input:
            return jsonify({"error": "Missing user input"}), 400
        
        # Determine the response type based on the input
        response_type = determine_response_type_logic(user_input)
        
        # Search Kendra for related documents based on the user input
        kendra_results = search_kendra(user_input)
        
        # Prepare the payload for the Llama model
        payload = {
            "input": user_input,
            "model": "llama_model_id"  # Replace with your actual model ID
        }

        # Query the Llama model (replace with your actual Llama model URL)
        llama_model_url = "http://your_llama_model_url"
        response = requests.post(f"{llama_model_url}/predict", json=payload)

        if response.status_code == 200:
            model_output = response.json().get("output", "")
            # Return the results from Kendra and Llama model
            return jsonify({
                "model_output": model_output,
                "response_type": response_type,
                "kendra_results": kendra_results
            }), 200
        else:
            logger.error(f"Llama model server error: {response.status_code} - {response.text}")
            return jsonify({"error": "Failed to get a response from the model."}), 500
    except Exception as error:
        logger.error(f"Error in getting response: {error}")
        return jsonify({"error": str(error)}), 500

# Function to determine the response type based on user input
def determine_response_type_logic(user_input):
    """
    Determine the type of response based on user input.

    Args:
        user_input (str): The input provided by the user.

    Returns:
        str: The category of the response.
    """
    user_input_lower = user_input.lower()

    # Check for keywords in the input and return the associated category
    for category, keywords in rag_data.items():
        if any(keyword in user_input_lower for keyword in keywords):
            return category

    return "default"  # Return "default" if no category matches

# Define /v1/get_response route
@app.route("/v1/get_response", methods=["POST"])
def get_response():
    try:
        # Extract user input from the request
        user_input = request.json.get("user_input", "")
        
        if not user_input:
            return jsonify({"error": "Missing user input"}), 400
        
        # Determine the response type based on the input
        response_type = determine_response_type_logic(user_input)
        
        # Search Kendra for related documents based on the user input
        kendra_results = search_kendra(user_input)
        
        # Prepare the payload for the Llama model
        payload = {
            "input": user_input,
            "model": "llama_model_id"  # Replace with your actual model ID
        }

        # Query the Llama model (replace with your actual Llama model URL)
        llama_model_url = "http://your_llama_model_url"
        response = requests.post(f"{llama_model_url}/predict", json=payload)

        if response.status_code == 200:
            model_output = response.json().get("output", "")
            # Return the results from Kendra and Llama model
            return jsonify({
                "model_output": model_output,
                "response_type": response_type,
                "kendra_results": kendra_results
            }), 200
        else:
            logger.error(f"Llama model server error: {response.status_code} - {response.text}")
            return jsonify({"error": "Failed to get a response from the model."}), 500
    except Exception as error:
        logger.error(f"Error in getting response: {error}")
        return jsonify({"error": str(error)}), 500


@app.route('/determine_response_type', methods=['POST'])
def determine_response_type():
    try:
<<<<<<< HEAD
        user_input = request.json.get("user_input", "").strip()
=======
        # Get the user input from the request
        user_input = request.json.get("user_input", "").strip()

>>>>>>> f95c480b9faf8462906002915c4aeea962ded177
        if not user_input:
            return jsonify({"error": "Missing user input"}), 400
        response_type = determine_response_type_logic(user_input)

        return jsonify({"response_type": response_type}), 200
    except Exception as error:
        return jsonify({"error": str(error)}), 500 

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    try:
        rag_data = request.json
        model_id = rag_data.get("model")
        messages = rag_data.get("messages", [])
        if not model_id or not messages:
            return jsonify({"error": "Missing required parameters: 'model' or 'messages'"}), 400

        tokenized_messages = rag_tokenizer(messages)

        if model_id == "qwen2.5-3b-instruct":
            response = {
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "The day is Thursday, I must say. It rhymes in a poetic way."
                        }
                    }
                ],
                "tokenized_input": tokenized_messages,
            }
            return jsonify(response), 200
        else:
            return jsonify({"error": f"Model '{model_id}' not supported"}), 400
    except Exception as error:
        logger.error(f"Error processing chat completions: {error}")
        return jsonify({"error": str(error)}), 500

# Route: /v1/trivia/questions
@app.route('/v1/trivia/questions', methods=['GET'])
def trivia_questions():
    try:
        trivia_data = [ {rag_data: "question"["response"]}]
        return jsonify({"success": True, "trivia": trivia_data}), 200
    except Exception as e:
        logger.error(f"Error fetching trivia questions: {e}")
        return jsonify({"error": str(e)}), 500

# Route: /v1/completions
@app.route('/v1/completions', methods=['POST'])
def completions():
    try:
        data = request.json
        prompt = data.get("prompt", "")
        if not prompt:
            return jsonify({"error": "Missing required parameter: 'prompt'"}), 400

        tokenized_prompt = tokenize_input(prompt)
        response = {
            "id": "cmpl-456",
            "object": "text_completion",
            "choices": [{"text": "The answer you seek lies in the lines; rhyming is fun, and it's always sublime."}],
            "tokenized_input": tokenized_prompt,
        }
        return jsonify(response), 200
    except Exception as error:
        logger.error(f"Error processing completions: {error}")
        return jsonify({"error": str(error)}), 500

# Route: /v1/embeddings
@app.route('/v1/embeddings', methods=['POST'])
def embeddings():
    try:
        data = request.json
        input_text = data.get("input", "")
        if not input_text:
            return jsonify({"error": "Missing required parameter: 'input'"}), 400

        tokenized_input = tokenize_input(input_text)
        response = {"object": "embedding", "data": [{"embedding": [0.1, 0.2, 0.3]}], "tokenized_input": tokenized_input}
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"Error processing embeddings: {e}")
        return jsonify({"error": str(e)}), 500

# Function to search Kendra
def search_kendra(query, index_id):
    """
    Queries AWS Kendra for the specified query.

    :param query: The query string to search in Kendra.
    :param index_id: The ID of the Kendra index to query.
    :return: A list of document titles that match the query, or an error message.
    """
    try:
        response = kendra_client.query(
            IndexId=index_id,
            QueryText=query
        )
        results = response.get('ResultItems', [])
        if results:
            return [result['DocumentTitle']['Text'] for result in results if 'DocumentTitle' in result]
        else:
            return ["No relevant documents found."]
    except ClientError as e:
        logger.error(f"Error querying Kendra: {e}")
        return ["Error querying Kendra."]
  
@app.route("/v1/get_response", methods=["POST"])
def get_response():
    try:
        user_input = request.json.get("user_input", "")
        response_type = determine_response_type(user_input)
        
        # Search Kendra for related documents based on user input
        kendra_results = search_kendra(user_input)

        # Prepare the payload for the Llama model
        payload = {
            "input": user_input,
            "model": llama_model_id
        }

        # Query the Llama model
        response = requests.post(f"{llama_model_url}/predict", json=payload)

        if response.status_code == 200:
            model_output = response.json().get("output", "")
            selected_response = response(response_type)
            return jsonify({
                "model_output": model_output,
                "selected_response": selected_response,
                "kendra_results": kendra_results  # Include Kendra search results in the response
            }), 200
        else:
            logger.error(f"Llama model server error: {response.status_code} - {response.text}")
            return jsonify({"error": "Failed to get a response from the model."}), 500
    except Exception as error:
        logger.error(f"Error in getting response: {error}")
        return jsonify({"error": str(error)}), 500

if __name__ == '__main__':
    app.run(debug=True)
