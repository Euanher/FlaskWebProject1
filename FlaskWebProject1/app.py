from ast import Global
import time
from flask import Flask, render_template, request, jsonify
import os
import logging
import json 
import requests
from transformers import pipeline  # type: ignore
import openai
import boto3
from botocore.exceptions import ClientError

# Flask App Initialization
app = Flask(__name__, render_template="templates", template_folder="static")

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

# Initialize text generation model
generator = pipeline("text-generation", model="gpt2")

# Logging setup
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Tokenizer initialization for OpenAI GPT models
def get_tokenizer(models="gpt-3.5-turbo"):
    """
    Get the tokenizer for a specific OpenAI model.
    """
    try:
        return tiktoken.encoding_for_model(models) # type: ignore
    except Exception as e:
        raise ValueError(f"Error initializing tokenizer for model {models}: {e}")

  
# Utility function to handle Kendra search
def search_kendra(query):
    try:
        response = kendra_client.query(
            IndexId=KENDRA_INDEX_ID,
            QueryText=query
        )
        return response.get('ResultItems', [])
    except ClientError as e:
        logger.error(f"Error querying Kendra: {e}")
        return ["Error querying Kendra."]
     
@app.route('/v1/rag_tokenizer', methods=['POST'])
def api_rag_tokenizer():
    try:
        data = request.json(models)
        messages = data.get("messages", [])
        models = data.get("model", "gpt-3.5-turbo")

        if not messages or not isinstance(messages, list):
            return jsonify({"error": "Invalid or missing 'messages'"}), 400

        tokenized_messages = rag_tokenizer(messages)
        return jsonify({"tokenized_messages": tokenized_messages}), 200
    except Exception as error:
        logger.error(f"Error in /v1/rag_tokenizer: {error}")
        return jsonify({"error": f"Failed to tokenize messages: {error}"}), 500
  
# Mock model information
models = [
    {"id": "qwen2.5-3b-instruct", "name": "Qwen 2.5 3B Instruct"},
    {"id": "gpt-3.5-turbo", "name": "GPT 3.5 Turbo"},
    {"id": "hermes-3-llama-3.2-3b", "name": "Hermes 3 LLaMA 3.2 3B"},
]

# RAG Data Initialization
rag_data = {}

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

load_rag_data()

# Utility: Tokenization
def tokenize_input(input_text):
    return input_text.split()

def rag_tokenizer(messages):
    return [{
        "role": msg.get("role", ""),
        "tokens": tokenize_input(msg.get("content", "")),
        "content": msg.get("content", "")
    } for msg in messages]

# Utility: Kendra Search
def search_kendra(query):
    try:
        response = kendra_client.query(
            IndexId=KENDRA_INDEX_ID,
            QueryText=query
        )
        return response.get('ResultItems', [])
    except ClientError as e:
        logger.error(f"Error querying Kendra: {e}")
        return ["Error querying Kendra."]

# Endpoints
@app.route('/v1/models', methods=['GET'])
def get_models():
    models = [
        {"id": "qwen2.5-3b-instruct", "name": "Qwen 2.5 3B Instruct"},
        {"id": "gpt-3.5-turbo", "name": "GPT 3.5 Turbo"},
        {"id": "hermes-3-llama-3.2-3b", "name": "Hermes 3 LLaMA 3.2 3B"}
    ]
    return jsonify({"models": models}), 200

@app.route('/v1/rag_tokenizer', methods=['POST'])
def api_rag_tokenizer():
    try:
        data = request.json
        messages = data.get("messages", [])
        if not messages or not isinstance(messages, list):
            return jsonify({"error": "Invalid or missing 'messages'"}), 400

        tokenized_messages = rag_tokenizer(messages)
        return jsonify({"tokenized_messages": tokenized_messages}), 200
    except Exception as error:
        logger.error(f"Error in /v1/rag_tokenizer: {error}")
        return jsonify({"error": f"Failed to tokenize messages: {error}"}), 500

@app.route('/v1/get_response', methods=['POST'])
def get_response():
    try:
        data = request.get_json()
        user_input = data.get("user_input", "").strip()

        if not user_input:
            return jsonify({"error": "Missing or invalid user input"}), 400

        # Perform Kendra search
        kendra_results = search_kendra(user_input)

        # Query the Llama model
        payload = {"input": user_input, "model": llama_model_id}
        llama_response = requests.post(f"{llama_model_url}/predict", json=payload)
        model_output = llama_response.json().get("output", "No response from model") if llama_response.status_code == 200 else "Error querying the Llama model."

        return jsonify({
            "response_type": "Trivia Question",
            "kendra_results": kendra_results,
            "model_output": model_output
        }), 200
    except Exception as error:
        logger.error(f"Error in /v1/get_response: {error}")
        return jsonify({"error": "Internal server error"}), 500
 

# API Endpoints
@app.route('/v1/models', methods=['GET'])
def get_models():
    """
    Fetch the list of available models.
    """
    return jsonify({"models": models}), 200

@app.route('/api/process_user_input', methods=['POST'])
def process_user_input():
    """Process the user's input."""
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        user_input = request.json.get('user_input')
        
        if not user_input:
            return jsonify({"error": "'user_input' is required"}), 400
        
        processed_input = f"Processed: {user_input}"
        return jsonify({"status": "Input processed", "processed_input": processed_input}), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500
 

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """Generate chat completions based on input messages and model ID."""
    try:
        data = request.json
        
        model_id = data.get("model")
        messages = data.get("messages", [])
        
        if not model_id or not messages:
            return jsonify({"error": "Missing required parameters: 'model' or 'messages'"}), 400

        supported_models = ["qwen2.5-3b-instruct", "hermes-3-llama-3.2-3b"]
        if model_id not in supported_models:
            return jsonify({"error": f"Model '{model_id}' not supported"}), 400
        
        tokenized_messages = rag_tokenizer(messages)
        
        if model_id == "qwen2.5-3b-instruct":
            response = {
                "id": "chatcmpl-" + str(int(time.time() * 1000)),
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Today's Thursday, my dear friend, Just like the song of Frère Jacques says so."
                        },
                        "logprobs": None,
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": len(tokenized_messages),
                    "completion_tokens": 19,
                    "total_tokens": len(tokenized_messages) + 19
                },
                "system_fingerprint": model_id
            }
            return jsonify(response), 200
        
        else:
            return jsonify({"error": f"Model '{model_id}' not supported"}), 400

    except Exception as error:
        logger.error(f"Error processing chat completions: {error}")
        return jsonify({"error": f"Internal server error: {str(error)}"}), 500


def determine_response_type_logic(user_input):
    """Determine the type of response based on user input."""
    user_input_lower = user_input.lower()

    for category, keywords in rag_data.items():
        if any(keyword.lower() in user_input_lower for keyword in keywords):
            return category

    if "trivia" in user_input_lower:
        return "Trivia"
    
    return "Unknown"


@app.route('/v1/trivia/questions', methods=['GET'])
def trivia_questions():
    """Fetch trivia questions."""
    try:
        trivia_data = rag_data.get("trivia", [])
        return jsonify({"success": True, "trivia": trivia_data}), 200
    except Exception as e:
        logger.error(f"Error fetching trivia questions: {e}")
        return jsonify({"error": str(e)}), 500
 
# Miscellaneous routes
@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

@app.route('/v1/embeddings', methods=['POST'])
def embeddings():
    try:
        # Assuming the input is in the request's JSON body
        request_data = request.json
        input_text = request_data.get("input", "")

        # Check if input is missing or invalid
        if not input_text:
            return jsonify({"error": "Missing required parameter: 'input'"}), 400

        # Tokenize the input (ensure tokenize_input is a valid function)
        input_text = tokenize_input(input_text)

        # Response with a mock embedding (you should replace this with actual embedding logic)
        response = {"object": "embedding", "data": [{"embedding": [0.1, 0.2, 0.3]}]}   
        
        return jsonify(response), 200

    except Exception as error:
        logger.error(f"Error processing embeddings: {error}")
        return jsonify({"error": str(error)}), 500


@app.route('/determine_response_type', methods=['POST'])
def determine_response_type():
    global rag_data
    try:
        data = request.rag_data()
        user_input = data.get('user_input', '').lower()

        # If the user wants to play trivia, return the first trivia question
        if user_input == "yes":
            trivia_question = rag_data[0]  # Get the first question
            response = {
                "response_type": "Trivia Question",
                "question": trivia_question["question"],
                "options": trivia_question["options"]
            }
        else:
            response = {
                "response_type": "Unknown",
                "message": "Please answer with 'yes' if you want to play trivia."
            }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
 
  
@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """
    Generate chat completions based on input messages and model ID.
    """
    try:
        data = request.json
        
        # Ensure all required parameters are present
        model_id = data.get("model")
        messages = data.get("messages", [])
        
        if not model_id or not messages:
            return jsonify({"error": "Missing required parameters: 'model' or 'messages'"}), 400

        # Optional: validate model_id, check against a predefined list
        supported_models = ["qwen2.5-3b-instruct", "hermes-3-llama-3.2-3b"]
        if model_id not in supported_models:
            return jsonify({"error": f"Model '{model_id}' not supported"}), 400
        
        # Tokenizing messages
        tokenized_messages = rag_tokenizer(messages)
        
        # Example of handling model-specific logic
        if model_id == "qwen2.5-3b-instruct":
            # Simulating a chat completion response (replace with actual model inference logic)
            response = {
                "id": "chatcmpl-" + str(int(time.time() * 1000)),  # Unique ID based on current time
                "object": "chat.completion",
                "created": int(time.time()),  # Current timestamp
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Today's Thursday, my dear friend, Just like the song of Frère Jacques says so."
                        },
                        "logprobs": None,
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": len(tokenized_messages),  # Tokens for the input message
                    "completion_tokens": 19,  # Tokens for the completion, adjust based on actual model output
                    "total_tokens": len(tokenized_messages) + 19  # Total tokens calculation
                },
                "system_fingerprint": model_id
            }
            return jsonify(response), 200
        
        # Additional model-specific handling could be added here
        else:
            return jsonify({"error": f"Model '{model_id}' not supported"}), 400

    except Exception as error:
        logger.error(f"Error processing chat completions: {error}")
        return jsonify({"error": f"Internal server error: {str(error)}"}), 500


if __name__ == "__main__": 
    app.run(debug=True)

