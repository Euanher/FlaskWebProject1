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
app = Flask(__name__, render_template="templates", static_url_path="static")

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
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s") 
logger = logging.getLogger(__name__)
logger.error(f"Error processing request: {request.json}, Error: {os.error}")

# Tokenizer initialization for OpenAI GPT models
def get_tokenizer(model_name="gpt-3.5-turbo"):
    """
    Get the tokenizer for a specific OpenAI model.
    """
    try:
        return tiktoken.encoding_for_model(model_name) # type: ignore
    except Exception as e:
        raise ValueError(f"Error initializing tokenizer for model {model_name}: {e}")

def rag_tokenizer(messages, model_name="gpt-3.5-turbo"):
    """
    Tokenize messages using OpenAI's tokenizer.
    Args:
        messages (list): List of message dictionaries with `role` and `content`.
        model_name (str): Model name for tokenization.

    Returns:
        list: Tokenized messages with token counts.
    """
    try:
        tokenizer = get_tokenizer(model_name)
        return [
            {
                "role": message.get("role", ""),
                "tokens": tokenizer.encode(message.get("content", "")),
                "token_count": len(tokenizer.encode(message.get("content", "")))
            }
            for message in messages
        ]
    except Exception as error:
        logger.error(f"Error tokenizing messages: {error}")
        raise

@app.route('/v1/rag_tokenizer', methods=['POST'])
def api_rag_tokenizer():
    try:
        data = request.json
        messages = data.get("messages", [])
        model_name = data.get("model", "gpt-3.5-turbo")

        if not messages or not isinstance(messages, list):
            return jsonify({"error": "Invalid or missing 'messages'"}), 400

        tokenized_messages = rag_tokenizer(messages)
        return jsonify({"tokenized_messages": tokenized_messages}), 200
    except Exception as error:
        logger.error(f"Error in /v1/rag_tokenizer: {error}")
        return jsonify({"error": f"Failed to tokenize messages: {error}"}), 500
 
# Global RAG data
rag_data = {}

# Load RAG data
def load_rag_data():
    """
    Load RAG (Retrieval-Augmented Generation) data from the configuration file.
    If the file is not found or there's an error, initialize RAG data with defaults.
    """
    global rag_data
    if not rag_data:
        rag_data = {"default": ["fallback_keyword"]}
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

# Initialize RAG data on startup
load_rag_data()

# Mock model information
models = [
    {"id": "qwen2.5-3b-instruct", "name": "Qwen 2.5 3B Instruct"},
    {"id": "gpt-3.5-turbo", "name": "GPT 3.5 Turbo"},
    {"id": "hermes-3-llama-3.2-3b", "name": "Hermes 3 LLaMA 3.2 3B"},
]

# Tokenizer functions
def tokenize_input(input_text):
    """Tokenize the input text."""
    return input_text.split()

def rag_tokenizer(messages):
    """
    Tokenize an array of messages, preserving roles and content.
    """
    return [
        {"role": msg.get("role", ""), "tokens": tokenize_input(msg.get("content", ""))}
        for msg in messages
    ]

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

# API Endpoints
@app.route('/v1/models', methods=['GET'])
def get_models():
    """
    Fetch the list of available models.
    """
    return jsonify({"models": models}), 200
 
@app.route('/v1/get_response', methods=['POST'])
def get_response():
    try:
        request_data = request.get_json()
        user_input = request_data.get("user_input", "").strip()
        
        if not user_input:
            return jsonify({"error": "Missing or invalid user input"}), 400

        response_type = determine_response_type_logic(user_input)

        # Query AWS Kendra
        kendra_results = search_kendra(user_input)

        # Query Llama model
        payload = {"input": user_input, "model": "llama_model_id"}
        llama_response = requests.post(f"{llama_model_url}/predict", json=payload)

        if llama_response.status_code == 200:
            model_output = llama_response.json().get("output", "No response from model")
        else:
            logger.error(f"Llama model error: {llama_response.status_code} - {llama_response.text}")
            model_output = "Error querying the Llama model."

        return jsonify({
            "response_type": response_type,
            "kendra_results": kendra_results,
            "model_output": model_output
        }), 200
    except Exception as error:
        logger.error(f"Error processing response: {error}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """
    Generate chat completions based on input messages and model ID.
    """
    try:
        data = request.json
        model_id = data.get("model")
        messages = data.get("messages", [])
        
        if not model_id or not messages:
            return jsonify({"error": "Missing required parameters: 'model' or 'messages'"}), 400

        tokenized_messages = rag_tokenizer(messages)

        # Process specific model logic
        if model_id == "qwen2.5-3b-instruct":
            response = {
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "choices": [{"message": {"role": "assistant", "content": "Here's your answer!"}}],
                "tokenized_input": tokenized_messages,
            }
            return jsonify({"response": "Here's your answer!", "system_messages": []}), 200
        else:
            return jsonify({"error": f"Model '{model_id}' not supported"}), 400

    except Exception as error:
        logger.error(f"Error processing chat completions: {error}")
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
  
# Miscellaneous routes
@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

@app.route('/v1/embeddings', methods=['POST'])
def embeddings():
    try:
        tokenized_input = request.json
        input_text = tokenized_input.get("input", "")
        if not input_text:
            return jsonify({"error": "Missing required parameter: 'input'"}), 400

        tokenized_input = tokenize_input(input_text)
        response = {"object": "embedding", "data": [{"embedding": [0.1, 0.2, 0.3]}]}   
        return jsonify(response), 200

    except Exception as error:
        logger.error(f"Error processing embeddings: {error}")
        return jsonify({"error": str(error)}), 500
      
@app.route('/determine_response_type', methods=['POST'])
def determine_response_type():
    try:  
        user_input = request.json.get("user_input", "").strip() 
        if not user_input:
            return jsonify({"error": "Missing user input"}), 400
        response_type = determine_response_type_logic(user_input)

        return jsonify({"message": response_type}), 200
    except Exception as error:         
        return jsonify({"error": str(error)}), 500 

@app.route('/v1/trivia/questions', methods=['GET'])
def trivia_questions():
    try:
        trivia_data = [{"question": "What is the capital of France?", "response": "Paris"}]
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

        # Tokenizing the input (real tokenization can be more complex depending on your needs)
        tokenized_prompt = tokenize_input(prompt)

        # This can be a mock embedding or replaced with actual embedding generation logic
        # For now, returning a mock embedding
        mock_embedding = [0.1, 0.2, 0.3]  

        # Creating the response structure similar to what you expect
        response = {
            "id": "cmpl-456",
            "object": "text_completion",
            "choices": [{
                "text": "The answer you seek lies in the lines; rhyming is fun, and it's always sublime."
            }],
            "tokenized_input": tokenized_prompt,
            "embedding": mock_embedding,  # Adding embedding data
        }

        return jsonify(response), 200

    except Exception as error:
        logger.error(f"Error processing completions: {error}")
        return jsonify({"error": str(error)}), 500
     
if __name__ == '__main__':
    app.run(debug=True)
