from flask import Flask, request, jsonify
import os
import logging
import json
import random
import requests
from transformers import pipeline  # type: ignore
import openai
import boto3
from botocore.exceptions import ClientError

app = Flask(__name__, '/static')

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
logging.basicConfig(level=logging.INFO)
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

app.route('/determine_response_type', methods=['POST'])
def determine_response_type():
    user_input = request.json.get("user_input", "")
    response_type = "initial"  # Default response type

    if "hello" in user_input.lower():
        response_type = "initial"
    # Add additional logic to determine response_type

    return jsonify({"response_type": response_type})
 
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
