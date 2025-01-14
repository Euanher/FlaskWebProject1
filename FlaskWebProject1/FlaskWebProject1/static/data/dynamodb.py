import boto3
import json
import os
from datetime import datetime

# Initialize LocalStack's mock AWS services
dynamodb = boto3.resource('dynamodb', endpoint_url="http://localstack/localstack:4566")

# Create a DynamoDB table for storing trivia questions and answers
def create_trivia_table():
    try:
        table = dynamodb.create_table(
            TableName='trivia-qa',
            KeySchema=[
                {
                    'AttributeName': 'question_id',
                    'KeyType': 'HASH'  # Partition key
                }
            ],
            AttributeDefinitions=[
                {
                    'AttributeName': 'question_id',
                    'AttributeType': 'S'  # String
                },
                {
                    'AttributeName': 'category',
                    'AttributeType': 'S'  # String (for filtering questions by category)
                }
            ],
            ProvisionedThroughput={
                'ReadCapacityUnits': 5,
                'WriteCapacityUnits': 5
            }
        )
        print(f"Table {table.name} created successfully!")
    except Exception as e:
        print(f"Error creating table: {e}")

# Function to add a trivia question to the table
def add_trivia_question(question_id, question, answer, category, difficulty="medium"):
    table = dynamodb.Table('trivia-qa')
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        table.put_item(Item={
            'question_id': question_id,
            'question': question,
            'answer': answer,
            'category': category,
            'difficulty': difficulty,
            'created_at': timestamp
        })
        print(f"Question '{question}' added successfully!")
    except Exception as e:
        print(f"Error adding question: {e}")

# Function to fetch a trivia question by ID
def get_trivia_question(question_id):
    table = dynamodb.Table('trivia-qa')
    try:
        response = table.get_item(Key={'question_id': question_id})
        return response.get('Item', None)
    except Exception as e:
        print(f"Error fetching question: {e}")
        return None

# Function to fetch all trivia questions by category
def get_trivia_questions_by_category(category):
    table = dynamodb.Table('trivia-qa')
    try:
        response = table.scan(FilterExpression='category = :category',
                              ExpressionAttributeValues={':category': category})
        return response.get('Items', [])
    except Exception as e:
        print(f"Error fetching questions by category: {e}")
        return []

if __name__ == '__main__':
    # Create table (run once)
    create_trivia_table()

    # Example of adding trivia questions
    add_trivia_question("1", "What is the capital of France?", "Paris", "Geography", "easy")
    add_trivia_question("2", "What is 2 + 2?", "4", "Math", "easy")
    add_trivia_question("3", "Who wrote 'Hamlet'?", "William Shakespeare", "Literature", "medium")

    # Example of fetching a trivia question by ID
    question = get_trivia_question("1")
    print("Fetched Question:", json.dumps(question, indent=2))

    # Example of fetching all questions from a specific category
    geography_questions = get_trivia_questions_by_category("Geography")
    print("Geography Questions:", json.dumps(geography_questions, indent=2))

