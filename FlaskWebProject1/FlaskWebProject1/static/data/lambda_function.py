import json

def lambda_handler(event, context):
    # Extracting message from the event
    message = event.get('message', 'No message provided')

    # Processing the message (this could be more sophisticated for your RAG chatbot)
    response_message = f"Processed message: {message}"

    # Returning the response as JSON
    return {
        'statusCode': 200,
        'body': json.dumps({'response': response_message})
    }
