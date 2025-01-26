import express from 'express';
import { json, urlencoded } from 'express';
import { config, DynamoDB, S3 } from 'aws-sdk';
import multer from 'multer';
import cors from 'cors';
import dotenv from 'dotenv';

dotenv.config();

const app = express();
const port = process.env.PORT || 5000;

// Middleware setup
app.use(cors());
app.use(json());
app.use(urlencoded({ extended: true }));

// Multer setup for file uploads
const upload = multer({ storage: multer.memoryStorage() });

// AWS SDK Configuration
config.update({
    region: 'us-east-1',
    accessKeyId: process.env.AWS_ACCESS_KEY_ID,
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
});

// Ensure AWS credentials are loaded
if (!process.env.AWS_ACCESS_KEY_ID || !process.env.AWS_SECRET_ACCESS_KEY) {
    console.error('AWS credentials are missing. Exiting...');
    process.exit(1);
}

// DynamoDB Client
const dynamoDB = new DynamoDB.DocumentClient();

// Fetch trivia data
const getTriviaData = async () => {
    try {
        const params = { TableName: 'trivia-questions' };
        const data = await dynamoDB.scan(params).promise();
        return data.Items || [];
    } catch (error) {
        console.error('Error fetching trivia data:', error);
        throw error; // Ensure calling functions handle errors
    }
};

// Fetch trivia endpoint
app.get('/get_trivia', async (req, res) => {
    try {
        const triviaData = await getTriviaData();
        if (!triviaData.length) {
            return res.status(404).json({ error: 'No trivia data found.' });
        }
        res.json(triviaData);
    } catch (error) {
        res.status(500).json({ error: 'Internal server error.' });
    }
});

// Upload file to S3
app.post('/upload_to_s3', upload.single('file'), (req, res) => {
    const { bucketName } = req.body;
    const file = req.file;

    if (!bucketName || !file) {
        return res.status(400).json({ error: 'Bucket name or file is missing.' });
    }

    const s3 = new S3();
    const params = {
        Bucket: bucketName,
        Key: file.originalname,
        Body: file.buffer,
        ContentType: file.mimetype,
    };

    s3.upload(params, (err, data) => {
        if (err) {
            console.error('Error uploading to S3:', err);
            return res.status(500).json({ error: 'Error uploading to S3.' });
        }
        res.json({ message: 'File uploaded successfully.', data });
    });
});

// Store chat history in DynamoDB
app.post('/store_chat_history', async (req, res) => {
    const { userInput, aiResponse } = req.body;

    if (!userInput || !aiResponse) {
        return res.status(400).json({ error: 'User input or AI response is missing.' });
    }

    const params = {
        TableName: 'chat-history',
        Item: {
            id: `${userInput}_${Date.now()}`,
            user_input: userInput,
            response: aiResponse,
            timestamp: Date.now(),
        },
    };

    try {
        await dynamoDB.put(params).promise();
        res.json({ message: 'Chat history stored successfully.' });
    } catch (error) {
        console.error('Error storing chat history:', error);
        res.status(500).json({ error: 'Error storing chat history.' });
    }
});

// Process user input endpoint
app.post('/api/process_user_input', (req, res) => {
    const { user_message } = req.body;

    if (!user_message) {
        return res.status(400).json({ error: 'User message is required.' });
    }

    const botResponse = `You said: "${user_message}"`;
    const systemMessages = ["System message 1", "System message 2"];

    res.json({
        assistant_response: { response: botResponse },
        system_messages: systemMessages,
    });
});

// Auto hello endpoint
app.post('/v1/auto_hello', (req, res) => {
    const autoHelloMessage = "Hello! Welcome to TRIVIA.io. How can I assist you today?";
    res.json({ auto_hello_response: autoHelloMessage });
});

// Generic error handler
app.use((err, req, res, next) => {
    console.error('Unhandled Error:', err.stack);
    res.status(500).send({ error: 'Internal server error.' });
});

// Start the server
app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
});

