document.addEventListener('DOMContentLoaded', () => {
    const sendButton = document.querySelector('#send-btn');
    const userInput = document.querySelector('#user-input');
    const messagesDiv = document.querySelector('#chat-box');
    const loader = document.querySelector('#loader');
    const startGameButton = document.querySelector('#start-game');
    const darkModeToggle = document.querySelector('#dark-mode-toggle');
    const avatars = document.querySelectorAll('.avatar');

    // Handle Send Button click
    sendButton.addEventListener('click', async () => {
        const userInputValue = userInput.value.trim();

        if (!userInputValue) {
            alert('Please enter a message.');
            return;
        }

        // Show loader during processing
        toggleLoader(true);

        try {
            const data = await fetchAPI('/api/process_user_input', { user_message: userInputValue });
            handleRAGResponse(data, userInputValue);
        } catch (error) {
            console.error('Error:', error);
            displayMessage('system_message', 'Sorry, there was an error processing your request.');
        } finally {
            toggleLoader(false);
        }
    });

    // Handle Enter key press to submit message
    userInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            event.preventDefault();
            sendButton.click();
        }
    });

    // Utility Function: Fetch API
    async function fetchAPI(url, payload) {
        const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        return response.json();
    }

    // Utility Function: Toggle Loader
    function toggleLoader(isVisible) {
        loader.style.display = isVisible ? 'block' : 'none';
    }

    // Handle RAG API Response
    function handleRAGResponse(data, userInputValue) {
        const assistantMessage = data.assistant_response?.response || "No response from assistant.";
        const systemMessages = data.system_messages || [];

        // Display user and bot messages
        displayMessage('user', userInputValue);
        displayMessage('bot', assistantMessage);

        systemMessages.forEach(msg => {
            displayMessage('system_message', msg);
        });

        userInput.value = ''; // Clear input field
    }

    // Utility Function: Display Messages
    function displayMessage(sender, message) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', sender);
        messageElement.textContent = message;
        messagesDiv.appendChild(messageElement);
        messagesDiv.scrollTop = messagesDiv.scrollHeight; // Auto-scroll
    }

    // Start game button functionality (Trivia Questions)
    startGameButton.addEventListener('click', async () => {
        const selectedAvatar = document.querySelector('.avatar.selected');
        if (!selectedAvatar) {
            alert("Please select an avatar!");
            return;
        }

        try {
            const avatarData = selectedAvatar.getAttribute('data-player');
            const data = await fetchAPI('/select_avatar', { avatar: avatarData });

            // Hide avatar selection and show chat interface
            document.querySelector('.avatar-selection').style.display = 'none';
            document.querySelector('#chat-box').style.display = 'block';
            document.querySelector('#user-input').style.display = 'block';
            document.querySelector('#send-btn').style.display = 'block';

            alert(data.message);
            loadTriviaQuestion(); // Start trivia game
        } catch (error) {
            console.error('Error:', error);
            alert('Error processing the request. Please try again.');
        }
    });

    // Load trivia question
    async function loadTriviaQuestion() {
        toggleLoader(true);

        try {
            const data = await fetchAPI('/determine_response_type', {});
            const triviaQuestion = data.question || "No question available.";
            displayMessage('bot', triviaQuestion);
        } catch (error) {
            console.error('Error:', error);
            displayMessage('system_message', 'Sorry, there was an error fetching the trivia question.');
        } finally {
            toggleLoader(false);
        }
    }

    // Avatar selection functionality
    avatars.forEach(avatar => {
        avatar.addEventListener('click', () => {
            avatars.forEach(av => av.classList.remove('selected'));
            avatar.classList.add('selected');
        });
    });

    // Dark Mode Toggle
    darkModeToggle.addEventListener('click', () => {
        document.body.classList.toggle('dark-mode');
    });
});
