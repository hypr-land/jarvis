// Initialize Socket.IO connection
const socket = io();

// DOM Elements
const chatContainer = document.getElementById('chat-container');
const queryInput = document.getElementById('query-input');
const sendButton = document.getElementById('send-btn');

// Stats Elements
const cpuBar = document.getElementById('cpu-bar');
const cpuText = document.getElementById('cpu-text');
const memoryBar = document.getElementById('memory-bar');
const memoryText = document.getElementById('memory-text');
const threadsText = document.getElementById('threads-text');
const connectionsText = document.getElementById('connections-text');
const statusText = document.getElementById('status-text');

// Helper function to create message elements
function createMessageElement(content, isUser = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
    messageDiv.textContent = content;
    return messageDiv;
}

// Helper function to create typing indicator
function createTypingIndicator() {
    const indicator = document.createElement('div');
    indicator.className = 'typing-indicator bot-message';
    indicator.innerHTML = `
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
    `;
    return indicator;
}

// Function to scroll chat to bottom
function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Handle sending messages
async function sendMessage() {
    const query = queryInput.value.trim();
    if (!query) return;

    // Clear input
    queryInput.value = '';

    // Add user message to chat
    chatContainer.appendChild(createMessageElement(query, true));
    scrollToBottom();

    // Show typing indicator
    const typingIndicator = createTypingIndicator();
    chatContainer.appendChild(typingIndicator);
    scrollToBottom();

    try {
        // Send via WebSocket
        socket.emit('query', { query });
    } catch (error) {
        console.error('Error sending message:', error);
        chatContainer.removeChild(typingIndicator);
        chatContainer.appendChild(createMessageElement('Error: Failed to send message. Please try again.'));
        scrollToBottom();
    }
}

// Event Listeners
sendButton.addEventListener('click', sendMessage);
queryInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Socket.IO Event Handlers
socket.on('connect', () => {
    console.log('Connected to server');
    statusText.textContent = 'Connected';
    statusText.className = 'text-lg text-green-500';
});

socket.on('disconnect', () => {
    console.log('Disconnected from server');
    statusText.textContent = 'Disconnected';
    statusText.className = 'text-lg text-red-500';
});

socket.on('response', (data) => {
    // Remove typing indicator if present
    const typingIndicator = chatContainer.querySelector('.typing-indicator');
    if (typingIndicator) {
        chatContainer.removeChild(typingIndicator);
    }

    // Add bot response
    if (data.type === 'response' && data.data && data.data.response) {
        chatContainer.appendChild(createMessageElement(data.data.response));
    } else {
        console.error('Invalid response format:', data);
        chatContainer.appendChild(createMessageElement('Error: Invalid response format from server'));
    }
    scrollToBottom();
});

socket.on('error', (data) => {
    // Remove typing indicator if present
    const typingIndicator = chatContainer.querySelector('.typing-indicator');
    if (typingIndicator) {
        chatContainer.removeChild(typingIndicator);
    }

    // Add error message
    const errorMessage = data.data && data.data.message 
        ? `Error: ${data.data.message}`
        : 'An unknown error occurred';
    
    const messageElement = createMessageElement(errorMessage);
    messageElement.style.color = '#ef4444'; // Red color for errors
    chatContainer.appendChild(messageElement);
    scrollToBottom();
});

// Handle stats updates
socket.on('stats_update', (stats) => {
    // Update CPU usage (convert to percentage with 1 decimal place)
    const cpuPercent = (stats.cpu_percent || 0).toFixed(1);
    cpuBar.style.width = `${Math.min(cpuPercent, 100)}%`;
    cpuText.textContent = `${cpuPercent}%`;

    // Update Memory usage (convert to percentage with 1 decimal place)
    const memPercent = (stats.memory_percent || 0).toFixed(1);
    memoryBar.style.width = `${Math.min(memPercent, 100)}%`;
    memoryText.textContent = `${memPercent}%`;

    // Update other stats
    threadsText.textContent = stats.threads || 0;
    connectionsText.textContent = stats.connections || 0;
    
    // Update status
    const status = stats.status || 'unknown';
    statusText.textContent = status;
    statusText.className = `text-lg ${status === 'running' ? 'text-green-500' : 'text-yellow-500'}`;
});

// Mobile viewport height fix
function setMobileHeight() {
    const vh = window.innerHeight * 0.01;
    document.documentElement.style.setProperty('--vh', `${vh}px`);
}

window.addEventListener('resize', setMobileHeight);
setMobileHeight(); 