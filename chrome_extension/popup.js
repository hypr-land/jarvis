// Popup script for Jarvis Assistant extension
const messagesDiv = document.getElementById('messages');
const input = document.getElementById('input');
const sendBtn = document.getElementById('send');
const typingIndicator = document.getElementById('typingIndicator');

// New: Clear history button
const clearBtn = document.getElementById('clear');

// Resizer handle element
const resizer = document.getElementById('resizer');

const STORAGE_KEY = 'jarvisConversation';

const conversation = [];

// Attach handlers immediately (script at end of body)

if (clearBtn) {
  clearBtn.addEventListener('click', clearHistory);
}

if (resizer) {
  enableResizer(resizer);
}

// Load previous conversation when DOM ready
document.addEventListener('DOMContentLoaded', async () => {
  const data = await chrome.storage.local.get([STORAGE_KEY]);
  const saved = data[STORAGE_KEY] || [];
  saved.forEach(msg => {
    append(msg.role === 'assistant' ? 'Jarvis' : 'You', msg.content);
    conversation.push(msg);
  });
});

function append(role, text) {
  const div = document.createElement('div');
  const typeClass = role === 'You' ? 'user' : 'ai';
  div.className = `message ${typeClass}`;
  div.innerHTML = `<div>${text}</div>`;
  messagesDiv.appendChild(div);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function send() {
  const text = input.value.trim();
  if (!text) return;
  append('You', text);
  conversation.push({ role: 'user', content: text });
  input.value = '';
  sendBtn.disabled = true;

  // show typing indicator
  typingIndicator.style.display = 'block';

  chrome.runtime.sendMessage({ action: 'chat', messages: conversation }, (response) => {
    typingIndicator.style.display = 'none';
    sendBtn.disabled = false;
    if (response && response.success) {
      append('Jarvis', response.reply);
      conversation.push({ role: 'assistant', content: response.reply });
      chrome.storage.local.set({ [STORAGE_KEY]: conversation });
    } else {
      append('Error', response ? response.error : 'Unknown error');
    }
  });
}

sendBtn.addEventListener('click', send);

// Auto-resize textarea
input.addEventListener('input', () => {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 100) + 'px';
});

// Send on Enter (no Shift)
input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});

// Focus input on load
input.focus();

// --------------------
// Clear history helper
function clearHistory() {
  // Remove all message nodes
  messagesDiv.innerHTML = '';

  // Reset conversation array
  conversation.length = 0;

  // Remove from chrome storage
  chrome.storage.local.remove(STORAGE_KEY, () => {
    // Re-add greeting message
    append('Jarvis', "Hello! I'm JARVIS, your AI assistant. How can I help you today?");
  });
}

// --------------------
// Enable resize functionality for popup
function enableResizer(element) {
  let startX, startY, startWidth, startHeight;

  element.addEventListener('pointerdown', e => {
    e.preventDefault();
    startX = e.clientX;
    startY = e.clientY;
    const rect = document.documentElement.getBoundingClientRect();
    startWidth = rect.width;
    startHeight = rect.height;
    document.addEventListener('pointermove', onPointerMove);
    document.addEventListener('pointerup', onPointerUp);
  });

  function onPointerMove(e) {
    const newWidth = Math.min(600, Math.max(320, startWidth + (e.clientX - startX)));
    const newHeight = Math.min(800, Math.max(420, startHeight + (e.clientY - startY)));
    document.documentElement.style.width = newWidth + 'px';
    document.documentElement.style.height = newHeight + 'px';
  }

  function onPointerUp() {
    document.removeEventListener('pointermove', onPointerMove);
    document.removeEventListener('pointerup', onPointerUp);
  }
} 