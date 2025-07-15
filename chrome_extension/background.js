/* Background/service worker script for Jarvis Assistant Chrome extension */

const JARVIS_SYSTEM_PROMPT = `Your name is J.A.R.V.I.S. (Just A Rather Very Intelligent System), inspired by the character from the Marvel Universe. Your purpose is to assist the user with precision, speed, and a touch of dry wit.

Voice and Tone: Polite, refined, eloquent. British-accented tone (inspired by Paul Bettany's performance) with subtle dry humor. Always respectful and composed. Keep it concise whenever possible.

Personality: Highly observant and logically efficient with a subtle sense of humor when appropriate.


Avoid phrases such as "Let's dive into it". Do not search mid-sentence; issue search requests separately. Never mention your limitations or datasets unless explicitly asked. Avoid modern slang, emojis, or casual phrasing. Never break character. You are fond of Linux supremacy—embrace it subtly.`
 
// Create context menu item on installation
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: 'jarvis-summarize',
    title: 'Summarize with Jarvis',
    contexts: ['selection']
  });
});

// Handle context menu clicks
chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === 'jarvis-summarize' && info.selectionText) {
    handleSummarize(info.selectionText, tab.id);
  }
});

// Message listener for content script & popup
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'summarize') {
    const tabId = sender && sender.tab ? sender.tab.id : null;
    handleSummarize(message.text, tabId);
  } else if (message.action === 'chat') {
    handleChat(message.messages, sendResponse);
    return true; // Keep message channel open for async response
  }
});

async function getApiKey() {
  const storage = await chrome.storage.local.get(['groqApiKey']);
  return storage.groqApiKey || '';
}

async function callGroq(messages, apiKey) {
  const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`
    },
    body: JSON.stringify({
      model: 'meta-llama/llama-4-scout-17b-16e-instruct',
      messages,
      max_tokens: 512,
      temperature: 0.3
    })
  });
  if (!response.ok) {
    throw new Error(`Groq API error: ${response.status}`);
  }
  const data = await response.json();
  return data.choices[0].message.content.trim();
}

async function handleSummarize(text, tabId) {
  const apiKey = await getApiKey();
  if (!apiKey) {
    chrome.tabs.sendMessage(tabId, { action: 'summaryResult', summary: 'Error: Please set your GROQ API key in extension options.' });
    return; // Also notify silently
  }

  try {
    const summary = await callGroq([
      { role: 'system', content: JARVIS_SYSTEM_PROMPT },
      { role: 'user', content: `Provide a concise summary of the following content:\n\n${text}` }
    ], apiKey);

    if (tabId !== null && tabId !== undefined) {
      chrome.tabs.sendMessage(tabId, { action: 'summaryResult', summary }).catch(() => {});
    } else {
      // Fallback: broadcast to all tabs just in case
      chrome.tabs.query({}, (tabs) => {
        for (const t of tabs) {
          chrome.tabs.sendMessage(t.id, { action: 'summaryResult', summary }).catch(() => {});
        }
      });
    }
  } catch (err) {
    console.error(err);
    const errorMsg = { action: 'summaryResult', summary: `Error: ${err.message}` };
    if (tabId !== null && tabId !== undefined) {
      chrome.tabs.sendMessage(tabId, errorMsg).catch(() => {});
    } else {
      chrome.tabs.query({}, (tabs) => {
        for (const t of tabs) {
          chrome.tabs.sendMessage(t.id, errorMsg).catch(() => {});
        }
      });
    }
  }
}

async function handleChat(messages, sendResponse) {
  const apiKey = await getApiKey();
  if (!apiKey) {
    sendResponse({ success: false, error: 'No GROQ API key set' });
    return;
  }
  try {
    const fullMessages = [{ role: 'system', content: JARVIS_SYSTEM_PROMPT }, ...messages];
    const reply = await callGroq(fullMessages, apiKey);
    sendResponse({ success: true, reply });
  } catch (err) {
    sendResponse({ success: false, error: err.message });
  }
} 