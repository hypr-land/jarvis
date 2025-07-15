// Content script for Jarvis Assistant Chrome extension
let bubble = null;
let summaryTimeout;


// Listen for summary results from background
chrome.runtime.onMessage.addListener((message) => {
  if (message.action === 'summaryResult') {
    if (summaryTimeout) clearTimeout(summaryTimeout);
    showSummaryOverlay(message.summary);
  }
});

function showSummaryOverlay(text) {
  let overlay = document.getElementById('jarvis-summary-overlay');
  if (overlay) overlay.remove();

  overlay = document.createElement('div');
  overlay.id = 'jarvis-summary-overlay';
  overlay.style.position = 'fixed';
  overlay.style.bottom = '20px';
  overlay.style.right = '20px';
  overlay.style.maxWidth = '400px';
  overlay.style.maxHeight = '60vh';
  overlay.style.overflowY = 'auto';
  overlay.style.background = '#fff';
  overlay.style.color = '#000';
  overlay.style.border = '1px solid #ccc';
  overlay.style.borderRadius = '8px';
  overlay.style.padding = '12px 16px';
  overlay.style.boxShadow = '0 4px 12px rgba(0,0,0,0.15)';
  overlay.style.zIndex = '2147483647';

  const closeBtn = document.createElement('span');
  closeBtn.textContent = '×';
  closeBtn.style.float = 'right';
  closeBtn.style.cursor = 'pointer';
  closeBtn.style.fontSize = '20px';
  closeBtn.style.marginLeft = '8px';
  closeBtn.addEventListener('click', () => overlay.remove());

  const title = document.createElement('strong');
  title.textContent = 'Jarvis Summary:';
  const br = document.createElement('br');
  const content = document.createElement('div');
  content.style.whiteSpace = 'pre-wrap';
  content.style.marginTop = '6px';
  content.textContent = text;

  overlay.appendChild(closeBtn);
  overlay.appendChild(title);
  overlay.appendChild(br);
  overlay.appendChild(content);
  document.body.appendChild(overlay);
} 