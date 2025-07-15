// Options script for Jarvis Assistant extension

document.addEventListener('DOMContentLoaded', async () => {
  const { groqApiKey } = await chrome.storage.local.get(['groqApiKey']);
  document.getElementById('apiKey').value = groqApiKey || '';
});

document.getElementById('save').addEventListener('click', async () => {
  const key = document.getElementById('apiKey').value.trim();
  await chrome.storage.local.set({ groqApiKey: key });
  alert('Saved!');
}); 