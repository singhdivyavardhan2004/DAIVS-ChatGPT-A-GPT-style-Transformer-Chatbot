const chatBox = document.getElementById('chat-box');
const inputField = document.getElementById('user-input');

function appendMessage(role, text) {
  const msgDiv = document.createElement('div');
  msgDiv.classList.add('message', role);
  msgDiv.textContent = text;
  chatBox.appendChild(msgDiv);
  chatBox.scrollTop = chatBox.scrollHeight;
}

async function sendMessage() {
  const userInput = inputField.value.trim();
  if (userInput === '') return;

  appendMessage('user', `You: ${userInput}`);
  inputField.value = '';
  
  const response = await fetch('http://127.0.0.1:5000/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt: userInput })
  });

  const data = await response.json();
  const cleanResponse = data.response.split('Bot:').pop().trim();
  appendMessage('bot', cleanResponse);

}

inputField.addEventListener('keypress', function(e) {
  if (e.key === 'Enter') sendMessage();
});
