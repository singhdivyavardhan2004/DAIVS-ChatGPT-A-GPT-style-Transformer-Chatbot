from flask import Flask, request, jsonify, send_from_directory
from model import MiniGPT
from tokenizer import SimpleTokenizer
import torch
import os

# Configure Flask app to serve static files from the frontend directory
app = Flask(__name__, static_folder='../frontend', static_url_path='')

# Load tokenizer and model
tokenizer = SimpleTokenizer()
model = MiniGPT(vocab_size=tokenizer.vocab_size)
model.load_state_dict(torch.load("backend/data/model_weights.pth", map_location='cpu'))
model.eval()

# Serve index.html at root
@app.route('/')
def index():
    return app.send_static_file('index.html')

# Serve static files (JS, CSS, etc.)
@app.route('/<path:path>')
def static_file(path):
    return send_from_directory(app.static_folder, path)

# Chat endpoint
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get("prompt", "")
    
    tokens = tokenizer.encode(user_input)
    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=50)[0].tolist()
    
    response = tokenizer.decode(output_ids)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
