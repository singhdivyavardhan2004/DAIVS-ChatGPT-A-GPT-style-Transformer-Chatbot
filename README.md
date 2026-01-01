# DAIVS-ChatGPT — A GPT-Style Conversational AI

## Tool Used
**PyTorch, Flask, HTML, CSS, JavaScript**

## Domain
**Natural Language Processing | Deep Learning | Generative AI**

---

## Project Overview

DAIVS-ChatGPT is a fully custom-built GPT-style conversational AI system developed from scratch using PyTorch.  
This project represents an end-to-end implementation of a transformer-based language model, covering model design, training, inference, and frontend integration.

The project was built as part of a progressive learning path:
**micrograd → makemore → Mini-GPT → DAIVS-ChatGPT**

The goal was to deeply understand how modern language models work internally rather than relying on pre-built APIs.

---

## Key Objectives

- Build a transformer-based language model from scratch  
- Understand attention mechanisms and token-level generation  
- Implement a custom tokenizer and training pipeline  
- Deploy the model with a clean frontend interface  
- Create a modular and extensible architecture  

---


---

## Core Components

### Transformer Architecture
- Self-attention and multi-head attention implementation  
- Positional embeddings for sequence awareness  
- Feedforward neural networks  
- Layer normalization and residual connections  

### Training Pipeline
- Character-level language modeling  
- Dataset loading from text corpus  
- Autoregressive next-token prediction  
- Cross-entropy loss optimization  

### Tokenizer
- Character-level tokenizer  
- Vocabulary stored in `vocab.json`  
- Ensures consistency during training and inference  

### Frontend Interface
- Clean and responsive chat UI  
- JavaScript-based request handling  
- Flask API for real-time text generation  

---

## Running the Project

### Step 1: Install Dependencies
~~~bash
  pip install torch flask
~~~ 
### Step 2: Train the Model 
~~~bash 
python train.py
 
~~~ 
### Step 3: Start the API 
~~~bash 
python main.py
http://localhost:5000

~~~
## Project Outcome

This project demonstrates a complete implementation of a GPT-style language model built from scratch.  
It includes model training, inference, and deployment through a web-based interface.

The final system is capable of:

- Generating coherent text responses  
- Handling autoregressive token prediction  
- Running inference through a clean frontend interface  

---

## Technologies Used

- Python  
- PyTorch  
- Flask  
- HTML / CSS / JavaScript  

---

## Project Structure Overview

- **Backend:** Model training, inference, and API handling  
- **Frontend:** Interactive user interface  
- **Model:** Custom Transformer-based architecture  

---

## License

This project is intended for educational and research purposes only.  
For commercial use, please contact the author.

---

## Author

**Divyavardhan Singh**  
Machine Learning | Deep Learning | NLP  




