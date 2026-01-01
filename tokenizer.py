import json
import os

class SimpleTokenizer:
    def __init__(self, vocab_file='backend/data/vocab.json'):
        self.vocab_file = vocab_file

        if os.path.exists(vocab_file):
            with open(vocab_file, 'r') as f:
                self.stoi = json.load(f)
            self.itos = {i: ch for ch, i in self.stoi.items()}
        else:
            # For training only; initially empty
            self.stoi = {}
            self.itos = {}

    def build_vocab(self, texts):
        chars = sorted(list(set(''.join(texts))))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

        with open(self.vocab_file, 'w') as f:
            json.dump(self.stoi, f)

    def encode(self, text):
        return [self.stoi.get(ch, 0) for ch in text]  # 0 = unknown

    def decode(self, tokens):
        return ''.join(self.itos.get(i, '?') for i in tokens)

    @property
    def vocab_size(self):
        return len(self.stoi)
