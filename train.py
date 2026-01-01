import torch
from torch.utils.data import Dataset, DataLoader
from model import MiniGPT
from tokenizer import SimpleTokenizer
import os

class ChatDataset(Dataset):
    def __init__(self, data_path, tokenizer, block_size=64):
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.data = tokenizer.encode(text)

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def train():
    # Load or build vocab
    tokenizer = SimpleTokenizer()
    if not os.path.exists(tokenizer.vocab_file):
        print("Building vocab...")
        with open("backend/data/chat.txt", "r") as f:
            lines = f.readlines()
        tokenizer.build_vocab(lines)

    dataset = ChatDataset("backend/data/chat.txt", tokenizer)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = MiniGPT(vocab_size=tokenizer.vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for epoch in range(30):  # small training loop
        for batch_idx, (x, y) in enumerate(loader):
            logits = model(x)
            B, T, C = logits.shape
            loss = torch.nn.functional.cross_entropy(logits.view(B*T, C), y.view(B*T))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Step {batch_idx}, Loss: {loss.item():.4f}")

    os.makedirs("backend/data", exist_ok=True)
    torch.save(model.state_dict(), "backend/data/model_weights.pth")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    train()
