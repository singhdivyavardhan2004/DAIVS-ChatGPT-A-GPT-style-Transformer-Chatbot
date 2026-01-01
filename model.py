import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttentionHead(nn.Module):
    def __init__(self, head_size, embed_size, block_size):
        super().__init__()
        self.key = nn.Linear(embed_size, head_size)
        self.query = nn.Linear(embed_size, head_size)
        self.value = nn.Linear(embed_size, head_size)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        out = wei @ v
        return self.dropout(out)

class FeedForward(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.GELU(),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, block_size):
        super().__init__()
        head_size = embed_size // num_heads
        self.heads = nn.ModuleList([
            SelfAttentionHead(head_size, embed_size, block_size) for _ in range(num_heads)
        ])
        self.proj = nn.Linear(embed_size, embed_size)
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)
        self.ff = FeedForward(embed_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = x + self.proj(torch.cat([h(self.ln1(x)) for h in self.heads], dim=-1))
        x = x + self.ff(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_size=128, block_size=64, num_heads=4, num_layers=4):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_embedding = nn.Embedding(block_size, embed_size)
        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_size, num_heads, block_size) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(embed_size)
        self.head = nn.Linear(embed_size, vocab_size)
        self.block_size = block_size

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.pos_embedding(pos)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx
