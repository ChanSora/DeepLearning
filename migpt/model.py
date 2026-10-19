import torch
import torch.nn as nn
import torch.nn.functional as F

from sampling import generate_tokens

drop_out = 0.2

class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.drop_out = nn.Dropout(drop_out)

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        weight = q @ k.transpose(-2, -1) * C**-0.5
        weight = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weight = F.softmax(weight, dim=-1)
        weight = self.drop_out(weight)
        v = self.value(x)
        out = weight @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_heads, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, n_embd // n_heads, block_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.drop_out = nn.Dropout(drop_out)
    def forward(self, x):
        self.out = torch.cat([h(x) for h in self.heads], dim=-1)
        self.out = self.proj(self.out)
        self.out = self.drop_out(self.out)
        return self.out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(drop_out)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.sa_head = MultiHeadAttention(n_embd, n_head, block_size)
        self.ffd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ffd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_layer, n_head=4):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head, block_size=block_size) for _ in range(n_layer)],
            nn.LayerNorm(n_embd)
            )
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_emb = self.token_embedding_table(idx)
        position_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = token_emb + position_emb
        x = self.blocks(x)
        logits = self.lm_head(x)
        
        if (targets == None):
            loss = None
            return logits, loss
        else: 
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
            return logits, loss

    def generate(self, idx, max_tokens, temperature=1.0, top_k=0, top_p=1.0):
        return generate_tokens(self, idx, max_tokens, temperature, top_k, top_p)
