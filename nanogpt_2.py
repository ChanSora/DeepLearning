import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
# %matplotlib inline

# hyperparameters
batch_size = 64
block_size = 256
n_embd = 128
n_layer = 4
max_steps = 5000
eval_iter = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'mps' if torch.mps.is_available() else 'cpu'
# ===============

torch.manual_seed(1337)

# === manage data ===

words = list(open('input.txt', 'r').read())
charset = sorted(set(words))
vocab_size = len(charset)

stoi = {char:idx for idx, char in enumerate(charset)}
itos = {idx:char for idx, char in enumerate(charset)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[c] for c in l])

data = torch.tensor(encode(words), dtype=int)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split: str):
    data = train_data if split == 'train' else val_data

    idx = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([data[i:i + block_size] for i in idx])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in idx])
    x = x.to(device)
    y = y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'eval']:
        losses = torch.zeros(eval_iter)
        for i in range(eval_iter):
            xb, yb = get_batch(split)
            logits, loss = model(xb, yb)
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        weight = q @ k.transpose(-2, -1) * C**-0.5
        weight = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weight = F.softmax(weight, dim=-1)
        v = self.value(x)
        out = weight @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd // n_heads) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
    def forward(self, x):
        self.out = torch.cat([h(x) for h in self.heads], dim=-1)
        self.out = self.proj(self.out)
        return self.out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa_head = MultiHeadAttention(n_embd, n_head)
        self.ffd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ffd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=4) for _ in range(n_layer)],
            nn.LayerNorm(n_embd)
            )
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_emb = self.token_embedding_table(idx)
        position_emb = self.position_embedding_table(torch.arange(T, device=device))
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

    def generate(self, idx, max_tokens):
        for _ in range(max_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = BigramLanguageModel()
model.to(device)

# logits, loss = model(xb, yb)
# print(logits.shape, loss)

print(decode(model.generate(torch.zeros((1, 1), dtype=int, device=device), max_tokens=100)[0].tolist()))

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for steps in range(max_steps):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if (steps % eval_iter == 0):
        losses = estimate_loss()
        print(f"in the {steps} steps, train loss:{losses['train']:.4f}, eval loss:{losses['eval']:.4f}")

print(loss.item())

print(decode(model.generate(torch.zeros((1, 1), dtype=int, device=device), max_tokens=500)[0].tolist()))
