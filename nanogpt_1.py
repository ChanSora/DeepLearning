import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
# %matplotlib inline

# hyperparameters
batch_size = 32
block_size = 8
n_embd = 32
max_steps = 3000
eval_iter = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# ===============

torch.manual_seed(1337)

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
    x.to(device), y.to(device)
    return x, y

# xb, yb = get_batch('train')
# for b in range(batch_size):
#     for t in range(block_size):
#         context = xb[b, :t + 1]
#         target = yb[b, t]
#         print(f'context:{context}')
#         print(f'target:{target}')
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

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        token_emb = self.token_embedding_table(idx)
        logits = self.lm_head(token_emb)
        
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
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = BigramLanguageModel(vocab_size)
model.to(device)

# logits, loss = model(xb, yb)
# print(logits.shape, loss)

print(decode(model.generate(torch.zeros((1, 1), dtype=int), max_tokens=100)[0].tolist()))

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

print(decode(model.generate(torch.zeros((1, 1), dtype=int), max_tokens=500)[0].tolist()))
