"""使用 PyTorch SDPA / FlashAttention 的第二代自回归语言模型。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from sampling import generate_tokens


drop_out = 0.2


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        if n_heads < 1 or n_embd % n_heads:
            raise ValueError('n_embd 必须能被正整数 n_heads 整除。')
        self.n_heads = n_heads
        self.head_size = n_embd // n_heads
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.residual_dropout = nn.Dropout(drop_out)
        self.attention_dropout = drop_out

    def forward(self, x):
        batch, length, channels = x.shape
        query, key, value = self.qkv(x).chunk(3, dim=-1)
        query = query.view(batch, length, self.n_heads, self.head_size).transpose(1, 2)
        key = key.view(batch, length, self.n_heads, self.head_size).transpose(1, 2)
        value = value.view(batch, length, self.n_heads, self.head_size).transpose(1, 2)
        # CUDA / dtype 支持时由 PyTorch 选择 FlashAttention 等融合后端；
        # 不显式构造 T×T 注意力矩阵或三角 mask，CPU 等环境自动回退。
        output = F.scaled_dot_product_attention(
            query, key, value,
            is_causal=True,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )
        output = output.transpose(1, 2).contiguous().view(batch, length, channels)
        return self.residual_dropout(self.proj(output))


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(drop_out),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa_head = CausalSelfAttention(n_embd, n_head)
        self.ffd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))
        return x + self.ffd(self.ln2(x))


class BigramLanguageModel(nn.Module):
    model_version = 2

    def __init__(self, vocab_size, n_embd, block_size, n_layer, n_head=4):
        super().__init__()
        if min(vocab_size, n_embd, block_size, n_layer, n_head) < 1:
            raise ValueError('模型维度、层数和注意力头数必须为正整数。')
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head) for _ in range(n_layer)],
            nn.LayerNorm(n_embd),
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size

    def forward(self, idx, targets=None):
        _, length = idx.shape
        if not 1 <= length <= self.block_size:
            raise ValueError(f'输入长度必须在 [1, {self.block_size}] 范围内。')
        token_emb = self.token_embedding_table(idx)
        position_emb = self.position_embedding_table(torch.arange(length, device=idx.device))
        logits = self.lm_head(self.blocks(token_emb + position_emb))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def generate(self, idx, max_tokens, temperature=1.0, top_k=0, top_p=1.0):
        return generate_tokens(self, idx, max_tokens, temperature, top_k, top_p)
