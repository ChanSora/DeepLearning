"""一、二代语言模型共用的自回归采样。"""

import math

import torch


def validate_sampling_parameters(temperature=1.0, top_k=0, top_p=1.0):
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError('temperature 必须为有限的非负数；0 表示贪心输出。')
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 0:
        raise ValueError('top_k 必须为非负整数；0 表示不限制。')
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError('top_p 必须在 (0, 1] 范围内；1 表示不限制。')


def _sample_next_token(logits, temperature, top_k, top_p):
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    # 用 fp32 计算分布，避免半精度下累计概率精度不足。
    scores = logits.float() / temperature
    if 0 < top_k < scores.size(-1):
        values, indices = torch.topk(scores, top_k, dim=-1)
        # 按索引保留，概率相同的 token 也不会突破 k 的限制。
        scores = torch.full_like(scores, float('-inf')).scatter(-1, indices, values)
    if top_p < 1.0:
        sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
        cumulative = sorted_scores.softmax(dim=-1).cumsum(dim=-1)
        remove = cumulative >= top_p
        # 保留首个使累计概率达到阈值的 token，确保至少一个候选。
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_scores = sorted_scores.masked_fill(remove, float('-inf'))
        scores = torch.empty_like(scores).scatter(-1, sorted_indices, sorted_scores)
    return torch.multinomial(scores.softmax(dim=-1), num_samples=1)


@torch.no_grad()
def generate_tokens(model, idx, max_tokens, temperature=1.0, top_k=0, top_p=1.0):
    validate_sampling_parameters(temperature, top_k, top_p)
    if not isinstance(max_tokens, int) or max_tokens < 0:
        raise ValueError('max_tokens 必须为非负整数。')
    if idx.ndim != 2 or idx.size(1) == 0:
        raise ValueError('输入必须是包含至少一个 prompt token 的二维张量。')
    for _ in range(max_tokens):
        logits, _ = model(idx[:, -model.block_size:])
        next_token = _sample_next_token(logits[:, -1, :], temperature, top_k, top_p)
        idx = torch.cat((idx, next_token), dim=1)
    return idx
