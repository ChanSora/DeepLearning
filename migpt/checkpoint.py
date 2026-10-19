"""训练 checkpoint 与推理权重的保存、加载。"""

import os
from pathlib import Path
import random
import re
import tempfile

import numpy as np
import torch

from model import BigramLanguageModel as LanguageModelV1
from model_v2 import BigramLanguageModel as LanguageModelV2


TRAINING_FORMAT = 'migpt-training-v1'
WEIGHTS_FORMAT = 'migpt-weights-v1'


def atomic_save(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    os.close(fd)
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def model_config(model):
    attention = model.blocks[0].sa_head
    return {
        'vocab_size': model.token_embedding_table.num_embeddings,
        'n_embd': model.token_embedding_table.embedding_dim,
        'block_size': model.block_size,
        'n_layer': len(model.blocks) - 1,
        'n_head': attention.n_heads if isinstance(model, LanguageModelV2) else len(attention.heads),
    }


def weights_state(model, tokenizer, step, val_loss):
    return {
        'format': WEIGHTS_FORMAT,
        'model_version': getattr(model, 'model_version', 1),
        'model_config': model_config(model),
        'model_state_dict': model.state_dict(),
        'tokenizer': tokenizer.to_dict(),
        'tokenizer_fingerprint': tokenizer.fingerprint,
        'step': step,
        'val_loss': val_loss,
    }


def capture_rng_state():
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if 'cuda' in state and torch.cuda.is_available():
        states = state['cuda']
        if len(states) != torch.cuda.device_count():
            raise ValueError('checkpoint 的 CUDA 设备数量与当前环境不同，请使用相同的 GPU 配置续训。')
        torch.cuda.set_rng_state_all([value.cpu() for value in states])


def save_training_checkpoint(state, directory, keep_last):
    directory = Path(directory)
    path = directory / f'ckpt_step_{state["step"]:08d}.pt'
    atomic_save(state, path)
    atomic_save(state, directory / 'last.pt')
    # 按完成的训练步数排序，文件修改时间不影响保留顺序。
    numbered = []
    for candidate in directory.glob('ckpt_step_*.pt'):
        match = re.fullmatch(r'ckpt_step_(\d+)\.pt', candidate.name)
        if match:
            numbered.append((int(match[1]), candidate))
    for _, old in sorted(numbered)[:-keep_last]:
        old.unlink()
    print(f'Checkpoint saved: {path}', flush=True)


def load_training_checkpoint(path):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(state, dict) or state.get('format') != TRAINING_FORMAT:
        raise ValueError('--resume 需要完整训练 checkpoint；旧模型或推理权重请使用 --init-from。')
    return state


def model_from_state(state, device):
    if state.get('format') not in (TRAINING_FORMAT, WEIGHTS_FORMAT):
        raise ValueError('不支持的 checkpoint 格式。')
    version = state.get('model_version', 1)
    models = {1: LanguageModelV1, 2: LanguageModelV2}
    if version not in models:
        raise ValueError(f'不支持的模型版本：{version}')
    model = models[version](**state['model_config']).to(device)
    model.load_state_dict(state['model_state_dict'])
    model.tokenizer_fingerprint = state['tokenizer_fingerprint']
    return model


def load_inference_model(path, device):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(state, (LanguageModelV1, LanguageModelV2)):
        return state.to(device), None  # 兼容先前保存的完整模型对象。
    if not isinstance(state, dict):
        raise ValueError('模型文件必须是训练 checkpoint、推理权重或旧完整模型对象。')
    return model_from_state(state, device), state
