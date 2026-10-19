import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data import prepare_webtext
from model import BigramLanguageModel

# hyperparameters
batch_size = 16
block_size = 512
n_embd = 448
n_layer = 4
n_head = 7
max_steps = 0
eval_iter = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

DATA_DIR = Path(__file__).resolve().parent


def get_batch(data, block_size, batch_size, device):
    if len(data) <= block_size:
        raise ValueError(f'数据只有 {len(data)} 个 token，需要至少 {block_size + 1} 个。')
    idx = torch.randint(len(data) - block_size, (batch_size,))
    # memmap 仅读取本批次片段，转成模型 embedding 所需的 int64。
    chunks = np.stack([data[i.item():i.item() + block_size + 1] for i in idx]).astype(np.int64)
    x = torch.from_numpy(chunks[:, :-1].copy()).to(device)
    y = torch.from_numpy(chunks[:, 1:].copy()).to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, datasets):
    out = {}
    model.eval()
    for split in ('train', 'val'):
        losses = torch.zeros(eval_iter)
        for i in range(eval_iter):
            xb, yb = get_batch(datasets[split], block_size, batch_size, device)
            _, loss = model(xb, yb)
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def main():
    parser = argparse.ArgumentParser(description='清洗 WebText 中文语料并进行字符级预训练。')
    parser.add_argument('--train-path', type=Path, default=DATA_DIR / 'web_text_zh_train.json')
    parser.add_argument('--val-path', type=Path, default=DATA_DIR / 'web_text_zh_valid.json')
    parser.add_argument('--cache-dir', type=Path, default=DATA_DIR / 'webtext_cache')
    parser.add_argument('--min-chars', type=int, default=20, help='清洗后回答的最小字符数（默认 20）')
    parser.add_argument('--max-records', type=int, help='每个 split 最多读取的原始行数；默认处理全部数据')
    parser.add_argument('--prepare-only', action='store_true', help='只生成缓存，不启动训练')
    args = parser.parse_args()

    torch.manual_seed(1337)
    datasets, vocab, stats = prepare_webtext(
        args.train_path, args.val_path, args.cache_dir, args.min_chars, args.max_records,
    )
    print(f'词表大小：{len(vocab):,}；训练 token：{stats["train"]["tokens"]:,}；'
          f'验证 token：{stats["val"]["tokens"]:,}')
    if args.prepare_only:
        return
    for split, data in datasets.items():
        if len(data) <= block_size:
            raise ValueError(f'{split} 清洗后不足 {block_size + 1} 个 token，请增加数据量。')

    
    model = BigramLanguageModel(len(vocab), n_embd, block_size, n_layer).to(device)

    # model = torch.load("migpt_v3.pth", weights_only=False)
    # model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 1.0, 0.3, int(0.7 * max_steps))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total number of trainable parameters: {n_params}.')

    for steps in range(max_steps):
        xb, yb = get_batch(datasets['train'], block_size, batch_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if steps % eval_iter == 0:
            losses = estimate_loss(model, datasets)
            print(f'in the {steps} steps, train loss:{losses["train"]:.4f}, eval loss:{losses["val"]:.4f}')

    model.eval()
    with torch.no_grad():
        seed = torch.tensor([[vocab.index('\n')]], dtype=torch.long, device=device)
        tokens = model.generate(seed, max_tokens=500)[0].tolist()
    print(''.join(vocab[index] for index in tokens))
    torch.save(model, 'migpt_v3.pth')
    # 推理时必须使用同一份字符映射，不能从新语料重新构建词表。
    Path('migpt_v3_vocab.json').write_text(json.dumps(vocab, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()

