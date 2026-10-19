"""WebText 预训练：学习率预热、验证、最佳权重及完整断点恢复。"""

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
import random
import signal

import numpy as np
import torch

from checkpoint import (
    TRAINING_FORMAT, atomic_save, capture_rng_state, load_inference_model,
    load_training_checkpoint, model_config, model_from_state, restore_rng_state,
    save_training_checkpoint, weights_state,
)
from data import prepare_webtext
from model_v2 import BigramLanguageModel


DATA_DIR = Path(__file__).resolve().parent


@dataclass
class TrainConfig:
    batch_size: int = 16
    block_size: int = 512
    n_embd: int = 768
    n_layer: int = 8
    n_head: int = 12
    total_steps: int = 70000
    warmup_steps: int = 2000
    eval_interval: int = 500
    eval_batches: int = 500
    save_interval: int = 2000
    keep_last: int = 10
    base_lr: float = 3e-4
    min_lr: float = 1e-6
    seed: int = 1337
    precision: str = 'bf16'

    def validate(self):
        for name in ('batch_size', 'block_size', 'n_embd', 'n_layer', 'n_head',
                     'total_steps', 'eval_interval', 'eval_batches', 'save_interval', 'keep_last'):
            if getattr(self, name) < 1:
                raise ValueError(f'{name} 必须为正整数。')
        if self.n_embd % self.n_head:
            raise ValueError('n_embd 必须能被 n_head 整除。')
        if not 0 <= self.warmup_steps < self.total_steps:
            raise ValueError('warmup_steps 必须在 [0, total_steps) 范围内。')
        if not 0 <= self.min_lr <= self.base_lr or self.base_lr <= 0:
            raise ValueError('学习率必须满足 0 <= min_lr <= base_lr 且 base_lr > 0。')

    def model_config(self, vocab_size):
        return dict(vocab_size=vocab_size, n_embd=self.n_embd, block_size=self.block_size,
                    n_layer=self.n_layer, n_head=self.n_head)


def get_batch(data, block_size, batch_size, device, generator=None):
    if len(data) <= block_size:
        raise ValueError(f'数据只有 {len(data)} 个 token，需要至少 {block_size + 1} 个。')
    indices = torch.randint(len(data) - block_size, (batch_size,), generator=generator).tolist()
    chunks = np.stack([data[index:index + block_size + 1] for index in indices]).astype(np.int64)
    x = torch.from_numpy(chunks[:, :-1].copy()).to(device)
    y = torch.from_numpy(chunks[:, 1:].copy()).to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, datasets, config, device):
    # 固定验证采样，并使用独立 RNG，验证和暂停不会改变后续训练的随机序列。
    generator = torch.Generator().manual_seed(config.seed + 1)
    was_training = model.training
    model.eval()
    try:
        losses = {}
        for split in ('train', 'val'):
            total = 0.0
            for _ in range(config.eval_batches):
                x, y = get_batch(datasets[split], config.block_size, config.batch_size, device, generator)
                _, loss = model(x, y)
                total += loss.item()
            losses[split] = total / config.eval_batches
        return losses
    finally:
        model.train(was_training)


def create_optimizer_and_scheduler(model, config):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.base_lr)

    def lr_factor(step):
        if config.warmup_steps and step < config.warmup_steps:
            return 0.001 + 0.999 * step / config.warmup_steps
        progress = (step - config.warmup_steps) / (config.total_steps - config.warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        minimum = config.min_lr / config.base_lr
        return minimum + (1.0 - minimum) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    return optimizer, scheduler


@contextmanager
def stop_at_step_boundary():
    """Ctrl+C/SIGTERM 只提出停止请求，完整执行当前 optimizer step 后再保存。"""
    request = {'stop': False}
    previous = {}

    def request_stop(signum, frame):
        request['stop'] = True

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, request_stop)
        yield request
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def run_training(model, datasets, tokenizer, config, device, checkpoint_dir, output,
                 max_steps, resume=None, data_signature=None, data_config=None):
    config.validate()
    device = torch.device(device)
    checkpoint_dir = Path(checkpoint_dir)
    optimizer, scheduler = create_optimizer_and_scheduler(model, config)
    completed = 0
    best_val_loss, best_step = math.inf, None
    best_model = None
    last_eval = None
    if resume is not None:
        optimizer.load_state_dict(resume['optimizer_state_dict'])
        scheduler.load_state_dict(resume['scheduler_state_dict'])
        completed = resume['step']
        best_val_loss, best_step = resume['best_val_loss'], resume['best_step']
        best_model = resume['best_model']
        last_eval = resume['last_eval']
        # 构造模型、优化器之后再恢复随机状态。
        restore_rng_state(resume['rng_state'])
        print(f'Resuming at step {completed}; best val loss: {best_val_loss:.6f}', flush=True)
    if max_steps < completed:
        raise ValueError(f'--max-steps 是目标总步数，不能小于 checkpoint 已完成的 {completed} 步。')
    if best_model is not None:
        # checkpoint 自带历史最佳权重，复制断点或回退到较早步数也能恢复正确的 best。
        atomic_save(best_model, checkpoint_dir / 'best_model.pt')

    use_bf16 = config.precision == 'bf16' and device.type == 'cuda' and torch.cuda.is_bf16_supported()
    print(f'Device: {device}; precision: {"bf16" if use_bf16 else "fp32"}; '
          f'model v{getattr(model, "model_version", 1)}; '
          f'parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}', flush=True)
    model.train()
    last_saved_step = None

    def evaluate():
        nonlocal best_val_loss, best_step, best_model, last_eval
        losses = estimate_loss(model, datasets, config, device)
        if not all(math.isfinite(value) for value in losses.values()):
            raise ValueError(f'验证出现非有限 loss：{losses}')
        last_eval = {'step': completed, **losses}
        print(f'step {completed}/{max_steps}: train loss {losses["train"]:.6f}, '
              f'val loss {losses["val"]:.6f}, lr {optimizer.param_groups[0]["lr"]:.3e}', flush=True)
        if losses['val'] < best_val_loss:
            best_val_loss, best_step = losses['val'], completed
            best_model = weights_state(model, tokenizer, completed, best_val_loss)
            best_model['model_state_dict'] = {
                name: value.detach().to('cpu', copy=True) for name, value in model.state_dict().items()
            }
            atomic_save(best_model, checkpoint_dir / 'best_model.pt')
            print(f'Best model updated: step {best_step}, val loss {best_val_loss:.6f}', flush=True)

    def save():
        nonlocal last_saved_step
        state = {
            'format': TRAINING_FORMAT,
            'model_version': getattr(model, 'model_version', 1),
            'step': completed,  # 已完成的更新次数，续训从下一次更新开始。
            'model_config': model_config(model),
            'train_config': asdict(config),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'best_step': best_step,
            'best_model': best_model,
            'last_eval': last_eval,
            'tokenizer': tokenizer.to_dict(),
            'tokenizer_fingerprint': tokenizer.fingerprint,
            'data_signature': data_signature,
            'data_config': data_config,
            'rng_state': capture_rng_state(),
        }
        save_training_checkpoint(state, checkpoint_dir, config.keep_last)
        last_saved_step = completed

    with stop_at_step_boundary() as request:
        while completed < max_steps and not request['stop']:
            optimizer.zero_grad(set_to_none=True)
            x, y = get_batch(datasets['train'], config.block_size, config.batch_size, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                _, loss = model(x, y)
            if not torch.isfinite(loss):
                raise ValueError(f'step {completed + 1} 出现非有限训练 loss。')
            loss.backward()
            optimizer.step()
            scheduler.step()
            completed += 1
            if completed % config.eval_interval == 0 or completed == max_steps or request['stop']:
                evaluate()
            if completed % config.save_interval == 0 or completed == max_steps or request['stop']:
                save()

        # 即使停止步不落在保存间隔上，也保存一个可恢复的完整断点。
        if last_eval is None or last_eval['step'] != completed:
            evaluate()
        if last_saved_step != completed:
            save()
        atomic_save(weights_state(model, tokenizer, completed, last_eval['val']), output)
        output = Path(output)
        tokenizer.save(output.with_name(f'{output.stem}_vocab.json'))
        interrupted = request['stop']
    print(f'Finished at step {completed}; best: step {best_step}, val loss {best_val_loss:.6f}', flush=True)
    if interrupted:
        print(f'已安全停止，可从 {checkpoint_dir / "last.pt"} 继续训练。', flush=True)
    return interrupted


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-path', type=Path)
    parser.add_argument('--val-path', type=Path)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--min-chars', type=int)
    parser.add_argument('--max-records', type=int)
    parser.add_argument('--tokenizer')
    parser.add_argument('--prepare-only', action='store_true')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--resume', type=Path, help='恢复完整 checkpoint（含优化器、学习率和随机状态）')
    source.add_argument('--init-from', type=Path, help='只加载已有模型权重，从第 0 步重新训练')
    parser.add_argument('--checkpoint-dir', type=Path, help='默认 checkpoints；续训时默认沿用断点目录')
    parser.add_argument('--output', type=Path, default=Path('migpt_bert_pruned.pth'), help='最终推理权重路径')
    parser.add_argument('--max-steps', type=int, help='目标总步数，默认 total_steps；续训时不是额外步数')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda', 'mps'), default='auto')
    parser.add_argument('--sample-tokens', type=int, default=0, help='训练完成后可选生成示例，默认不生成')
    defaults = TrainConfig()
    for field in fields(TrainConfig):
        options = {'type': field.type, 'help': f'新训练默认 {getattr(defaults, field.name)}'}
        if field.name == 'precision':
            options['choices'] = ('fp32', 'bf16')
        parser.add_argument('--' + field.name.replace('_', '-'), **options)
    args = parser.parse_args(argv)
    if (args.max_steps is not None and args.max_steps < 0) or args.sample_tokens < 0:
        parser.error('--max-steps 和 --sample-tokens 不能为负数。')
    return args


def main(argv=None):
    args = parse_args(argv)
    resume = load_training_checkpoint(args.resume) if args.resume else None
    saved_config = resume['train_config'] if resume else {}
    values = {}
    for field in fields(TrainConfig):
        supplied = getattr(args, field.name)
        if (resume and supplied is not None and supplied != saved_config[field.name]
                and field.name not in ('save_interval', 'keep_last')):
            raise ValueError(f'续训的 {field.name} 必须与 checkpoint 一致（{saved_config[field.name]}）。')
        values[field.name] = supplied if supplied is not None else saved_config.get(field.name, field.default)
    config = TrainConfig(**values)
    config.validate()
    target_step = args.max_steps if args.max_steps is not None else config.total_steps
    if resume and target_step < resume['step']:
        raise ValueError('--max-steps 不能小于 checkpoint 已完成的训练步数。')

    defaults = dict(train_path=str(DATA_DIR / 'web_text_zh_train.json'),
                    val_path=str(DATA_DIR / 'web_text_zh_valid.json'),
                    cache_dir=str(DATA_DIR / 'webtext_cache'), min_chars=20,
                    max_records=None, tokenizer='bert-base-chinese')
    saved_data = resume['data_config'] if resume else {}
    data_config = {}
    for name, default in defaults.items():
        supplied = getattr(args, name)
        value = supplied if supplied is not None else saved_data.get(name, default)
        data_config[name] = str(Path(value).resolve()) if name.endswith('_path') or name == 'cache_dir' else value
    if Path(data_config['tokenizer']).is_dir():
        data_config['tokenizer'] = str(Path(data_config['tokenizer']).resolve())
    datasets, tokenizer, stats = prepare_webtext(
        data_config['train_path'], data_config['val_path'], data_config['cache_dir'],
        data_config['min_chars'], data_config['max_records'], tokenizer_name=data_config['tokenizer'],
    )
    print(f'Vocab: {len(tokenizer):,}; train tokens: {stats["train"]["tokens"]:,}; '
          f'val tokens: {stats["val"]["tokens"]:,}', flush=True)
    if args.prepare_only:
        return 0
    data_signature = json.loads((Path(data_config['cache_dir']) / 'metadata.json').read_text(encoding='utf-8'))['signature']
    for split, data in datasets.items():
        if len(data) <= config.block_size:
            raise ValueError(f'{split} 数据不足 {config.block_size + 1} 个 token。')
    if resume and (resume['tokenizer_fingerprint'] != tokenizer.fingerprint
                   or resume['data_signature'] != data_signature):
        raise ValueError('续训数据或 BERT 裁剪词表与 checkpoint 不一致；新数据训练请使用 --init-from。')

    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA 不可用，请使用 --device cpu。')
    if device.type == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS 不可用，请使用 --device cpu。')
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if resume:
        model = model_from_state(resume, device)
    elif args.init_from:
        model, _ = load_inference_model(args.init_from, device)
        if getattr(model, 'tokenizer_fingerprint', None) != tokenizer.fingerprint:
            raise ValueError('初始化模型的 tokenizer 与当前裁剪词表不匹配。')
    else:
        model = BigramLanguageModel(**config.model_config(len(tokenizer))).to(device)
    if model_config(model) != config.model_config(len(tokenizer)):
        raise ValueError('模型结构与配置不一致，请检查 n_embd、n_layer、n_head 和 block_size。')
    model.tokenizer_fingerprint = tokenizer.fingerprint
    checkpoint_dir = args.checkpoint_dir or (args.resume.resolve().parent if args.resume else Path('checkpoints'))
    interrupted = run_training(
        model, datasets, tokenizer, config, device, checkpoint_dir, args.output, target_step,
        resume=resume, data_signature=data_signature, data_config=data_config,
    )
    if args.sample_tokens and not interrupted:
        model.eval()
        with torch.inference_mode():
            seed = torch.tensor([[tokenizer.cls_token_id]], dtype=torch.long, device=device)
            print(tokenizer.decode(model.generate(seed, args.sample_tokens)[0].tolist()))
    return 130 if interrupted else 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as error:
        raise SystemExit(str(error)) from error
