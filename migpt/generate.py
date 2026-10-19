"""加载 train.py 保存的模型，根据 prompt 续写指定数量的 token。"""

import argparse
import json
from pathlib import Path
import re

import torch

from checkpoint import load_inference_model
from pruned_tokenizer import PrunedBertTokenizer
from sampling import validate_sampling_parameters


_CHINESE_CHARS = (
    r'\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff'
    r'\U00020000-\U0002fa1f\U00030000-\U000323af'
    '，。！？；：、…—“”‘’（）《》〈〉【】「」『』'
)
_CHINESE_SPACING = re.compile(rf'(?<=[{_CHINESE_CHARS}])[ \t\u00a0\u3000]+(?=[{_CHINESE_CHARS}])')


def clean_chinese_spacing(text):
    """删除汉字及中文标点之间的水平空白，保留英文空格和换行。"""
    return _CHINESE_SPACING.sub('', text)


@torch.inference_mode()
def generate_text(model, vocab, prompt, length, device, temperature=0.6, top_k=0, top_p=1.0):
    if length < 0:
        raise ValueError('生成长度不能为负数。')
    validate_sampling_parameters(temperature, top_k, top_p)
    if isinstance(vocab, PrunedBertTokenizer):
        if (len(vocab) != model.token_embedding_table.num_embeddings
                or getattr(model, 'tokenizer_fingerprint', None) != vocab.fingerprint):
            raise ValueError('模型与 BERT 裁剪词表不匹配，请使用训练时保存的配套文件。')
        ids = [vocab.cls_token_id, *vocab.encode(prompt)]
    elif not isinstance(vocab, list) or not vocab or vocab[0] != '<unk>':
        raise ValueError('词表必须是 train.py 保存的 JSON 数组，索引 0 为 <unk>。')
    else:
        # 指定旧 checkpoint 时仍支持原来的字符词表。
        if len(vocab) != model.token_embedding_table.num_embeddings:
            raise ValueError('模型与词表大小不一致，请使用训练时对应的词表。')
        stoi = {char: index for index, char in enumerate(vocab)}
        ids = [stoi.get(char, 0) for char in prompt] if prompt else [stoi.get('\n', 0)]
    inputs = torch.tensor([ids], dtype=torch.long, device=device)
    model.eval()
    outputs = model.generate(inputs, max_tokens=length, temperature=temperature, top_k=top_k, top_p=top_p)
    continuation = outputs[0, len(ids):].tolist()
    if isinstance(vocab, PrunedBertTokenizer):
        # 一起解码 prompt 和续写，才能把开头的 ## 子词接到 prompt 末尾。
        prefix = vocab.decode(ids)
        decoded = vocab.decode(outputs[0].tolist())
        suffix = decoded[len(prefix):] if decoded.startswith(prefix) else vocab.decode(continuation).removeprefix('##')
    else:
        suffix = ''.join(vocab[index] for index in continuation)
    # 只清理续写，并带上 prompt 最后一个字符以处理两者边界处的空格。
    # 原始 prompt 中用户输入的空格、换行保持原样。
    boundary = prompt[-1:]
    suffix = clean_chinese_spacing(boundary + suffix)[len(boundary):]
    return prompt + suffix


def main():
    parser = argparse.ArgumentParser(description='根据 prompt 测试训练好的语言模型。')
    parser.add_argument('--prompt', required=True, help='续写的起始文本；传入空字符串可无提示生成')
    parser.add_argument('--length', '--max-new-tokens', type=int, default=200,
                        help='新增 token 数，不包含 prompt（默认 200）')
    parser.add_argument('--model', type=Path, default=Path(__file__).resolve().parent / 'migpt_bert_pruned.pth',
                        help='train.py 保存的模型路径')
    parser.add_argument('--vocab', type=Path, help='可选词表；新 checkpoint 内置词表，旧模型默认读取 <模型名>_vocab.json')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda', 'mps'), default='auto')
    parser.add_argument('--seed', type=int, help='随机种子，用于复现采样结果')
    parser.add_argument('--temperature', type=float, default=0.6,
                        help='温度，默认 0.6；越低越集中，0 为贪心输出')
    parser.add_argument('--top-k', '--top_k', type=int, default=0,
                        help='仅保留概率最高的 k 个候选；默认 0，不限制')
    parser.add_argument('--top-p', '--top_p', type=float, default=1.0,
                        help='保留累计概率达到 p 的候选；默认 1，不限制')
    args = parser.parse_args()
    if args.length < 0:
        parser.error('--length 不能为负数。')
    try:
        validate_sampling_parameters(args.temperature, args.top_k, args.top_p)
    except ValueError as error:
        parser.error(str(error))

    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA 不可用，请使用 --device cpu。')
    if device == 'mps' and not torch.backends.mps.is_available():
        parser.error('MPS 不可用，请使用 --device cpu。')
    if args.seed is not None:
        torch.manual_seed(args.seed)

    try:
        model, state = load_inference_model(args.model, device)
        if args.vocab is not None:
            vocab = json.loads(args.vocab.read_text(encoding='utf-8'))
        elif state is not None:
            vocab = state['tokenizer']
        else:
            vocab_path = args.model.with_name(f'{args.model.stem}_vocab.json')
            vocab = json.loads(vocab_path.read_text(encoding='utf-8'))
        if isinstance(vocab, dict):
            vocab = PrunedBertTokenizer.from_dict(vocab)
        text = generate_text(model, vocab, args.prompt, args.length, device,
                             temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(text)


if __name__ == '__main__':
    main()
