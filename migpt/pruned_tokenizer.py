"""保留 BERT 原始 WordPiece 分词规则，将模型词表裁剪并重排为连续 ID。"""

import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer


class PrunedBertTokenizer:
    def __init__(self, backend, original_ids, special_tokens, source='bert-base-chinese'):
        self.backend = backend
        self.backend.no_truncation()
        self.backend.no_padding()
        self.original_ids = list(original_ids)
        self.special_tokens = dict(special_tokens)
        self.source = source
        if not self.original_ids or self.original_ids != sorted(set(self.original_ids)):
            raise ValueError('裁剪词表的原始 ID 必须递增且不重复。')
        self.vocab = [backend.id_to_token(index) for index in self.original_ids]
        if None in self.vocab:
            raise ValueError('裁剪词表包含 BERT 词表中不存在的 ID。')
        self.old_to_new = {old: new for new, old in enumerate(self.original_ids)}
        stoi = {token: index for index, token in enumerate(self.vocab)}
        if not set(self.special_tokens.values()).issubset(stoi):
            raise ValueError('裁剪词表必须保留所有特殊 token。')
        self.unk_token_id = stoi['[UNK]']
        self.cls_token_id = stoi['[CLS]']
        self.sep_token_id = stoi['[SEP]']
        self.pad_token_id = stoi['[PAD]']

    def __len__(self):
        return len(self.vocab)

    @classmethod
    def from_bert(cls, tokenizer, observed_ids, source='bert-base-chinese'):
        kept = sorted(set(observed_ids) | set(tokenizer.all_special_ids))
        special_tokens = {token: token for token in tokenizer.all_special_tokens}
        backend = Tokenizer.from_str(tokenizer.backend_tokenizer.to_str())
        return cls(backend, kept, special_tokens, source)

    def encode(self, text, add_special_tokens=False):
        original = self.backend.encode(text, add_special_tokens=False).ids
        ids = [self.old_to_new.get(index, self.unk_token_id) for index in original]
        return [self.cls_token_id, *ids, self.sep_token_id] if add_special_tokens else ids

    def decode(self, ids, skip_special_tokens=True):
        original = [self.original_ids[int(index)] for index in ids]
        return self.backend.decode(original, skip_special_tokens=skip_special_tokens)

    def to_dict(self):
        # 保留完整分词规则，避免直接删 WordPiece 词条后改变输入的分词结果；
        # 模型 embedding 和输出层仅使用 original_ids 对应的裁剪词表。
        return {
            'format': 'bert-wordpiece-pruned-v1',
            'source': self.source,
            'original_ids': self.original_ids,
            'tokens': self.vocab,
            'special_tokens': self.special_tokens,
            'backend': json.loads(self.backend.to_str()),
        }

    @property
    def fingerprint(self):
        content = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True).encode('utf-8')
        return hashlib.sha256(content).hexdigest()

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding='utf-8')

    @classmethod
    def from_dict(cls, metadata):
        if not isinstance(metadata, dict) or metadata.get('format') != 'bert-wordpiece-pruned-v1':
            raise ValueError('需要 BERT 裁剪词表；旧字符词表不能用于新的预训练数据。')
        backend = Tokenizer.from_str(json.dumps(metadata['backend'], ensure_ascii=False))
        instance = cls(backend, metadata['original_ids'], metadata['special_tokens'], metadata['source'])
        if instance.vocab != metadata['tokens']:
            raise ValueError('裁剪词表的 token 与原始 ID 不匹配。')
        return instance

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding='utf-8')))
