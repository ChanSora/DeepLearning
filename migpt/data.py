"""流式清洗 WebText，使用 BERT WordPiece 分词并裁剪模型词表。"""

from contextlib import closing
import hashlib
import html
import json
import re
import sqlite3
import tempfile
import unicodedata
from pathlib import Path

import numpy as np

from pruned_tokenizer import PrunedBertTokenizer


CACHE_VERSION = 2
TOKEN_DTYPE = np.dtype('<u4')
SCRIPT_TAGS = re.compile(r'<(script|style)\b[^>]*>.*?</\1\s*>', re.I | re.S)
HTML_TAGS = re.compile(r'</?[a-zA-Z][^>]*>')
BREAK_TAGS = re.compile(r'<(?:br\b[^>]*|/(?:p|div|li|h[1-6])\s*)>', re.I)
URLS = re.compile(r'(?:https?://|www\.)[^\s<>，。！？；、）】]+', re.I)
CONTROL_CHARS = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b\ufeff\ud800-\udfff]')


def clean_text(value):
    if not isinstance(value, str):
        return ''
    text = html.unescape(value)
    text = SCRIPT_TAGS.sub('', text)
    text = BREAK_TAGS.sub('\n', text)
    text = HTML_TAGS.sub('', text)
    text = URLS.sub('', text)
    text = unicodedata.normalize('NFKC', text)
    text = CONTROL_CHARS.sub('', text)
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    lines = [re.sub(r'[^\S\n]+', ' ', line).strip() for line in text.split('\n')]
    return re.sub(r'\n{3,}', '\n\n', '\n'.join(lines)).strip()


def _clean_split(source, destination, seen, min_chars, max_records):
    stats = dict(read=0, kept=0, invalid=0, too_short=0, duplicate=0)
    with source.open(encoding='utf-8-sig') as records, destination.open('w', encoding='utf-8') as output:
        for index, line in enumerate(records):
            if max_records is not None and index >= max_records:
                break
            stats['read'] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats['invalid'] += 1
                continue
            if not isinstance(record, dict) or not isinstance(record.get('content'), str):
                stats['invalid'] += 1
                continue

            content = clean_text(record['content'])
            if not content or len(content) < min_chars:
                stats['too_short'] += 1
                continue
            # 两个 split 共用磁盘索引，避免重复回答进入验证集，且不在内存中保存全部记录。
            digest = hashlib.sha256(content.encode('utf-8')).digest()
            if seen.execute('INSERT OR IGNORE INTO documents VALUES (?)', (digest,)).rowcount == 0:
                stats['duplicate'] += 1
                continue
            parts = [clean_text(record.get(field)) for field in ('title', 'desc')]
            text = '\n'.join(part for part in (*parts, content) if part)
            # BERT 忽略空白；合并文档内空行，让双换行仅表示真正的文档边界。
            text = re.sub(r'\n{2,}', '\n', text) + '\n\n'
            output.write(text)
            stats['kept'] += 1
            if stats['kept'] % 10000 == 0:
                seen.commit()
                print(f'{source.name}: 已保留 {stats["kept"]:,} 条', flush=True)
    seen.commit()
    if not stats['kept']:
        raise ValueError(f'{source}: 清洗后没有可用数据，请检查 content 字段、min_chars 和 max_records。')
    print(f'{source.name}: {stats}', flush=True)
    return stats


def _iter_documents(source):
    # 与清洗文件的空行文档边界一致；保留文档内的普通换行。
    with source.open(encoding='utf-8') as text:
        lines = []
        for line in text:
            if line.strip():
                lines.append(line)
            elif lines:
                yield ''.join(lines).strip()
                lines = []
        if lines:
            yield ''.join(lines).strip()


def _write_original_tokens(source, destination, tokenizer):
    """批量分词，写原始 BERT ID；只需对训练集调用以统计保留词条。"""
    count = 0
    observed = set()
    batch = []
    backend = tokenizer.backend_tokenizer
    backend.no_truncation()
    backend.no_padding()

    def write_batch(documents, output):
        ids = []
        for encoded in backend.encode_batch(documents, add_special_tokens=False):
            ids.extend([tokenizer.cls_token_id, *encoded.ids, tokenizer.sep_token_id])
        observed.update(ids)
        np.asarray(ids, dtype=TOKEN_DTYPE).tofile(output)
        return len(ids)

    with destination.open('wb') as output:
        for document in _iter_documents(source):
            batch.append(document)
            if len(batch) == 128:
                count += write_batch(batch, output)
                batch = []
        if batch:
            count += write_batch(batch, output)
    return count, observed


def _remap_tokens(source, destination, tokenizer):
    mapping = np.full(tokenizer.backend.get_vocab_size(), tokenizer.unk_token_id, dtype=TOKEN_DTYPE)
    mapping[tokenizer.original_ids] = np.arange(len(tokenizer), dtype=TOKEN_DTYPE)
    with source.open('rb') as original, destination.open('wb') as output:
        while (chunk := np.fromfile(original, dtype=TOKEN_DTYPE, count=1024 * 1024)).size:
            mapping[chunk].tofile(output)


def _write_tokens(source, destination, tokenizer):
    count = 0
    with destination.open('wb') as output:
        for document in _iter_documents(source):
            ids = tokenizer.encode(document, add_special_tokens=True)
            np.asarray(ids, dtype=TOKEN_DTYPE).tofile(output)
            count += len(ids)
    return count


def prepare_webtext(train_path, val_path, cache_dir, min_chars=20, max_records=None,
                    tokenizer_name='bert-base-chinese'):
    """只保留训练集出现的 BERT token 和全部特殊 token，重排为连续 ID。

    max_records 限制每个 split 读取的原始行数，适合先小规模验证。
    返回 train/val 的只读 memmap、PrunedBertTokenizer 和处理统计。
    """
    if min_chars < 1 or (max_records is not None and max_records < 1):
        raise ValueError('min_chars 和 max_records 必须为正整数。')
    sources = {'train': Path(train_path).resolve(), 'val': Path(val_path).resolve()}
    if sources['train'] == sources['val']:
        raise ValueError('训练集和验证集必须使用不同的文件。')
    signature = {
        'version': CACHE_VERSION,
        'min_chars': min_chars,
        'max_records': max_records,
        'tokenizer': str(tokenizer_name),
        'sources': {
            split: {'path': str(path), 'size': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}
            for split, path in sources.items()
        },
    }
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / 'metadata.json'
    metadata = None
    try:
        candidate = json.loads(manifest_path.read_text(encoding='utf-8'))
        tokenizer = PrunedBertTokenizer.load(cache_dir / 'vocab.json')
        if (candidate['signature'] == signature
                and len(tokenizer) == candidate['vocab_size']
                and tokenizer.fingerprint == candidate['tokenizer_fingerprint']
                and all((cache_dir / f'{split}.bin').stat().st_size
                        == candidate['stats'][split]['tokens'] * TOKEN_DTYPE.itemsize
                        for split in sources)):
            metadata = candidate
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        pass

    if metadata is None:
        print('开始清洗并编码 WebText，首次处理完整语料可能需要较长时间。', flush=True)
        # 先在临时目录完成全部处理，失败时保留已有缓存。
        with tempfile.TemporaryDirectory(prefix='prepare-', dir=cache_dir) as temporary:
            staging = Path(temporary)
            stats = {}
            with closing(sqlite3.connect(staging / 'seen.sqlite3')) as seen:
                seen.execute('CREATE TABLE documents (digest BLOB PRIMARY KEY) WITHOUT ROWID')
                for split, source in sources.items():
                    stats[split] = _clean_split(
                        source, staging / f'{split}.txt', seen, min_chars, max_records,
                    )
            from transformers import BertTokenizerFast

            # 优先读取本地缓存；首次使用时才请求下载 tokenizer，不加载 BERT 模型权重。
            try:
                base = BertTokenizerFast.from_pretrained(tokenizer_name, local_files_only=True)
            except OSError:
                base = BertTokenizerFast.from_pretrained(tokenizer_name)
            print(f'开始 BERT 分词，原始词表：{len(base):,}', flush=True)
            original_path = staging / 'train.original.bin'
            stats['train']['tokens'], observed = _write_original_tokens(staging / 'train.txt', original_path, base)
            tokenizer = PrunedBertTokenizer.from_bert(base, observed, str(tokenizer_name))
            _remap_tokens(original_path, staging / 'train.bin', tokenizer)
            stats['val']['tokens'] = _write_tokens(staging / 'val.txt', staging / 'val.bin', tokenizer)
            tokenizer.save(staging / 'vocab.json')
            metadata = {
                'signature': signature, 'stats': stats, 'vocab_size': len(tokenizer),
                'original_vocab_size': len(base), 'tokenizer_fingerprint': tokenizer.fingerprint,
            }
            print(f'词表裁剪：{len(base):,} -> {len(tokenizer):,}（含全部特殊 token）', flush=True)
            (staging / 'metadata.json').write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8',
            )
            # 最后提交元数据；中途退出时下次会重新处理。
            manifest_path.unlink(missing_ok=True)
            for name in ('train.txt', 'val.txt', 'train.bin', 'val.bin', 'vocab.json', 'metadata.json'):
                (staging / name).replace(cache_dir / name)
    else:
        print(f'复用 WebText 缓存：{cache_dir}', flush=True)

    datasets = {
        split: np.memmap(cache_dir / f'{split}.bin', dtype=TOKEN_DTYPE, mode='r')
        for split in sources
    }
    return datasets, tokenizer, metadata['stats']
