import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from data import clean_text, prepare_webtext
from pruned_tokenizer import PrunedBertTokenizer
from train import get_batch
from transformers import BertTokenizerFast


class WebTextTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.train = self.root / 'train.json'
        self.val = self.root / 'val.json'
        self.cache = self.root / 'cache'
        self.tokenizer_dir = self.root / 'bert'
        self.tokenizer_dir.mkdir()
        characters = sorted(set('标题描述中文回答验证训练内容新增第一条二Ω'))
        self.original_vocab = ['[PAD]', '[unused1]', '[UNK]', '[CLS]', '[SEP]', '[MASK]',
                               *characters, 'play', '##ing', 'unusedword']
        (self.tokenizer_dir / 'vocab.txt').write_text('\n'.join(self.original_vocab) + '\n', encoding='utf-8')
        self.base = BertTokenizerFast.from_pretrained(self.tokenizer_dir, local_files_only=True)

    def write_records(self, path, records):
        path.write_text(''.join(json.dumps(record, ensure_ascii=False) + '\n' for record in records), encoding='utf-8')

    def prepare(self, **options):
        return prepare_webtext(self.train, self.val, self.cache, min_chars=2,
                               tokenizer_name=str(self.tokenizer_dir), **options)

    def test_cleaning_preserves_paragraphs_and_chinese(self):
        raw = ('<p>ＡＢＣ&nbsp;  中文\x00\u200b</p><script>广告()</script>'
               '<style>隐藏</style><br>正文 https://example.com/a，保留后文\r\n\n\n  结束')
        self.assertEqual(clean_text(raw), 'ABC 中文\n\n正文 ,保留后文\n\n结束')
        self.assertEqual(clean_text(None), '')
        self.assertEqual(clean_text(123), '')
        self.assertEqual(clean_text('a < b 且 c > d'), 'a < b 且 c > d')

    def test_filtering_deduplication_vocab_and_encoding(self):
        self.write_records(self.train, [
            {'title': '标题', 'desc': '描述', 'content': '<b>中文回答</b>'},
            {'content': '中文回答'}, {'content': '短'}, {'content': None}, [],
        ])
        with self.train.open('a', encoding='utf-8') as output:
            output.write('not json\n')
        self.write_records(self.val, [{'content': '中文回答'}, {'content': '验证Ω'}])
        datasets, tokenizer, stats = self.prepare()
        self.assertEqual(stats['train'], dict(read=6, kept=1, invalid=3, too_short=1, duplicate=1, tokens=10))
        self.assertEqual(stats['val']['kept'], 1)
        self.assertEqual(stats['val']['duplicate'], 1)
        self.assertIsInstance(datasets['train'], np.memmap)
        self.assertEqual(datasets['train'].tolist(), tokenizer.encode('标题\n描述\n中文回答', add_special_tokens=True))
        self.assertNotIn('Ω', tokenizer.vocab)
        self.assertIn(tokenizer.unk_token_id, datasets['val'])
        observed = set(self.base.backend_tokenizer.encode('标题\n描述\n中文回答', add_special_tokens=False).ids)
        self.assertEqual(tokenizer.original_ids, sorted(observed | set(self.base.all_special_ids)))
        self.assertTrue(set(self.base.all_special_tokens).issubset(tokenizer.vocab))
        self.assertNotIn('[unused1]', tokenizer.vocab)
        self.assertNotIn('unusedword', tokenizer.vocab)
        for data in datasets.values():
            self.assertGreaterEqual(int(data.min()), 0)
            self.assertLess(int(data.max()), len(tokenizer))

    def test_wordpiece_roundtrip_and_document_boundaries(self):
        self.write_records(self.train, [{'content': '中文\n\nplaying'}, {'content': '回答'}])
        self.write_records(self.val, [{'content': '验证内容'}])
        datasets, tokenizer, stats = self.prepare()
        self.assertIn('play', tokenizer.vocab)
        self.assertIn('##ing', tokenizer.vocab)
        self.assertEqual(tokenizer.decode(tokenizer.encode('playing')), 'playing')
        self.assertEqual(datasets['train'].tolist().count(tokenizer.cls_token_id), 2)
        self.assertEqual(datasets['train'].tolist().count(tokenizer.sep_token_id), 2)
        restored = PrunedBertTokenizer.load(self.cache / 'vocab.json')
        self.assertEqual(restored.fingerprint, tokenizer.fingerprint)
        self.assertEqual(restored.encode('中文 playing 未见'), tokenizer.encode('中文 playing 未见'))
        self.assertEqual(restored.decode(restored.encode('playing')), 'playing')
        self.assertEqual(restored.encode('unusedword'), [restored.unk_token_id])

    def test_cache_reuse_and_source_change(self):
        self.write_records(self.train, [{'content': '训练内容'}])
        self.write_records(self.val, [{'content': '验证内容'}])
        self.prepare()
        binary = self.cache / 'train.bin'
        before = binary.stat().st_mtime_ns
        with patch('transformers.BertTokenizerFast.from_pretrained', side_effect=AssertionError('不应重新加载 BERT')):
            self.prepare()
        self.assertEqual(before, binary.stat().st_mtime_ns)
        self.write_records(self.train, [{'content': '训练内容'}, {'content': '新增内容'}])
        _, _, stats = self.prepare()
        self.assertEqual(stats['train']['kept'], 2)

    def test_long_documents_are_not_truncated(self):
        self.write_records(self.train, [{'content': '中文' * 300}])
        self.write_records(self.val, [{'content': '验证内容'}])
        datasets, tokenizer, stats = self.prepare()
        self.assertEqual(stats['train']['tokens'], 602)
        self.assertEqual(datasets['train'][0], tokenizer.cls_token_id)
        self.assertEqual(datasets['train'][-1], tokenizer.sep_token_id)

    def test_record_limit_and_options_invalidate_cache(self):
        self.write_records(self.train, [{'content': '第一条'}, {'content': '第二条'}])
        self.write_records(self.val, [{'content': '验证条'}])
        _, _, stats = self.prepare(max_records=1)
        self.assertEqual(stats['train']['read'], 1)
        _, _, stats = self.prepare()
        self.assertEqual(stats['train']['read'], 2)

    def test_failure_preserves_existing_cache(self):
        self.write_records(self.train, [{'content': '训练内容'}])
        self.write_records(self.val, [{'content': '验证内容'}])
        self.prepare()
        original = (self.cache / 'metadata.json').read_bytes()
        self.write_records(self.val, [{'content': ''}])
        with self.assertRaisesRegex(ValueError, '没有可用数据'):
            self.prepare()
        self.assertEqual((self.cache / 'metadata.json').read_bytes(), original)

    def test_batch_alignment_dtype_and_minimum_length(self):
        # 恰好 block_size + 1 个 token 时也必须能采样。
        data = np.arange(5, dtype=np.uint32)
        x, y = get_batch(data, block_size=4, batch_size=2, device='cpu')
        self.assertEqual(x.dtype, torch.long)
        self.assertEqual(tuple(x.shape), (2, 4))
        self.assertTrue(torch.equal(x[:, 1:], y[:, :-1]))
        self.assertEqual(y[0, -1].item(), 4)
        with self.assertRaises(ValueError):
            get_batch(data[:4], block_size=4, batch_size=2, device='cpu')


if __name__ == '__main__':
    unittest.main()
