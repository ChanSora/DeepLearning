import unittest

import torch

from generate import generate_text
from model_v2 import BigramLanguageModel
from pruned_tokenizer import PrunedBertTokenizer
import test_data


class GenerateTests(unittest.TestCase):
    def test_pruned_tokenizer_generation_and_mismatch(self):
        fixture = test_data.WebTextTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        base = fixture.base
        observed = base.backend_tokenizer.encode('中文 playing', add_special_tokens=False).ids
        tokenizer = PrunedBertTokenizer.from_bert(base, observed)
        model = BigramLanguageModel(len(tokenizer), 8, 4, 1)
        model.tokenizer_fingerprint = tokenizer.fingerprint
        # 强制输出两个 WordPiece，验证解码会正确合并 ## 前缀。
        play, ing = tokenizer.encode('playing')
        def generate(inputs, max_tokens, **sampling):
            suffix = torch.tensor([[play, ing]], dtype=torch.long)
            return torch.cat([inputs, suffix[:, :max_tokens]], dim=1)
        model.generate = generate
        self.assertEqual(generate_text(model, tokenizer, '提示🙂', 2, 'cpu'), '提示🙂playing')
        self.assertEqual(generate_text(model, tokenizer, '', 2, 'cpu'), 'playing')
        self.assertEqual(generate_text(model, tokenizer, '中文', 0, 'cpu'), '中文')
        model.generate = lambda inputs, max_tokens, **sampling: torch.cat([inputs, torch.tensor([[ing]])], dim=1)
        self.assertEqual(generate_text(model, tokenizer, 'play', 1, 'cpu'), 'playing')
        model.tokenizer_fingerprint = 'wrong mapping'
        with self.assertRaisesRegex(ValueError, '不匹配'):
            generate_text(model, tokenizer, '中文', 2, 'cpu')


if __name__ == '__main__':
    unittest.main()
