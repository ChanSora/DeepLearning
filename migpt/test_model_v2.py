import copy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from checkpoint import WEIGHTS_FORMAT, load_inference_model, model_config, model_from_state
from model import BigramLanguageModel as LanguageModelV1
from model_v2 import BigramLanguageModel, CausalSelfAttention


class ModelV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_sdpa_matches_explicit_causal_attention_and_gradients(self):
        torch.manual_seed(42)
        attention = CausalSelfAttention(12, 3).double().eval()
        reference = copy.deepcopy(attention)
        inputs = torch.randn(2, 5, 12, dtype=torch.float64, requires_grad=True)
        reference_inputs = inputs.detach().clone().requires_grad_()
        actual = attention(inputs)

        query, key, value = reference.qkv(reference_inputs).chunk(3, dim=-1)
        query = query.view(2, 5, 3, 4).transpose(1, 2)
        key = key.view(2, 5, 3, 4).transpose(1, 2)
        value = value.view(2, 5, 3, 4).transpose(1, 2)
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(4)
        mask = torch.ones(5, 5, dtype=torch.bool).tril()
        weights = scores.masked_fill(~mask, float('-inf')).softmax(dim=-1)
        expected = reference.proj((weights @ value).transpose(1, 2).contiguous().view(2, 5, 12))
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        actual.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(inputs.grad, reference_inputs.grad, atol=1e-11, rtol=1e-11)
        for parameter, reference_parameter in zip(attention.parameters(), reference.parameters()):
            torch.testing.assert_close(parameter.grad, reference_parameter.grad, atol=1e-11, rtol=1e-11)

    def test_causality_and_eval_determinism(self):
        model = BigramLanguageModel(17, 12, 5, 2, n_head=3).eval()
        inputs = torch.tensor([[1, 2, 3, 4, 5]])
        changed = torch.tensor([[1, 2, 9, 10, 11]])
        with torch.no_grad():
            first, _ = model(inputs)
            repeated, _ = model(inputs)
            future_changed, _ = model(changed)
        self.assertTrue(torch.equal(first, repeated))
        torch.testing.assert_close(first[:, :2], future_changed[:, :2], atol=0, rtol=0)

    def test_sdpa_dropout_is_disabled_in_eval(self):
        attention = CausalSelfAttention(8, 2)
        inputs = torch.randn(2, 4, 8)
        with patch('model_v2.F.scaled_dot_product_attention', wraps=F.scaled_dot_product_attention) as sdpa:
            attention.train()
            attention(inputs)
            attention.eval()
            attention(inputs)
        self.assertEqual(sdpa.call_args_list[0].kwargs['dropout_p'], 0.2)
        self.assertEqual(sdpa.call_args_list[1].kwargs['dropout_p'], 0.0)
        self.assertTrue(all(call.kwargs['is_causal'] for call in sdpa.call_args_list))

    def test_loss_backward_and_generation_beyond_context(self):
        model = BigramLanguageModel(17, 12, 4, 2, n_head=3)
        inputs = torch.randint(17, (2, 4))
        logits, loss = model(inputs, inputs.flip(-1))
        self.assertEqual(tuple(logits.shape), (2, 4, 17))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                            for parameter in model.parameters()))
        model.eval()
        prompt = torch.tensor([[1, 2, 3, 4, 5]])
        with torch.inference_mode():
            output = model.generate(prompt, max_tokens=7)
        self.assertEqual(tuple(output.shape), (1, 12))
        self.assertTrue(torch.equal(output[:, :5], prompt))
        with self.assertRaises(ValueError):
            model(prompt)
        with self.assertRaises(ValueError):
            BigramLanguageModel(17, 12, 4, 2, n_head=5)

    def test_checkpoint_versions_and_legacy_model_objects(self):
        with tempfile.TemporaryDirectory() as temporary:
            for version, cls in ((1, LanguageModelV1), (2, BigramLanguageModel)):
                original = cls(17, 12, 4, 2, n_head=3).eval()
                original.tokenizer_fingerprint = 'test-tokenizer'
                state = {
                    'format': WEIGHTS_FORMAT, 'model_config': model_config(original),
                    'model_state_dict': original.state_dict(), 'tokenizer_fingerprint': 'test-tokenizer',
                }
                if version == 2:
                    state['model_version'] = 2
                # 无版本标记的旧 checkpoint 必须仍按第一代模型加载。
                restored = model_from_state(state, 'cpu').eval()
                self.assertIsInstance(restored, cls)
                inputs = torch.tensor([[1, 2, 3, 4]])
                torch.testing.assert_close(original(inputs)[0], restored(inputs)[0], atol=0, rtol=0)
                path = Path(temporary) / f'model_v{version}.pth'
                torch.save(original, path)
                legacy, metadata = load_inference_model(path, 'cpu')
                self.assertIsInstance(legacy, cls)
                self.assertIsNone(metadata)
            state['model_version'] = 999
            with self.assertRaisesRegex(ValueError, '不支持的模型版本'):
                model_from_state(state, 'cpu')

    @unittest.skipUnless(torch.cuda.is_available(), '当前环境没有 CUDA，无法实测 FlashAttention 内核')
    def test_cuda_flash_attention_forward_and_backward(self):
        from torch.nn.attention import SDPBackend, sdpa_kernel

        attention = CausalSelfAttention(128, 2).to(device='cuda', dtype=torch.float16)
        inputs = torch.randn(2, 16, 128, device='cuda', dtype=torch.float16, requires_grad=True)
        # 强制 FlashAttention，避免本测试在 GPU 上悄悄回退到 math 后端。
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            output = attention(inputs)
            output.float().square().mean().backward()
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(inputs.grad).all())


if __name__ == '__main__':
    unittest.main()
