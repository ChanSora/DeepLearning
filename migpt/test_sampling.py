import unittest
from unittest.mock import patch

import torch

from model import BigramLanguageModel as LanguageModelV1
from model_v2 import BigramLanguageModel as LanguageModelV2
from sampling import generate_tokens


class FixedLogitsModel(torch.nn.Module):
    block_size = 3

    def __init__(self, probabilities):
        super().__init__()
        self.register_buffer('logits', torch.tensor(probabilities, dtype=torch.float32).log())
        self.context_lengths = []

    def forward(self, inputs):
        self.context_lengths.append(inputs.size(1))
        return self.logits.expand(inputs.size(0), inputs.size(1), -1), None


class SamplingTests(unittest.TestCase):
    def distribution(self, probabilities=(0.4, 0.3, 0.2, 0.1), **parameters):
        model = FixedLogitsModel(probabilities)
        prompt = torch.zeros((2, 1), dtype=torch.long)
        with patch('sampling.torch.multinomial', side_effect=lambda probs, num_samples: probs.argmax(-1, keepdim=True)) as sample:
            generate_tokens(model, prompt, max_tokens=1, **parameters)
        return sample.call_args.args[0]

    def test_temperature_and_unfiltered_distribution(self):
        original = torch.tensor([0.4, 0.3, 0.2, 0.1])
        torch.testing.assert_close(self.distribution()[0], original)
        warmer = original.sqrt()
        torch.testing.assert_close(self.distribution(temperature=2.0)[0], warmer / warmer.sum())
        torch.testing.assert_close(self.distribution(top_k=999)[0], original)

    def test_top_k_limits_candidates_even_when_logits_are_tied(self):
        expected = torch.tensor([4 / 7, 3 / 7, 0, 0])
        torch.testing.assert_close(self.distribution(top_k=2)[0], expected)
        tied = self.distribution(probabilities=(0.25, 0.25, 0.25, 0.25), top_k=2)
        self.assertTrue(torch.equal((tied > 0).sum(-1), torch.tensor([2, 2])))
        self.assertTrue(torch.equal(tied.max(-1).values, torch.tensor([0.5, 0.5])))

    def test_top_p_keeps_threshold_crossing_token_and_at_least_one(self):
        expected = torch.tensor([4 / 7, 3 / 7, 0, 0])
        torch.testing.assert_close(self.distribution(top_p=0.6)[0], expected)
        exact = self.distribution(probabilities=(0.5, 0.25, 0.125, 0.125), top_p=0.75)
        torch.testing.assert_close(exact[0], torch.tensor([2 / 3, 1 / 3, 0, 0]))
        torch.testing.assert_close(self.distribution(top_p=1e-8)[0], torch.tensor([1., 0, 0, 0]))

    def test_filters_are_combined_after_temperature_in_order(self):
        torch.testing.assert_close(self.distribution(top_k=2, top_p=0.5)[0], torch.tensor([1., 0, 0, 0]))
        torch.testing.assert_close(self.distribution(temperature=0.5, top_k=2, top_p=0.6)[0],
                                   torch.tensor([1., 0, 0, 0]))

    def test_greedy_generation_preserves_prompt_and_crops_context(self):
        model = FixedLogitsModel((0.1, 0.6, 0.3))
        prompt = torch.tensor([[2, 0, 2, 0]])
        with patch('sampling.torch.multinomial') as sample:
            generated = generate_tokens(model, prompt, 5, temperature=0, top_k=2, top_p=0.9)
        sample.assert_not_called()
        self.assertTrue(torch.equal(generated[:, :4], prompt))
        self.assertTrue(torch.equal(generated[:, 4:], torch.ones((1, 5), dtype=torch.long)))
        self.assertEqual(model.context_lengths, [3] * 5)

    def test_invalid_parameters_are_rejected_even_for_zero_length(self):
        model = FixedLogitsModel((0.5, 0.5))
        prompt = torch.zeros((1, 1), dtype=torch.long)
        invalid = [dict(temperature=-1), dict(temperature=float('nan')), dict(temperature=float('inf')),
                   dict(top_k=-1), dict(top_k=1.5), dict(top_p=0), dict(top_p=1.01),
                   dict(top_p=float('nan')), dict(top_p=float('inf'))]
        for parameters in invalid:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                generate_tokens(model, prompt, 0, **parameters)
        self.assertTrue(torch.equal(generate_tokens(model, prompt, 0), prompt))

    def test_both_model_versions_support_sampling_and_seed_reproducibility(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for cls in (LanguageModelV1, LanguageModelV2):
                with self.subTest(model=cls.__module__):
                    model = cls(11, 8, 4, 1, n_head=2).eval()
                    prompt = torch.tensor([[1, 2, 3]])
                    greedy = model.generate(prompt, 3, temperature=0)
                    self.assertTrue(torch.equal(greedy, model.generate(prompt, 3, top_k=1)))
                    torch.manual_seed(42)
                    first = model.generate(prompt, 3, temperature=0.8, top_k=5, top_p=0.9)
                    torch.manual_seed(42)
                    second = model.generate(prompt, 3, temperature=0.8, top_k=5, top_p=0.9)
                    self.assertTrue(torch.equal(first, second))
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == '__main__':
    unittest.main()
