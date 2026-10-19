from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import random
import signal
import unittest
from unittest.mock import patch

import numpy as np
import torch

from checkpoint import (
    atomic_save, load_inference_model, load_training_checkpoint, model_from_state,
    save_training_checkpoint,
)
from generate import main as generate_main
from model_v2 import BigramLanguageModel
from pruned_tokenizer import PrunedBertTokenizer
import test_data
from train import TrainConfig, estimate_loss, get_batch, main, run_training


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.fixture = test_data.WebTextTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        base = self.fixture.base
        observed = base.backend_tokenizer.encode('中文训练内容 playing', add_special_tokens=False).ids
        self.tokenizer = PrunedBertTokenizer.from_bert(base, observed)
        data = np.tile(np.arange(len(self.tokenizer), dtype=np.uint32), 8)
        self.datasets = {'train': data, 'val': data[::-1]}
        self.config = TrainConfig(batch_size=2, block_size=4, n_embd=8, n_layer=1, n_head=2,
                                  total_steps=6, warmup_steps=2, eval_interval=2, eval_batches=1,
                                  save_interval=1, keep_last=2, precision='fp32', seed=17)

    def new_model(self):
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        model = BigramLanguageModel(**self.config.model_config(len(self.tokenizer)))
        model.tokenizer_fingerprint = self.tokenizer.fingerprint
        return model

    def run_model(self, model, directory, steps, resume=None):
        directory = self.root / directory
        with redirect_stdout(io.StringIO()):
            interrupted = run_training(
                model, self.datasets, self.tokenizer, self.config, 'cpu', directory,
                directory / 'final.pth', steps, resume=resume,
            )
        return load_training_checkpoint(directory / 'last.pt'), interrupted

    def assert_state_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_state_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_state_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_resume_matches_uninterrupted_training_including_optimizer_and_rng(self):
        full, _ = self.run_model(self.new_model(), 'full', 6)
        for split_step in (2, 3):
            with self.subTest(split_step=split_step):
                partial, _ = self.run_model(self.new_model(), f'partial_{split_step}', split_step)
                resumed_model = model_from_state(partial, 'cpu')
                resumed, _ = self.run_model(resumed_model, f'resumed_{split_step}', 6, resume=partial)
                self.assertEqual(resumed['step'], 6)
                self.assertEqual(partial['scheduler_state_dict']['last_epoch'], split_step)
                self.assertEqual(resumed['scheduler_state_dict']['last_epoch'], 6)
                for key in ('model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict',
                            'rng_state', 'last_eval'):
                    self.assert_state_equal(full[key], resumed[key])
                if split_step == 2:
                    for key in ('best_val_loss', 'best_step', 'best_model'):
                        self.assert_state_equal(full[key], resumed[key])
                self.assertEqual(resumed['model_config']['n_head'], 2)
                self.assertEqual(resumed['model_version'], 2)
                self.assertAlmostEqual(resumed['optimizer_state_dict']['param_groups'][0]['lr'], self.config.min_lr)

    def test_best_weights_do_not_drift_and_survive_copying_checkpoint(self):
        self.config.eval_interval = 1
        metrics = [{'train': value, 'val': value} for value in (3.0, 1.0, 2.0)]
        with patch('train.estimate_loss', side_effect=metrics):
            saved, _ = self.run_model(self.new_model(), 'best', 3)
        best = torch.load(self.root / 'best/best_model.pt', weights_only=False)
        self.assertEqual((best['step'], best['val_loss']), (2, 1.0))
        self.assertNotIn('optimizer_state_dict', best)
        self.assertTrue(any(not torch.equal(value, saved['model_state_dict'][key])
                            for key, value in best['model_state_dict'].items()))
        model = model_from_state(saved, 'cpu')
        with patch('train.estimate_loss', return_value={'train': 4.0, 'val': 4.0}):
            resumed, _ = self.run_model(model, 'copied', 4, resume=saved)
        restored_best = torch.load(self.root / 'copied/best_model.pt', weights_only=False)
        self.assertEqual(resumed['best_step'], 2)
        self.assertEqual(resumed['best_val_loss'], 1.0)
        self.assert_state_equal(best, restored_best)

    def test_evaluation_preserves_rng_and_model_mode(self):
        model = self.new_model()
        before = torch.get_rng_state().clone()
        first = estimate_loss(model, self.datasets, self.config, 'cpu')
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        model.eval()
        self.assertEqual(first, estimate_loss(model, self.datasets, self.config, 'cpu'))
        self.assertFalse(model.training)

    def test_sigint_saves_a_completed_step_and_restores_signal_handler(self):
        self.config.eval_interval = self.config.save_interval = 4
        previous = signal.getsignal(signal.SIGINT)
        requested = False

        def interrupt_batch(*args, **kwargs):
            nonlocal requested
            if not requested:
                requested = True
                os.kill(os.getpid(), signal.SIGINT)
            return get_batch(*args, **kwargs)

        with patch('train.get_batch', side_effect=interrupt_batch):
            state, interrupted = self.run_model(self.new_model(), 'interrupt', 6)
        self.assertTrue(interrupted)
        self.assertEqual(state['step'], 1)
        self.assertEqual(state['scheduler_state_dict']['last_epoch'], 1)
        self.assertEqual(state['last_eval']['step'], 1)
        self.assertEqual(signal.getsignal(signal.SIGINT), previous)

    def test_retention_orders_by_step_and_preserves_best(self):
        directory = self.root / 'retention'
        atomic_save({'marker': 'best'}, directory / 'best_model.pt')
        with redirect_stdout(io.StringIO()):
            for step in (2, 10, 11):
                save_training_checkpoint({'step': step}, directory, keep_last=2)
        self.assertEqual(sorted(path.name for path in directory.glob('ckpt_step_*.pt')),
                         ['ckpt_step_00000010.pt', 'ckpt_step_00000011.pt'])
        self.assertEqual(torch.load(directory / 'last.pt', weights_only=False)['step'], 11)
        self.assertTrue((directory / 'best_model.pt').is_file())

    def test_atomic_save_failure_preserves_previous_checkpoint(self):
        path = self.root / 'atomic.pt'
        atomic_save({'step': 1}, path)
        with patch('checkpoint.torch.save', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                atomic_save({'step': 2}, path)
        self.assertEqual(torch.load(path, weights_only=False)['step'], 1)
        self.assertFalse(list(self.root.glob('.atomic.pt.*.tmp')))

    def test_cli_resume_inherits_configuration_and_embedded_tokenizer_generates(self):
        fixture = self.fixture
        fixture.write_records(fixture.train, [{'content': '中文训练内容' * 4}])
        fixture.write_records(fixture.val, [{'content': '验证中文内容' * 4}])
        initial = [
            '--train-path', str(fixture.train), '--val-path', str(fixture.val),
            '--cache-dir', str(fixture.cache), '--tokenizer', str(fixture.tokenizer_dir),
            '--min-chars', '2', '--batch-size', '2', '--block-size', '4', '--n-embd', '8',
            '--n-layer', '1', '--n-head', '2', '--total-steps', '4', '--warmup-steps', '1',
            '--eval-interval', '2', '--eval-batches', '1', '--save-interval', '2',
            '--device', 'cpu', '--precision', 'fp32', '--checkpoint-dir', str(self.root / 'cli'),
            '--output', str(self.root / 'first.pth'), '--max-steps', '2',
        ]
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(initial), 0)
            self.assertEqual(main(['--resume', str(self.root / 'cli/last.pt'), '--max-steps', '4',
                                   '--device', 'cpu', '--checkpoint-dir', str(self.root / 'cli_resumed'),
                                   '--output', str(self.root / 'final.pth')]), 0)
        checkpoint = self.root / 'cli_resumed/last.pt'
        state = load_training_checkpoint(checkpoint)
        self.assertEqual(state['step'], 4)
        self.assertEqual(state['model_version'], 2)
        self.assertEqual(state['train_config']['n_embd'], 8)
        for artifact in (checkpoint, self.root / 'cli_resumed/best_model.pt', self.root / 'final.pth'):
            model, metadata = load_inference_model(artifact, 'cpu')
            self.assertIsInstance(model, BigramLanguageModel)
            self.assertEqual(model.tokenizer_fingerprint, metadata['tokenizer_fingerprint'])
            captured = io.StringIO()
            with patch('sys.argv', ['generate.py', '--model', str(artifact), '--prompt', '中文',
                                    '--length', '2', '--device', 'cpu', '--seed', '42',
                                    '--temperature', '0.8', '--top-k', '5', '--top-p', '0.9']), redirect_stdout(captured):
                generate_main()
            self.assertTrue(captured.getvalue().startswith('中文'))

        # 旧完整模型对象可只加载权重重新训练，优化器和步数从头开始。
        legacy_model, _ = load_inference_model(self.root / 'final.pth', 'cpu')
        legacy_path = self.root / 'legacy.pth'
        torch.save(legacy_model, legacy_path)
        init_args = initial.copy()
        for flag, value in (('--max-steps', '1'), ('--checkpoint-dir', str(self.root / 'initialized')),
                            ('--output', str(self.root / 'initialized.pth'))):
            init_args[init_args.index(flag) + 1] = value
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main([*init_args, '--init-from', str(legacy_path)]), 0)
        initialized = load_training_checkpoint(self.root / 'initialized/last.pt')
        self.assertEqual(initialized['step'], 1)
        self.assertEqual(initialized['scheduler_state_dict']['last_epoch'], 1)
        with self.assertRaisesRegex(ValueError, '必须与 checkpoint 一致'):
            main(['--resume', str(checkpoint), '--batch-size', '999'])
        with self.assertRaisesRegex(ValueError, '小于'):
            main(['--resume', str(checkpoint), '--max-steps', '3'])
        with self.assertRaisesRegex(ValueError, '完整训练 checkpoint'):
            load_training_checkpoint(self.root / 'final.pth')
        fixture.write_records(fixture.val, [{'content': '验证内容发生变化' * 4}])
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, '续训数据'):
            main(['--resume', str(checkpoint), '--max-steps', '4', '--device', 'cpu'])


if __name__ == '__main__':
    unittest.main()
