"""Offline regression tests for document selection before pretraining."""
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datasets import Dataset, DatasetDict
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from ipe.data import build_pretrain_dataset
from ipe.conflict_data import ALL_PREF_BY_ID, build_conflict_pretrain_dataset
from ipe.spo_data import SPODataOptions, build_spo_dataset


class DataSelectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        env = patch.dict('os.environ', {'IPE_TOKENIZED_DATA_DIR': self.tmp.name})
        env.start()
        self.addCleanup(env.stop)
        words = ['[UNK]', '[PAD]', 'normal', 'flipped'] + [f'doc{i}' for i in range(30)]
        backend = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token='[UNK]'))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]', pad_token='[PAD]')
        self.raw = Dataset.from_dict({'text': [f'doc{i}' for i in range(30)]})
        pref_id = next(iter(ALL_PREF_BY_ID))
        self.conflict_raw = Dataset.from_list([
            {'uid': f'{i:02d}', 'variant': variant, 'text': f'doc{i} {variant}',
             'preference_id': pref_id, 'topic': 'test'}
            for i in range(30) for variant in ('normal', 'flipped')
        ])

    def build(self, kind, seed=None, limit=5, disable_cache=False):
        raw = self.conflict_raw if kind == 'conflict' else self.raw
        module = {'regular': 'ipe.data', 'conflict': 'ipe.conflict_data', 'spo': 'ipe.spo_data'}[kind]
        with patch(f'{module}._load_dataset_local_or_hub', return_value=DatasetDict(train=raw)):
            selection = {} if seed is None else {'data_selection_seed': seed}
            if kind == 'spo':
                result = build_spo_dataset('synthetic', '', self.tokenizer, 'tiny', limit,
                                           SPODataOptions(seq_len=16, use_reflection=False, **selection),
                                           disable_cache=disable_cache)
            else:
                build = build_conflict_pretrain_dataset if kind == 'conflict' else build_pretrain_dataset
                extra = {'conflict_ratio': 0.5, 'conflict_seed': 13} if kind == 'conflict' else {}
                result = build(dataset_name='synthetic', dataset_config='', seq_len=16,
                               model_source='tiny', tokenizer=self.tokenizer, num_train_samples=limit,
                               use_reflection=False, disable_cache=disable_cache, **extra, **selection)
            return [self.tokenizer.decode(row['input_ids']) for row in result]

    def test_disabled_preserves_existing_selection(self):
        for kind in ('regular', 'conflict', 'spo'):
            with self.subTest(kind=kind):
                implicit = self.build(kind)
                self.assertEqual(implicit, self.build(kind, -1))
                if kind == 'conflict':
                    ids = list(range(30))
                    random.Random(13).shuffle(ids)
                    expected = [f'doc{i} flipped' for i in ids[:5]]
                else:
                    expected = [f'doc{i}' for i in range(5)]
                self.assertEqual(implicit, expected)

    def test_shuffle_selects_from_full_dataset_and_is_reproducible(self):
        for kind in ('regular', 'conflict', 'spo'):
            with self.subTest(kind=kind):
                original = self.build(kind, -1)
                selected = self.build(kind, 42)
                self.assertEqual(len(selected), 5)
                self.assertEqual(len(set(selected)), 5)
                self.assertNotEqual(set(selected), set(original))
                self.assertNotEqual(set(selected), set(self.build(kind, 43)))
                self.assertEqual(selected, self.build(kind, 42, disable_cache=True))
                if kind != 'conflict':
                    self.assertEqual(selected, list(self.raw.shuffle(seed=42).select(range(5))['text']))

    def test_cache_reuse_does_not_mix_selections(self):
        for kind in ('regular', 'conflict', 'spo'):
            with self.subTest(kind=kind):
                results = {seed: self.build(kind, seed) for seed in (-1, 0, 42, 43)}
                for seed in (43, 42, 0, -1):
                    self.assertEqual(results[seed], self.build(kind, seed))
                self.assertEqual(len({tuple(rows) for rows in results.values()}), 4)

    def test_conflict_assignments_and_pairs_are_preserved(self):
        original = self.build('conflict', -1, limit=30)
        shuffled = self.build('conflict', 42, limit=30)
        self.assertNotEqual(original, shuffled)
        self.assertEqual(set(original), set(shuffled))
        self.assertEqual(len({row.split()[0] for row in shuffled}), 30)

    def test_limit_larger_than_dataset(self):
        for kind in ('regular', 'conflict', 'spo'):
            with self.subTest(kind=kind):
                self.assertEqual(len(self.build(kind, 42, limit=100)), 30)

    def test_invalid_negative_seed(self):
        for kind in ('regular', 'conflict', 'spo'):
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, 'data_selection_seed'):
                self.build(kind, -2)


if __name__ == '__main__':
    unittest.main()
