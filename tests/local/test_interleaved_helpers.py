"""Regression checks for the IEPE mechanics shared with SPO."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from ipe.trainer_iepe import InterleavedEPETrainer


class InterleavedRegressionTests(unittest.TestCase):
    def trainer(self, mode="full", masked=True, bos=False):
        trainer = object.__new__(InterleavedEPETrainer)
        trainer.mask_reflection = masked
        trainer.reflection_attention_mode = mode
        trainer.reflection_attention_k = 2
        trainer.reflection_attention_p = 0.5
        trainer.reflection_attention_include_bos = bos
        return trainer

    def test_suffix_positions(self):
        valid = torch.tensor([[1]*9+[0], [1]*7+[0]*3])
        positions = self.trainer()._build_position_ids(valid, torch.tensor([3, -1]), torch.tensor([6, -1]))
        self.assertEqual(positions.tolist(), [[0,1,2,3,4,5,6,3,4,0], [0,1,2,3,4,5,6,0,0,0]])

    def test_attention_modes(self):
        valid = torch.tensor([[1]*9+[0]])
        for mode in ("full", "last_k", "random_p"):
            for masked in (False, True):
                for bos in (False, True):
                    with self.subTest(mode=mode, masked=masked, bos=bos):
                        torch.manual_seed(37)
                        actual = self.trainer(mode,masked,bos)._build_4d_attention_mask(
                            valid, torch.tensor([4]), torch.tensor([6]), torch.float32)[0,0] == 0
                        expected = torch.ones(10,10,dtype=torch.bool).tril()
                        expected[:,9] = False
                        if masked:
                            expected[7:,4:7] = False
                        if mode == "last_k":
                            expected[4:7, int(bos):2] = False
                        elif mode == "random_p":
                            torch.manual_seed(37)
                            n = 4-int(bos)
                            blocked = torch.randperm(n)[max(1,int(n*0.5)):] + int(bos)
                            expected[4:7,blocked] = False
                        torch.testing.assert_close(actual,expected)


if __name__ == "__main__":
    unittest.main()
