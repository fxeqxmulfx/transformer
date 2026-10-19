"""The standard namespace exposes the actual installed PyTorch classes."""

import unittest

import torch

import lean_optim


class StandardTests(unittest.TestCase):
    def test_standard_is_torch_optim_and_paper_variants_have_explicit_names(self):
        self.assertIs(lean_optim.standard, torch.optim)
        for name in ("Adam", "AdamW", "SGD", "Adagrad", "RMSprop"):
            self.assertIs(getattr(lean_optim.standard, name), getattr(torch.optim, name))
        self.assertIs(lean_optim.PaperAdam, lean_optim.Adam)
        self.assertIs(lean_optim.PaperAMSGrad, lean_optim.AMSGrad)
        self.assertIs(lean_optim.PaperAdaGrad, lean_optim.AdaGrad)
        self.assertIsNot(lean_optim.PaperAdam, torch.optim.Adam)


if __name__ == "__main__":
    unittest.main()
