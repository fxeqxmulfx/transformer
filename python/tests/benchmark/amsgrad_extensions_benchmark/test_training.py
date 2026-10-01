"""Screening cannot read test loss; final runs restore exact selected weights."""

import os
from pathlib import Path
import tempfile
import unittest

import torch

from gpt_mini.gpt_mini import Config
from gpt_mini.infrastructure.benchmark.optimizer_benchmark.data import TextData
from gpt_mini.infrastructure.benchmark.patience_benchmark.stopping import StopConfig, choose_best_step
from gpt_mini.infrastructure.benchmark.amsgrad_extensions_benchmark.training import train_one


@unittest.skipUnless(os.environ.get("OPTIMIZER_BENCH_CUDA") == "1", "CUDA is explicit")
class TrainingTests(unittest.TestCase):
    def test_screening_and_final_checkpoint_contract_on_both_attentions(self):
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        cfg = Config(vocab_size=8,n_layers=1,n_heads=2,d_model=16,d_ff=32,max_seq_len=8)
        tokens = (torch.arange(400) % 8).to("cuda")
        # Empty test makes an accidental screening test evaluation fail.
        screen = TextData(tokens,tokens[:80],tokens[:0],"abcdefgh","fixture",(1,2,3))
        final = TextData(tokens,tokens[:80],tokens[:80].roll(1),"abcdefgh","fixture",(1,2,3))
        stop = StopConfig(every=2,patience=3,min_steps=6,max_steps=6)
        with tempfile.TemporaryDirectory() as directory:
            for attention in ("softmax","sparsemax"):
                row = train_one(cfg,screen,attention,"amsgradw",.001,0,2,stop,
                    Path(directory)/attention/"screen",evaluate_test=False,phase="screen")
                self.assertEqual(row["status"],"ok",row)
                self.assertIsNone(row["test_loss"])
                self.assertEqual(row["test_evaluations"],0)
                self.assertEqual(row["best_step"],choose_best_step(row["curves"]))
                row = train_one(cfg,final,attention,"amsgradmd_guarded",.3,0,2,stop,
                    Path(directory)/attention/"final")
                self.assertEqual(row["status"],"ok",row)
                self.assertEqual(row["test_evaluations"],1)
                self.assertEqual(row["best_step"],choose_best_step(row["curves"]))
                self.assertEqual(row["compile"]["final"]["graph_breaks"],{})
                self.assertGreater(row["compile"]["final"]["recorded_graph_nodes"],0)
