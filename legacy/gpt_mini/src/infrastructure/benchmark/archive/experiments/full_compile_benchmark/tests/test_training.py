"""Full compiled stopping loop and fresh best-checkpoint reload."""

import json
import os
from pathlib import Path
import tempfile
import unittest

import torch

from experiments.gpt_mini import Config
from experiments.optimizer_benchmark.data import TextData
from experiments.patience_benchmark.stopping import StopConfig, choose_best_step
from experiments.full_compile_benchmark.training import train_one
from experiments.full_compile_benchmark.validation import check_checkpoint


@unittest.skipUnless(os.environ.get("OPTIMIZER_BENCH_CUDA") == "1", "CUDA is explicit")
class TrainingTests(unittest.TestCase):
    def test_stopping_curves_and_checkpoint_replay_on_fisher_and_magma(self):
        torch.set_num_threads(4)
        cfg=Config(vocab_size=8,n_layers=1,n_heads=2,d_model=16,d_ff=32,max_seq_len=8)
        stop=StopConfig(every=2,patience=3,min_steps=1,max_steps=6)
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)
            text=directory/'text.txt'
            text.write_text('abcdefgh'*200)
            data=TextData.load(text,'cuda')
            for attention in ('softmax','sparsemax'):
                for method in ('magma_muon','adafisher'):
                    with self.subTest(attention=attention,method=method):
                        row=train_one(cfg,data,attention,method,.001,0,2,stop,directory)
                        self.assertEqual(row['status'],'ok',row.get('recovery_error'))
                        self.assertEqual(row['actual_steps'],6)
                        self.assertEqual(row['test_evaluations'],1)
                        self.assertEqual(row['best_step'],choose_best_step(row['curves']))
                        self.assertTrue(row['compile']['fullgraph'])
                        self.assertTrue(row['compile']['model_loss_gradient_optimizer_compiled'])
                        self.assertEqual(row['compile']['final']['graph_breaks'],{})
                        self.assertGreater(row['compile']['final']['recorded_graph_nodes'],0)
                        if method=='magma_muon':
                            self.assertEqual(row['magma']['mask_draws'],24)
                        actual=check_checkpoint(row,cfg,data,2)
                        self.assertEqual(actual['test_loss'],row['test_loss'])
                        json.dumps(row,allow_nan=False)
