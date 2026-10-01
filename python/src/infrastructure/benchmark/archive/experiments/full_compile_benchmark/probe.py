"""Paired eager/full-graph timing on the actual RTX3050 GPTMini configuration."""

import gc
import math
import os
from pathlib import Path
import time

root=Path(__file__).resolve().parents[2]
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR",str(root/'experiments/runs/inductor_cache'))
os.environ.setdefault("TRITON_CACHE_DIR",str(root/'experiments/runs/triton_cache'))
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS","2")

import torch
from torch.nn import functional as F

from experiments.gpt_mini import Config
from experiments.optimizer_benchmark.attention import make_model
from experiments.optimizer_benchmark.data import TextData, training_starts, windows
from experiments.optimizer_benchmark.storage import write_json
from experiments.patience_benchmark.registry import make_optimizer, selected_rates
from experiments.compiled_benchmark.telemetry import snapshot
from .step import FullStep


def main():
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    cfg=Config(vocab_size=65,n_layers=2,n_heads=4,d_model=128,d_ff=512,max_seq_len=64)
    data=TextData.load('experiments/tinyshakespeare.txt','cuda')
    starts=training_starts(len(data.train),64,32,220,0).to('cuda')
    directory=root/'experiments/full_compile_benchmark/results/preflight'
    rows=[]
    for attention in ('softmax','sparsemax'):
        for method in ('adamw','magma_muon','dash_ndb_guarded','dash_evd','dash_chebyshev','adafisher'):
            for compiled in (False,True):
                torch.compiler.reset()
                from torch._dynamo.utils import counters
                counters.clear()
                model=make_model(cfg,attention,0,'cuda')
                rate=selected_rates()[attention][method]
                runtime=FullStep(model,attention,method,rate,0,32,220) if compiled else None
                opt=runtime.opt if compiled else make_optimizer(method,model,rate,0)
                torch.cuda.reset_peak_memory_stats()
                losses=[]

                def update(index):
                    if compiled:
                        loss=runtime.train(data.train,starts)
                    else:
                        opt.zero_grad(set_to_none=True)
                        inputs,targets=windows(data.train,starts[index],64)
                        loss=F.cross_entropy(model(inputs).flatten(0,1),targets.flatten())
                        loss.backward()
                        opt.step()
                    value=loss.item()
                    if not math.isfinite(value): raise RuntimeError('Nonfinite probe loss')
                    losses.append(value)
                    torch.cuda.synchronize()

                began=time.perf_counter()
                update(0)
                cold=time.perf_counter()-began
                for i in range(1,20): update(i)
                warm=snapshot()
                began=time.perf_counter()
                for i in range(20,220): update(i)
                ms=(time.perf_counter()-began)*1000/200
                final=snapshot()
                if compiled and (warm!=final or final['graph_breaks'] or final['recorded_graph_nodes']==0):
                    raise RuntimeError(f'Unstable or uncaptured full step: {attention} {method}')
                row={'attention':attention,'method':method,'compiled':compiled,'lr':rate,
                    'cold_first_step_seconds':cold,'warmup_steps':20,'measured_steps':200,
                    'warmed_ms_per_step':ms,'first_loss':losses[0],'final_train_loss':losses[-1],
                    'peak_memory_mib':torch.cuda.max_memory_allocated()/2**20,
                    'warm_counters':warm,'final_counters':final}
                rows.append(row)
                write_json(directory/'throughput.json',rows)
                print(f'{attention} {method} compiled={compiled}: {ms:.3f} ms cold={cold:.2f}s '
                      f'FX={final["unique_fx_graphs"]} CUDA_nodes={final["recorded_graph_nodes"]}',flush=True)
                opt.close()
                del runtime,opt,model
                torch.compiler.reset()
                gc.collect()
                torch.cuda.empty_cache()


if __name__=='__main__': main()
