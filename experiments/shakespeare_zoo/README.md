# GPTMini on Tiny Shakespeare under the optimizer zoo

Which optimizer trains GPTMini to the lowest test cross entropy on Tiny
Shakespeare, under softmax and under sparsemax attention? These are the 27
recipes of the GPTMini benchmarks, the 24 of the
[full-compile benchmark](../archive/gpt_mini/full_compile_benchmark) and
the 3 of the [AMSGrad extensions](../archive/gpt_mini/amsgrad_extensions_benchmark),
in the lab's language.

## Runs

Model, data and stopping are those of
[`shakespeare_amsgradw`](../shakespeare_amsgradw): GPTMini of width 128,
with 2 layers of 4 heads, reads 32 windows of 64 characters per update;
validation is evaluated every 250 updates, and a run stops after 8
observations without an improvement of 1e-4, or after 3 consecutive
observations 0.1 above its best, both from update 1,000 on, and at the
latest after 20,000 updates. Each recipe trains at the rate its validation
selected from a grid of three, under each attention:

| Recipe | Optimizer | Rate, softmax / sparsemax |
| --- | --- | --- |
| `sgd` | SGD | 0.1 / 0.3 |
| `adagrad` | AdaGrad | 0.03 |
| `adam` | Adam without debiasing | 3e-4 |
| `adamw` | PyTorch's AdamW, decay 0.01 | 1e-3 |
| `amsgrad` | AMSGrad without debiasing | 3e-4 |
| `amsgrad_inverse` | AMSGrad with b1_t = b1 / t and step size lr / sqrt(t) | 3e-3 |
| `amsgrad_geometric` | AMSGrad with b1_t = b1 0.99^(t - 1) and step size lr / sqrt(t) | 3e-3 |
| `adamx` | AdamX | 3e-3 |
| `adamnc` | AdamNC | 0.01 |
| `rmsprop` | RMSProp | 1e-4 / 3e-4 |
| `muon` | Muon | 0.03 |
| `dash_evd` | DASH, inverse roots by eigendecomposition | 1e-3 |
| `dash_ndb` | DASH, by Newton-Denman-Beavers iterations | 1e-3 |
| `dash_cn` | DASH, by coupled Newton iterations | 1e-3 |
| `dash_chebyshev` | DASH, by a Chebyshev fit | 1e-3 |
| `adafisher` | AdaFisher | 1e-3 |
| `adafisherw` | AdaFisher, decay 0.01 | 1e-3 |
| `magma_rmsprop` | MAGMA over RMSProp | 3e-4 |
| `magma_adam` | MAGMA over Adam | 9e-4 |
| `magma_adamw` | MAGMA over AdamW | 3e-3 |
| `magma_muon` | MAGMA over Muon | 0.03 |
| `magma_sgd` | MAGMA over SGD | 0.3 |
| `muon_guarded` | the descent guard over Muon | 0.1 / 0.3 |
| `dash_ndb_guarded` | the descent guard over `dash_ndb` | 0.1 / 0.3 |
| `amsgradw` | raw AMSGradW, decay 0.01 | 3e-4 |
| `amsgradmd` | AMSGradMD | 3e-4 |
| `amsgradmd_guarded` | the descent guard over AMSGradMD, its directions at 3e-4, sigma 0.25 | 0.1 |

A run is labeled `<attention>-<recipe>-seed<seed>`, as `softmax-sgd-seed0`
or `sparsemax-amsgradmd_guarded-seed2`, from model seeds 0, 1 and 2: 162
runs. Every recipe replays from CUDA graphs except `dash_evd`, whose
eigendecomposition torch cannot capture, and which runs eagerly.

## Running

```sh
./make.py check experiments/shakespeare_zoo              # the runs, and what differs between them
./make.py run experiments/shakespeare_zoo [label ...]    # train every run, or the labeled ones
./make.py report experiments/shakespeare_zoo [label ...] # what the runs recorded, as JSON
```

A run trains into `runs/<label>/` here, which git ignores, and continues
from its checkpoint when started again.

## Archived runs

The [GPTMini benchmarks](../archive/gpt_mini/README.md#tiny-shakespeare-benchmarks)
ran on an RTX 3050. The 144 runs of the
[full-compile benchmark](../archive/gpt_mini/full_compile_benchmark/results/rtx3050_all/REPORT.md)
ranked AMSGrad lowest of its 24 methods in mean test cross entropy, with its
sample SD over the seeds: 1.62787 ± 0.00075 under softmax and 1.64600 ±
0.01707 under sparsemax. The
[extensions](../archive/gpt_mini/amsgrad_extensions_benchmark/results/rtx3050/REPORT.md)
ranked AMSGradW lowest of all 27, at 1.625375 ± 0.001731 and 1.638935 ±
0.016796.

The guard over Muon and over DASH rejected every proposal, so both trained
as SGD (`Transformer.OptimizerBenchmark.guardedBatchRun_eq_sgd`): at SGD's
rates, each equaled SGD to the bit in all six of its runs, in every
validation loss, the selected model and its test loss.
The guard over AMSGradMD accepted 0.16% to 0.67% of its proposals.

No run here reproduces its archived run bit for bit: the historical steps
were compiled by Inductor.
