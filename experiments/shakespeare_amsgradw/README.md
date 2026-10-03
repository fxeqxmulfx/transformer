# GPTMini on Tiny Shakespeare under AMSGradW

How well does GPTMini model Tiny Shakespeare under AMSGradW, the best of the
27 optimizers its benchmarks compared, and how much does sparsemax attention
cost it? This is the AMSGradW arm of the
[AMSGrad extensions benchmark](../archive/gpt_mini/amsgrad_extensions_benchmark),
in the lab's language.

## Runs

The text is the 1,115,393 characters of Tiny Shakespeare, split at 90% and
95% into train, validation and test. GPTMini, the
[reference GPTMini](../archive/gpt_mini/gpt_mini.py) of width 128 with 2
layers of 4 heads, reads 32 windows of 64 characters per update, at random
starts, and predicts every next character. It trains under AMSGradW at rate
3e-4, the rate its validation selected, with betas (0.9, 0.999), epsilon
1e-8 and weight decay 0.01.

Validation is evaluated every 250 updates, in chunks of 32 windows. A run
stops after 8 observations without an improvement of 1e-4, or after 3
consecutive observations 0.1 above its best, both from update 1,000 on, and
at the latest after 20,000 updates; its best validation model is then
evaluated once on test.

| Labels | Attention weights | Model seeds |
| --- | --- | --- |
| `softmax-seed0`, `softmax-seed1`, `softmax-seed2` | softmax | 0, 1, 2 |
| `sparsemax-seed0`, `sparsemax-seed1`, `sparsemax-seed2` | sparsemax | 0, 1, 2 |

Every run replays from CUDA graphs.

## Running

```sh
./make.py check experiments/shakespeare_amsgradw              # the runs, and what differs between them
./make.py run experiments/shakespeare_amsgradw [label ...]    # train every run, or the labeled ones
./make.py report experiments/shakespeare_amsgradw [label ...] # what the runs recorded, as JSON
```

A run trains into `runs/<label>/` here, which git ignores, and continues
from its checkpoint when started again.

## Archived runs

The [AMSGrad extensions benchmark](../archive/gpt_mini/amsgrad_extensions_benchmark)
(RTX 3050) screened AMSGradW at three rates for 250 updates on seed 0, then
trained these six runs. Every one stopped on patience. The mean test cross
entropy of their best models, with its sample SD over the seeds, was
1.625375 ± 0.001731 under softmax and 1.638935 ± 0.016796 under sparsemax:
the lowest of all 27 methods of the GPTMini benchmarks, under both
attentions ([`REPORT.md`](../archive/gpt_mini/amsgrad_extensions_benchmark/results/rtx3050/REPORT.md)).
Retrained after the historical codebase was restructured, the six runs
equaled the originals in every curve, stopping decision and selected model
([`REPRODUCTION.md`](../archive/gpt_mini/amsgrad_extensions_benchmark/results/reproduction_20261002/REPRODUCTION.md)).
[`shakespeare_zoo`](../shakespeare_zoo) trains all 27.

They are not reproduced here bit for bit: the historical step was compiled
by Inductor, and it moved a parameter as x - lr m / (eps + sqrt(vmax)) -
(lr decay) x, which rounds otherwise than x - lr (m / (sqrt(vmax) + eps) +
decay x).
