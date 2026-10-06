# ANSR on the ordinary softmax Basis GPTMini

Does the Cohesive-like behavior of the 796-parameter copy-task transformer
in [fxeqxmulfx/ansr](https://github.com/fxeqxmulfx/ansr/tree/9cb98c12b3184368c80fea72f3b81432123d96dd)
transfer to the real depth, associative recall and parity tasks of Basis?

All runs use the unchanged small Basis GPTMini: width 64, two layers, four
heads, fused QKV, QKNorm, softmax, XSA, RoPE, RMSNorm, ReLU squared FFN and
tied readout. The easy Basis data, model/data/batch seeds and task-specific
training batch sizes are retained. Validation and test splits remain untouched.
Execution is eager CUDA on the GTX 1050; the original Basis compiled on CPU.

| Labels | Optimizer | Budget | Other differences |
| --- | --- | --- | --- |
| `adamw-easy-small-<task>-seed0` | Original Basis AdamW, rate, decay and warmup | Original task budget, or solved | Eager CUDA |
| `ansr-low-easy-small-<task>-seed0` | ANSR, population 64, sigma 0.05, p_self 0.05, bound 20 | 100,000 generations, or solved | No rate, decay or backward pass; evaluate every 32 generations |
| `ansr-high-easy-small-<task>-seed0` | Same ANSR with p_self 0.95 | Same | Same |

ANSR is ported from `src/ansr/ansr_torch.py` at the linked commit. Its
coordinate-wise mutation, neighbour choice and restart rules are retained.
Because Basis changes batches, personal attractors are reevaluated on the
current batch before comparison; comparing stored losses from different
batches would be invalid. Counts include this refresh, usually two
population evaluations per generation after the first. Both restart frequency
definitions and total function calls are recorded. Population and RNG state
are checkpointed. Deterministic functional evaluation preserves tied weights;
the initial GPTMini is retained as particle zero and must fit the box.

The main budget is 100,000 generations per ANSR arm, about 12.8 million
training-objective evaluations including attractor refreshes, less any skipped
invalid attractors after a restart. The six ANSR arms run serially on one GPU.
Their checkpoints preserve the completed preparation work. This is not a full
parameter sweep. Generations, distinct sampled batches, forward calls, training
seconds and accuracies measure different costs and outcomes.

## Found

All nine preparation runs completed. AdamW passed all three easy Basis tasks;
ANSR's 256-generation preparation checks have been archived in
[pilot_summary.json](pilot_summary.json) with descriptions and source hashes,
and [pilot_report.json](pilot_report.json) with complete metrics. The user
requested a much larger main budget; every ANSR arm now resumes toward
100,000 generations. The short checks do not settle ANSR's training efficacy.

| Task | AdamW updates to pass | AdamW validation loss | ANSR low p_self, best loss at 256 | ANSR high p_self, best loss at 256 |
| --- | ---: | ---: | ---: | ---: |
| depth | 200 | 0.018909 | 3.451107 | 3.274674 |
| recall | 1,700 | 0.012766 | 6.320269 | 6.320269 |
| parity | 4,800 | 0.003552 | 4.290004 | 4.290004 |

AdamW validation sequence accuracies were 100%, 99.61% and 100%; test
accuracies were 100%, 99.61% and 99.80%. High-p_self ANSR reached 4.30%
validation sequence accuracy on depth, from 0%; the other short ANSR checks
retained the initial model as their best observation. Each ANSR check used
32,704 training forward calls, except high-p_self recall (32,702 because of
restarts). A depth checkpoint confirmed that the low-p_self incumbent changed
only by unit-box float32 roundoff, with maximum parameter difference below
3e-6. These observations concern the preparation budget only. Measurements
overlapped with the lab's validation suite and are not an isolated timing study.

The active queue writes `runs/long_training.log`; each run's `history.jsonl`
and `checkpoint.pt` record progress independently. Running the same labels
again resumes their populations, generator state and data order.

```
./make.py check experiments/basis_ansr
./make.py run experiments/basis_ansr ansr-low-easy-small-depth-seed0
./make.py report experiments/basis_ansr
```
