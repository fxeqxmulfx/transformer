# The basis: depth, recall and parity, easy and hard

Does a transformer of width 64 and two layers pass three tasks that ask
different things of a transformer, depth, recall and parity, in an easy
mode, while their hard mode takes one of width 128 and six layers; how fast
do the runs go, and at what batch size and rate? `basis(model, mode, seed)`
of the lab ([`basis.py`](../../python/src/lab/domain/basis.py)) is a
model's three runs of a mode. Here they run on both models from three
seeds, beside the runs that set the recipes: sweeps of batch size and rate,
studies of E_2 at length 128, of recall's rate, schedule and rows, and of
parity of 24 bits, and the time of an update.

## Tasks

A task is passed when its selection split reaches sequence accuracy 0.99,
and its run stops there or at its budget. A model passes a mode when it
passes the mode's three tasks.

| Task | Easy | Hard |
| --- | --- | --- |
| depth | E_2, judged at the trained lengths, 32 to 64 | E_4, judged at length 128 |
| recall | 8 keys, each bound once and queried | 16 writes to 8 keys, each key queried for its latest value |
| parity | 1 to 16 bits | 1 to 16 bits |

**Depth.** E_k (`AlternatingBlocks`) holds the strings over a, b and a
neutral letter that, the neutral letters deleted, are k alternating blocks
of a and b from a, and the model labels every prefix (arXiv:2506.16055v3,
Appendix F); most examples come in pairs with the same letter counts and
different labels. E_(k+1) is definable at depth k + 1 of TL[◁#]^pos and
not at depth k (`Transformer.CRASP.definablePos_altPlusNeutral`), and no
fixed-precision transformer of k layers of one head whose RoPE angles are
rational multiples of π recognizes it
(`Transformer.CRASP.not_recognizes_altPlusNeutral_rope`); one of fewer
layers is one of k whose last layers pass their input on, so two layers
reach neither E_3 nor E_4. At bounded lengths E_k is a finite language,
which a shallow transformer can fit, so a run trains at lengths 32 to 64
and the hard mode judges E_4 at 128. The easy mode judges E_2 at the
trained lengths and tests it at 128 alone (`test/length-128`): the small
model fits E_2 there from every seed, and carries it to 128 from about
half (Found). The models here have four heads a layer, compute in floating
point, and their `RoPE` turns by 10,000^(-2i/h) radians a position at head
width h, no rational multiple of π: the theorem predicts the small model's
failure without covering it.

**Recall.** Multi-query associative recall (`MQAR`, arXiv:2312.04927v1,
Appendix E.1): after BOS, writes bind keys to values, out of 256 key and 256
value tokens, more than either width; random values follow, among which the
keys recur as queries, each supervised with its key's value. The tokens
outnumber the width because over few tokens recall can be counting: over c
tokens, once each key and its value share a position, it is defined at
depth 1 of TL[◁#] by c² counts, and a one-layer transformer with uniform
attention, no query matched to a key, does it
(`Transformer.CRASP.exists_rtfr_answers`). Attention that ignores the query
needs, over n + 1 ≥ c rows, at k layers of width d and p bits,
c − 1 ≤ k (2d + 1) log₂(2^p (n + 1) + 1)
(`Transformer.CRASP.two_pow_le_of_queryFree`), a width growing with c,
while one layer of width 5 whose attention compares the query with each key
recalls at p = O(log c) bits at every length
(`Transformer.CRASP.matcher_answers`), so
the two separate at every depth and width
(`Transformer.CRASP.exists_matcher_not_queryFree`). The runs bind 8 or 16
keys, far fewer rows than tokens, where the bound says nothing, and that more
tokens than width rule counting out there is not proved. In the hard mode 8 of the 16 writes rebind a key, and a query asks for the latest
value: a head that adds a recency term to a write's score returns the later
write when one step of that term outweighs the rounding of both scores
(`Transformer.ALM.latest_wins`), and for every recency scale and bound on
the rounding, past some position a rounding within the bound can make the
earlier write win (`Transformer.ALM.past_the_window_rounding_decides`). No
theorem bounds the size recall needs; its separation is measured. Recall
trains on 20,000 rows, a fifth of Zoology's (Appendix E.2): on 100,000 the
small model passes later, or not within twice the budget (Found).

**Parity.** Whether 1 to 16 bits hold an odd number of ones, answered after
the bits without a scratchpad (`Parity`). RASP-L has no program for it
(arXiv:2310.16028v1, §5.2), and without a scratchpad the paper's
transformer fits not even its training set (Appendix C.1). No formula of
TL[◁#] defines it, so no future-masked rounded transformer recognizes it at
every length (`Transformer.CRASP.not_recognizes_parity`); up to 16 bits it
is a finite language, which a transformer can fit, and no run judges a
longer string. It is the same in both modes: the large model learns 16 bits in more updates than the
small one, and 20 bits from seed 0 not within 12,000 updates, where the
small one does; at 24 bits neither passes reliably (Found).

## Runs

Both models are GPTMini with four heads: fused QKV projections, QK-norm,
softmax with exclusive self-attention (XSA), a ReLU² feed-forward layer of
four times the width, pre-norm RMSNorm, RoPE (base 10,000), a tied readout
and initialization at 0.02; `small` has width 64 and two layers, `large`
width 128 and six. A run trains under AdamW with betas (0.9, 0.98) and
weight decay 0.1, the rate warmed up linearly over 50 updates and then held
unless a label says otherwise, on splits drawn from data seed 1: 20,000
training rows, 512 validation and 512 test. It is observed after every
3,200 examples, in chunks of 256 rows, and replays CUDA graphs.

| Labels | Runs | What they train |
| --- | ---: | --- |
| `<mode>-<model>-<task>-seed<seed>` | 30 | each mode's tasks on both models from model seeds 0, 1 and 2, at the mode's recipe; a hard parity run is its easy one, `easy-<model>-parity-seed<seed>` |
| `sweep-easy-<task>-b<batch>-lr<rate>` | 48 | the easy tasks on the small model from seed 0, depth judged at length 128, over 192,000 examples, at 16, 32, 64 and 128 rows per update and rates 3e-4, 1e-3, 3e-3 and 1e-2 |
| `sweep-hard-<task>-<model>-b<batch>-lr<rate>` | 20 | the hard depth and recall, and parity of 1 to 20 and 1 to 24 bits (`parity20`, `parity24`), from seed 0 over 192,000 examples, at the rows and rates `TRIED` lists |
| `depth128-<model>-b<batch>-lr<rate>-seed<seed>` | 23 | E_2 judged at length 128: the small model from five seeds at four recipes, the large one from three at one |
| `recall-small-lr3e-4-seed<seed>` | 3 | the easy recall on the small model at rate 3e-4, for 6,400 updates |
| `recall-annealed-<model>-seed<seed>` | 6 | the easy recall, its rate annealed by a half cosine to a tenth at the budget |
| `recall-rows100k-<model>-seed<seed>` | 6 | the easy recall on 100,000 training rows, Zoology's, for 9,600 updates |
| `parity24-[<variant>-]<model>-seed<seed>` | 18 | parity of 1 to 24 bits for 16,000 updates: at the parity recipe, with the rate annealed by a half cosine to a tenth at the budget (`anneal`), and on 100,000 training rows (`rows100k`) |
| `time-<mode>-<task>-<model>-b<batch>` | 26 | 300 updates of a task, observed after every 100: the easy tasks on the small model at 16 to 256 rows per update, the hard ones on the large model at 16 to 128, depth to 64 |

The recipes, rows per update, rate and budget in updates:

| Task | Easy | Hard |
| --- | --- | --- |
| depth | 16, 1e-3, 2,000 | 16, 1e-3, 10,000 |
| recall | 64, 1e-3, 4,800 | 64, 3e-4, 4,800 |
| parity | 16, 3e-4, 20,000 | 16, 3e-4, 20,000 |

## Running

```sh
./make.py check experiments/basis              # the runs, and what differs between them
./make.py run experiments/basis [label ...]    # train every run, or the labeled ones
./make.py report experiments/basis [label ...] # what the runs recorded, as JSON; `stop` says solved or budget
./make.py profile experiments/basis <label>    # train one run afresh under Scalene's CPU profiler
```

A run trains into `runs/<label>/` here, which git ignores, and continues
from its checkpoint when started again. Another model takes the benchmark
in an experiment file of its own, whose runs are `basis(model, "easy")` and
`basis(model, "hard")`.

## Found

The runs ran on a GeForce GTX 1050 (2 GB) under torch 2.14.1, on 2026-10-03
and 04. A run's time is its wall clock from drawing its splits to its stop;
starting the process, about 2 s, is not counted.

### The benchmark

The update at which each run passed, or the best selection accuracy of one
that did not, with its seconds; a hard parity run is its easy one:

| Mode | Model | Seed | Depth | Recall | Parity | Seconds |
| --- | --- | ---: | --- | --- | --- | ---: |
| easy | small | 0 | 200 (5 s) | 1,600 (31 s) | 3,000 (9 s) | 45 |
| easy | small | 1 | 200 (5 s) | 2,150 (40 s) | 3,400 (10 s) | 55 |
| easy | small | 2 | 200 (5 s) | 1,250 (26 s) | 6,000 (16 s) | 47 |
| easy | large | 0 | 200 (9 s) | 2,200 (184 s) | 6,000 (64 s) | 258 |
| easy | large | 1 | 200 (9 s) | fails, 0.977 (396 s) | 8,200 (87 s) | 493 |
| easy | large | 2 | 200 (10 s) | 1,450 (124 s) | 11,600 (122 s) | 256 |
| hard | small | 0 | fails, 0.371 (53 s) | fails, 0.000 (84 s) | 3,000 (9 s) | 147 |
| hard | small | 1 | fails, 0.363 (54 s) | fails, 0.000 (85 s) | 3,400 (10 s) | 149 |
| hard | small | 2 | fails, 0.354 (53 s) | fails, 0.000 (84 s) | 6,000 (16 s) | 153 |
| hard | large | 0 | 2,600 (70 s) | 3,000 (251 s) | 6,000 (64 s) | 385 |
| hard | large | 1 | 1,000 (30 s) | 2,350 (198 s) | 8,200 (87 s) | 315 |
| hard | large | 2 | 4,200 (109 s) | 2,500 (210 s) | 11,600 (122 s) | 442 |

The small model passes the easy mode from every seed, in 45 to 55 s, and
fails the hard one: at length 128 it labels at most 0.371 of the E_4
sequences right, and no hard recall sequence. The large model passes the
hard mode from every seed, in 315 to 442 s, and the easy one from seeds 0
and 2: from seed 1 its easy recall stops at 0.977 (below). A run stops at
the first observation at which its selection split reaches 0.99, so the
test split of a passing run, drawn the same way, scores 0.963 to 1.0. Easy
depth stops by update 200, before the small model carries E_2 to length
128: its test there is 0.604, 0.637 and 0.904.

### How the recipes were set

**Depth.** Judged at length 128, E_2 is a draw for the small model: in
`depth128-*` it fits the trained lengths within 400 updates in all 20 runs
and reaches 0.99 at 128 in 11, from all five seeds at no recipe, while the
large model reaches it from all three seeds. So the easy mode judges E_2 at
the trained lengths, which both models fit by update 200, and only tests it
at 128. E_4 at 128 separates the models: the small one stays below 0.38
over 12,000 updates at rate 1e-3 or 3e-4 (`sweep-hard-depth-small-*`), and
the large one passes by update 4,200 of its 10,000. On the large model, 16
rows per update at 1e-3 pass in 2,600 updates (70 s), 32 in 2,300 (112 s),
16 at 3e-4 in 5,200 (134 s), and 8 at 1e-3 not within 24,000 (0.922).

**Recall.** On the easy recall the small model's validation loss creeps
down from 6.3 nats to 3.4 to 3.8, and then falls below 3 within 50 updates,
after 1.9 to 3.7 passes over its training rows; at rate 3e-4 it passes from
two seeds of three, and on 100,000 rows later, from seed 2 not within twice
the budget. The large model's loss falls below 3 by update 650 on either
number of rows, and its accuracy then climbs slowly. The update each run
passed at, by seed, or its best selection accuracy:

| Easy recall | Small | Large |
| --- | --- | --- |
| 64 rows per update, rate 1e-3: the recipe | 1,600; 2,150; 1,250 | 2,200; 0.977; 1,450 |
| rate 3e-4, for 6,400 updates | 3,400; 0.553; 2,300 | |
| rate annealed to a tenth at the budget | 1,600; 1,950; 1,600 | 2,100; 0.971; 1,450 |
| 100,000 rows, for 9,600 updates | 2,250; 5,350; 0.334 | 8,950; 4,850; 5,950 |

From seed 1 the large model's last model at the recipe gets 0.968 of its
training rows and 0.951 of validation right, fitting neither; annealed,
0.995 and 0.953, memorizing the rows. On 100,000 rows the large model
passes from every seed, but past the budget, in 424 to 758 s a run against
124 to 184 s for its passes on 20,000 (two of them trained in two sittings,
which count the setup twice). The easy mode keeps the recipe the small
model passes with from every seed, and the large model's miss from seed 1
is its price.

The hard recall takes the large model 8,400 updates at 16 rows per update
and rate 3e-4 (200 s), 5,400 at 32 (237 s), and 2,350 to 3,000 at 64, its
recipe (198 to 251 s); at 64 rows and 1e-3 it does not leave the plateau
within 3,000 updates, and at 16 rows and 1e-4 it reaches 0.963 within
12,000. The small model, at 16 rows and 3e-4 or 1e-3, reaches 0 and 0.055
within 12,000.

**Parity.** 16 rows per update at 3e-4 is the sweep's fastest on the small
model (below), and the budget of 20,000 updates covers the large model's
slowest pass, at update 11,600.

### Parity does not grow with the model

At 24 bits for 16,000 updates, the runs that pass, and the best selection
accuracy of those that do not:

| Recipe | Small | Large |
| --- | --- | --- |
| the parity recipe | none; 0.951, 0.977, 0.889 | seed 0 at update 10,400; 0.984, 0.928 |
| rate annealed | seed 1 at 13,400; 0.984, 0.953 | seed 0 at 14,200; 0.977, 0.953 |
| 100,000 rows | none; 0.977, 0.953, 0.977 | none; 0.984, 0.908, 0.760 |

The small model passes one run in nine and the large two, and an update of
the large model takes four to five times as long (a small run of 16,000
updates 39 to 50 s, a large one 135 to 213 s). Neither annealing, which
smooths the climb, nor five times the rows, which rules out memorizing the
training split, makes 24 bits reliable within the budget. At 20 bits from
seed 0 the small model passes at update 11,800, and the large one does not
within 12,000 updates, at rate 3e-4 (0.936) or 1e-4 (0.971). A longer
parity in the hard mode would leave its outcome to the seed and slow every
run, and no theorem here gives width or depth a part in parity at a
bounded length; so parity stays at 16 bits in both modes.

### Batch size and rate

The easy tasks on the small model from seed 0, over 192,000 examples: the
update each passed at, the fewest of a batch in bold, and the seconds to the
fastest pass of the batch:

| Task | Rows | 3e-4 | 1e-3 | 3e-3 | 1e-2 | Fastest |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| depth at 128 | 16 | 7,600 | **400** | – | 5,000 | 6 s |
|  | 32 | 700 | **200** | – | – | 6 s |
|  | 64 | 1,450 | **150** | 1,600 | – | 6 s |
|  | 128 | **150** | **150** | 600 | – | 9 s |
| recall | 16 | 9,000 | **6,800** | – | – | 37 s |
|  | 32 | 5,700 | **4,700** | 5,800 | – | 45 s |
|  | 64 | – | **1,600** | – | – | 31 s |
|  | 128 | – | 1,100 | **975** | – | 35 s |
| parity | 16 | **3,000** | 9,200 | 8,800 | – | 9 s |
|  | 32 | **2,800** | – | 4,100 | – | 12 s |
|  | 64 | 2,950 | **2,400** | 3,000 | – | 17 s |
|  | 128 | **1,325** | – | – | – | 17 s |

The fewest updates of a batch, over rates tuned on a grid, stand for the
batch (McCandlish et al., arXiv:1812.06162v1, Appendix A.2 and A.3). Their
eqs. 2.11 and 2.12 with E = BS make S = S_min (1 + B_crit / B), which,
fitted to them by least squares in 1 / B, gives S_min 85 and B_crit 56 for
depth at length 128, 311 and 352 for recall, and 1,650 and 15 for parity. A batch well
below B_crit takes about as many examples as any smaller one, and one well
above about as many updates as any larger one.

Adam's best rate at batch B, by Theorem 3 of Li et al.
(arXiv:2405.14578v5, §2.1), is ε_max / (½(√(B_noise / B) + √(B / B_noise))):
it rises with the batch up to B_noise and falls beyond, and their Theorem 5
(§2.2) gives Adam McCandlish's tradeoff, B_noise standing for B_crit. With
the fitted B_crit, from 16 to 128 rows the best rate grows 2.2 times for
recall, changes by at most a fifth for depth, and falls by two fifths for
parity: against the grid's step of √10, at most one step up for recall, and
none for the others. The sweep's best rate is 1e-3 for depth at every
batch (at 128 with 3e-4), 1e-3 for recall up to 64 rows and 3e-3 at 128,
as predicted, and 3e-4 for parity at every batch but 64, where 1e-3 passed
in 2,400 updates against 2,950.

An update's time between observations 100 and 300 of the `time-*` runs
fits t(B) = c (B_sat + B): past about B_sat rows the device is busy, and
the time grows in proportion to the rows. The wall time to the target,
S(B) t(B), is least at B* = √(B_sat B_crit):

| Task | Model | ms per row, c | B_sat | B_crit | B* | Recipe |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| easy depth | small | 0.209 | 5.3 | 56 | 17 | 16 |
| easy recall | small | 0.224 | 5.4 | 352 | 43 | 64 |
| parity | small | 0.067 | 12.7 | 15 | 14 | 16 |
| parity | large | 0.353 | 9.9 | 15 | 12 | 16 |
| hard depth | large | 1.129 | 3.4 | 5 | 4 | 16 |
| hard recall | large | 1.138 | 3.8 | 75 | 17 | 64 |

Parity's B_crit is the small model's. The hard depth's comes from 16 and 32
rows at 1e-3 on the large model, and the hard recall's from 16 and 32 rows
at 3e-4 and the benchmark's seed 0 at 64: S_min 1,500. The large model's
hard depth at 128 rows per update does not fit in the GTX 1050 beside its
evaluation at length 128, and is not timed. The easy depth and parity
train at their B*. The easy recall's B* of 43 lies between 32 and 64 rows,
whose fitted wall times differ by under 1%, and 64 passed fastest. The hard
depth's fit rests on two batches and misses that 8 rows do not pass within
24,000 updates: 16 is the fewest that pass. The hard recall's fit makes 16
rows 24% faster than 64; the one run at 16 took 200 s, within the 198 to
251 s of the three at 64, so the recipe keeps 64.

### Where the time goes

A benchmark run's wall clock, the mean over its three seeds: setup, to the
first observation; training; and observations, with the final evaluation
and checkpoints:

| Run | Setup | Training | Observations | Wall |
| --- | ---: | ---: | ---: | ---: |
| easy depth, small | 4.4 s | 0.9 s | 0.1 s | 5.4 s |
| easy recall, small | 4.7 s | 26.1 s | 1.5 s | 32.3 s |
| parity, small | 2.7 s | 8.1 s | 0.8 s | 11.6 s |
| easy depth, large | 4.7 s | 4.4 s | 0.4 s | 9.5 s |
| easy recall, large | 5.1 s | 218.2 s | 11.6 s | 234.9 s |
| parity, large | 3.3 s | 79.4 s | 8.3 s | 91.1 s |
| hard depth, small | 3.7 s | 43.1 s | 6.5 s | 53.4 s |
| hard recall, small | 5.1 s | 75.0 s | 4.2 s | 84.3 s |
| hard depth, large | 4.4 s | 56.5 s | 8.8 s | 69.6 s |
| hard recall, large | 6.4 s | 202.7 s | 10.9 s | 219.9 s |

Of setup, drawing the splits takes 1.1 s for parity, 1.9 s for depth and
2.3 s for recall on 20,000 rows; the first AdamW of a process 1.1 s, as it
imports TorchDynamo; and warming up the updates before their capture 0.5 s
on the small model and 1.0 s on the large one. Under Scalene's CPU profiler
(`./make.py profile`, `runs/<label>.scalene.json`) the easy recall on the
small model, 43 s there, spends 61% of its time with the host waiting for
the device, 12% drawing the splits, 11% building AdamW and 2% issuing the
updates' graphs; the easy depth, 16 s there and 5 s without, spends 37%
drawing its splits and 31% building AdamW. Training is bound by the device,
and the host could gain only on setup: drawing a split once for the runs
that share it.
