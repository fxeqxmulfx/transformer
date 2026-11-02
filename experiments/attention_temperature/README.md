# Learned attention temperature in current GPT

Task162's next comparison changes one component after the matched historical
GPTMini control: a learned positive score gain per attention head. All current
GPT weights, normalization, accurate RMS backward, residual coefficients,
SwiGLU, RoPE, centered XSA and initialization remain as in df1e9f7.

| Component | Current reference | Temperature candidate |
| --- | --- | --- |
| Normalized query entering SDPA | q | exp(log_gain) * q |
| Added parameters | None | One zero-initialized log-gain per head |
| Width / depth / heads | 128 / 6 / 4 | 128 / 6 / 4 |
| Parameter count | 1,249,024 | 1,249,048 |
| Data and seeds | Task160 hard recall, seeds0/1 | Identical |
| Training recipe | Schedule-Free AdamW, rate0.0003, original warmup/batches | Identical |

The gain is applied after both Q/K normalization and rotation. Its initial
value is one and it consumes no random draws. This preserves every common
initial parameter and the initial function. The finite Q/K RMS epsilon remains;
this is not a substitution of GPTMini's unit-vector normalization. Applying
the gain before normalization would define a different intervention.

The historical GPTMini comparison changes several architectural components.
Its first seed solves under the same optimizer, but both successful and slow
seeds learn larger temperatures. Those observations motivate this isolated
test; they do not establish temperature as the cause of the gap.

Compare the candidate with the fixed current-model task160 endpoints: seed0
censored at100,000 and seed1 first99% at123,550. Start100,000-update chunks,
continue promising states explicitly, and stop at the first99% exact validation
sequences. Retain all outcomes, full states and gradient traces. New final-test
predictions remain sealed until the complete candidate cohort closes. These
are the original public Basis rows, not a fresh distribution across studies.

This is a screening comparison, with no production replacement from two runs.
A useful gain must be confirmed on the broader Basis and arithmetic settings,
with a declared acceptance rule, direct GPT tests and no architecture flag.
The frozen source is committed at3f71f7c8. GPU learning is active; both new
final tests remain sealed. A copied seed0 state at55,800 updates fits the
first512 sampled training sequences in averaged mode, while validation has
27.51% token accuracy and no exact sequences. The complete prediction record
reproduces from the copied state. This is a diagnostic snapshot, not a completed
learning outcome or a model-selection result. Use
`./run.py attention-temperature-probe <label> [--snapshot <path>]` for the
read-only working/averaged-state check.

Seed0's initial100,000-update chunk ends with39.5264% validation token accuracy
and no exact sequences. Seed1 is still learning. This is a continuation
boundary, not a censored result or a completed cohort; both final tests remain
sealed. Complete-state continuation decisions follow the initial pair.
