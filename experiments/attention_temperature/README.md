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
The frozen source is committed at3f71f7c8. Learning is now closed under the
user's later cost rule; both final tests remain unopened. A copied seed0 state
at55,800 updates fits the
first512 sampled training sequences in averaged mode, while validation has
27.51% token accuracy and no exact sequences. The complete prediction record
reproduces from the copied state. This is a diagnostic snapshot, not a completed
learning outcome or a model-selection result. Use
`./run.py attention-temperature-probe <label> [--snapshot <path>]` for the
read-only working/averaged-state check.

The user subsequently required stopping at twice the best comparable run and
reasoning before further experiments. The matched Schedule-Free GPTMini seed0
first reaches99% at12,200 updates, making the new update cap24,400. Both
temperature candidates have zero exact validation sequences at that point.
This retrospective cost comparison rejects the candidate; it does not mean
either run actually stopped at24,400 or could never grok later.

| Seed | Exact validation at24,400 | Retained actual updates | Endpoint exact validation | Endpoint token validation |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0/512 | 100,000 | 0/512 (0%) | 39.5264% |
| 1 | 0/512 | 62,900 | 416/512 (81.25%) | 97.3389% |

The new instruction arrived after these later states had trained. Seed0 had
already reached its original boundary; seed1 was interrupted on receipt. All
162,900 saved gradient records are consecutive, finite and nonzero, and both
complete checkpoints are retained. The original interrupted phase and its
traceback are preserved. No checkpoint, raw history or optimizer state was
rewritten to apply the new rule. Only study status changes to closed/censored.

[The closure record](analysis/cost_cap_closure.json) contains both phases,
checkpoint and raw-record hashes, timing and the observations at the new cap.
It is a validation-only report; the usual final-test report was not run.
The existing cohort runner refuses to resume a closed study. There is no
production model change. The next step is an analytical mechanism check using
saved states, not another long continuation or a wider parameter sweep.
