# Fresh softmax late failures and sampled recoveries

This partial scientific witness retains the complete 1,121-observation prefix
through 280,000 of the unchanged 300,000-update softmax control. The memorization
plateau and long confirmation remain observed; two actual late failures now
make the frozen all-observations final-window criterion impossible to satisfy.
The full budget and paired review continue, with no new scientific selection.

| Canonical update | Batch | Train accuracy | Held-out accuracy | Event |
| ---: | ---: | ---: | ---: | --- |
| 274,000 | 512 | 51.198187% | 49.179620% | First final-window failure |
| 274,250 | 512 | 100% | 99.974813% | First sampled recovery |
| 275,500 | 48 | 81.951641% | 81.768135% | Second final-window failure |
| 275,750 | 512 | 99.535838% | 99.107657% | Second sampled recovery |

EOS accuracy is 100% throughout the boundary triplets; numeric answers fail.
Both episodes recover at the next canonical observation, after 250 sampled
updates. The first failure's before-probe 273,999 uses a short batch of 48 and
fails; canonical 274,000 and after-probe 274,001 both use 512 and still fail.
Before 275,500, full-batch probe 275,499 passes; full-batch probe 275,501 fails.
These associations do not establish a sampling or optimizer cause.

The recovery before-probes 274,249 and 275,749 already pass. Probe 275,751 is
below target again (held-out 98.657887%) despite canonical 275,750 passing.
Thus exact first recovery time and continuous persistence between observations
are unmeasured; neighbor probes never certify or replace the canonical gate.

The three complete 10,000-update final-window bins have failure counts 0/40,
0/40 and 2/40: episode-onset frequencies 0, 0 and 2 per 10,000 updates.
Their boundaries are [250,000,260,000), [260,000,270,000) and [270,000,280,000).
The preceding eleven complete bins from 160,000 through 270,000 have no new
episodes. The later return of two episodes refutes a monotone decrease of this
observed frequency series; it supplies no population-statistical or future
stability conclusion. Only one observation of the next bin is captured.

The prefix includes 121 of the required 201 final-window observations, with two
failures. All six post-long-onset sampled episodes so far recover. Learning and
delayed generalization remain visible; a stable final benchmark does not pass.

Four boundary triplets preserve eight actual neighbors, twelve gradient/rate
records and twelve full-tensor diagnostics. All 1,121 non-time canonical records
exactly match the SHA256-pinned old same-seed/split reference prefix. This is an
observed metric A/A check, not independent confirmation or tensor/unobserved
trajectory identity. Every 43 Python/seventeen native/nine Lean/two paper/audit
fingerprint remains unchanged. The capture imports no Torch and adds zero updates.

An actual second capture reproduces all nine core files byte-for-byte, with only
the validation timestamp differing. The separate standard-library verifier
recomputes episodes and bin frequencies from stored observations. See its
[executed receipt](softmax-late-tail-recoveries/verification-receipt.json).

```bash
python3 -m experiments.synthetic_trainers.protocols.adamw_stability_20261002.softmax_tail_recovery_capture /tmp/softmax-tail-fresh
python3 experiments/synthetic_trainers/protocols/adamw_stability_20261002/softmax-late-tail-recoveries/verify-snapshot.py experiments/synthetic_trainers/protocols/adamw_stability_20261002/softmax-late-tail-recoveries /tmp/softmax-tail-fresh
```

Portable verification uses the second command without its replay directory.
Finish, archive, verify, visually review and commit the full control/pair and
consume both terminal handles before conditionally freezing the next protocol.
