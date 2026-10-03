# Fresh softmax phase confirmation and first sampled recovery

This partial witness captures the first 105,750 updates of the fresh native
AdamW/softmax control in the frozen attention pair. It preserves 424 canonical
observations, four boundary triplets, twelve gradient/rate and twelve full-tensor
diagnostic records, both original plans and the exact capture program. It adds
no training updates and leaves the 300,000-update budget and criteria unchanged.

The qualifying memorization plateau is 4,500–27,250 (92 observations). Twenty
consecutive joint train/held-out targets run from 95,750 through confirmation
at 100,500. Measured confirmation costs are 2,491.825568 training seconds and
3,092.410553 wall seconds. These are descriptive same-machine measurements;
sparsemax has no eligible target time, so no paired speed ratio is available.

| Canonical update | Train accuracy | Exhaustive held-out accuracy | Event |
| ---: | ---: | ---: | --- |
| 100,500 | 100% | 99.769718% | Long confirmation |
| 105,250 | 100% | 97.481290% | First sampled joint failure |
| 105,500 | 100% | 98.672280% | Last sampled failure in this episode |
| 105,750 | 100% | 99.122050% | First canonical recovery |

EOS accuracy is 100% at all four boundaries. The sampled recovery duration is
500 updates from first failure to first canonical recovery. Neighbor 105,749
already passes while 105,501 fails; the actual first recovery between scheduled
evaluations is unobserved. Neighbor probes never certify the canonical gate.
One episode supplies no decreasing-frequency or future-stability conclusion.

All 424 observed canonical records, excluding measured time fields, exactly
match the immutable previous same-seed/split 300,000-update reference prefix.
The capture verifies that reference's frozen SHA256 before comparison and keeps
its corresponding non-time records. This is an observed metric A/A check;
it does not establish parameter-tensor or unobserved-trajectory identity and
does not count as independent confirmation. Future reference observations are
not outcomes of the current run until actually measured.

The frozen final window is 250,000–300,000. This witness contains zero of its
201 required observations; final persistence is unmeasured, and the scientific
run/pair remain incomplete. This early recovered episode alone cannot reject
or certify the later final-window criterion. No scientific confirmation,
schedule or architecture is selected by this prefix.

The capture checks all 43 Python/nine Lean sources, seventeen native trainer
sources, both local papers and the existing Lean audit receipt. It runs without
Torch. A second fresh capture reproduces all eight core files byte-for-byte;
the timestamp is the sole validation difference. A separate standard-library
verifier recomputes the phase, episode and neighbor checks from stored records.
The executed programs, verification receipt and hashes are in
[the witness](softmax-first-long-and-recovery/validation.json).

Reproduce the captured prefix while the original raw case remains available:

```bash
python3 -m experiments.synthetic_trainers.protocols.adamw_stability_20261002.softmax_long_recovery_capture /tmp/softmax-prefix-fresh
python3 experiments/synthetic_trainers/protocols/adamw_stability_20261002/softmax-first-long-and-recovery/verify-snapshot.py experiments/synthetic_trainers/protocols/adamw_stability_20261002/softmax-first-long-and-recovery /tmp/softmax-prefix-fresh
```

For portable verification, run the second command without its replay-directory
argument. Complete, archive, review and commit the whole softmax control and
pair before the next scientific freeze; both live terminal handles remain to
be consumed after completion.
