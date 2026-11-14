# Original declared-readout comparison

The frozen CM35 readout adapter completes all 35 original cases and their
35 replays with zero process failures. It keeps 35/35 primary and auxiliary
successes in full batch and minibatch, and 34/35 in forward-only, for both
initializer policies. No selected model changes. These are this earlier
readout branch's results; the later context-mixture comparisons are separate
experiments and already retain 35/35 in every mode.

| Original mode | Primary successes | Auxiliary successes | Standalone readout successes | Selected replacements |
| --- | ---: | ---: | ---: | ---: |
| Full batch | 35/35 | 35/35 | 7/7 available endpoints | 0 |
| Minibatch | 35/35 | 35/35 | 7/7 available endpoints | 0 |
| Forward only | 34/35 | 34/35 | 7/7 available endpoints | 0 |

The readout candidates cover six arithmetic cases and Fibonacci, all at
prime97. All 21 available endpoints pass their primary and auxiliary checks
and are eligible offers. All 3774 refused mode outcomes remain explicit.
There are 1265 staged trials: every one of 55 stages for each of 23 planned
tasks. The other 12 original tasks retain their reviewed endpoints. This
7/7 candidate result does not turn the original denominator into seven.

The six arithmetic candidates have 100% primary/raw accuracy and ordinary CE
3.624130329e-9 in every mode. Each Fibonacci candidate has 4699/4704 field
correct predictions, or 99.8937%, 3119/4704 raw correct predictions, or
66.3053%, and validation CE8.086734704. Its observed raw CE is9.626594307e-9.
The earlier kernel Fibonacci endpoint has 4698 field and3118 raw correct
predictions, with CE8.091836744. These small standalone gains remain visible,
but the readout candidates do not improve observed raw training CE by the
selector's required more-than1e-12. The frozen selector retains the existing
parent on those ties. No held-out metric overrides that decision.

The selected forward associative-divisor endpoint therefore remains at
53.8660% primary accuracy, the sole failed original task/mode. It has no
available admitted readout candidate here. There are no selected accuracy or
CE regressions, no newly selected success, and no unavailable model hidden
behind a success flag. The readout's many-to-one output interpretation does
not identify missing raw aliases; Fibonacci's low raw accuracy remains
separate from its successful declared field metric.

Every prior cost and all staged work remain in the complete summary. The
readout comparison performs no new discovery or head training; it imports
reviewed task259 endpoints and evaluates them. The staged record retains
14.7734 discovery seconds,59.4624 head seconds,131 gradient batches,19 grouped
kernels and7 admitted raw expansions. Forward-only keeps its staged
zero-gradient semantics. This does not make replay or metric computation
free, and no speed or matched-compute claim follows.

All 35 conventional transformer endpoints and336 depth outcomes are retained.
Historical DiffGPT's recorded three-seed35/35 remains distinct. No baseline
is retrained or replaced. Both105-choice policies and all original short/long
splits remain; `original/report.jq` is only a read-only projection and cannot
select models or alter the frozen success flags.

Preparation takes1:14:04, original cases total572.38seconds, their replays
609.45seconds, and complete summary reconstruction1:14:58. The long preparation
and summary include recursive verification of frozen parent comparisons and
staged training. Peak recorded RSS is4338116KiB, about4.14GiB, during summary;
case/replay peaks are3665760/3665736KiB. Every process has a64GiB address-space
limit and the runner checks a16GiB disk reserve. Long verification was allowed
to finish; no completed case was dropped or restarted to change the result.

Reproduce frozen producer `d341fdf86aefcce7aa71ce978821aed4af9ab74a`, reviewed
training `6419ecd237755049ddd0c2bbb9ab715615552f78` and parent findings
`5806f3771da0a281641de195bae2a5e1b5b6c6ab`. The source fingerprint is
`291dac2098e16960708d8bfd0b8ddd48687cc9bc3b94f6df307477e3272b5266`;
the executable hash is
`6a626903f6f269d0fd23a3acb4d02befb226e191e5f0e5216825b64c8db8e0ac`.
The inventory hash is
`13d219bff08179ee2c71e4554cc4d2a34bef780851ec454b63607af466f505fe`;
the full summary hash is
`c23aeac21f399b6b19c0ab536c2e80f944c633e36c00e4daa9a8a525ae978365`;
the report hash is
`ace52a7ff37de5b24972278d249d76b05f8876e47903445765918018edfa8777`.

The432-file completed manifest binds every original artifact and the report.
Ignored evidence is on storage in `transformer-runs/convex_stress_readout_compare/`;
the executable/request/protocol are in `transformer-runs/freezes/readout-compare-original-v1/`.
Use the parent README's frozen native commands to reproduce; generate the
interpretation with:

```sh
prlimit --as=68719476736:68719476736 -- jq \
  -f experiments/convex_stress_readout_compare/original/report.jq ORIGINAL_ROOT/summary.json
```

Exact-commit review must rerun all35 original replays, reconstruct the complete
summary including every stage/cost, compare the report exactly, and rerun
native/Lean/axiom/index/source checks before task261 approval. Task267 remains
behind that review barrier. This branch does not close the forward-only gap.
