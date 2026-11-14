# Shared categorical encoders

Can a shared encoder across input positions remove an ambiguity that a
complete observed kernel retains? CM40 in `review/flat_spectrum.md` proves
conditional identification after adding actual linear role equations.
This directory contains a training-only diagnostic. It contains no trained
new model, original evaluation adapter or new benchmark score.

The diagnostic reads the saved grouped code from the reviewed task 259
training stage. Row and column names are separate categorical dictionaries.
Only identical token IDs occurring in both dictionaries are aligned. For
every projective direction of the complete two-dimensional code, it checks
observed class separation and whether all aligned row-minus-column codes
have one common offset. That offset remains free; setting it to zero would
break invariance under the original gauges. There is no arithmetic rule for
turning the token names into answers, and no held-out query is evaluated.

| Training-only condition | Surviving projective directions |
| --- | ---: |
| Complete code over the field of 97 elements | 98 |
| Separate all 97 observed answer groups | 97 |
| Also share one row-minus-column offset across 191 aligned tokens | 1 |

The remaining direction is `[1, 36]`, with common offset 62 in the saved
basis. Both original basis coordinates separately have two different role
offsets. These coordinate values depend on the saved basis. Uniqueness here
is a diagnostic about the reviewed training code, not a proof of the unknown
target's global shared-role assumption or a measured benchmark improvement.
The diagnostic trusts the reviewed certificate; it does not replace its
complete-basis audit. A native model must construct and audit the restricted
kernel itself, preserve original record multiplicity, and bind its actual
complex heads, ordinary CE traces and resource limits.

The Lean witnesses use actual matrices. Sharing resolves a previously
unmatched cross cell over the field of three elements. Raw aliases remain
ambiguous under a many-to-one readout. An independent-role target fits the
same observations with a different decoded answer, showing the extra global
premise matters. Over the field of two elements the shared kernel still
refuses the cross cell; the coverage witness explicitly needs nonzero two.
Flatness remains a bound on the complex feature model, not an identification
assumption. Frozen comparisons 255, 261 and 267 and component 270 are unchanged.

From the repository root, reproduce the diagnostic with native `jq` under
the 64 GiB process limit. The saved training input is generated evidence
retained outside temporary worktrees. Its producing source is
`21e86e7c0054ce16397998726b715155be1cbebd`, reviewed at
`6419ecd237755049ddd0c2bbb9ab715615552f78`. Regeneration of the training input uses
the commands and frozen protocol in `../convex_stress_readout_kernel/`.

```sh
stage=experiments/convex_stress_readout_kernel/cohort/runs/training-v1/recurrence-fibonacci-k1-c2-lgt/stage-24-p97.json
printf '%s  %s\n' \
  f35d6ff7c7375e908000f535a27bc4f2ee508a95ca902ef10a6b463d38fb1116 "$stage" \
  | sha256sum -c -
out=/run/media/misha/storage/transformer-runs/role-sharing-diagnostic-v1
mkdir -p "$out"
prlimit --as=68719476736:68719476736 -- \
  jq -e -f experiments/convex_stress_role_sharing/diagnose.jq "$stage" \
  > "$out/diagnostic.json"
LAKE_ARTIFACT_CACHE=false prlimit --as=68719476736:68719476736 -- \
  lake build Transformer.Grokking.FlatSpectrum.CoupledMoments.SharedRoles
```

The script deliberately caps its field at 997 and each dictionary at 10,000
entries and requires dimension two. These are diagnostic resource limits,
not claims that a higher-dimensional native model is impossible. All outputs
remain local; only the generator, pins and interpretation are committed.
