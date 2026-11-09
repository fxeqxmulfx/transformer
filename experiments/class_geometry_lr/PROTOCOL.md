# Higher peak learning rates with the fixed main-model recipe

Task #201, frozen 2026-11-09 before any higher-rate candidate training. The
user asks whether more learning rate is needed while task #200 is running.
The task #200 seed-7 endpoint regression was already observed; this is a new
follow-up comparison, not a retrospective change to that protocol.

Use the exact task #200 model, data, initialization, sampler, FP32 AdamW
kernels, beta1=.9, beta2=.95, epsilon1e-8, independent-gradient clip1 and
weight decay .1. Change only the peak learning rate from .001 to each of
the two fixed values .003 and .01. Keep the final LR1e-8 and the schedule's
zero-based 2% warmup, 78% stable interval, 20% cosine cooldown. No other
hyperparameter, numerical tolerance, model feature or data split changes.

Run .003 then .01, each on all seeds [7,11,1,3,5,13,17,19] in that order,
10000 updates each, with the same 31 probes, ranks [0,32,33,128], 21 late
probes and current-state certificate rules. Each arm has a fixed1800-second
cap, including every failure, capped and unrun slot. No early success stop,
replacement seed, continuation from another arm or increased cap. Run on
the GPU sequentially after #200 finishes, retaining its .001 cohort intact.

Use separate Cargo feature builds `lr003` and `lr01` of the shared
`class_geometry_wsd` native source, with mutually exclusive peaks and the
correct protocol embedded. Freeze the peak in report metadata and the source
fingerprint. Require all paired initialization, training, validation and
sampler hashes to match the .001 report before measuring outcomes.

Run the same16-case GPU preflight at each peak before training. Preserve
the4-ulp parameter/moment comparison and2e-6 independent norm allowance;
all actual trajectory probes also compare CPU/GPU optimizer steps. All
update energies, prefix identities, damping signs, rounding limits and
FP64 query-margin checks retain #200's numerical allowances. Any failure
is reported without adjusting allowances from outcomes.

Report all8paired endpoints and late minima against .001, every50/90/99
crossing and all regressions. Primary metrics are final validation accuracy,
final rank-32 certificate coverage, and minimum rank-32 coverage over all
21late probes. No single winner is declared from a selected seed or metric;
present tradeoffs if faster discovery sacrifices coverage or persistence.
The completed constant-rate #199 cohort remains separate background context.
All results concern this fixed task/budget and do not establish convergence.

Commit native source, protocols, commands and authored findings. Keep generated
evidence ignored under root `experiments/class_geometry_lr/runs/` and
`.kanban/artifacts/201/`. No Python, no changes to the external GPT repository,
and no changes to the current running #200 executable.
