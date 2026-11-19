# What short training sequences identify about distance

The distinct-symbol copy model fits every original training and ordinary
test sequence but misses every length-32 and length-64 test sequence.
Can its short training data distinguish different positional rules? CM118
applies a nonnegative coefficient to the absolute distance of each
destination from its source. Both reads use the same effective logits.
All model coefficients are held at their fitted CM116 values during this
training-only intervention. [Original copy results](../rewrite/README.md)

The six Rewrite tasks contain training words of lengths 8 through 16.
Their largest supervised prefix has 34 objects and therefore 33 memory
positions. Only 65 signed offsets, from −32 to 32, occur in its positional
rows. All 192 remaining entries of the 257-entry relative table stayed at
their initial zero in every fit. The lookup overlap exposed distances
through 22 and left 144 relative entries unobserved. For distinct-symbol
copy, the largest trained relative logit at seeds 0, 1 and 2 occurred at
offsets +28, +30 and +31, respectively. Each exceeded the trained logit
at +1.

Each arm reconstructs the complete training logits of all 42 endpoints
from its sealed weights, including every teacher-forced and greedy call.
For each frozen model it subtracts $a|\delta|$ from its signed relative
potential, with $a$ in 0, 1/16, 1/8, 1/4, 1/2, 1 and 2. It records
ordinary teacher-forced CE, correct objects and complete generated
sequences. Two feasible scalar queries at $2^{-16}$ and $2^{-15}$ verify
the analytic CE derivative at zero. The largest absolute difference
was below $1.05\times10^{-8}$. Both archived arms produced identical reports.

| Original training case | Evidence from the distance grid |
| --- | --- |
| Copy, distinct symbols | All seven settings preserved all 128 exact generated sequences and all 1,732 correct objects at every seed. Zero had the lowest CE of the tested settings. |
| Reverse, distinct symbols | The largest exact counts at seeds 0 / 1 / 2 increased from 94 / 97 / 94 to 106 / 101 / 107 of 128 in both methods. The tested positive settings increased teacher-forced CE. |
| Two-hop lookup | The only six fits with a lower CE on the grid were the lookup fits. Their exact counts remained 128/128. |
| Copy or reverse with repeats, both sort tasks | Across these 24 fits, no tested positive setting improved the number of exact training sequences or lowered CE. |

Eighteen of the 42 fits had a negative right derivative at zero, including
fits whose CE at 1/16 already exceeded its original value. At every
length, changing the residual by $t|\delta|$ compensates a change $t$ in
the coefficient. The short-copy observations leave many extrapolations
compatible with perfect training accuracy. A restricted relative
potential with shared coefficients across distances can be tested by a
fresh fit. [Physical bound and identifiability](../../../review/convex/moments.md)

```sh
base=/run/media/misha/storage/transformer-runs
model="$base/composed-retrieval/output-rewrite7-v1"
manifest=experiments/composed_retrieval/Cargo.toml
for arm in original repeat; do
  cargo run --release --locked --offline --manifest-path "$manifest" \
    --features rewrite-cohort --bin distance-decay -- "$PWD" /run/media/misha/storage \
    "$model" "$arm" > "$base/distance-decay-$arm.json"
done
cmp "$base/distance-decay-original.json" "$base/distance-decay-repeat.json"
```
