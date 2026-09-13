# todo2 — theorems for a Quake bot

The task these are for, in one paragraph, and only so that the statements
below have a referent: one policy that plays ioquake3 duel on maps held out
of training, reading a sequence of *event* tokens — item seen, item taken,
damage from a direction, enemy sighted, death — never one token per tick, and
carrying no absolute tick index and no absolute world coordinate in any token
field.  The head that matters is a retrieval over that event memory; the
computation that matters is the item timer: find the last pickup of an item,
add its respawn interval, compare against now.

The harness, the observation format, the training loop and the experiment
that would decide between attention and a recurrent core are engineering and
are not in this file.  What is here is only what can be checked against the
build.

§0 comes first because it is not this task's invention: it is the list of
claims the ALM authors make and do not prove, and several of them are the same
statements the bot needs.  §1-§3 then say what the repository supplies, what it
does not, and what is missing.

## 0. Do first: what the ALM authors asserted and did not prove

Source: the two Percepta posts of 2026-03-11, recovered under `papers/`
(gitignored) from https://www.percepta.ai/blog/can-llms-be-computers and its
companion.  The companion says plainly, "A formal write-up with the key theory
will be released shortly" — so every claim below is prose and released code,
with no proof behind it anywhere.  Quotes are verbatim from the recovered text,
whose header records what did not survive the extraction.

The released implementation is `transformer-vm/` (gitignored), read 2026-09-14.
Every constant quoted below comes from it, because the posts give none.  What is
wrong with that code *as code*, rather than unproved, is `todo3.md`.

Ordered by what is both provable and load-bearing here.

- [ ] **Softmax approximates hard-max — with which constant?**  "Standard
      softmax attention approximates hard-max when the scores are scaled up by
      **a large constant**, so the same construction carries over with
      exponentially small approximation error."

      The constant is `HARD_K = 1e10` (`model/weights.py:22`), and at that
      value the claim fails in their own headline regime.  Latest-write
      separates consecutive positions by only
      `δ(p) = 0.3 (1/log(p+2) − 1/log(p+3)) ≈ 0.3/(p log² p)`, so `HARD_K · δ`
      is 226 at `p = 10^5`, **15.7** at `10^6` and **1.16** at `10^7`, and the
      mass left off the winner, `(n−1) e^{−βδ}`, goes ~0 → 0.15 → above 1
      across those three.  "Millions of steps" is exactly where it breaks, and
      no path in the release ever runs a softmax (todo3 §1).

      `ALM.SoftmaxValue` already has that `(n-1) e^{-β} C` shape: the error
      carries the trace length, so `β` must grow like `log n + log(1/δ)` and is
      not a constant.  The constant-in-`n` form exists —
      `ALM.VectorInt.softmax_winner_int_sharp` — but needs **distinct integer**
      keys and pays the dimension in the exponent, a hypothesis the post never
      states and the perturbation below destroys on purpose.

      This is T1 of §3.  Doing it settles the authors' claim and the bot's head
      in one statement.

- [ ] **Latest-write by perturbation: the gap it destroys.**  "we add a small
      position-dependent perturbation to each key.  Among entries with the same
      logical key, the latest one then scores strictly highest."

      The perturbation is `0.3 · (1/log 2 − 1/log(p+2))` added to the key's
      `y` coordinate (`graph/core.py`, `LATEST_ALPHA`, `_to_2d_key`).  A
      perturbation that separates `n` positions shrinks the score gap `δ`, and
      the softmax error is `e^{-βδ}` at the `β` of the item above.  The two
      claims pull in opposite directions and the post never multiplies them
      together.

      "Strictly highest" is also false once the key is large: the perturbation
      is added to `-k²` in float64 and survives only while
      `p log² p · k² ≲ 1.4·10^15`, so at `k = 10^5, p = 10^4` two consecutive
      writes to one key are already bit-identical.  What implements latest-write
      in the released system is an integer sequence number inside the data
      structure, outside the model entirely (todo3 §2).

      What to prove, and the gap is bigger than it looked.  `ALM.SoftmaxLatestMass`
      already bounds the softmax head against the log's latest resolution, and its
      `‖V b − V c‖ / 2` term is tight rather than slack — at a tie the head returns
      the mean and the mean is not the later payload.  But that covers less of the
      machine than its name suggests:

      - `softmax_head_resolves_latest_of_int` assumes `Function.Injective K`, and
        `ALM.SoftmaxTieInt.sScore_tie_gap_one` needs it: the proof runs through
        `2q = K c + K b`, so the tie it is about is a *reflection* tie between two
        distinct keys straddling the query — the miss of todo3 §3, not two writes
        under one key.  The duplicate-key case is covered only by the general
        `softmax_head_resolves_latest_of_gap`, where the caller supplies `δ = 1`
        by hand.
      - Neither version reaches three or more writes under one key, which is what
        a memory byte ordinarily gets.  For `m` copies the head returns their mean
        and the error is `‖mean − V_last‖`, up to `(1 − 1/m)` of the payload
        spread; what is needed is the same statement with the gap hypothesis taken
        over the complement of the tie *set* rather than of a pair.
      - And the claim that latest-write cannot live in the weights at all is
        currently one theorem plus one measurement: the theorem is about an exact
        tie, which the perturbation is designed to prevent, and what defeats the
        perturbation is arithmetic (todo3 §2).  The missing theorem is the grid
        budget — any key encoding that separates `S` recencies multiplies the key
        magnitude by `S`, so `fp_eval_exact_of_grid` caps `k · S < √(2^53)`, which
        is 95 logical keys on a `10^6`-token trace.  That is what would make the
        obstruction a theorem rather than a table.

- [ ] **The ceiling nobody states: where float64 ends.**  Neither post bounds
      the trace length, and the code carries no assertion.  There is a bound:
      the score `2qk − k²` is an integer, and float64 holds integers exactly
      only below `2^53`, so exact retrieval needs `n < √(2^53) =` **`94 906 266`**.

      Measured, and the bound is attained to the unit: computing the score on
      the integer grid, the first query that loses to its own neighbour is
      `q = 94 906 266 = ⌈√(2^53)⌉`.  Their arithmetic does not reach it — the
      `HARD_K·√2` scaling takes the score off the grid and costs 0.86 of a bit,
      so the shipped head fails from `q = 52 301 885` (2^25.64) and is 2.1 %
      wrong at `n = 10^8`.  The Sudoku demo is ~5.4·10^6 tokens, a factor of 10
      below the shipped wall and 18 below the grid one.

      The smallest item in this file and the only one that ends in a number.
      `ALM.FloatGrid.fp_eval_exact_of_grid` is the tool: it says a float
      evaluation is exact on a grid, and what is wanted is that read against
      grid spacing `1` at magnitude `n²`.

- [ ] **The queries are not integers, and the margin is one ulp.**
      "`p` is known from the embedding, multiplying that average by `p`
      recovers the **exact** cumulative sum … This is how quantities such as the
      instruction pointer, stack depth, and call-stack depth are maintained
      exactly over time."

      It is not exact.  The head divides (`HullMeta::resolve`: `inv = 1.0/count;
      out = vsum*inv`) and the FFN multiplies back, and `fl(fl(s·fl(1/p))·p) = s`
      fails for **25.8 %** of round trips — first at `s = 3, p = 5` — and at
      `p ≈ 10^6` **37.3 %** of the cumulative sums are not integers.  Since
      `cursor` is one of them and `5·cursor + 1` is the query of the instruction
      fetch (`wasm/interpreter.py:316,320`), the `q : ℤ` hypothesis of
      `fp_exact_of_grid` is false on the machine's main path.

      Nothing is wrong with the construction — the division is forced, because
      the alternative recurrence costs a layer per token (todo3 §8).  What is
      wrong is that exactness is the wrong statement for this path.  The right
      one is a margin: a lookup whose query is within `ε` of an integer still
      returns the same key, provided `2kε + ρ < 1/2`.  `ALM.FloatHull.cmp_of_sep`
      is already that shape — `|a'−a| ≤ δ₁`, `|b'−b| ≤ δ₂`, `δ₁+δ₂ < |a−b|` — and
      what is missing is its instantiation at `δ = ulp(q)·k`, which turns the
      measured fact that one ulp is survivable and two are not into a theorem.

      Small, self-contained, and it is the hypothesis every other exactness
      result in `ALM.FloatGrid` silently assumes.  Do it directly after the
      ceiling above; together they say how long the machine actually runs.

- [ ] **Differentiability through the executed program.**  "Because the
      execution trace is part of the forward pass, the whole process remains
      differentiable: we can even propagate gradients through the computation
      itself."

      By their own statement, "our construction uses hard-max attention".
      Gradients flow through the retrieved *values*; through *which key was
      retrieved* there is none.  The released code does not even pose the
      question: generation is `@torch.no_grad()`, picks tokens with
      `.argmax().item()`, and passes keys through `.numpy()` into C++
      (`model/transformer.py:41,69`, `attention/hull_cache.py:48`) — three
      independent severings of the autograd graph.  Substituting softmax at
      large `β` to recover a gradient makes the off-winner weights `e^{-βδ}` by
      the bound above — the same `β` that makes execution exact makes the
      training signal vanish.  What to prove: that trade-off as an inequality,
      gradient magnitude against retrieval error, at one lookup head.

      Likely a negative result, which is why it comes early: it decides whether
      the compiled-fast-path branch of §2 is worth anything.

- [ ] **The cost of `k`-sparse retrieval.**  "it is easy to approximate it with
      k-sparse softmax attention: retrieve the top-`k` keys and perform the
      softmax only over those.  By storing points across nested convex hulls,
      this yields a decoding cost of …", and alongside it the admission "we do
      not yet know whether exact softmax attention can be maintained with the
      same efficiency".

      The nested-hull structure is one sentence, never built and never
      analysed.  `ALM.SparseSoftmax` prices the *answer* exactly — its total
      variation is an equality, not a bound — and says in its own docstring that
      nothing there bounds the cost of producing the retained set.  That cost is
      the open half.

- [ ] **Where the dimension boundary actually is.**  "The same machinery also
      extends naturally to 3D heads via 3D convex hulls, although higher
      dimensions quickly become less efficient", and "The key question is
      whether 2D already captures most of the speedup, or whether slightly
      larger heads unlock substantially more capability."

      No bound is given, and "quickly becomes less efficient" is not the shape
      of it: `ALM.FixedDim.OVHard_needs_growing_dimension` says the barrier
      needs a dimension growing like `log n` and is simply absent at any fixed
      `d`.  What is missing is the query cost at fixed `d ≥ 3` between those two
      regimes.

- [ ] **Turing completeness of ALM.**  "ALMs **can be shown** to be Turing
      Complete", and "for Turing completeness, 2D attention is all you need!"

      "Can be shown" is the entire proof.  Neither post sketches it and it is
      not in this repository.  Large: it needs the ALM machine as a syntactic
      object, which `Transformer.ALM` does not have — every file there is about
      one head, not about a machine.

- [ ] **Compiler correctness.**  "if the compiled solver is correct, the
      transformer's execution is correct as well" — offered as a guarantee that
      "is universal rather than benchmark-specific".

      The chain is CALM semantics → gate graph → MILP schedule → slot
      allocation → weight matrices → float execution, and no link of it is
      proved.  The authors concede this by listing "Formally verifying the logic
      a transformer implements" as future work.  The step most likely to be
      wrong gets one sentence in the post: "When a slot is reused, the stale
      value must be subtracted before the new value is written."

      The largest item by far, and the last: only the final link is in reach
      today (`ALM.FloatHead.fp_head_output` proves the float head against their
      own `hull2d_cht.h`), and the rest needs CALM and the scheduler formalized
      first.  Note also that the last link does not hold as released: their hull
      cache and their own brute-force head disagree on ties (todo3 §3), so
      "the transformer's execution is correct as well" is false before the
      compiler is even reached.

- [ ] **Not a theorem, now or later: 2D heads under training.**  "we find that
      it is still flexible enough to train efficiently and can capture very
      complicated logic", hedged in the same paragraph by "we are still
      exploring how limiting this is in practice for training" and "The real
      question is how capable such models become when trained at scale".

      Every weight in both posts was *compiled*, never trained, so there is no
      evidence in them either way.  Listed only so that it is not mistaken for
      an open problem someone could close here: it is an experiment, and §2
      applies to it.

## 1. What this repository supplies

`Transformer.ALM` is 67 modules and 366 theorems with no `sorry`, no vacuous
statement and no placeholder definition (`INDEX.md`).  Of it, these are the
statements that could bear weight, each with the condition under which it
applies:

- `ALM.SAHead.SAOutput_eq_softmax_head` — the lookup head is ordinary causal
  self-attention at particular projections.  This is what licenses reading any
  bound below as a bound on a real head rather than on a bespoke operator.
- `RASPL.Attention.argmax_shift` — hard attention against a shifted score
  matrix returns the largest value *among the selected*, i.e. a masked argmax.
  "The most recent pickup of item `x`" is that statement with the mask
  `item = x` and the value the tick count.  This is the primitive the item
  timer is built from, and it is proved rather than assumed.  Note that its
  statement is pure arithmetic — `v : Fin m → ℕ`, a mask `P`, a bound `C` —
  with no RASP syntax in it, which is why it applies to a head whose scores
  were trained rather than written.
- `ALM.SparseSoftmax.sparse_softmax_output_close` — truncating a softmax to a
  retained set costs, on the head's *output*, at most twice the mass left
  outside times the value spread.  Use it to price any top-`k` memory head.
  Note the limit stated in that file: it prices the answer, not the retrieval
  — nothing there bounds the cost of producing the retained set.
- `ALM.FixedDim.OVHard_needs_growing_dimension` — the hardness barrier for
  exact lookup requires a dimension growing like `log n`; at fixed dimension
  the conjecture behind it is false.  So low-dimensional retrieval keys are
  safe from that barrier by a theorem, not by a remark.
- `ALM.VectorInt.softmax_winner_int_sharp` — for **distinct integer** keys in
  dimension `m`, the winner's softmax weight is at least
  `1 / (1 + 2 e^{-β}/(1 - e^{-3β}))^m`, with no dependence on the number of
  keys.  Distinctness is a real hypothesis and it is not satisfied by a masked
  argmax, where many positions share a score by construction; see T1.
- `ALM.PlanarHead.planar_head_argmax` and `ALM.HullCost.hullQuery_cost_total`
  — one binary search answers any planar head.  **Trigger condition:** this
  buys nothing below roughly `n ~ 10^4` keys, and a match's event memory is
  hundreds.  It becomes relevant only for retrieval across a corpus of
  matches.  Do not build it early.

## 2. What this repository does not supply

- **Anything about training.**  No optimization, no sample complexity, no
  generalization bound, no reinforcement learning.  The largest risk in the
  task — whether one policy transfers across maps — has no theorem here and
  will not acquire one.
- **Anything that needs a hand-written program.**  `Transformer.RASP` splits
  in two, and only one half is of any use here.  `Selector`, `selected` and
  `aggregate` are plain definitions — a `Fin n → Fin n → Bool` and a mean over
  a `Finset` — and T1 uses `aggregate` as a name for the right-hand side, which
  costs nothing.  `Expr`, `eval`, `heads` and `layers` are compilation
  bookkeeping: they count what a program *someone wrote* needs, and a trained
  policy is not an `Expr`, so no induction over that syntax reaches it.  The
  budget theorems (`layers_le_heads`, `layers_lt_agg_value`) are therefore
  inert for this task.  They stop being inert only on the branch below, where
  the program is written and compiled rather than trained — which is the only
  honest reason to keep RASP in view at all.
- **Compiling a program into the policy's weights.**  The differentiable fast
  path — a compiled timer or path search inside the forward pass, trained
  through — is the most interesting idea in reach, and it is research, not a
  step.  The construction it comes from was compiled analytically, never
  trained, and its authors state that the 2D-head restriction is untested
  under training.

## 3. The theorems the task needs

Three, in order of what they buy.  Everything in §1 is about *one head in
isolation*; none of it survives contact with a program of several heads, and
that gap is what this section names.

- [ ] **T1 — the bridge: a softmax head computes `aggregate`.**

      `‖softmaxHead β s val i − RASP.aggregate S val d i‖ ≤ 2 (n − |selected S i|)
      e^{−βδ} · spread`, where `selected S i` is the argmax set of the score
      row `s i ·` and `δ` its gap to the runner-up.

      This is a softmax-to-hardmax bound and nothing more grand; `aggregate`
      appears only because it is already the name in this repository for
      "uniform mean over the selected set".  A softmax at large `β`
      concentrates on the argmax set and averages it uniformly — so the two
      agree in the limit, ties included.  It subsumes
      `RASP.aggregate_of_selected_eq_singleton` (the case `|selected| = 1`)
      and prices the case that lemma does not cover.

      Mostly assembled already: `ALM.SparseSoftmax.sparse_softmax_output_close`
      with `T := selected S i`, and `softmax_mass_outside_le` for the mass.
      What it costs: with integer scores `δ ≥ 1`, so `β ≥ log n + log(1/ε)`
      suffices — the temperature grows like the *logarithm* of the event
      count, which for a match of ~10³ tokens is `β ≈ 10`.  That number is the
      answer to "does the head still work at the end of a long match", and it
      cannot be claimed before T1 is proved.

      It also settles the first item of §0: the published claim is that a large
      *constant* suffices, and `log n` is not a constant.

- [ ] **T2 — error does not accumulate with depth.**

      Per-layer error `ε`, depth `L`: naively `C^L ε`, which makes every bound
      in §1 worthless for a program of more than one head.  The way out is that
      the intermediate values are integers — tick counts, item identifiers —
      so: if each layer's exact output lies on a lattice of gap `δ` and the
      approximation is within `δ/2`, a threshold read-out recovers it exactly,
      and a depth-`L` program is exactly right whenever every layer satisfies
      T1 with `ε < δ/2`.  Consequence: `β ~ log n`, *independent of `L`*.

      Hypothesis that must be discharged by the architecture, not by the proof:
      that the read-out is a threshold.  Without T2, the item timer — last
      pickup, plus respawn interval, compared against now — is three heads deep
      and has no error bound at all.

      Not the same as the C-RASP depth hierarchy (`src/Transformer/CRASP/`,
      arXiv:2506.16055): that is about what depth `L` can and cannot express,
      this is about numerical error in a program that is already expressible.

- [ ] **T3 — equivariance, in place of the generalization conjecture.**

      `RASPL.Conjecture.RASPGeneralizationConjecture` is a conjecture and must
      remain one, so the map-transfer claim will never be a theorem.  What is
      provable is the invariance that the rule on absolute indices and
      coordinates is really about, and it has to be stated about the *head*,
      not about a syntax: for the group `G` generated by time translation, yaw
      rotation and permutation of simultaneous events,

          score (g · q) (g · k) = score q k   for all g ∈ G,

      together with the corresponding invariance of the token encoding.  That
      is a property of the learned projections, checkable on a trained model,
      and it turns "no absolute index" from a style rule into a condition with
      a statement.

      What this replaces: an earlier version proved the same closure by
      induction over `RASP.Expr`, with `Expr.indices` as the one primitive that
      breaks it.  True, and about the wrong object — the bot's encoder is not
      an `Expr`.

- [ ] **Nothing for training.**  Not a gap to be filled later — there is no
      route from anything in `src/` to a statement about optimization, sample
      complexity or transfer, and §2 should be read as permanent.
