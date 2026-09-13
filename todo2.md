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
