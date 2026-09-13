# todo2 — a Quake bot

One agent, trained once, that plays duel on maps it has not seen.  The engine
is Quake III (ioquake3), chosen because it is the only candidate with source:
headless stepping, reset, determinism, many instances, no anti-cheat.

This is not a formalization task.  It is written here because §5 and §6 record
what `src/` does and does not supply to it, and those two sections are the
only ones that can be checked against the build.  Everything else is an
engineering plan and carries no proof.

## 0. The task, stated so that it can fail

- **Target.**  1v1 duel against the engine's built-in bots at a fixed skill,
  on maps held out of training.  Generalization across maps is the objective,
  not the record on any one map.
- **Not the target.**  Beating a human, teams, pixels, The Finals.  A bot that
  wins on its training maps and loses on held-out ones has failed, even at
  100% win rate on the former.
- **The premise under test.**  That attention over a match's event history
  beats a recurrent core at maintaining belief over the unobserved.  This is a
  hypothesis, not a design decision: see §3.

## 1. Harness

No Lean in this section.  It is the majority of the work and the only part
that is certain to be needed.

- [ ] Headless dedicated server, stepped from the outside, resettable to a
      fixed state, deterministic under a fixed seed and input sequence.
- [ ] `N` parallel instances.  Gate: measure env-steps/sec/core before writing
      any model code, and record the number here.  If it is below ~10^4, the
      plan changes and no architecture will save it.
- [ ] Observation extraction from engine state, filtered to what the player
      could see (potentially-visible set plus a trace), with the *unfiltered*
      state also exported for the critic (§4).
- [ ] Action interface: view angles, movement axes, buttons, at the server's
      own tick rate.
- [ ] Opponent: the built-in bots, skill level as a curriculum axis.
- [ ] Map corpus split into train and held-out.  Count the maps; the number of
      distinct training maps is the hyperparameter that decides §0, and it is
      not in any config file.

## 2. The observation

The decision that determines everything downstream, as the track-token
definition would have in a racing bot.

- [ ] **Event token.**  One token per event, never per tick: item seen, item
      taken, damage taken from a direction, enemy sighted, death, timer
      expiry.  A ten-minute match must compress to hundreds of tokens.
- [ ] **Map tokens.**  Static geometry, precomputable once per map — the
      engine's own navigation data if it can be read out.  Unlike a
      destructible-environment game, nothing invalidates these mid-match.
- [ ] **Index-independence, as a hard rule.**  No absolute tick index, no
      absolute world coordinate, no map identifier in any token field.  Only
      relative quantities: elapsed time since, displacement from, direction
      to.  The rule is what `RASPL.Conjecture` conjectures to govern
      generalization, and violating it is the most likely single cause of a
      bot that memorizes maps.  Note that it is a *conjecture*
      (`RASPGeneralizationConjecture`), stated as one in `src/`, and is used
      here as a design heuristic, never as a guarantee.

## 3. The experiment that tests the premise

Item timing: predict the opponent's position and armour from sparse
observations.  This is the cleanest instance of belief over unobserved state
the game offers, it is a skill humans name and rank each other by, and it is
scored by a number rather than by a win rate.

- [ ] Ground truth from the engine; input restricted to the filtered
      observation of §1.
- [ ] Two models on identical observations: a two-timescale recurrent core,
      and attention over the event tokens of §2.
- [ ] Metric: prediction error on held-out maps, reported separately from
      training maps.

The published result on this engine (team capture-the-flag, population-based
training) used a recurrent core, not attention — see §8, this is from memory
and must be checked.  So the honest prior is that the recurrent baseline wins,
and the experiment exists to overturn it or to accept it.

Nothing in §4 is worth building before this returns a number.

## 4. Architecture, contingent on §3

- [ ] Fast contour at the server tick rate: aim, dodge, movement.  Reactive,
      shallow, no attention.  It is the simplest part of the problem and must
      not consume the model budget.
- [ ] Slow contour at ~2 Hz: item timing, map control, engage or disengage.
      This is where the winner of §3 goes.
- [ ] Asymmetric actor-critic: the critic sees the unfiltered state, the actor
      only the filtered one.  Under partial observability this is worth more
      than any architectural change, and it is why §1 exports both.

## 5. What this repository supplies

`Transformer.ALM` is 67 modules and 366 theorems with no `sorry`, no vacuous
statement and no placeholder definition (`INDEX.md`).  Of it, these are the
statements that could bear weight here, each with the condition under which it
applies:

- `ALM.SAHead.SAOutput_eq_softmax_head` — the lookup head is ordinary causal
  self-attention at particular projections.  This is what licenses reading any
  bound below as a bound on a real head rather than on a bespoke operator.
- `ALM.VectorInt.softmax_winner_int_sharp` — for **distinct integer** keys in
  dimension `m`, the winner's softmax weight is at least
  `1 / (1 + 2 e^{-β}/(1 - e^{-3β}))^m`, with no dependence on the number of
  keys.  Applies to a retrieval head over the event memory of §2 exactly when
  its keys are integers and distinct — item identifiers, tick counts.  The
  consequence worth having: the temperature needed for exact retrieval does
  not have to grow with match length.
- `RASPL.Attention.argmax_shift` — hard attention against a shifted score
  matrix returns the largest value *among the selected*, i.e. a masked argmax.
  "The most recent pickup of item `x`" is that statement with the mask
  `item = x` and the value the tick count.  This is the primitive the item
  timer of §3 is built from, and it is proved rather than assumed.
- `ALM.SparseSoftmax.sparse_softmax_output_close` — truncating a softmax to a
  retained set costs, on the head's *output*, at most twice the mass left
  outside times the value spread.  Use it to price any top-`k` memory head.
  Note the limit stated in that file: it prices the answer, not the retrieval
  — nothing there bounds the cost of producing the retained set.
- `ALM.FixedDim.OVHard_needs_growing_dimension` — the hardness barrier for
  exact lookup requires a dimension growing like `log n`; at fixed dimension
  the conjecture behind it is false.  So low-dimensional retrieval keys are
  safe from that barrier by a theorem, not by a remark.
- `ALM.PlanarHead.planar_head_argmax` and `ALM.HullCost.hullQuery_cost_total`
  — one binary search answers any planar head.  **Trigger condition:** this
  buys nothing below roughly `n ~ 10^4` keys, and §2 puts a match at hundreds.
  It becomes relevant only for retrieval across a corpus of matches, which is
  not in this plan.  Do not build it early.

## 6. What this repository does not supply

- **Anything about training.**  No optimization, no sample complexity, no
  generalization bound, no reinforcement learning.  The single largest risk in
  §0 — does one policy transfer across maps — has no theorem here and will not
  acquire one.
- **A connection between head counts and a trained policy.**  `RASP.Expr.heads`
  and `RASP.Expr.layers` count what a written program needs, and
  `RASP.aggregate` is an average over a selected set, not a softmax.  The
  bridge between the two (`aggregate` against `ALM.SAHead` at the selectors
  that are argmax sets) is not written.  Until it is, the head counts say
  nothing about a model that was trained rather than compiled.
- **Compiling a program into the policy's weights.**  The differentiable fast
  path — a compiled timer or path search inside the forward pass, trained
  through — is the most interesting idea in reach, and it is research, not a
  step.  The construction it comes from was compiled analytically, never
  trained, and its authors state that the 2D-head restriction is untested
  under training.  Out of scope for §0; revisit only after §3 and §4 produce a
  working bot.

## 7. Deliberately out of scope

- Pixels.  Engine state with a visibility filter answers the question of §0 at
  a fraction of the cost, and pixels can be added later without invalidating
  anything above.
- Self-play.  The built-in bots are a fixed, reproducible opponent; self-play
  introduces non-stationarity that would confound the measurement of §3.
- Teams, objectives, destructible geometry.  Each is a separate project.

## 8. Claims above that are from memory and must be checked first

Listed separately because the plan depends on them and none has been verified
against a source in this repository.

- [ ] The server's tick rate and whether it can be decoupled from wall clock.
- [ ] Whether the engine's bot navigation data is readable as a map
      representation for §2, and in what form.
- [ ] Whether built-in bot skill levels are stable enough to be a curriculum.
- [ ] The architecture of the published capture-the-flag agent on this engine
      (§3 asserts it was recurrent).
- [ ] Whether a headless build steps deterministically under a fixed seed.
