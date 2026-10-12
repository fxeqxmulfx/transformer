/-
# Metastability — the staircase questions of 2410.06833v1

The energy-level question of §4 is proved for a nonempty window near
consensus at sufficiently large `β` in `Section4_EnergyWindow`.

The full-trajectory normalized form of §6's `conj: saddle-to-saddle`
is false with uniform convergence on entire open plateaux. The actual
attention counterexample is `saddle_to_saddle_counterexample`, proved in
`Section6_StaircaseCounterexample`. It retains the time-change requirements
and replaces the previous unproved universal assertion of that version.

The printed question uses the energy without the `2β` normalization and
omits the last plateau; its different reading remains documented in
`Section6_Profile` and `EnergyScale`. The counterexample does not refute
convergence on compact subintervals away from the jump times.

The separate question about the prescribed gradient clock is still a
`theorem` with an unproved proof. Its gradient-norm and clock inputs are
predicates of their arguments in `Section6_Profile`, with a complete
stationary witness there. No theorem may rest on this unproved claim.
-/

import Transformer.Metastability.Section4_EnergyWindow
import Transformer.Metastability.Section6_StaircaseCounterexample
