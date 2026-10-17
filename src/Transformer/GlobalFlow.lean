/-
# Complete trajectories of Lipschitz ODEs

An autonomous field on a finite-dimensional space that is locally Lipschitz
and grows at most linearly, `‖F x‖ ≤ C ‖x‖`, has a unique solution through
each point, defined for all time.  `GlobalFlow/Basic.lean` holds the radial
retraction onto a ball and the Grönwall a priori bound;
`GlobalFlow/Existence.lean` holds existence and uniqueness on `ℝ`.
`GlobalFlow/TimeDependent.lean` constructs unique complete trajectories for
bounded time-dependent fields with a uniform spatial Lipschitz constant;
`GlobalFlow/Flow.lean` assembles them into the flow `Φ_{s,t}`, Lipschitz in
the initial point, and `GlobalFlow/FlowDeriv.lean` and
`GlobalFlow/Variational.lean` show it is `C¹` in the initial point when the
field is, with a first-order expansion in the initial time.

Source: the standard proof of global existence under linear growth, as used by
arXiv:2305.05465v6, `p:wellposedparticles`.
-/

import Transformer.GlobalFlow.Basic
import Transformer.GlobalFlow.Existence
import Transformer.GlobalFlow.TimeDependent
import Transformer.GlobalFlow.Flow
import Transformer.GlobalFlow.FlowDeriv
import Transformer.GlobalFlow.Variational
