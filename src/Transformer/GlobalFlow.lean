/-
# Global flows of autonomous ODEs

An autonomous field on a finite-dimensional space that is locally Lipschitz
and grows at most linearly, `‖F x‖ ≤ C ‖x‖`, has a unique solution through
each point, defined for all time.  `GlobalFlow/Basic.lean` holds the radial
retraction onto a ball and the Grönwall a priori bound;
`GlobalFlow/Existence.lean` holds existence and uniqueness on `ℝ`.

Source: the standard proof of global existence under linear growth, as used by
arXiv:2305.05465v6, `p:wellposedparticles`.
-/

import Transformer.GlobalFlow.Basic
import Transformer.GlobalFlow.Existence
