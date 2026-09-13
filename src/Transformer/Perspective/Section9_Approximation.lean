/-
# §10 — Approximation, control, training

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

The original §10 is a brief survey of known results on universal
approximation, control and training of Transformers.  This file records the
relevant statements as `True`-placeholders, so the structure of the survey is
mirrored faithfully and the file can be extended if/when these results are
formalized in Mathlib.
-/

import Transformer.Basic
import Transformer.Perspective.Section1_IPS

namespace Transformer
namespace Perspective

variable (d n : ℕ)

/-- *Universal approximation (Yun et al.).*  Discrete-time Transformers with
translation parameters can approximate arbitrary continuous maps as the
number of layers tends to `+∞`. -/
theorem universal_approximation_discrete :
    True := by trivial

/-- *Measure-to-measure universal approximation* (Agrachev et al., Adu-Hajj,
Furuya et al.).  Transformers viewed as measure-to-measure flow maps satisfy
a universal approximation property. -/
theorem universal_approximation_measure :
    True := by trivial

end Perspective
end Transformer
