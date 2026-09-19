/-
# The randomized Hadamard transform, EDEN rescaling, and `MS-EDEN`

arXiv:2601.22813v2, "Quartet II: Accurate LLM Pre-Training in NVFP4 by
Improved Unbiased Gradient Estimation" (ICML 2026), §3.2, §3.3 and
Algorithm 1, together with the Corollary that closes §3.3.

EDEN (Vargaftik et al., arXiv:2108.08842) makes a *biased* quantizer unbiased
by rotating first and rescaling afterwards: with `S = ⟨x,x⟩ / ⟨RHT(x), Q(RHT
x)⟩`, the estimate `S · Q(RHT x)` is co-linear with `RHT x` in expectation.
The obstacle §3.2 names is precision: `S` lands in `[0.94, 1.06]`, while the
smallest relative step of an E4M3 group scale is `1.0625`.  `MS-EDEN`
(Algorithm 1) folds `S` into the group scale by *stochastic* rounding, so that
the scale is right in expectation, and that is all the correction needs.

The rotation itself, and the fact that it is invertible and leaves inner
products alone, are `Transformer.Quartet.Hadamard`.

**What the Corollary claims.**  It is stated for every `d` and every `s ≠ 0`,
while the EDEN theorem it rests on ("Based on Theorem 2.1 of Vargaftik et
al.") only gives unbiasedness in the limit `d → ∞`; Appendix A says as much,
and lists the further compromises of the implementation.  The statement below
is the paper's, at finite dimension.
-/

import Transformer.Quartet.Section3_NVFP4
import Transformer.Quartet.Hadamard
import Mathlib.MeasureTheory.Constructions.Pi
import Mathlib.MeasureTheory.Measure.Lebesgue.Basic

namespace Transformer
namespace Quartet

variable {k : ℕ} {s : ℝ}

/-- The EDEN bias correction of one group, `S_g = ⟨x^RHT_g, x^RHT_g⟩ /
⟨x^RHT_g, x^RTN_g⟩` (Algorithm 1), taken over the `16` entries of an NVFP4
group rather than over a whole rotation group: "Performing unbiasing in smaller
groups allows us to reduce the amount of inter-thread communications". -/
noncomputable def edenScale (s : ℝ) (y : Fin (2 ^ k) → Fin 16 → ℝ) (i : Fin (2 ^ k)) : ℝ :=
  (∑ j, y i j * y i j) / ∑ j, y i j * qRTN s y i j

/-- **`MS-EDEN`** (Algorithm 1), dequantized: rotate, round to nearest, then
fold the EDEN correction into the E4M3 group scale by stochastic rounding with
the coin `u i` of that group.  The per-tensor FP32 scale is untouched. -/
noncomputable def msEden (k : ℕ) (s : ℝ) (x : Fin (2 ^ k) → Fin 16 → ℝ)
    (ε : Fin (2 ^ k) → Fin 16 → Bool) (u : Fin (2 ^ k) → ℝ) (i : Fin (2 ^ k)) (j : Fin 16) : ℝ :=
  let y := rht k ε x
  rtn fp4 (y i j / (groupScaleRTN s y i * tensorScaleRTN s y)) *
      sr fp8 (edenScale s y i * groupScaleRTN s y i) (u i) * tensorScaleRTN s y

/-- The expectation over both seeds of Algorithm 1: the sign seed `ω_RHT` is
uniform over the sign patterns, and the rounding seed `ω_SR` is one coin per
group, uniform on `[0,1]`. -/
noncomputable def mean (k : ℕ)
    (f : (Fin (2 ^ k) → Fin 16 → Bool) → (Fin (2 ^ k) → ℝ) → ℝ) : ℝ :=
  (Fintype.card (Fin (2 ^ k) → Fin 16 → Bool) : ℝ)⁻¹ *
    ∑ ε : Fin (2 ^ k) → Fin 16 → Bool,
      ∫ u in Set.univ.pi fun _ : Fin (2 ^ k) => Set.Icc (0 : ℝ) 1, f ε u

/-- **The Corollary of §3.3: `MS-EDEN` is unbiased.**  "For all `x ∈ ℝ^d` and
scale `s ≠ 0`: `E_{ω_RHT, ω_SR} RHT⁻¹(x̂, ω_RHT) = x`", where `x̂` is the
output of Algorithm 1.  The paper derives it from Theorem 2.1 of Vargaftik et
al. together with the unbiasedness of stochastic rounding, and that theorem
holds only as `d → ∞`; at the finite `d` the statement quantifies over, it is
open here. -/
theorem mean_rhtInv_msEden (hs : s ≠ 0) (x : Fin (2 ^ k) → Fin 16 → ℝ) (i : Fin (2 ^ k))
    (j : Fin 16) : mean k (fun ε u => rhtInv k ε (msEden k s x ε u) i j) = x i j :=
  sorry

/-- The hypothesis of the Corollary is satisfiable at the clipping factor the
paper uses, `s = 6 · (16/17) / 0.93`, the numerical minimizer of the expected
error over `𝒩(0,1)` (§3.3). -/
example : (6 * (16 / 17) / 0.93 : ℝ) ≠ 0 := by norm_num

end Quartet
end Transformer
