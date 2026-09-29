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

**What became of the Corollary.**  It is stated for every `d` and every `s ≠ 0`, and it is
false in both parameters, so this file defines the algorithm and states no theorem about it.
Over `s`: a large clipping factor overflows the E4M3 headroom the EDEN correction needs
(`Transformer.Quartet.not_mean_rhtInv_msEden`), and a small one rounds every entry to `0`,
which the paper's own "Guarantees" paragraph excludes (`Q(x) ≠ 0`).  Over `d`: inside the
window `6 · 16/17 ≤ s ≤ (1/0.93) · 6 · 16/17` of clipping factors the paper uses, the mean of
`MS-EDEN` is off by the factor `101/102` at every dimension `16 · 2^k`
(`Transformer.Quartet.not_mean_rhtInv_msEden_window`); the EDEN guarantee §3.2 quotes for the
`RHT` is a limit `d → ∞`.
-/

import Transformer.Quartet.Section3_NVFP4
import Transformer.Quartet.Hadamard
import Mathlib.MeasureTheory.Constructions.Pi
import Mathlib.MeasureTheory.Measure.Lebesgue.Basic

namespace Transformer
namespace Quartet

variable {k : ℕ}

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

end Quartet
end Transformer
