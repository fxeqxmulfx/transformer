/-
# Metastability — the within-cap variance (`rem: variance` of 2410.06833v1)

The remark closing §2: inside a cap `𝒮_q(2ε)` the concentration

  `η_q(t) = min_{x_i(t) ∈ 𝒮_q(2ε)} ⟨x_i(t), w_q⟩`

grows at a rate controlled by the attention-weighted variance of the cap
around the token that attains the minimum, up to the leakage `n e^{-(1-α)β}`
from outside.

`η_q` and the reference token `i(t)` are not free here: `IsCapMin` pins them
down as the minimum and its argmin, exactly as §2 defines them.  Read over an
arbitrary `η` the inequality would be false — `η t = -t` refutes it as soon
as `n e^{-(1-α)β} < 1` — so the pinning is what makes the statement the
paper's.
-/

import Transformer.Metastability.MainTheorem

open scoped BigOperators Classical
open Real

namespace Transformer
namespace Metastability

variable (d n : ℕ)

/-- `η` is the within-cap minimum of `⟨·, w⟩` and `i` attains it, as §2
defines them:

  `η_q(t) = min_{x_i(t) ∈ 𝒮_q(2ε)} ⟨x_i(t), w_q⟩`,
  `i(t) ∈ argmin_{i : x_i(t) ∈ 𝒮_q(2ε)} ⟨x_i(t), w_q⟩`.

Source: arXiv:2410.06833v1, §2, `sec: direct.proof`. -/
def IsCapMin (ε : ℝ) (X : ℝ → SphereTuple d n) (w : SSphere d)
    (η : ℝ → ℝ) (i : ℝ → Idx n) : Prop :=
  ∀ t : ℝ,
    X t (i t) ∈ sphericalCap d w (2 * ε) ∧
    η t = inner (𝕜 := ℝ) ((X t (i t) : EucSpace d)) ((w : EucSpace d)) ∧
    ∀ j : Idx n, X t j ∈ sphericalCap d w (2 * ε) →
      η t ≤ inner (𝕜 := ℝ) ((X t j : EucSpace d)) ((w : EucSpace d))

/-- **Remark (rem: variance).** *Variance inequality.*

  `η̇_q(t) ≥ η_q(t) Σ_{j : x_j ∈ 𝒮_q(2ε)} a_{i(t)j}(t) ‖x_j - x_{i(t)}‖²/2
              - n e^{-(1-α)β}`,

the subtracted term being the leakage from the tokens outside the cap.

The paper reads the sum as the variance of the cap around `x_{i(t)}` and
observes that it therefore governs the rate of convergence inside the cap;
its monotonicity, it notes, is not straightforward and a cap splitting into
two separated sub-caps makes it rise and then fall.

Not proved here.

Source: arXiv:2410.06833v1, §2, `rem: variance`. -/
theorem variance_inequality (β α ε : ℝ) (X : ℝ → SphereTuple d n)
    (w : SSphere d) (η : ℝ → ℝ) (i : ℝ → Idx n) (Tesc : ℝ)
    (hX : Perspective.SA d n β X) (hη : IsCapMin d n ε X w η i) :
    ∀ t : ℝ, 0 ≤ t → t ≤ Tesc →
      η t * (∑ j : Idx n,
          if (X t j) ∈ sphericalCap d w (2 * ε) then
            attn d n β X t (i t) j
              * ‖((X t j : EucSpace d)) - ((X t (i t) : EucSpace d))‖ ^ 2 / 2
          else 0)
        - (n : ℝ) * Real.exp (-((1 - α) * β))
      ≤ deriv η t := by
  sorry

/-- The hypotheses of `variance_inequality` are satisfiable: one token
sitting at `basePoint 0`, which solves `SA`, with the cap centred on it, so
that `η ≡ 1` is the within-cap minimum and the token attains it. -/
example (β : ℝ) :
    Perspective.SA 1 1 β (fun _ _ => Transformer.basePoint 0) ∧
      IsCapMin 1 1 (1 / 32) (fun _ _ => Transformer.basePoint 0)
        (Transformer.basePoint 0) (fun _ => 1) (fun _ => 0) := by
  have hx : ‖((Transformer.basePoint 0 : SSphere 1) : EucSpace 1)‖ = 1 :=
    mem_sphere_zero_iff_norm.mp (Transformer.basePoint 0).2
  have hxx : inner (𝕜 := ℝ) (((Transformer.basePoint 0 : SSphere 1)) : EucSpace 1)
      (((Transformer.basePoint 0 : SSphere 1)) : EucSpace 1) = 1 := by
    rw [real_inner_self_eq_norm_mul_norm, hx]; ring
  have hcap : (Transformer.basePoint 0 : SSphere 1)
      ∈ sphericalCap 1 (Transformer.basePoint 0) (2 * (1 / 32)) := by
    show (1 : ℝ) - 2 * (1 / 32) ≤ _
    rw [hxx]; norm_num
  exact ⟨Perspective.SA_const_consensus 1 1 one_pos β (Transformer.basePoint 0),
    fun _ => ⟨hcap, hxx.symm, fun _ _ => le_of_eq hxx.symm⟩⟩

end Metastability
end Transformer
