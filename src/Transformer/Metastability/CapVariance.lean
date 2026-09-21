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

What is proved here is the envelope argument — below the escape time `η_q`
touches the height of the attaining token from below, so its derivative is
that token's velocity — and nothing else: the estimate on the velocity itself
is `Metastability.inner_proj_softmax_ge`, a statement about a plain tuple of
unit vectors.
-/

import Transformer.Metastability.MainTheorem
import Transformer.Metastability.CapVelocityBound
import Mathlib.Analysis.InnerProductSpace.Calculus

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

**What the source says and what is changed here.**  Three things.

*The `α`-separation is a hypothesis.*  The remark is stated with no relation
between `α` and the configuration, and without one the leakage term is not a
bound on anything: nothing makes the weight `a_{i(t)j}` of a token outside the
cap small.  `h_far` supplies what the leakage estimate needs and what the
section's other statements assume — `rho_diff_ineq` carries the same
hypothesis — namely `⟨x_{i(t)}, x_j⟩ ≤ α` for `x_j` outside the cap on
`[0, T_esc]`.  Together with `β ≥ 0` it gives
`a_{i(t)j} ≤ e^{βα}/e^{β} = e^{-(1-α)β}`, the partition function being at
least its own diagonal term.

*The constant is the paper's `n`.*  A token outside the cap contributes to
the gap sum and to the variance sum together
`a_{i(t)j} ⟨x_j, w - η_q x_{i(t)}⟩`, and `‖w - η_q x_{i(t)}‖ = √(1 - η_q²) ≤ 1`
bounds this below by `-a_{i(t)j} ≥ -e^{-(1-α)β}`; there are at most `n` such
tokens.

*`η_q` is differentiable and the reference token does not leave the cap.*  The
minimum of finitely many smooth functions has corners, so `hη_diff` is a
hypothesis exactly as in `rho_diff_ineq`; and the envelope argument that
identifies `η̇_q` with the derivative of the attaining token needs that token to
stay admissible for nearby times, which is `h_stay`.  Below the escape time
that is what `T_esc` means; stated for an arbitrary `i` it has to be assumed.

Source: arXiv:2410.06833v1, §2, `rem: variance`. -/
theorem variance_inequality (β α ε : ℝ) (hβ : 0 ≤ β)
    (X : ℝ → SphereTuple d n)
    (w : SSphere d) (η : ℝ → ℝ) (i : ℝ → Idx n) (Tesc : ℝ)
    (hX : Perspective.SA d n β X) (hη : IsCapMin d n ε X w η i)
    (hη_diff : ∀ t : ℝ, 0 ≤ t → t ≤ Tesc → DifferentiableAt ℝ η t)
    (h_stay : ∀ t : ℝ, 0 ≤ t → t ≤ Tesc →
      ∀ᶠ s in nhds t, X s (i t) ∈ sphericalCap d w (2 * ε))
    (h_far : ∀ t : ℝ, 0 ≤ t → t ≤ Tesc → ∀ j : Idx n,
      X t j ∉ sphericalCap d w (2 * ε) →
      inner (𝕜 := ℝ) ((X t (i t) : EucSpace d)) ((X t j : EucSpace d)) ≤ α) :
    ∀ t : ℝ, 0 ≤ t → t ≤ Tesc →
      η t * (∑ j : Idx n,
          if (X t j) ∈ sphericalCap d w (2 * ε) then
            attn d n β X t (i t) j
              * ‖((X t j : EucSpace d)) - ((X t (i t) : EucSpace d))‖ ^ 2 / 2
          else 0)
        - (n : ℝ) * Real.exp (-((1 - α) * β))
      ≤ deriv η t := by
  intro t ht0 htT
  obtain ⟨-, hηt, hmin⟩ := hη t
  have hnorm : ∀ j : Idx n, ‖((X t j : EucSpace d))‖ = 1 :=
    fun j => mem_sphere_zero_iff_norm.mp (X t j).2
  -- the attaining token moves with the `SA` velocity
  have hvel : HasDerivAt (fun s => ((X s (i t) : EucSpace d)))
      (proj d ((X t (i t) : EucSpace d))
        ((∑ l : Idx n, Real.exp (β * inner (𝕜 := ℝ) ((X t (i t) : EucSpace d))
            ((X t l : EucSpace d))))⁻¹ •
          ∑ j : Idx n, Real.exp (β * inner (𝕜 := ℝ) ((X t (i t) : EucSpace d))
            ((X t j : EucSpace d))) • ((X t j : EucSpace d)))) t := hX t (i t)
  have hFderiv : HasDerivAt
      (fun s => inner (𝕜 := ℝ) ((X s (i t) : EucSpace d)) ((w : EucSpace d)))
      (inner (𝕜 := ℝ)
        (proj d ((X t (i t) : EucSpace d))
          ((∑ l : Idx n, Real.exp (β * inner (𝕜 := ℝ) ((X t (i t) : EucSpace d))
              ((X t l : EucSpace d))))⁻¹ •
            ∑ j : Idx n, Real.exp (β * inner (𝕜 := ℝ) ((X t (i t) : EucSpace d))
              ((X t j : EucSpace d))) • ((X t j : EucSpace d)))) ((w : EucSpace d))) t := by
    have h := hvel.inner ℝ (hasDerivAt_const t ((w : EucSpace d)))
    simp only [inner_zero_right, zero_add] at h
    exact h
  -- the envelope argument: `η` touches the attaining token's height from below
  have hmax : IsLocalMax
      (fun s => η s
        - inner (𝕜 := ℝ) ((X s (i t) : EucSpace d)) ((w : EucSpace d))) t := by
    filter_upwards [h_stay t ht0 htT] with s hs
    have hs' := (hη s).2.2 (i t) hs
    have hEq : η t
        = inner (𝕜 := ℝ) ((X t (i t) : EucSpace d)) ((w : EucSpace d)) := hηt
    linarith
  have hderiv : deriv η t
      = inner (𝕜 := ℝ)
          (proj d ((X t (i t) : EucSpace d))
            ((∑ l : Idx n, Real.exp (β * inner (𝕜 := ℝ) ((X t (i t) : EucSpace d))
                ((X t l : EucSpace d))))⁻¹ •
              ∑ j : Idx n, Real.exp (β * inner (𝕜 := ℝ) ((X t (i t) : EucSpace d))
                ((X t j : EucSpace d))) • ((X t j : EucSpace d)))) ((w : EucSpace d)) := by
    have hsub := (hη_diff t ht0 htT).hasDerivAt.sub hFderiv
    have h0 : deriv η t
        - inner (𝕜 := ℝ)
            (proj d ((X t (i t) : EucSpace d))
              ((∑ l : Idx n, Real.exp (β * inner (𝕜 := ℝ) ((X t (i t) : EucSpace d))
                  ((X t l : EucSpace d))))⁻¹ •
                ∑ j : Idx n, Real.exp (β * inner (𝕜 := ℝ) ((X t (i t) : EucSpace d))
                  ((X t j : EucSpace d))) • ((X t j : EucSpace d)))) ((w : EucSpace d)) = 0 := by
      rw [← hsub.deriv]
      exact hmax.deriv_eq_zero
    linarith
  rw [hderiv]
  exact inner_proj_softmax_ge d n β α (η t) hβ (fun j => ((X t j : EucSpace d)))
    ((w : EucSpace d)) (i t) hnorm (mem_sphere_zero_iff_norm.mp w.2)
    (fun j => X t j ∈ sphericalCap d w (2 * ε)) hηt hmin
    (fun j hj => h_far t ht0 htT j hj)

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
