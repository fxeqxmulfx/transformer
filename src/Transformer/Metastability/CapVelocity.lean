/-
# Metastability — the softmax velocity, tested against a direction
  (behind `rem: variance` of 2410.06833v1)

The computation the remark of §2 rests on.  Tested against a fixed direction
`w`, the `SA` velocity of a token splits into the attention-weighted gap to
the other tokens and the attention-weighted variance around it
(`inner_proj_softmax_eq`); bounding the first sum below by the leakage from
the `α`-separated tokens and the second by its restriction to a subset gives
the remark's inequality (`inner_proj_softmax_ge`).

Both are statements about a plain tuple of unit vectors: no dynamics, no cap,
no time.  `Metastability.variance_inequality` is what is left once the
envelope argument has identified `η̇_q(t)` with this velocity.
-/

import Transformer.Basic

open scoped BigOperators
open Real

namespace Transformer
namespace Metastability

variable (d n : ℕ)

/-- **The `SA` velocity of a token, resolved against a fixed direction.**

For unit vectors `x_1,…,x_n` and any `w`,

  `⟨Proj_{x_k} (Σ_j a_{kj} x_j), w⟩
     = Σ_j a_{kj} (⟨x_j, w⟩ - ⟨x_k, w⟩)
       + ⟨x_k, w⟩ Σ_j a_{kj} ‖x_j - x_k‖² / 2`,

with `a_{kj} = e^{β⟨x_k,x_j⟩} / Σ_l e^{β⟨x_k,x_l⟩}`.  The only input is
`⟨x_k, x_j⟩ = 1 - ‖x_j - x_k‖²/2`, which holds because the `x_j` are unit
vectors; the weights need not even sum to one for the identity to hold, the
term `Σ_j a_{kj}` cancelling between the two sums.

This is the whole computation behind `rem: variance`: the remark's inequality
is what is left of it after the first sum is bounded below by the leakage and
the second is restricted to the cap. -/
theorem inner_proj_softmax_eq (β : ℝ) (x : Idx n → EucSpace d)
    (w : EucSpace d) (k : Idx n) (hx : ∀ j : Idx n, ‖x j‖ = 1) :
    inner (𝕜 := ℝ)
        (proj d (x k)
          ((∑ l : Idx n, Real.exp (β * inner (𝕜 := ℝ) (x k) (x l)))⁻¹ •
            ∑ j : Idx n, Real.exp (β * inner (𝕜 := ℝ) (x k) (x j)) • x j)) w
      = (∑ j : Idx n,
            Real.exp (β * inner (𝕜 := ℝ) (x k) (x j)) /
                (∑ l : Idx n, Real.exp (β * inner (𝕜 := ℝ) (x k) (x l))) *
              (inner (𝕜 := ℝ) (x j) w - inner (𝕜 := ℝ) (x k) w))
        + inner (𝕜 := ℝ) (x k) w *
            ∑ j : Idx n,
              Real.exp (β * inner (𝕜 := ℝ) (x k) (x j)) /
                  (∑ l : Idx n, Real.exp (β * inner (𝕜 := ℝ) (x k) (x l))) *
                (‖x j - x k‖ ^ 2 / 2) := by
  have hnormsq : ∀ j : Idx n,
      ‖x j - x k‖ ^ 2 / 2 = 1 - inner (𝕜 := ℝ) (x k) (x j) := by
    intro j
    rw [norm_sub_sq_real, hx j, hx k, real_inner_comm]
    ring
  have h1 : (∑ j : Idx n,
        Real.exp (β * inner (𝕜 := ℝ) (x k) (x j)) /
            (∑ l : Idx n, Real.exp (β * inner (𝕜 := ℝ) (x k) (x l))) *
          (inner (𝕜 := ℝ) (x j) w - inner (𝕜 := ℝ) (x k) w))
      = (∑ j : Idx n,
          Real.exp (β * inner (𝕜 := ℝ) (x k) (x j)) /
              (∑ l : Idx n, Real.exp (β * inner (𝕜 := ℝ) (x k) (x l))) *
            inner (𝕜 := ℝ) (x j) w)
        - inner (𝕜 := ℝ) (x k) w *
            ∑ j : Idx n, Real.exp (β * inner (𝕜 := ℝ) (x k) (x j)) /
              (∑ l : Idx n, Real.exp (β * inner (𝕜 := ℝ) (x k) (x l))) := by
    rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun j _ => by ring
  have h2 : (∑ j : Idx n,
        Real.exp (β * inner (𝕜 := ℝ) (x k) (x j)) /
            (∑ l : Idx n, Real.exp (β * inner (𝕜 := ℝ) (x k) (x l))) *
          (‖x j - x k‖ ^ 2 / 2))
      = (∑ j : Idx n, Real.exp (β * inner (𝕜 := ℝ) (x k) (x j)) /
            (∑ l : Idx n, Real.exp (β * inner (𝕜 := ℝ) (x k) (x l))))
        - ∑ j : Idx n,
            Real.exp (β * inner (𝕜 := ℝ) (x k) (x j)) /
                (∑ l : Idx n, Real.exp (β * inner (𝕜 := ℝ) (x k) (x l))) *
              inner (𝕜 := ℝ) (x k) (x j) := by
    rw [← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun j _ => by rw [hnormsq j]; ring
  have h3 : inner (𝕜 := ℝ)
        (proj d (x k)
          ((∑ l : Idx n, Real.exp (β * inner (𝕜 := ℝ) (x k) (x l)))⁻¹ •
            ∑ j : Idx n, Real.exp (β * inner (𝕜 := ℝ) (x k) (x j)) • x j)) w
      = (∑ j : Idx n,
          Real.exp (β * inner (𝕜 := ℝ) (x k) (x j)) /
              (∑ l : Idx n, Real.exp (β * inner (𝕜 := ℝ) (x k) (x l))) *
            inner (𝕜 := ℝ) (x j) w)
        - (∑ j : Idx n,
            Real.exp (β * inner (𝕜 := ℝ) (x k) (x j)) /
                (∑ l : Idx n, Real.exp (β * inner (𝕜 := ℝ) (x k) (x l))) *
              inner (𝕜 := ℝ) (x k) (x j)) * inner (𝕜 := ℝ) (x k) w := by
    simp only [proj, inner_sub_left, real_inner_smul_left, real_inner_smul_right,
      sum_inner, inner_sum, Finset.mul_sum]
    rw [Finset.sum_mul, Finset.sum_mul]
    refine congrArg₂ (· - ·) (Finset.sum_congr rfl fun j _ => by ring)
      (Finset.sum_congr rfl fun j _ => by ring)
  rw [h1, h2, h3]
  ring


/-- **The arithmetic of the variance inequality.**

Over opaque reals: if every term of `b` is dominated by the corresponding
`a_j v_j`, and every `a_j (c_j - η)` is at least `-2E`, then

  `η Σ_j b_j - 2 n E ≤ Σ_j a_j (c_j - η) + η Σ_j a_j v_j`.

This is the whole inequality reasoning of `inner_proj_softmax_ge`, separated
from the vectors so that it is `n` additions and one multiplication. -/
theorem cap_variance_bound (η E : ℝ) (a c v b : Idx n → ℝ) (hη0 : 0 ≤ η)
    (hb : ∀ j : Idx n, b j ≤ a j * v j)
    (hac : ∀ j : Idx n, -(2 * E) ≤ a j * (c j - η)) :
    η * (∑ j : Idx n, b j) - 2 * (n : ℝ) * E
      ≤ (∑ j : Idx n, a j * (c j - η)) + η * ∑ j : Idx n, a j * v j := by
  have h1 : -(2 * (n : ℝ) * E) ≤ ∑ j : Idx n, a j * (c j - η) := by
    calc -(2 * (n : ℝ) * E) = ∑ _j : Idx n, -(2 * E) := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
          ring
      _ ≤ _ := Finset.sum_le_sum fun j _ => hac j
  have h2 : ∑ j : Idx n, b j ≤ ∑ j : Idx n, a j * v j :=
    Finset.sum_le_sum fun j _ => hb j
  have h3 := mul_le_mul_of_nonneg_left h2 hη0
  linarith

/-- The hypotheses of `cap_variance_bound` are satisfiable: three indices,
`η = E = a = c = v = b = 1`. -/
example : (1 : ℝ) * (∑ _ : Idx 3, (1 : ℝ)) - 2 * ((3 : ℕ) : ℝ) * 1
    ≤ (∑ _ : Idx 3, (1 : ℝ) * ((1 : ℝ) - 1)) + 1 * ∑ _ : Idx 3, (1 : ℝ) * 1 :=
  cap_variance_bound 3 1 1 (fun _ => 1) (fun _ => 1) (fun _ => 1) (fun _ => 1)
    zero_le_one (fun _ => by norm_num) (fun _ => by norm_num)

/-- **The velocity, bounded below by the variance of a subset.**

For unit vectors `x_1,…,x_n`, a unit direction `w`, a reference index `K`
with `η = ⟨x_K, w⟩ ≥ 0`, and a subset `P` on which `⟨x_j, w⟩ ≥ η` whose
complement is `α`-separated from `x_K`,

  `⟨Proj_{x_K}(Σ_j a_{Kj} x_j), w⟩
      ≥ η Σ_{j ∈ P} a_{Kj} ‖x_j - x_K‖²/2 - 2 n e^{-(1-α)β}`.

The two sums of `inner_proj_softmax_eq` are estimated separately: a token of
`P` contributes a nonnegative gap, a token outside contributes at worst `-2`
against a weight `a_{Kj} ≤ e^{βα}/e^{β} = e^{-(1-α)β}` — the partition
function is at least its own diagonal term `e^{β}` — and the variance sum is
only made larger by restoring the terms outside `P`. -/
theorem inner_proj_softmax_ge (β α η : ℝ) (hβ : 0 ≤ β)
    (x : Idx n → EucSpace d) (w : EucSpace d) (K : Idx n)
    (hx : ∀ j : Idx n, ‖x j‖ = 1) (hw : ‖w‖ = 1)
    (P : Idx n → Prop) [DecidablePred P]
    (hη0 : 0 ≤ η) (hηK : η = inner (𝕜 := ℝ) (x K) w)
    (hin : ∀ j : Idx n, P j → η ≤ inner (𝕜 := ℝ) (x j) w)
    (hout : ∀ j : Idx n, ¬ P j → inner (𝕜 := ℝ) (x K) (x j) ≤ α) :
    η * (∑ j : Idx n,
        if P j then
          Real.exp (β * inner (𝕜 := ℝ) (x K) (x j)) /
              (∑ l : Idx n, Real.exp (β * inner (𝕜 := ℝ) (x K) (x l)))
            * ‖x j - x K‖ ^ 2 / 2
        else 0)
      - 2 * (n : ℝ) * Real.exp (-((1 - α) * β))
    ≤ inner (𝕜 := ℝ)
        (proj d (x K)
          ((∑ l : Idx n, Real.exp (β * inner (𝕜 := ℝ) (x K) (x l)))⁻¹ •
            ∑ j : Idx n, Real.exp (β * inner (𝕜 := ℝ) (x K) (x j)) • x j)) w := by
  have hSpos : 0 < ∑ l : Idx n, Real.exp (β * inner (𝕜 := ℝ) (x K) (x l)) :=
    Finset.sum_pos (fun j _ => Real.exp_pos _) ⟨K, Finset.mem_univ K⟩
  have hSge : Real.exp β
      ≤ ∑ l : Idx n, Real.exp (β * inner (𝕜 := ℝ) (x K) (x l)) := by
    have hdiag : Real.exp (β * inner (𝕜 := ℝ) (x K) (x K)) = Real.exp β := by
      rw [real_inner_self_eq_norm_mul_norm, hx K]; ring_nf
    rw [← hdiag]
    exact Finset.single_le_sum (f := fun l : Idx n =>
      Real.exp (β * inner (𝕜 := ℝ) (x K) (x l)))
      (fun l _ => (Real.exp_pos _).le) (Finset.mem_univ K)
  rw [inner_proj_softmax_eq d n β x w K hx, ← hηK]
  set S : ℝ := ∑ l : Idx n, Real.exp (β * inner (𝕜 := ℝ) (x K) (x l)) with hSdef
  clear_value S
  have hEpos : (0 : ℝ) < Real.exp (-((1 - α) * β)) := Real.exp_pos _
  have hanonneg : ∀ j : Idx n,
      0 ≤ Real.exp (β * inner (𝕜 := ℝ) (x K) (x j)) / S :=
    fun j => div_nonneg (Real.exp_pos _).le hSpos.le
  have hac : ∀ j : Idx n,
      -(2 * Real.exp (-((1 - α) * β)))
        ≤ Real.exp (β * inner (𝕜 := ℝ) (x K) (x j)) / S
            * (inner (𝕜 := ℝ) (x j) w - η) := by
    intro j
    by_cases hj : P j
    · have h := mul_nonneg (hanonneg j) (sub_nonneg.mpr (hin j hj))
      linarith
    · have hcj : |inner (𝕜 := ℝ) (x j) w| ≤ 1 := by
        have h := abs_real_inner_le_norm (x j) w
        rwa [hx j, hw, one_mul] at h
      have hηle : |η| ≤ 1 := by
        rw [hηK]
        have h := abs_real_inner_le_norm (x K) w
        rwa [hx K, hw, one_mul] at h
      have hgap : (-2 : ℝ) ≤ inner (𝕜 := ℝ) (x j) w - η := by
        rw [abs_le] at hcj hηle; linarith
      have haj : Real.exp (β * inner (𝕜 := ℝ) (x K) (x j)) / S
          ≤ Real.exp (-((1 - α) * β)) := by
        rw [div_le_iff₀ hSpos]
        have hnum : Real.exp (β * inner (𝕜 := ℝ) (x K) (x j)) ≤ Real.exp (β * α) :=
          Real.exp_le_exp.mpr (by nlinarith [hout j hj])
        have hprod : Real.exp (-((1 - α) * β)) * Real.exp β = Real.exp (β * α) := by
          rw [← Real.exp_add]; congr 1; ring
        have hmul : Real.exp (-((1 - α) * β)) * Real.exp β
            ≤ Real.exp (-((1 - α) * β)) * S :=
          mul_le_mul_of_nonneg_left hSge hEpos.le
        linarith
      have hstep := mul_le_mul_of_nonneg_left hgap (hanonneg j)
      nlinarith
  have hb : ∀ j : Idx n,
      (if P j then
          Real.exp (β * inner (𝕜 := ℝ) (x K) (x j)) / S * ‖x j - x K‖ ^ 2 / 2
        else 0)
        ≤ Real.exp (β * inner (𝕜 := ℝ) (x K) (x j)) / S * (‖x j - x K‖ ^ 2 / 2) := by
    intro j
    by_cases hj : P j
    · rw [ite_eq_left hj]
      exact le_of_eq (mul_div_assoc _ _ _)
    · rw [ite_eq_right hj]
      exact mul_nonneg (hanonneg j) (by positivity)
  exact cap_variance_bound n η (Real.exp (-((1 - α) * β)))
    (fun j => Real.exp (β * inner (𝕜 := ℝ) (x K) (x j)) / S)
    (fun j => inner (𝕜 := ℝ) (x j) w)
    (fun j => ‖x j - x K‖ ^ 2 / 2)
    (fun j => if P j then
        Real.exp (β * inner (𝕜 := ℝ) (x K) (x j)) / S * ‖x j - x K‖ ^ 2 / 2
      else 0)
    hη0 hb hac

/-- The hypotheses of `inner_proj_softmax_ge` are satisfiable: a single unit
vector, tested against itself — `w = x_0`, `K = 0`, `η = 1`, `P` everywhere
true, `α = 1`. -/
example (β : ℝ) (hβ : 0 ≤ β) (v : EucSpace 1) (hv : ‖v‖ = 1) :
    (1 : ℝ) * (∑ _ : Idx 1,
        if True then
          Real.exp (β * inner (𝕜 := ℝ) v v) /
              (∑ _ : Idx 1, Real.exp (β * inner (𝕜 := ℝ) v v))
            * ‖v - v‖ ^ 2 / 2
        else 0)
      - 2 * ((1 : ℕ) : ℝ) * Real.exp (-((1 - 1) * β))
    ≤ inner (𝕜 := ℝ)
        (proj 1 v
          ((∑ _ : Idx 1, Real.exp (β * inner (𝕜 := ℝ) v v))⁻¹ •
            ∑ _ : Idx 1, Real.exp (β * inner (𝕜 := ℝ) v v) • v)) v := by
  have hvv : inner (𝕜 := ℝ) v v = 1 := by
    rw [real_inner_self_eq_norm_mul_norm, hv]; ring
  exact inner_proj_softmax_ge 1 1 β 1 1 hβ (fun _ => v) v 0 (fun _ => hv) hv
    (fun _ => True) zero_le_one hvv.symm (fun _ _ => hvv.ge) (fun _ h => absurd trivial h)

end Metastability
end Transformer
