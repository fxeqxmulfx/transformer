/-
# Metastability — the softmax velocity, bounded below by a cap's variance
  (behind `rem: variance` of 2410.06833v1)

`inner_proj_softmax_eq` splits the tested `SA` velocity into a gap sum and a
variance sum; here both are bounded below token by token, which gives the
remark's inequality with the paper's leakage constant `n e^{-(1-α)β}`
(`inner_proj_softmax_ge`).  A statement about a plain tuple of unit vectors:
no dynamics, no cap, no time.
-/

import Transformer.Metastability.CapVelocity

open scoped BigOperators
open Real

namespace Transformer
namespace Metastability

variable (d n : ℕ)

/-- **The arithmetic of the variance inequality.**

Over opaque reals: if every index satisfies
`η b_j - E ≤ a_j (c_j - η) + η a_j v_j`, then summing,

  `η Σ_j b_j - n E ≤ Σ_j a_j (c_j - η) + η Σ_j a_j v_j`.

This is the summation step of `inner_proj_softmax_ge`, separated from the
vectors. -/
theorem cap_variance_bound (η E : ℝ) (a c v b : Idx n → ℝ)
    (hj : ∀ j : Idx n, η * b j - E ≤ a j * (c j - η) + η * (a j * v j)) :
    η * (∑ j : Idx n, b j) - (n : ℝ) * E
      ≤ (∑ j : Idx n, a j * (c j - η)) + η * ∑ j : Idx n, a j * v j := by
  have h := Finset.sum_le_sum fun j (_ : j ∈ Finset.univ) => hj j
  rw [Finset.sum_sub_distrib, Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum,
    Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at h
  exact h

/-- The hypothesis of `cap_variance_bound` is satisfiable: three indices,
`η = E = a = c = v = b = 1`. -/
example : (1 : ℝ) * (∑ _ : Idx 3, (1 : ℝ)) - ((3 : ℕ) : ℝ) * 1
    ≤ (∑ _ : Idx 3, (1 : ℝ) * ((1 : ℝ) - 1)) + 1 * ∑ _ : Idx 3, (1 : ℝ) * 1 :=
  cap_variance_bound 3 1 1 (fun _ => 1) (fun _ => 1) (fun _ => 1) (fun _ => 1)
    (fun _ => by norm_num)

/-- **The velocity, bounded below by the variance of a subset.**

For unit vectors `x_1,…,x_n`, a unit direction `w`, a reference index `K`
with `η = ⟨x_K, w⟩`, and a subset `P` on which `⟨x_j, w⟩ ≥ η` whose
complement is `α`-separated from `x_K`,

  `⟨Proj_{x_K}(Σ_j a_{Kj} x_j), w⟩
      ≥ η Σ_{j ∈ P} a_{Kj} ‖x_j - x_K‖²/2 - n e^{-(1-α)β}`.

Token by token, in the two sums of `inner_proj_softmax_eq`: a token of `P`
contributes a nonnegative gap `⟨x_j, w⟩ - η` and its variance term.  A token
outside contributes, gap and variance term together,
`a_{Kj} ⟨x_j, w - η x_K⟩ ≥ -a_{Kj} ‖w - η x_K‖ = -a_{Kj} √(1 - η²) ≥ -a_{Kj}`,
against a weight `a_{Kj} ≤ e^{βα}/e^{β} = e^{-(1-α)β}` — the partition
function is at least its own diagonal term `e^{β}`. -/
theorem inner_proj_softmax_ge (β α η : ℝ) (hβ : 0 ≤ β)
    (x : Idx n → EucSpace d) (w : EucSpace d) (K : Idx n)
    (hx : ∀ j : Idx n, ‖x j‖ = 1) (hw : ‖w‖ = 1)
    (P : Idx n → Prop) [DecidablePred P]
    (hηK : η = inner (𝕜 := ℝ) (x K) w)
    (hin : ∀ j : Idx n, P j → η ≤ inner (𝕜 := ℝ) (x j) w)
    (hout : ∀ j : Idx n, ¬ P j → inner (𝕜 := ℝ) (x K) (x j) ≤ α) :
    η * (∑ j : Idx n,
        if P j then
          Real.exp (β * inner (𝕜 := ℝ) (x K) (x j)) /
              (∑ l : Idx n, Real.exp (β * inner (𝕜 := ℝ) (x K) (x l)))
            * ‖x j - x K‖ ^ 2 / 2
        else 0)
      - (n : ℝ) * Real.exp (-((1 - α) * β))
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
  have hwK : ‖w - η • x K‖ ≤ 1 := by
    have hsq : ‖w - η • x K‖ ^ 2 = 1 - η ^ 2 := by
      rw [norm_sub_sq_real, norm_smul, real_inner_smul_right, hw, hx K, Real.norm_eq_abs,
        mul_pow, sq_abs, real_inner_comm (x K) w, ← hηK]
      ring
    nlinarith [norm_nonneg (w - η • x K), sq_nonneg η]
  have hj : ∀ j : Idx n,
      η * (if P j then
          Real.exp (β * inner (𝕜 := ℝ) (x K) (x j)) / S * ‖x j - x K‖ ^ 2 / 2
        else 0) - Real.exp (-((1 - α) * β))
        ≤ Real.exp (β * inner (𝕜 := ℝ) (x K) (x j)) / S
            * (inner (𝕜 := ℝ) (x j) w - η)
          + η * (Real.exp (β * inner (𝕜 := ℝ) (x K) (x j)) / S
            * (‖x j - x K‖ ^ 2 / 2)) := by
    intro j
    have ha := hanonneg j
    by_cases hP : P j
    · simp only [hP, ↓reduceIte]
      rw [mul_div_assoc]
      have h := mul_nonneg ha (sub_nonneg.mpr (hin j hP))
      linarith
    · simp only [hP, ↓reduceIte]
      rw [mul_zero, zero_sub]
      -- gap and variance term together: `⟨x_j, w - η x_K⟩ ≥ -1`
      have hv : ‖x j - x K‖ ^ 2 / 2 = 1 - inner (𝕜 := ℝ) (x K) (x j) := by
        rw [norm_sub_sq_real, hx j, hx K, real_inner_comm]
        ring
      have hcs : -1 ≤ inner (𝕜 := ℝ) (x j) w - η * inner (𝕜 := ℝ) (x K) (x j) := by
        have h := abs_real_inner_le_norm (x j) (w - η • x K)
        rw [hx j, one_mul, inner_sub_right, real_inner_smul_right,
          real_inner_comm (x K) (x j)] at h
        linarith [(abs_le.mp (h.trans hwK)).1]
      have haj : Real.exp (β * inner (𝕜 := ℝ) (x K) (x j)) / S
          ≤ Real.exp (-((1 - α) * β)) := by
        rw [div_le_iff₀ hSpos]
        have hnum : Real.exp (β * inner (𝕜 := ℝ) (x K) (x j)) ≤ Real.exp (β * α) :=
          Real.exp_le_exp.mpr (by nlinarith [hout j hP])
        have hprod : Real.exp (-((1 - α) * β)) * Real.exp β = Real.exp (β * α) := by
          rw [← Real.exp_add]; congr 1; ring
        have hmul : Real.exp (-((1 - α) * β)) * Real.exp β
            ≤ Real.exp (-((1 - α) * β)) * S :=
          mul_le_mul_of_nonneg_left hSge hEpos.le
        linarith
      have hstep := mul_le_mul_of_nonneg_left hcs ha
      rw [hv]
      nlinarith
  exact cap_variance_bound n η (Real.exp (-((1 - α) * β)))
    (fun j => Real.exp (β * inner (𝕜 := ℝ) (x K) (x j)) / S)
    (fun j => inner (𝕜 := ℝ) (x j) w)
    (fun j => ‖x j - x K‖ ^ 2 / 2)
    (fun j => if P j then
        Real.exp (β * inner (𝕜 := ℝ) (x K) (x j)) / S * ‖x j - x K‖ ^ 2 / 2
      else 0)
    hj

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
      - ((1 : ℕ) : ℝ) * Real.exp (-((1 - 1) * β))
    ≤ inner (𝕜 := ℝ)
        (proj 1 v
          ((∑ _ : Idx 1, Real.exp (β * inner (𝕜 := ℝ) v v))⁻¹ •
            ∑ _ : Idx 1, Real.exp (β * inner (𝕜 := ℝ) v v) • v)) v := by
  have hvv : inner (𝕜 := ℝ) v v = 1 := by
    rw [real_inner_self_eq_norm_mul_norm, hv]; ring
  exact inner_proj_softmax_ge 1 1 β 1 1 hβ (fun _ => v) v 0 (fun _ => hv) hv
    (fun _ => True) hvv.symm (fun _ _ => hvv.ge) (fun _ h => absurd trivial h)

end Metastability
end Transformer
