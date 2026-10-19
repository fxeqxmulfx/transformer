module

public import Transformer.Landscape.PhaseModel.Generating
public import Transformer.Landscape.PhaseModel.Lift
public import Transformer.Landscape.SaddleAvoidance.SecondOrder

/-!
# Phase model — twisted states

A critical point of a phase energy where every cosine `cos ⟪m_r, z⟫` is
positive and no nonzero vector is orthogonal to every row is a local
minimizer (`isLocalMin_phaseEnergy_of_cos_pos`): the Hessian
`h ↦ Σ_r cos ⟪m_r, z⟫ ⟪m_r, h⟫ m_r` is then positive definite. It need not
be a zero. For the division table, take integer phases `j` and
`ψ = (π/3) j`: the relation of `p = (x, y)` takes the value `π d_p / 3`, with
the defect `d_p = j_x - j_y - j_{x-y}` (`defect`). When every defect of `S`
is `0` or `±1` modulo `6`, every cosine is `1` or `1/2` and every sine `0` or
`±√3/2` (`sin_cos_pi_div_three_mul`), so `ψ` is critical as soon as the rows
of defect `1` and those of defect `-1` modulo `6` have the same sum: a
twisted state. For `S` generating the full table it is a local minimizer
(`isLocalMin_divisionEnergy_twisted`), of energy half the number of nonzero
defects (`divisionEnergy_twisted`).
-/

@[expose] public section

open Real
open scoped Finset

namespace Transformer.Landscape.PhaseModel

local notation "⟪" x ", " y "⟫" => inner ℝ x y

section Positive

variable {E R : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
  [Fintype R] (m : R → E)

/-- **No point with nonnegative cosines is a strict saddle of a phase
energy**: there `⟪H h, h⟫ = Σ_r cos ⟪m_r, z⟫ ⟪m_r, h⟫² ≥ 0`, while an
eigenvector `w` of `H` for `μ < 0` has `⟪H w, w⟫ = μ ‖w‖² < 0`. -/
theorem not_isStrictSaddle_phaseEnergy {z : E} (hcos : ∀ r, 0 ≤ cos ⟪m r, z⟫) :
    ¬Landscape.SaddleAvoidance.IsStrictSaddle (phaseEnergy m) z := by
  rintro ⟨-, μ, hμ, hev⟩
  obtain ⟨w, hw⟩ := hev.exists_hasEigenvector
  have h1 : ⟪fderiv ℝ (gradient (phaseEnergy m)) z w, w⟫ = μ * ‖w‖ ^ 2 := by
    rw [show fderiv ℝ (gradient (phaseEnergy m)) z w = μ • w from hw.apply_eq_smul,
      real_inner_smul_left, real_inner_self_eq_norm_sq]
  have h2 : 0 ≤ ⟪fderiv ℝ (gradient (phaseEnergy m)) z w, w⟫ := by
    rw [inner_fderiv_gradient_phaseEnergy]
    exact Finset.sum_nonneg fun r _ => mul_nonneg (hcos r) (sq_nonneg _)
  have h3 := mul_neg_of_neg_of_pos hμ (pow_pos (norm_pos_iff.2 hw.2) 2)
  linarith

/-- The hypothesis of `not_isStrictSaddle_phaseEnergy` holds for the single row
`1` on the line at `0`. -/
example : let m := fun _ : Unit => (1 : ℝ)
    ∀ r, 0 ≤ cos ⟪m r, 0⟫ := by
  intro m r
  simp [m]

/-- **A critical point with positive cosines is a local minimizer of the phase
energy** when no nonzero vector is orthogonal to every row: the Hessian is
injective (`injective_fderiv_gradient_phaseEnergy`) and the point is not a
strict saddle, so the second-order test applies. -/
theorem isLocalMin_phaseEnergy_of_cos_pos [FiniteDimensional ℝ E] {z : E}
    (hz : gradient (phaseEnergy m) z = 0) (hcos : ∀ r, 0 < cos ⟪m r, z⟫)
    (hspan : ∀ h, (∀ r, ⟪m r, h⟫ = 0) → h = 0) : IsLocalMin (phaseEnergy m) z := by
  obtain ⟨c, hc, hpos⟩ := Landscape.SaddleAvoidance.exists_pos_mul_norm_sq_le_inner
    (contDiff_phaseEnergy m) hz (not_isStrictSaddle_phaseEnergy m fun r => (hcos r).le)
    (injective_fderiv_gradient_phaseEnergy m (s := 1) (fun r => by simpa using hcos r) hspan)
  exact Landscape.SaddleAvoidance.isLocalMin_of_mul_norm_sq_le_inner
    ((contDiff_phaseEnergy m (k := 1)).differentiable one_ne_zero)
    (hasFDerivAt_gradient_phaseEnergy m z).differentiableAt.hasFDerivAt hz hc hpos

/-- The hypotheses of `isLocalMin_phaseEnergy_of_cos_pos` hold for the single
row `1` on the line at `0`, where `V(ψ) = 1 - cos ψ`. -/
example : let m := fun _ : Unit => (1 : ℝ)
    gradient (phaseEnergy m) 0 = 0 ∧ (∀ r, 0 < cos ⟪m r, 0⟫) ∧
      ∀ h, (∀ r, ⟪m r, h⟫ = 0) → h = 0 := by
  intro m
  refine ⟨by simp [m, gradient_phaseEnergy], fun _ => by simp [m], fun h hh => ?_⟩
  simpa [m] using hh ()

end Positive

/-- **The twist of an integer**: `1` for `d ≡ 1`, `-1` for `d ≡ -1` and `0`
otherwise modulo `6`; for `d ≡ 0, ±1` it is the sign of `sin (π d / 3)`. -/
def twist (d : ℤ) : ℤ :=
  if d % 6 = 1 then 1 else if d % 6 = 5 then -1 else 0

/-- **Sines and cosines of `π d / 3` for `d ≡ 0, ±1 (mod 6)`**: the sine is
`√3/2` times the twist, the cosine `1` for `d ≡ 0` and `1/2` otherwise. -/
theorem sin_cos_pi_div_three_mul {d : ℤ} (hd : d % 6 = 0 ∨ d % 6 = 1 ∨ d % 6 = 5) :
    sin (π / 3 * d) = √3 / 2 * twist d ∧ cos (π / 3 * d) = if d % 6 = 0 then 1 else 1 / 2 := by
  have h : π / 3 * d = π / 3 * ((d % 6 : ℤ) : ℝ) + (d / 6 : ℤ) * (2 * π) := by
    have : (d : ℝ) = ((d % 6 : ℤ) : ℝ) + ((d / 6 : ℤ) : ℝ) * 6 := by
      exact_mod_cast (Int.emod_add_ediv_mul d 6).symm
    rw [this]
    ring
  rw [h, sin_add_int_mul_two_pi, cos_add_int_mul_two_pi]
  rcases hd with hd | hd | hd <;> simp only [twist, hd]
  · simp
  · simp
  · rw [show π / 3 * ((5 : ℤ) : ℝ) = 2 * π - π / 3 by push_cast; ring, sin_two_pi_sub,
      cos_two_pi_sub]
    simp

/-- The hypothesis of `sin_cos_pi_div_three_mul` holds for `d = -5`. -/
example : (-5 : ℤ) % 6 = 0 ∨ (-5 : ℤ) % 6 = 1 ∨ (-5 : ℤ) % 6 = 5 := by decide

variable {n : ℕ}

/-- **The defect of integer phases** `j` on the pair `p = (x, y)`:
`j_x - j_y - j_{x-y}`, the relation of `p` at `j`. -/
def defect (j : Fin n → ℤ) (p : Fin n × Fin n) : ℤ :=
  j p.1 - j p.2 - j (p.1 - p.2)

/-- At the phases `t j` the relation of `p` takes the value `t d_p`. -/
theorem inner_intVec_divisionRow_smul_intVec (p : Fin n × Fin n) (t : ℝ) (j : Fin n → ℤ) :
    ⟪intVec (divisionRow p), t • intVec j⟫ = t * defect j p := by
  rw [inner_intVec_divisionRow]
  simp [intVec, defect]
  ring

variable {S : Finset (Fin n × Fin n)} {j : Fin n → ℤ}

/-- **Twisted states are local minimizers of generating partial tables**: if
the rows of `S` generate those of the full table, every defect of `S` is
`0` or `±1` modulo `6`, and the rows of `S` weighted by the twists of their
defects sum to `0`, then the phases `(π/3) j` are a local minimizer of `V_S`.
The gradient there is `√3/2` times that sum, every cosine is `1` or `1/2`,
and only `0` is orthogonal to every row of `S`
(`eq_zero_of_forall_inner_divisionRow_of_mem_span`). -/
theorem isLocalMin_divisionEnergy_twisted
    (hS : ∀ p, divisionRow p ∈ Submodule.span ℤ (divisionRow '' (S : Set (Fin n × Fin n))))
    (hd : ∀ p ∈ S, defect j p % 6 = 0 ∨ defect j p % 6 = 1 ∨ defect j p % 6 = 5)
    (hsum : ∑ p ∈ S, twist (defect j p) • divisionRow p = 0) :
    IsLocalMin (divisionEnergy S) ((π / 3) • intVec j) := by
  have hsin : ∀ p : S, sin ⟪intVec (divisionRow p.1), (π / 3) • intVec j⟫ =
      √3 / 2 * twist (defect j p.1) := fun p => by
    rw [inner_intVec_divisionRow_smul_intVec, (sin_cos_pi_div_three_mul (hd p.1 p.2)).1]
  refine isLocalMin_phaseEnergy_of_cos_pos _ ?_ (fun p => ?_) fun h hh =>
    eq_zero_of_forall_inner_divisionRow_of_mem_span hS h fun p hp => hh ⟨p, hp⟩
  · simp only [gradient_phaseEnergy, hsin]
    refine PiLp.ext fun i => ?_
    have hi := congrFun hsum i
    simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul, Pi.zero_apply] at hi
    simp only [WithLp.ofLp_sum, WithLp.ofLp_smul, Finset.sum_apply, Pi.smul_apply,
      smul_eq_mul, intVec, PiLp.zero_apply]
    rw [Finset.sum_coe_sort S fun p => √3 / 2 * (twist (defect j p) : ℝ) *
      (divisionRow p i : ℝ)]
    simp only [mul_assoc, ← Finset.mul_sum]
    exact mul_eq_zero_of_right _ (by exact_mod_cast hi)
  · rw [inner_intVec_divisionRow_smul_intVec, (sin_cos_pi_div_three_mul (hd p.1 p.2)).2]
    split_ifs <;> norm_num

/-- **The energy of a twisted state** is half the number of pairs of `S`
with a nonzero defect modulo `6`: their cosines are `1/2`, the others `1`. -/
theorem divisionEnergy_twisted
    (hd : ∀ p ∈ S, defect j p % 6 = 0 ∨ defect j p % 6 = 1 ∨ defect j p % 6 = 5) :
    divisionEnergy S ((π / 3) • intVec j) = #{p ∈ S | defect j p % 6 ≠ 0} / 2 := by
  rw [divisionEnergy, phaseEnergy, Finset.sum_coe_sort S fun p =>
    1 - cos ⟪intVec (divisionRow p), (π / 3) • intVec j⟫, Finset.card_filter, Nat.cast_sum,
    Finset.sum_div]
  refine Finset.sum_congr rfl fun p hp => ?_
  rw [inner_intVec_divisionRow_smul_intVec, (sin_cos_pi_div_three_mul (hd p hp)).2]
  by_cases h : defect j p % 6 = 0
  · simp [h]
  · norm_num [h]

/-- The hypotheses of `isLocalMin_divisionEnergy_twisted` and
`divisionEnergy_twisted` hold for the full table and the phases `0`, where
every defect is `0`; `Landscape.PhaseModel.Spurious` gives nonzero defects. -/
example : let S := (Finset.univ : Finset (Fin 3 × Fin 3)); let j : Fin 3 → ℤ := 0
    (∀ p, divisionRow p ∈ Submodule.span ℤ (divisionRow '' (S : Set (Fin 3 × Fin 3)))) ∧
      (∀ p ∈ S, defect j p % 6 = 0 ∨ defect j p % 6 = 1 ∨ defect j p % 6 = 5) ∧
      ∑ p ∈ S, twist (defect j p) • divisionRow p = 0 := by
  intro S j
  exact ⟨fun p => Submodule.subset_span ⟨p, by simp [S], rfl⟩, by decide, by decide⟩

end Transformer.Landscape.PhaseModel
