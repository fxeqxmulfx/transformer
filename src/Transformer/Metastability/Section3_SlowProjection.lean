module

public import Transformer.Metastability.Section3_SharpEnergy
public import Transformer.Metastability.Section3_SlowGradient

/-!
# A uniform energy bound for a projection onto the slow manifold

ArXiv:2410.06833v1, §3.1 (H1), and §3.2, `eq: otto.attention`.
The Euclidean lower bound in (H1), combined with the actual energy's
global Taylor remainder, controls the energy discrepancy by the
gradient at the projection. No dynamics or regularity of that
projection is needed for this static estimate.
-/

@[expose] public section

open scoped BigOperators
open Real

namespace Transformer.Metastability

/-- **For at least two tokens the Taylor coefficient is at most `1/4`.**
Source: arXiv:2410.06833v1, §3.2, the assumption `n≥2` of
`lem: PL.borjan` and the Euclidean projection in `eq: otto.attention`. -/
theorem angular_taylor_coefficient_le_quarter (n : ℕ) (hn : 2 ≤ n) :
    1 / (2 * (n : ℝ)) ≤ (1 / 4 : ℝ) := by
  have hnR : (2 : ℝ) ≤ n := by exact_mod_cast hn
  apply (div_le_iff₀ (by positivity : 0 < 2 * (n : ℝ))).mpr
  linarith

/-- The two-token endpoint meets the coefficient hypothesis. -/
example : 2 ≤ 2 := le_rfl

/-- **The lower half of (H1) is realized by an actual gradient step
backwards from its target.** Source: arXiv:2410.06833v1, §3.1 (H1),
as used in §3.2, `eq: otto.attention`. The sharp global remainder
shows this for every target, including noncritical configurations. -/
theorem angular_lower_projection_of_gradient_step (n : ℕ) (hn : 2 ≤ n)
    (β : ℝ) (hβ : 0 < β) (V : Idx n → ℝ) :
    (1 / 2 : ℝ) * ∑ i : Idx n, ((V - angularGrad n β V) i - V i) ^ 2 ≤
      angularEβ n β V - angularEβ n β (V - angularGrad n β V) := by
  let g := angularGrad n β V
  let G := ∑ i : Idx n, (g i) ^ 2
  have hG : 0 ≤ G := Finset.sum_nonneg fun _ _ => sq_nonneg _
  have h := (abs_le.mp (angularEβ_remainder_half n (by omega) β hβ V (-g))).2
  have hvec : V + -g = V - g := by abel
  have hlin : (∑ i : Idx n, g i * (-g) i) = -G := by
    simp only [Pi.neg_apply, mul_neg, ← pow_two, Finset.sum_neg_distrib, G]
  have hsq : (∑ i : Idx n, ((-g) i) ^ 2) = G := by
    simp only [Pi.neg_apply, neg_sq, G]
  have hdisp : (∑ i : Idx n, ((V - g) i - V i) ^ 2) = G := by
    simp only [Pi.sub_apply, sub_sub_cancel_left, neg_sq, G]
  rw [hvec, hlin, hsq] at h
  have hcoef := mul_le_mul_of_nonneg_right (angular_taylor_coefficient_le_quarter n hn) hG
  change (1 / 2 : ℝ) * ∑ i : Idx n, ((V - g) i - V i) ^ 2 ≤
    angularEβ n β V - angularEβ n β (V - g)
  rw [hdisp]
  linarith

/-- The gradient-step witness is realized on two orthogonal circle
tokens at `β=1`. These inputs do not require a critical target. -/
example : (1 / 2 : ℝ) * ∑ i : Idx 2,
    ((![0, π / 2] - angularGrad 2 1 ![0, π / 2]) i - (![0, π / 2] : Idx 2 → ℝ) i) ^ 2 ≤
      angularEβ 2 1 ![0, π / 2] -
        angularEβ 2 1 (![0, π / 2] - angularGrad 2 1 ![0, π / 2]) :=
  angular_lower_projection_of_gradient_step 2 le_rfl 1 one_pos _

/-- **The gradient pairing has the Young bound with coefficient `1/8`.**
Source: arXiv:2410.06833v1, §3.1 (H1), in the Euclidean norm
used by `eq: otto.attention`. -/
theorem sum_mul_le_two_and_eighth_squares (n : ℕ) (g v : Idx n → ℝ) :
    (∑ i : Idx n, g i * v i) ≤
      2 * ∑ i : Idx n, (g i) ^ 2 + (1 / 8 : ℝ) * ∑ i : Idx n, (v i) ^ 2 := by
  calc
    _ ≤ ∑ i : Idx n, (2 * (g i) ^ 2 + (1 / 8 : ℝ) * (v i) ^ 2) :=
      Finset.sum_le_sum fun i _ => by nlinarith [sq_nonneg (2 * g i - v i / 2)]
    _ = _ := by rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum]

/-- **A point satisfying the lower half of (H1) has discrepancy at
most eight times the squared gradient at its projection.**
Source: arXiv:2410.06833v1, §3.1 (H1), and §3.2,
`eq: otto.attention`. The estimate uses the actual normalized
exponential energy, so its constant is uniform in temperature. -/
theorem angularEβ_gain_le_eight_target_gradient (n : ℕ) (hn : 2 ≤ n)
    (β : ℝ) (hβ : 0 < β) (U V : Idx n → ℝ)
    (hproj : (1 / 2 : ℝ) * ∑ i : Idx n, (U i - V i) ^ 2 ≤
      angularEβ n β V - angularEβ n β U) :
    angularEβ n β V - angularEβ n β U ≤
      8 * ∑ i : Idx n, (angularGrad n β V i) ^ 2 := by
  let D := ∑ i : Idx n, (V i - U i) ^ 2
  let G := ∑ i : Idx n, (angularGrad n β V i) ^ 2
  have hD : 0 ≤ D := Finset.sum_nonneg fun _ _ => sq_nonneg _
  have hsq : (∑ i : Idx n, (U i - V i) ^ 2) = D := by
    apply Finset.sum_congr rfl
    intro i _
    ring
  rw [hsq] at hproj
  have htaylor := angularEβ_gain_le_target_gradient n (by omega) β hβ U V
  have hcoef := mul_le_mul_of_nonneg_right (angular_taylor_coefficient_le_quarter n hn) hD
  have hyoung := sum_mul_le_two_and_eighth_squares n (angularGrad n β V) (V - U)
  change (∑ i : Idx n, angularGrad n β V i * (V i - U i)) ≤ 2 * G + (1 / 8 : ℝ) * D at hyoung
  change angularEβ n β V - angularEβ n β U ≤
    (∑ i : Idx n, angularGrad n β V i * (V i - U i)) + (1 / (2 * (n : ℝ))) * D at htaylor
  have hgap : angularEβ n β V - angularEβ n β U ≤ 2 * G + (3 / 8 : ℝ) * D := by
    linarith
  have hspread : D ≤ 16 * G := by linarith
  linarith

/-- All inputs of the energy-projection estimate hold on two
consensus tokens at `β=1`, taking the projection to be the same tuple. -/
example : 2 ≤ 2 ∧ (0 : ℝ) < 1 ∧
    (1 / 2 : ℝ) * ∑ i : Idx 2,
      ((fun _ => 0 : Idx 2 → ℝ) i - (fun _ => 0 : Idx 2 → ℝ) i) ^ 2 ≤
      angularEβ 2 1 (fun _ => 0) - angularEβ 2 1 (fun _ => 0) := by
  norm_num

/-- **The square-root energy discrepancy of an in-cap slow projection
is bounded by `3 exp(-λβ/2)`.** Source: arXiv:2410.06833v1,
§3.2, `eq: otto.attention`, with the missing cap containment stated
explicitly. The upper half of (H1) and a flow equation are unnecessary
for this stronger pointwise conclusion. -/
theorem angularEβ_sqrt_gain_le_at_slow_point (n : ℕ) (hn : 2 ≤ n) (k : ℕ)
    (β : ℝ) (hβ : 0 < β) (α τ lam : ℝ) (hrate : lam / 2 ≤ 1 - α)
    (ω : Idx k → ℝ)
    (hα : ∀ q q' : Idx k, q ≠ q' → ∀ u v : ℝ,
      1 - 2 * τ ≤ Real.cos (u - ω q) → 1 - 2 * τ ≤ Real.cos (v - ω q') →
        Real.cos (u - v) ≤ α)
    (U V : Idx n → ℝ) (hcaps : ∀ i : Idx n, ∃ q : Idx k,
      1 - 2 * τ ≤ Real.cos (V i - ω q))
    (hN : V ∈ slowManifold n β τ lam k ω)
    (hproj : (1 / 2 : ℝ) * ∑ i : Idx n, (U i - V i) ^ 2 ≤
      angularEβ n β V - angularEβ n β U) :
    Real.sqrt (angularEβ n β V - angularEβ n β U) ≤
      3 * Real.exp (-(lam * β / 2)) := by
  have hgap := angularEβ_gain_le_eight_target_gradient n hn β hβ U V hproj
  have hgrad := angularGrad_sq_sum_le_at_slow_point n (by omega) k β hβ.le
    α τ lam hrate ω hα V hcaps hN
  apply Real.sqrt_le_iff.mpr
  refine ⟨by positivity, ?_⟩
  nlinarith [sq_nonneg (Real.exp (-(lam * β / 2)))]

/-- Every input of the square-root projection estimate is realized
simultaneously by a two-token consensus in one cap. -/
example : 2 ≤ 2 ∧ (0 : ℝ) < 1 ∧ (1 / 2 : ℝ) ≤ 1 - 0 ∧
    (∀ q q' : Idx 1, q ≠ q' → ∀ u v : ℝ,
      1 - 2 * (1 / 100 : ℝ) ≤ Real.cos (u - (fun _ => 0 : Idx 1 → ℝ) q) →
      1 - 2 * (1 / 100 : ℝ) ≤ Real.cos (v - (fun _ => 0 : Idx 1 → ℝ) q') →
        Real.cos (u - v) ≤ 0) ∧
    (∀ i : Idx 2, ∃ q : Idx 1,
      1 - 2 * (1 / 100 : ℝ) ≤ Real.cos ((fun _ => 0 : Idx 2 → ℝ) i -
        (fun _ => 0 : Idx 1 → ℝ) q)) ∧
    (fun _ : Idx 2 => (0 : ℝ)) ∈ slowManifold 2 1 (1 / 100) 1 1 (fun _ => 0) ∧
    (1 / 2 : ℝ) * ∑ i : Idx 2,
      ((fun _ => 0 : Idx 2 → ℝ) i - (fun _ => 0 : Idx 2 → ℝ) i) ^ 2 ≤
      angularEβ 2 1 (fun _ => 0) - angularEβ 2 1 (fun _ => 0) := by
  refine ⟨le_rfl, one_pos, by norm_num,
    fun q q' hne => (hne (Subsingleton.elim q q')).elim,
    fun _ => ⟨0, by norm_num⟩, ?_, by norm_num⟩
  intro q i j _ _
  simp only [sub_self, abs_zero]
  exact (Real.exp_pos _).le

end Transformer.Metastability
