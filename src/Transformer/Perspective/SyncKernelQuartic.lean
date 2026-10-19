module

public import Transformer.Perspective.SyncKernel
public import Mathlib.Algebra.Polynomial.Roots
public import Mathlib.Tactic.ComputeDegree

/-!
# The quartic behind the sign of `f_β'''`

Polyanskiy, Rigollet, Yao — arXiv:2507.22857v1, *Synchronization of mean-field
models on the circle*, Appendices A–B, proofs of `lemma:positiveregion` and
`lemma:positiveregion2`.

`f_β'''(x) = p(cos x) e^{β(cos x - 1)}` with the quartic

  `p(z) = -(z + 3βz² - 4β(1 - z²) - 6β²z(1 - z²) + β³(1 - z²)²)`,

so the sign of `f_β'''` is the sign of `p` on `[-1, 1]`. The source evaluates
`p` at a few points; each sign change gives a root by the intermediate value
theorem, and a quartic has no more than four.

* `β > 1/3`: two roots below `-1` and two roots `r₁ < r₂` in `(-1, 1)`; on
  `[-1, 1]`, `p > 0` exactly on `(r₁, r₂)` (`exists_kernelQuartic_roots`).
* `0 < β ≤ 1/3`: three roots in `(-∞, -1]` and one root `r ∈ (0, 1)`; on
  `[-1, 1]`, `p ≥ 0` exactly on `[-1, r]` (`exists_kernelQuartic_root`).
-/

@[expose] public section

open Real Polynomial

namespace Transformer
namespace Perspective

/-- **The quartic** `p(z) = -(z + 3βz² - 4β(1 - z²) - 6β²z(1 - z²) + β³(1 - z²)²)`.

Source: arXiv:2507.22857v1, Appendix A, proof of `lemma:positiveregion`. -/
noncomputable def kernelQuartic (β z : ℝ) : ℝ :=
  -(z + 3 * β * z ^ 2 - 4 * β * (1 - z ^ 2) - 6 * β ^ 2 * z * (1 - z ^ 2) +
    β ^ 3 * (1 - z ^ 2) ^ 2)

/-- `f_β'''(x) = p(cos x) e^{β(cos x - 1)}`, as the source observes. -/
theorem f_β'''_eq_kernelQuartic (β x : ℝ) :
    f_β''' β x = kernelQuartic β (Real.cos x) * Real.exp (β * (Real.cos x - 1)) := by
  rw [f_β''', kernelQuartic, show Real.sin x ^ 4 = (Real.sin x ^ 2) ^ 2 by ring, Real.sin_sq]

theorem continuous_kernelQuartic (β : ℝ) : Continuous (kernelQuartic β) := by
  unfold kernelQuartic
  fun_prop

/-- `p` expanded: `-β³z⁴ - 6β²z³ + (2β³ - 7β)z² + (6β² - 1)z + 4β - β³`. -/
theorem eval_kernelQuartic (β z : ℝ) :
    (C (-β ^ 3) * X ^ 4 + C (-6 * β ^ 2) * X ^ 3 + C (2 * β ^ 3 - 7 * β) * X ^ 2 +
      C (6 * β ^ 2 - 1) * X + C (4 * β - β ^ 3)).eval z = kernelQuartic β z := by
  simp only [eval_add, eval_mul, eval_C, eval_pow, eval_X, kernelQuartic]
  ring

theorem kernelQuartic_one (β : ℝ) : kernelQuartic β 1 = -1 - 3 * β := by
  simp only [kernelQuartic]
  ring

theorem kernelQuartic_neg_one (β : ℝ) : kernelQuartic β (-1) = 1 - 3 * β := by
  simp only [kernelQuartic]
  ring

theorem kernelQuartic_zero (β : ℝ) : kernelQuartic β 0 = 4 * β - β ^ 3 := by
  simp only [kernelQuartic]
  ring

/-- **Five distinct roots are too many**: `p` is a quartic, and `p(1) = -1 - 3β ≠ 0`
for `β > 0`, so `p ≠ 0`. -/
theorem exists_kernelQuartic_ne_zero {β : ℝ} (hβ : 0 < β) {z : Fin 5 → ℝ}
    (hz : Function.Injective z) : ∃ i, kernelQuartic β (z i) ≠ 0 := by
  refine not_forall.mp fun h => ?_
  have h0 := eq_zero_of_natDegree_lt_card_of_eval_eq_zero
    (C (-β ^ 3) * X ^ 4 + C (-6 * β ^ 2) * X ^ 3 + C (2 * β ^ 3 - 7 * β) * X ^ 2 +
      C (6 * β ^ 2 - 1) * X + C (4 * β - β ^ 3)) hz
    (fun i => (eval_kernelQuartic β (z i)).trans (h i))
    (by rw [Fintype.card_fin]; exact Nat.lt_succ_of_le (by compute_degree))
  have h1 := eval_kernelQuartic β 1
  rw [h0, eval_zero, kernelQuartic_one] at h1
  linarith

/-- Four roots `s₁ < s₂ < s₃ < s₄` of `p` are all of its roots. -/
theorem eq_of_kernelQuartic_eq_zero {β : ℝ} (hβ : 0 < β) {s₁ s₂ s₃ s₄ w : ℝ}
    (h₁₂ : s₁ < s₂) (h₂₃ : s₂ < s₃) (h₃₄ : s₃ < s₄) (h₁ : kernelQuartic β s₁ = 0)
    (h₂ : kernelQuartic β s₂ = 0) (h₃ : kernelQuartic β s₃ = 0)
    (h₄ : kernelQuartic β s₄ = 0) (hw : kernelQuartic β w = 0) :
    w = s₁ ∨ w = s₂ ∨ w = s₃ ∨ w = s₄ := by
  by_contra hne
  simp only [not_or] at hne
  obtain ⟨i, hi⟩ := exists_kernelQuartic_ne_zero hβ (z := ![s₁, s₂, s₃, s₄, w]) (by
    intro i j hij
    fin_cases i <;> fin_cases j <;> simp at hij ⊢ <;> simp_all <;> linarith)
  fin_cases i <;> simp_all

/-- **A sign change gives a root** strictly between, by the intermediate value
theorem. -/
theorem exists_kernelQuartic_eq_zero (β : ℝ) {u v : ℝ} (huv : u ≤ v)
    (h : kernelQuartic β u * kernelQuartic β v < 0) :
    ∃ w ∈ Set.Ioo u v, kernelQuartic β w = 0 := by
  have hc := (continuous_kernelQuartic β).continuousOn (s := Set.Icc u v)
  rcases lt_or_gt_of_ne (left_ne_zero_of_mul h.ne) with hu | hu
  · exact intermediate_value_Ioo huv hc ⟨hu, by nlinarith⟩
  · exact intermediate_value_Ioo' huv hc ⟨by nlinarith, hu⟩

/-- `p(-2 - 6/β) < 0`: the leading coefficient `-β³` takes over. -/
theorem kernelQuartic_neg_two_sub_lt {β : ℝ} (hβ : 0 < β) :
    kernelQuartic β (-2 - 6 / β) < 0 := by
  have h : kernelQuartic β (-2 - 6 / β) =
      -(9 * β ^ 4 + 108 * β ^ 3 + 420 * β ^ 2 + 598 * β + 246) / β := by
    simp only [kernelQuartic]
    field_simp
    ring
  rw [h]
  exact div_neg_of_neg_of_pos (by nlinarith [pow_pos hβ 3, pow_pos hβ 4]) hβ

/-- `p(-1 - 2/β) = (β + 2)(5β + 3)/β > 0`. -/
theorem kernelQuartic_neg_one_sub_two_div_pos {β : ℝ} (hβ : 0 < β) :
    0 < kernelQuartic β (-1 - 2 / β) := by
  have h : kernelQuartic β (-1 - 2 / β) = (β + 2) * (5 * β + 3) / β := by
    simp only [kernelQuartic]
    field_simp
    ring
  rw [h]
  positivity

/-- `p(-1 - 1/β) = (5β² + β - 1)/β < 0` for `0 < β ≤ 1/3`. -/
theorem kernelQuartic_neg_one_sub_inv_lt {β : ℝ} (hβ : 0 < β) (hβ' : β ≤ 1 / 3) :
    kernelQuartic β (-1 - 1 / β) < 0 := by
  have h : kernelQuartic β (-1 - 1 / β) = (5 * β ^ 2 + β - 1) / β := by
    simp only [kernelQuartic]
    field_simp
    ring
  rw [h]
  exact div_neg_of_neg_of_pos (by nlinarith) hβ

/-- `p(1 - 0.3/β) = 1.688 - 0.1761/β + 0.24β > 0` for `β > 1/3`, as the source
computes. -/
theorem kernelQuartic_one_sub_div_pos {β : ℝ} (hβ : 1 / 3 < β) :
    0 < kernelQuartic β (1 - 3 / (10 * β)) := by
  have hβ0 : 0 < β := by linarith
  have h : kernelQuartic β (1 - 3 / (10 * β)) =
      (2400 * β ^ 2 + 16880 * β - 1761) / (10000 * β) := by
    simp only [kernelQuartic]
    field_simp
    ring
  rw [h]
  exact div_pos (by nlinarith) (by positivity)

/-- The hypotheses of `exists_kernelQuartic_ne_zero` are satisfiable: `i ↦ i` is
injective on `Fin 5`. -/
example : ∃ i : Fin 5, kernelQuartic 1 ((i : ℕ) : ℝ) ≠ 0 :=
  exists_kernelQuartic_ne_zero one_pos (Nat.cast_injective.comp Fin.val_injective)

/-- The hypotheses of `eq_of_kernelQuartic_eq_zero` and
`exists_kernelQuartic_eq_zero` are satisfiable: at `β = 1`,
`p(z) = -z⁴ - 6z³ - 5z² + 5z + 3` changes sign on `(-8, -3)`, `(-3, -1)`,
`(-1, 0)` and `(0, 1)`, so it has four roots. -/
example : ∃ s₁ s₂ s₃ s₄ : ℝ, s₁ < s₂ ∧ s₂ < s₃ ∧ s₃ < s₄ ∧ kernelQuartic 1 s₁ = 0 ∧
    kernelQuartic 1 s₂ = 0 ∧ kernelQuartic 1 s₃ = 0 ∧ kernelQuartic 1 s₄ = 0 := by
  obtain ⟨s₁, h₁, e₁⟩ := exists_kernelQuartic_eq_zero 1 (show (-8 : ℝ) ≤ -3 by norm_num)
    (by norm_num [kernelQuartic])
  obtain ⟨s₂, h₂, e₂⟩ := exists_kernelQuartic_eq_zero 1 (show (-3 : ℝ) ≤ -1 by norm_num)
    (by norm_num [kernelQuartic])
  obtain ⟨s₃, h₃, e₃⟩ := exists_kernelQuartic_eq_zero 1 (show (-1 : ℝ) ≤ 0 by norm_num)
    (by norm_num [kernelQuartic])
  obtain ⟨s₄, h₄, e₄⟩ := exists_kernelQuartic_eq_zero 1 zero_le_one
    (by norm_num [kernelQuartic])
  exact ⟨s₁, s₂, s₃, s₄, h₁.2.trans h₂.1, h₂.2.trans h₃.1, h₃.2.trans h₄.1, e₁, e₂, e₃, e₄⟩

/-- The hypotheses `0 < β` and `β ≤ 1/3` of the lemmas above are satisfiable
together, and so is `1/3 < β`. -/
example : (0 : ℝ) < 1 / 3 ∧ (1 : ℝ) / 3 ≤ 1 / 3 ∧ (1 : ℝ) / 3 < 1 := by norm_num

end Perspective
end Transformer
