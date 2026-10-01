/-
# IC-EoT: the shared latent gate-value product

arXiv:2603.22095v2, §3.3, Lemma 2 and Eqs. (18)–(20).
The Jensen proof below works for every convex monotone non-negative
activation, including non-smooth ReLU, without differentiability assumptions.
The two branches must share the same scalar latent variable.
-/

import Transformer.ICEoT.Section3_Closure

noncomputable section

namespace Transformer.ICEoT

/-- One scalar channel of Eq. (20), with the diagonal coefficients of
Eqs. (18), (19); §3.3, Lemma 2. -/
def gateValue (φ : ℝ → ℝ) (dA dV b z : ℝ) : ℝ := φ (dA * z + b) * (dV * z)

/-- The non-negativity conclusion of §3.3, Lemma 2. -/
theorem gateValue_nonnegative (φ : ℝ → ℝ) (dA dV b z : ℝ)
    (hφ : ∀ x, 0 ≤ φ x) (hV : 0 ≤ dV) (hz : 0 ≤ z) :
    0 ≤ gateValue φ dA dV b z := mul_nonneg (hφ _) (mul_nonneg hV hz)

example : (∀ x, 0 ≤ relu x) ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 2 :=
  ⟨relu_conditions.2.2, zero_le_one, by norm_num⟩

/-- The monotonicity conclusion of §3.3, Lemma 2, on the entire
non-negative half-line; the gate bias has no sign restriction. -/
theorem gateValue_monotone (φ : ℝ → ℝ) (dA dV b : ℝ)
    (hφ : ActivationConditions φ) (hA : 0 ≤ dA) (hV : 0 ≤ dV) :
    MonotoneOn (gateValue φ dA dV b) (Set.Ici 0) := by
  intro x hx y hy hxy
  apply mul_le_mul
  · exact hφ.2.1 (add_le_add (mul_le_mul_of_nonneg_left hxy hA) le_rfl)
  · exact mul_le_mul_of_nonneg_left hxy hV
  · exact mul_nonneg hV hx
  · exact hφ.2.2 _

example : ActivationConditions relu ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 :=
  ⟨relu_conditions, zero_le_one, zero_le_one⟩

/-- The convexity conclusion of §3.3, Lemma 2. An algebraic Jensen
argument proves the non-smooth case directly, completing the source's
smooth-approximation argument without changing its hypotheses. -/
theorem gateValue_convex (φ : ℝ → ℝ) (dA dV b : ℝ)
    (hφ : ActivationConditions φ) (hA : 0 ≤ dA) (hV : 0 ≤ dV) :
    ConvexOn ℝ (Set.Ici 0) (gateValue φ dA dV b) := by
  refine ⟨convex_Ici 0, ?_⟩
  intro x hx y hy a c ha hc hac
  simp only [smul_eq_mul]
  have hj := hφ.1.2 (Set.mem_univ (dA * x + b))
    (Set.mem_univ (dA * y + b)) ha hc hac
  simp only [smul_eq_mul] at hj
  have harg : a * (dA * x + b) + c * (dA * y + b) =
      dA * (a * x + c * y) + b := by
    calc
      _ = dA * (a * x + c * y) + (a + c) * b := by ring
      _ = _ := by rw [hac, one_mul]
  rw [harg] at hj
  have hcross : 0 ≤ (y - x) * (φ (dA * y + b) - φ (dA * x + b)) := by
    rcases le_total x y with hxy | hyx
    · exact mul_nonneg (sub_nonneg.mpr hxy) (sub_nonneg.mpr
        (hφ.2.1 (add_le_add (mul_le_mul_of_nonneg_left hxy hA) le_rfl)))
    · exact mul_nonneg_of_nonpos_of_nonpos (sub_nonpos.mpr hyx)
        (sub_nonpos.mpr (hφ.2.1
          (add_le_add (mul_le_mul_of_nonneg_left hyx hA) le_rfl)))
  have hweighted := mul_nonneg (mul_nonneg (mul_nonneg ha hc) hV) hcross
  have hz : 0 ≤ dV * (a * x + c * y) :=
    mul_nonneg hV (add_nonneg (mul_nonneg ha hx) (mul_nonneg hc hy))
  have hfirst := mul_le_mul_of_nonneg_right hj hz
  unfold gateValue
  calc
    φ (dA * (a * x + c * y) + b) * (dV * (a * x + c * y)) ≤
        (a * φ (dA * x + b) + c * φ (dA * y + b)) *
          (dV * (a * x + c * y)) := hfirst
    _ ≤ a * (φ (dA * x + b) * (dV * x)) +
        c * (φ (dA * y + b) * (dV * y)) := by
      have hca : c = 1 - a := by linarith
      rw [hca] at hweighted ⊢
      nlinarith only [hweighted]

example : ActivationConditions relu ∧ (0 : ℝ) ≤ 2 ∧ (0 : ℝ) ≤ 3 :=
  ⟨relu_conditions, by norm_num, by norm_num⟩

variable {E : Type*} [AddCommGroup E] [Module ℝ E]

/-- The restricted monotone composition used in §3.3, Proposition 1:
the shared latent variable lies in `Ici 0` because of Eq. (17). -/
theorem nonnegative_composition_convex (g f : ℝ → ℝ)
    (hg : ConvexOn ℝ (Set.Ici 0) g) (hm : MonotoneOn g (Set.Ici 0))
    (hf : ConvexOn ℝ Set.univ f) (hn : ∀ x, 0 ≤ f x) :
    ConvexOn ℝ Set.univ (fun x => g (f x)) := by
  refine ⟨convex_univ, ?_⟩
  intro x hx y hy a b ha hb hab
  exact (hm (hn _) (add_nonneg (smul_nonneg ha (hn x))
    (smul_nonneg hb (hn y))) (hf.2 hx hy ha hb hab)).trans
      (hg.2 (hn x) (hn y) ha hb hab)

example : ConvexOn ℝ (Set.Ici 0) relu ∧ MonotoneOn relu (Set.Ici 0) ∧
    ConvexOn ℝ Set.univ relu ∧ (∀ x, 0 ≤ relu x) :=
  ⟨relu_conditions.1.subset (Set.subset_univ _) (convex_Ici 0),
    relu_conditions.2.1.monotoneOn _, relu_conditions.1, relu_conditions.2.2⟩

/-- The general input-space version of the composition step in §3.3,
Proposition 1, used for every latent channel. -/
theorem gateValue_composition [Preorder E] (φ : ℝ → ℝ) (dA dV b : ℝ)
    (z : E → ℝ) (hφ : ActivationConditions φ) (hA : 0 ≤ dA) (hV : 0 ≤ dV)
    (hz : ConvexOn ℝ Set.univ z ∧ Monotone z) (hn : ∀ x, 0 ≤ z x) :
    ConvexOn ℝ Set.univ (fun x => gateValue φ dA dV b (z x)) ∧
      Monotone (fun x => gateValue φ dA dV b (z x)) := by
  have hg := gateValue_convex φ dA dV b hφ hA hV
  have hm := gateValue_monotone φ dA dV b hφ hA hV
  constructor
  · refine ⟨convex_univ, ?_⟩
    intro x hx y hy a c ha hc hac
    exact (hm (hn _) (add_nonneg (smul_nonneg ha (hn x))
      (smul_nonneg hc (hn y))) (hz.1.2 hx hy ha hc hac)).trans
        (hg.2 (hn x) (hn y) ha hc hac)
  · intro x y h
    exact hm (hn x) (hn y) (hz.2 h)

example : ActivationConditions relu ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 ∧
    (ConvexOn ℝ Set.univ relu ∧ Monotone relu) ∧ (∀ x, 0 ≤ relu x) :=
  ⟨relu_conditions, zero_le_one, zero_le_one,
    ⟨relu_conditions.1, relu_conditions.2.1⟩, relu_conditions.2.2⟩

end Transformer.ICEoT
