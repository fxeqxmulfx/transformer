module

public import Transformer.SequenceModel.Attention.Expressivity.Products
public import Transformer.SequenceModel.Attention.Expressivity.Geometry

/-!
# Finite interpolation on the actual normalized ordered-pair codes

Source: `review/flat_spectrum.md`, *Transformer learning bridges*, E4.
Squares identify diagonal pairs; context/current cross-products identify
unequal pairs and cancel the square contribution. Positive RMS epsilon
only changes two nonzero scales, which fixed output weights compensate.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Expressivity

open GPTMini
open scoped BigOperators

/-- One square per token and one context/current product per ordered pair. -/
abbrev Features (p : ℕ) := Fin p ⊕ (Fin p × Fin p)

/-- Gate selects the square coordinate or the context coordinate. -/
def interpGate {p d : ℕ} (e : (Fin p ⊕ Fin p) ↪ Fin d) :
    Features p → EucSpace d →L[ℝ] ℝ
  | .inl b => PiLp.proj 2 (fun _ : Fin d => ℝ) (e (.inl b))
  | .inr (a, _) => PiLp.proj 2 (fun _ : Fin d => ℝ) (e (.inr a))

/-- Value always selects the current-token coordinate. -/
def interpValue {p d : ℕ} (e : (Fin p ⊕ Fin p) ↪ Fin d) :
    Features p → EucSpace d →L[ℝ] ℝ
  | .inl b => PiLp.proj 2 (fun _ : Fin d => ℝ) (e (.inl b))
  | .inr (_, b) => PiLp.proj 2 (fun _ : Fin d => ℝ) (e (.inl b))

/-- Fixed coefficients compensate the two input scales and cancel the diagonal feature. -/
def interpOut {p d : ℕ} (s t : ℝ) (B : Fin p → Fin p → EucSpace d) : Features p → EucSpace d
  | .inl b => (s ^ 2)⁻¹ • B b b
  | .inr (a, b) => (t ^ 2)⁻¹ • B a b - (s ^ 2)⁻¹ • B b b

/-- E4: finitely many genuine quadratic features interpolate an arbitrary
vector table on all ordered pairs, without input-dependent weights. -/
theorem interpolate_pair {p d m : ℕ} (e : (Fin p ⊕ Fin p) ↪ Fin d)
    (s t : ℝ) (hs : s ≠ 0) (ht : t ≠ 0) (B : Fin p → Fin p → EucSpace m) (a b : Fin p) :
    let x := (if a = b then s else t) • pairCode e a b
    ∑ r, (interpGate e r x * interpValue e r x) • interpOut s t B r = B a b := by
  dsimp only
  rw [Fintype.sum_sum_type, Fintype.sum_prod_type]
  simp only [interpGate, interpValue, PiLp.proj_apply, PiLp.smul_apply, smul_eq_mul,
    pairCode_left, pairCode_right, interpOut]
  by_cases hab : a = b
  · subst a
    simp only [↓reduceIte, mul_zero, zero_mul, zero_smul, Finset.sum_const_zero, add_zero]
    simp only [mul_ite, ite_mul, mul_one, mul_zero, zero_mul, ite_smul, zero_smul]
    simp only [Finset.sum_ite_eq', Finset.mem_univ, ↓reduceIte]
    rw [← sq, smul_smul, mul_inv_cancel₀ (pow_ne_zero 2 hs), one_smul]
  · simp only [hab, ↓reduceIte]
    simp only [mul_ite, ite_mul, mul_one, mul_zero, zero_mul, ite_smul, zero_smul]
    simp only [Finset.sum_ite_eq', Finset.mem_univ, ↓reduceIte]
    simp only [smul_sub, smul_smul]
    rw [← sq, mul_inv_cancel₀ (pow_ne_zero 2 ht), one_smul]
    abel

/-- E4: the actual positive-epsilon branch input is one of precisely those
two scales. The diagonal case is included, rather than dropped by XSA. -/
theorem normalized_pair {p d : ℕ} (e : (Fin p ⊕ Fin p) ↪ Fin d)
    (eps : ℝ) (a b : Fin p) :
    rmsNormEps eps (pairCode e a b) =
      (if a = b then normScale d eps 1 else normScale d eps 2) • pairCode e a b := by
  rw [rms_of_normSq d eps _ _ (pairCode_normSq e a b)]
  split_ifs <;> rfl

end Transformer.SequenceModel.Attention.Expressivity
