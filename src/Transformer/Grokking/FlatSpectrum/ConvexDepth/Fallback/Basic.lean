module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.ComplexHead

/-!
# Actual bounded heads with a fixed input fallback

Source: `review/flat_spectrum.md`, CM38. A fixed support predicate selects
candidate features on supported inputs and parent features elsewhere. The
actual combined coefficient tensor stays linear and independently represents
both complex coordinates. Flat component codes supply a feature bound, not
an assertion that the routed feature matrix itself is unitary.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.Fallback

open Matrix
open scoped BigOperators

variable {K L X C : Type*} [Fintype K] [Fintype L]

/-- Routing is a fixed input function, independent of the trained head,
CM38. The parent coordinates are active exactly outside candidate support. -/
def features (known : X → Bool) (candidate : K → X → ℝ) (parent : L → X → ℝ) :
    K ⊕ L → X → ℝ :=
  Sum.elim (fun k x => if known x = true then candidate k x else 0)
    (fun l x => if known x = true then 0 else parent l x)

/-- Store both actual head blocks without a learned factorization, CM38. -/
def coefficients (W : K → C → ℝ) (V : L → C → ℝ) : K ⊕ L → C → ℝ := Sum.elim W V

/-- The sum over actual routed coordinates equals the selected head's
logits on every input, CM38. This includes previously unseen inputs. -/
theorem logits (known : X → Bool) (candidate : K → X → ℝ) (parent : L → X → ℝ)
    (W : K → C → ℝ) (V : L → C → ℝ) (x : X) :
    headLogits (features known candidate parent) (coefficients W V) x =
      if known x = true then headLogits candidate W x else headLogits parent V x := by
  funext c
  unfold headLogits
  rw [Fintype.sum_sum_type]
  by_cases h : known x = true <;> simp [features, coefficients, h]

/-- Both stored blocks are charged even though only one predicts at an
input, CM38. A gate does not justify omitting the parent's storage budget. -/
theorem coefficients_budget {W : K → C → ℝ} {V : L → C → ℝ} {R T : ℝ}
    (hW : W ∈ headBudget R) (hV : V ∈ headBudget T) :
    coefficients W V ∈ headBudget (R + T) := by
  intro c
  rw [Fintype.sum_sum_type]
  exact add_le_add (hW c) (hV c)

omit [Fintype K] [Fintype L] in
/-- Zeroing inactive features preserves a common nonnegative coordinate
bound, CM38. The gate need not be linear in the input. -/
theorem features_bound (known : X → Bool) (candidate : K → X → ℝ) (parent : L → X → ℝ)
    (x : X) {M : ℝ} (hM : 0 ≤ M) (hc : ∀ k, |candidate k x| ≤ M)
    (hp : ∀ l, |parent l x| ≤ M) : ∀ i, |features known candidate parent i x| ≤ M := by
  intro i
  cases i with
  | inl k => by_cases h : known x = true <;> simp [features, h, hc k, hM]
  | inr l => by_cases h : known x = true <;> simp [features, h, hp l, hM]

variable {I J : Type*} [Fintype I] [Fintype J]

/-- Gating the realifications still computes the real parts of the
actual complex matrix products, CM38. Both coefficient blocks remain real
and imaginary parameters; routing does not replace their forward pass. -/
theorem complex_logits (known : X → Bool) (U : Matrix I I ℂ) (V : Matrix J J ℂ)
    (candidateState : X → I) (parentState : X → J)
    (H : Matrix I C ℂ) (G : Matrix J C ℂ) (x : X) (c : C) :
    headLogits (features known (fun k x => complexFeatures U k (candidateState x))
      (fun l x => complexFeatures V l (parentState x)))
      (coefficients (complexCoefficients H) (complexCoefficients G)) x c =
      if known x = true then ((U * H) (candidateState x) c).re
        else ((V * G) (parentState x) c).re := by
  rw [logits]
  change (if known x = true then
    headLogits (complexFeatures U) (complexCoefficients H) (candidateState x) else
    headLogits (complexFeatures V) (complexCoefficients G) (parentState x)) c = _
  by_cases h : known x = true <;> simp [h, complex_head_logits]

omit [Fintype I] [Fintype J] in
/-- Every combined real coefficient tensor is represented by two actual
complex blocks, CM38. No independence restriction is imposed on its entries. -/
theorem complex_coefficients_surjective (W : (I × Bool) ⊕ (J × Bool) → C → ℝ) :
    ∃ (H : Matrix I C ℂ) (G : Matrix J C ℂ),
      coefficients (complexCoefficients H) (complexCoefficients G) = W := by
  obtain ⟨H, hH⟩ := complexCoefficients_surjective (fun k c => W (.inl k) c)
  obtain ⟨G, hG⟩ := complexCoefficients_surjective (fun l c => W (.inr l) c)
  refine ⟨H, G, ?_⟩
  funext k c
  cases k <;> simp [coefficients, hH, hG]

variable [DecidableEq I] [DecidableEq J]

/-- The flat component codes retain their common feature bound after
routing, CM38. No spectral equality for the routed input matrix is claimed. -/
theorem complex_features_bound (known : X → Bool) (U : Matrix I I ℂ) (V : Matrix J J ℂ)
    (hU : U ∈ Matrix.unitaryGroup I ℂ) (hV : V ∈ Matrix.unitaryGroup J ℂ)
    (candidateState : X → I) (parentState : X → J) (x : X) :
    ∀ k, |features known (fun k x => complexFeatures U k (candidateState x))
      (fun l x => complexFeatures V l (parentState x)) k x| ≤ 1 := by
  exact features_bound known _ _ x (by norm_num)
    (fun k => complexFeatures_bound U hU k _) (fun l => complexFeatures_bound V hV l _)

/-- Flat component bounds control every feasible routed logit, CM38.
The bound covers the whole learned real/imaginary tensor, not only an initializer. -/
theorem complex_logits_bound (known : X → Bool) (U : Matrix I I ℂ) (V : Matrix J J ℂ)
    (hU : U ∈ Matrix.unitaryGroup I ℂ) (hV : V ∈ Matrix.unitaryGroup J ℂ)
    (candidateState : X → I) (parentState : X → J)
    {W : (I × Bool) ⊕ (J × Bool) → C → ℝ} {R : ℝ} (hW : W ∈ headBudget R)
    (x : X) (c : C) :
    |headLogits (features known (fun k x => complexFeatures U k (candidateState x))
      (fun l x => complexFeatures V l (parentState x))) W x c| ≤ R := by
  simpa only [mul_one] using headLogits_bound _ hW (M := 1) (by norm_num) x
    (complex_features_bound known U V hU hV candidateState parentState x) c

omit [DecidableEq I] [DecidableEq J] in
/-- Gating retains both stored complex blocks: this is the exact number
of real coefficient slots, CM38. Inference sparsity does not remove parameters. -/
theorem coefficient_count [Fintype C] :
    Fintype.card (((I × Bool) ⊕ (J × Bool)) × C) =
      2 * (Fintype.card I + Fintype.card J) * Fintype.card C := by
  simp only [Fintype.card_prod, Fintype.card_sum, Fintype.card_bool]
  ring

example : coefficients (fun (_ : Unit) (_ : Bool) => (1 : ℝ))
    (fun (_ : Unit) (_ : Bool) => (2 : ℝ)) ∈ headBudget 3 := by
  have h := coefficients_budget
    (W := fun (_ : Unit) (_ : Bool) => (1 : ℝ))
    (V := fun (_ : Unit) (_ : Bool) => (2 : ℝ))
    (R := 1) (T := 2) (by intro c; simp) (by intro c; simp)
  norm_num at h ⊢
  exact h

example := features_bound (fun x : ℝ => decide (x ≤ 0)) (fun (_ : Unit) x => x)
  (fun (_ : Unit) x => x ^ 2) 2 (M := 4) (by norm_num)
  (by intro k; norm_num) (by intro k; norm_num)

example := complex_features_bound (fun x : Bool => x) fourier4 fourier4
  fourier4_unitary fourier4_unitary (fun _ => 0) (fun _ => 1) false

example := complex_logits_bound (C := Bool) (fun x : Bool => x) fourier4 fourier4
  fourier4_unitary fourier4_unitary (fun _ => 0) (fun _ => 1)
  (W := fun _ _ => 1 / 2) (R := 8)
  (by intro c; norm_num [Fintype.sum_sum_type, Fintype.sum_prod_type, Fintype.sum_bool]) false true

example : Fintype.card (((Fin 4 × Bool) ⊕ (Fin 4 × Bool)) × Bool) = 32 := by
  norm_num [coefficient_count]

end Transformer.Grokking.FlatSpectrum.ConvexDepth.Fallback
