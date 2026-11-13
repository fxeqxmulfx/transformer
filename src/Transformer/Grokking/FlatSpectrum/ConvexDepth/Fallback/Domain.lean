module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.Fallback.Basic

/-!
# Freeze the parent in a convex routed-head domain

Source: `review/flat_spectrum.md`, CM38. Affine coordinate constraints freeze
the parent while the candidate may change. The actual logit identity gives
exact parent preservation outside fixed support. Ordinary full-vocabulary CE
remains convex; this is not a claim that covered unseen inputs are correct.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.Fallback

variable {K L X C S : Type*} [Fintype K] [Fintype L]

/-- The whole combined budget and affine equations for the saved parent,
CM38. No prediction or desired-answer assertion is hidden in this domain. -/
def parentDomain (parent : L → C → ℝ) (R : ℝ) : Set (K ⊕ L → C → ℝ) :=
  {W | W ∈ headBudget R ∧ ∀ l c, W (.inr l) c = parent l c}

/-- Fixing one coefficient block is affine and retains joint convexity
of every remaining real/imaginary coordinate, CM38. -/
theorem parentDomain_convex (parent : L → C → ℝ) (R : ℝ) :
    Convex ℝ (parentDomain (K := K) parent R) := by
  intro W hW V hV a b ha hb hab
  refine ⟨(headBudget_convex R) hW.1 hV.1 ha hb hab, ?_⟩
  intro l c
  change a * W (.inr l) c + b * V (.inr l) c = parent l c
  rw [hW.2, hV.2, ← add_mul, hab, one_mul]

/-- Feasible candidate and parent heads give a concrete feasible routed
point at the charged sum of their budgets, CM38. -/
theorem coefficients_mem {W : K → C → ℝ} {parent : L → C → ℝ} {R T : ℝ}
    (hW : W ∈ headBudget R) (hp : parent ∈ headBudget T) :
    coefficients W parent ∈ parentDomain parent (R + T) :=
  ⟨coefficients_budget hW hp, fun _ _ => rfl⟩

/-- Every feasible head preserves the actual parent logits outside the
fixed support, CM38. The parent need not be a correct classifier there. -/
theorem parent_logits (known : X → Bool) (candidate : K → X → ℝ) (previous : L → X → ℝ)
    {parent : L → C → ℝ} {R : ℝ} {W : K ⊕ L → C → ℝ}
    (hW : W ∈ parentDomain parent R) {x : X} (hx : known x = false) :
    headLogits (features known candidate previous) W x = headLogits previous parent x := by
  have heq : W = coefficients (fun k c => W (.inl k) c) parent := by
    funext k c
    cases k with
    | inl k => rfl
    | inr l => exact hW.2 l c
  rw [heq, logits]
  simp only [hx, Bool.false_eq_true, ite_false]

/-- The complete ordinary observed CE stays convex on the frozen-parent
domain, CM38. The support predicate and both feature families are fixed. -/
theorem loss_convex [Fintype C] [Fintype S] (known : X → Bool)
    (candidate : K → X → ℝ) (previous : L → X → ℝ)
    (inputs : S → X) (labels : S → C) (parent : L → C → ℝ) (R : ℝ) :
    ConvexOn ℝ (parentDomain parent R)
      (headLoss (features known candidate previous) inputs labels) := by
  exact (headLoss_convex _ inputs labels R).subset (fun _ h => h.1)
    (parentDomain_convex parent R)

/-- Any feasible finite sequence retains parent logits on unsupported
inputs, CM38. Feasibility is an explicit hypothesis, not a convergence claim. -/
theorem trace_parent_logits (known : X → Bool) (candidate : K → X → ℝ)
    (previous : L → X → ℝ) (parent : L → C → ℝ) (R : ℝ)
    (trace : ℕ → K ⊕ L → C → ℝ) (htrace : ∀ n, trace n ∈ parentDomain parent R)
    {x : X} (hx : known x = false) (n : ℕ) :
    headLogits (features known candidate previous) (trace n) x =
      headLogits previous parent x :=
  parent_logits known candidate previous (htrace n) hx

private noncomputable def candidateExample (k : Fin 2) (c : Bool) : ℝ :=
  if c then (k.val + 1) / 3 else -(k.val + 1) / 3
private noncomputable def parentExample (k : Fin 2) (c : Bool) : ℝ :=
  if c then 2 * (k.val + 1) / 3 else -2 * (k.val + 1) / 3

private theorem candidateExample_mem : candidateExample ∈ headBudget 1 := by
  intro c
  cases c <;> norm_num [candidateExample, Fin.sum_univ_two]

private theorem parentExample_mem : parentExample ∈ headBudget 2 := by
  intro c
  cases c <;> norm_num [parentExample, Fin.sum_univ_two]

example : coefficients candidateExample parentExample ∈ parentDomain parentExample 3 := by
  have h := coefficients_mem candidateExample_mem parentExample_mem
  norm_num at h ⊢
  exact h

example : headLogits
    (features (fun x : Bool => x) (fun (_ : Fin 2) _ => 3) (fun (_ : Fin 2) _ => 4))
    (coefficients candidateExample parentExample) false =
      headLogits (fun (_ : Fin 2) (_ : Bool) => 4) parentExample false := by
  apply parent_logits (R := 1 + 2)
  · exact coefficients_mem candidateExample_mem parentExample_mem
  · rfl

example := loss_convex (fun x : Bool => x) (fun (_ : Fin 2) _ => 3)
  (fun (_ : Fin 2) _ => 4) id id parentExample 3

example := trace_parent_logits (fun x : Bool => x) (fun (_ : Fin 2) _ => 3)
  (fun (_ : Fin 2) _ => 4) parentExample (1 + 2)
  (fun _ => coefficients candidateExample parentExample)
  (fun _ => coefficients_mem candidateExample_mem parentExample_mem) (x := false) rfl 0

example := trace_parent_logits (fun x : Bool => x) (fun (_ : Fin 2) _ => 3)
  (fun (_ : Fin 2) _ => 4) parentExample (1 + 2)
  (fun _ => coefficients candidateExample parentExample)
  (fun _ => coefficients_mem candidateExample_mem parentExample_mem) (x := false) rfl 4

end Transformer.Grokking.FlatSpectrum.ConvexDepth.Fallback
