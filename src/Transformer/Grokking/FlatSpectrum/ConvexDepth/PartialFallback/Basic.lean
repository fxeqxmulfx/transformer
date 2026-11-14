module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.Fallback.Batch

/-!
# Route only to an available saved predecessor

Source: `review/flat_spectrum.md`, CM41. The parent features are genuinely
partial: their input type is the subtype accepted by a fixed domain predicate.
Zero extension and a fixed gate preserve the candidate outside that domain.
Both actual coefficient blocks remain charged; neither support nor availability
asserts that an unseen answer is correct.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.PartialFallback

variable {K L X C : Type*} [Fintype K] [Fintype L]

/-- The candidate also handles inputs outside the saved parent's domain,
CM41. Both predicates are fixed independently of the trained coefficients. -/
def useCandidate (known available : X → Bool) (x : X) : Bool :=
  known x || !(available x)

/-- Extend actual partial parent coordinates by zero, CM41. The parent is
never applied to an input without a proof that its fixed domain accepts it. -/
def zeroExtend (available : X → Bool) (parent : L → {x // available x = true} → ℝ)
    (l : L) (x : X) : ℝ :=
  if h : available x = true then parent l ⟨x, h⟩ else 0

/-- Actual disjoint candidate and parent feature coordinates, CM41. -/
def features (known available : X → Bool) (candidate : K → X → ℝ)
    (parent : L → {x // available x = true} → ℝ) : K ⊕ L → X → ℝ :=
  Fallback.features (useCandidate known available) candidate (zeroExtend available parent)

/-- Expand the actual finite dot product on all three routing regions,
CM41. In particular an unavailable predecessor cannot suppress the candidate. -/
theorem logits (known available : X → Bool) (candidate : K → X → ℝ)
    (parent : L → {x // available x = true} → ℝ)
    (W : K → C → ℝ) (P : L → C → ℝ) (x : X) :
    headLogits (features known available candidate parent) (Fallback.coefficients W P) x =
      if known x = true then headLogits candidate W x
      else if h : available x = true then headLogits parent P ⟨x, h⟩
      else headLogits candidate W x := by
  rw [features, Fallback.logits]
  by_cases hk : known x = true
  · simp [useCandidate, hk]
  · by_cases ha : available x = true
    · simp [useCandidate, hk, ha]
      funext c
      unfold headLogits
      simp only [zeroExtend, ha, dite_true]
    · simp [useCandidate, hk, ha]

omit [Fintype K] [Fintype L] in
/-- Extending a partial feature by zero preserves its nonnegative bound,
CM41. No value or bound outside the parent's original domain is assumed. -/
theorem zeroExtend_bound (available : X → Bool)
    (parent : L → {x // available x = true} → ℝ) {M : ℝ} (hM : 0 ≤ M)
    (hp : ∀ l x, |parent l x| ≤ M) (l : L) (x : X) :
    |zeroExtend available parent l x| ≤ M := by
  by_cases ha : available x = true
  · simpa only [zeroExtend, ha, dite_true] using hp l ⟨x, ha⟩
  · simp [zeroExtend, ha, hM]

omit [Fintype K] [Fintype L] in
/-- The actual routed feature coordinates obey the same common bound,
CM41. The saved parent is only required to be bounded on its own domain. -/
theorem features_bound (known available : X → Bool) (candidate : K → X → ℝ)
    (parent : L → {x // available x = true} → ℝ) {M : ℝ} (hM : 0 ≤ M)
    (hc : ∀ k x, |candidate k x| ≤ M) (hp : ∀ l x, |parent l x| ≤ M) (x : X) :
    ∀ i, |features known available candidate parent i x| ≤ M := by
  apply Fallback.features_bound (useCandidate known available) candidate
    (zeroExtend available parent) x hM (fun k => hc k x)
  exact fun l => zeroExtend_bound available parent hM hp l x

/-- Both stored heads pay their full budgets even when the parent is
unavailable at a query, CM41. The bound controls the actual combined logits. -/
theorem logits_bound (known available : X → Bool) (candidate : K → X → ℝ)
    (parent : L → {x // available x = true} → ℝ)
    {W : K → C → ℝ} {P : L → C → ℝ} {R T M : ℝ}
    (hW : W ∈ headBudget R) (hP : P ∈ headBudget T) (hM : 0 ≤ M)
    (hc : ∀ k x, |candidate k x| ≤ M) (hp : ∀ l x, |parent l x| ≤ M) (x : X) (c : C) :
    |headLogits (features known available candidate parent)
      (Fallback.coefficients W P) x c| ≤ (R + T) * M := by
  apply headLogits_bound _ (Fallback.coefficients_budget hW hP) hM x
  exact features_bound known available candidate parent hM hc hp x

variable {I J : Type*} [Fintype I] [Fintype J] [DecidableEq I] [DecidableEq J]

/-- Flat complex component codes provide a bound for the actual routed
real/imaginary features, CM41. The routed input matrix need not be unitary. -/
theorem complex_features_bound (known available : X → Bool)
    (U : Matrix I I ℂ) (V : Matrix J J ℂ)
    (hU : U ∈ Matrix.unitaryGroup I ℂ) (hV : V ∈ Matrix.unitaryGroup J ℂ)
    (candidateState : X → I) (parentState : {x // available x = true} → J) (x : X) :
    ∀ k, |features known available (fun k x => complexFeatures U k (candidateState x))
      (fun l x => complexFeatures V l (parentState x)) k x| ≤ 1 := by
  apply features_bound known available _ _ (by norm_num)
  · exact fun k x => complexFeatures_bound U hU k (candidateState x)
  · exact fun l x => complexFeatures_bound V hV l (parentState x)

/-- Support of the concrete CM41 domain-routing witness. -/
def witnessKnown (x : Fin 4) : Bool := decide (x.val < 2)
/-- The saved parent's strictly larger but still proper input domain, CM41. -/
def witnessAvailable (x : Fin 4) : Bool := decide (x.val < 3)
/-- The two nonzero candidate coordinates in the CM41 witness. -/
def witnessCandidate (k : Fin 2) (x : Fin 4) : ℝ := k.val + x.val + 1
/-- The actual two parent coordinates, defined only on its domain, CM41. -/
def witnessParent (k : Fin 2) (x : {x // witnessAvailable x = true}) : ℝ :=
  2 * k.val + x.val.val + 1

-- The supported input 0 uses the candidate, unsupported 2 uses the available
-- parent, and unsupported 3 still uses the candidate when the parent is absent.
example := logits witnessKnown witnessAvailable witnessCandidate witnessParent
  (fun _ (_ : Bool) => 1) (fun _ _ => 2) 0
example := logits witnessKnown witnessAvailable witnessCandidate witnessParent
  (fun _ (_ : Bool) => 1) (fun _ _ => 2) 2
example := logits witnessKnown witnessAvailable witnessCandidate witnessParent
  (fun _ (_ : Bool) => 1) (fun _ _ => 2) 3

private theorem witnessCandidate_bound (k : Fin 2) (x : Fin 4) : |witnessCandidate k x| ≤ 5 := by
  fin_cases k <;> fin_cases x <;> norm_num [witnessCandidate]

private theorem witnessParent_bound (k : Fin 2) (x : {x // witnessAvailable x = true}) :
    |witnessParent k x| ≤ 5 := by
  rcases x with ⟨x, hx⟩
  have hx' : x.val < 3 := by simpa [witnessAvailable] using hx
  fin_cases k <;> fin_cases x <;> norm_num [witnessParent] at *

example := zeroExtend_bound witnessAvailable witnessParent (by norm_num : (0 : ℝ) ≤ 5)
  witnessParent_bound 0 3
example := features_bound witnessKnown witnessAvailable witnessCandidate witnessParent
  (by norm_num : (0 : ℝ) ≤ 5) witnessCandidate_bound witnessParent_bound 3
example := logits_bound witnessKnown witnessAvailable witnessCandidate witnessParent
  (W := fun _ (_ : Bool) => 1) (P := fun _ _ => 2) (R := 2) (T := 4)
  (by intro c; norm_num [Fin.sum_univ_two]) (by intro c; norm_num [Fin.sum_univ_two])
  (by norm_num : (0 : ℝ) ≤ 5) witnessCandidate_bound witnessParent_bound 3 true
example := complex_features_bound witnessKnown witnessAvailable fourier4 fourier4
  fourier4_unitary fourier4_unitary (fun x => x) (fun x => x.val) 3

/-- A genuine unavailable-parent query keeps a nonzero candidate logit,
CM41. Routing only by support would instead select the parent's zero extension. -/
theorem unavailable_parent_witness :
    headLogits (features witnessKnown witnessAvailable witnessCandidate witnessParent)
      (Fallback.coefficients (fun _ (_ : Bool) => 1) (fun _ _ => 2)) 3 true = 9 ∧
    headLogits (Fallback.features witnessKnown witnessCandidate
      (zeroExtend witnessAvailable witnessParent))
      (Fallback.coefficients (fun _ (_ : Bool) => 1) (fun _ _ => 2)) 3 true = 0 := by
  constructor
  · rw [logits]
    norm_num [witnessKnown, witnessAvailable, headLogits, witnessCandidate, Fin.sum_univ_two]
  · rw [Fallback.logits]
    norm_num [witnessKnown, witnessAvailable, headLogits, zeroExtend, Fin.sum_univ_two]

end Transformer.Grokking.FlatSpectrum.ConvexDepth.PartialFallback
