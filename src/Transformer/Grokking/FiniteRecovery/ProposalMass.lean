module

public import Transformer.Grokking.FiniteRecovery.LabelOrbit
public import Transformer.Grokking.FiniteRecovery.GaussianBox
public import Transformer.Grokking.FiniteRecovery.DivisionMask
public import Transformer.Grokking.FiniteRecovery.DivisionNative
public import Mathlib.Data.Fintype.Perm

/-! Quantify the uniform single-proposal constant used by
`compact_shift_mass`. A vocabulary-symmetric center can make the mass of
strict modular division success small even for an identifiable mask. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Set MeasureTheory ProbabilityTheory Metric
open scoped NNReal ENNReal

variable {A : Type*} [Fintype A] {H d L T : ℕ}

/-- An iid proposal treats all real parameter coordinates alike, including
the embeddings of unused answer tokens. This is the noise law at a fixed
checkpoint, with every gate and frequency included. -/
theorem proposal_relabel_preimage (μ : Measure ℝ) [IsProbabilityMeasure μ] (e : Equiv.Perm A)
    (U : Set (Weights A H d L)) :
    Measure.pi (fun _ : Parameter A H d L => μ) U =
      Measure.pi (fun _ : Parameter A H d L => μ) (relabelWeights e ⁻¹' U) := by
  let E := parameterTokenEquiv (H := H) (d := d) (L := L) e
  let M := MeasurableEquiv.piCongrLeft (fun _ : Parameter A H d L => ℝ) E
  have hm := measurePreserving_piCongrLeft (fun _ : Parameter A H d L => μ) E
  have he (z : Weights A H d L) : M z = relabelWeights e z := by
    funext j
    obtain ⟨i,rfl⟩ := E.surjective j
    change (MeasurableEquiv.piCongrLeft (fun _ => ℝ) E z) (E i) = z (E.symm (E i))
    rw [MeasurableEquiv.piCongrLeft_apply_apply,E.symm_apply_apply]
  rw [← hm.measure_preimage_equiv]
  apply congrArg (Measure.pi (fun _ : Parameter A H d L => μ))
  ext z
  exact iff_of_eq (congrArg (· ∈ U) (he z))

variable [DecidableEq A]

omit [Fintype A] in
/-- Distinct assignments of absent labels have disjoint strict-win events.
This uses only invariance of the draw law and equivariance of the scores. -/
theorem symmetric_assignment_mass {D : Type*} [MeasurableSpace D] (law : Measure D)
    [IsProbabilityMeasure law] (free : Finset A) (score : D → free → A → ℝ)
    (meas : ∀ f, MeasurableSet {z | StrictOn (score z) f})
    (move : Equiv.Perm free → D → D)
    (invariant : ∀ σ U, law U = law (move σ ⁻¹' U))
    (equivariant : ∀ σ z x a, score (move σ z) x (Equiv.Perm.ofSubtype σ a) = score z x a) :
    law.real {z | StrictOn (score z) (fun x : free => x.val)} ≤ 1/(free.card.factorial : ℝ) := by
  classical
  let label (σ : Equiv.Perm free) (x : free) : A := σ x
  have hinj : Function.Injective label := fun _ _ eq => Equiv.ext fun x => Subtype.ext (congrFun eq x)
  have h := strict_assignment_first_hit law ({0} : Finset ℕ) Finset.univ (Equiv.refl free)
    (Finset.mem_univ _) label hinj.injOn (fun z _ => score z) (fun _ σ => meas (label σ))
    (fun σ _ => ⟨Equiv.Perm.ofSubtype σ,move σ,Equiv.Perm.ofSubtype_apply_coe σ,invariant σ,
      fun z _ => equivariant σ z⟩)
  simpa [label,Fintype.card_perm] using h

/-- One hidden prompt for each of k distinct correct tokens, none of those
tokens in any of the selected prompts. At a center invariant under their
permutations, at most one of k! equally probable strict assignments wins. -/
theorem symmetric_proposal_mass (μ : Measure ℝ) [IsProbabilityMeasure μ] (r : ℕ)
    (free : Finset A) (prompt : free → Word A (T+1))
    (unused : ∀ x j, prompt x j ∉ free) (center : Weights A H d L)
    (fixed : ∀ σ : Equiv.Perm free, relabelWeights (Equiv.Perm.ofSubtype σ) center = center) :
    let law := Measure.pi (fun _ : Parameter A H d L => μ)
    let score (z : Weights A H d L) x := (Model.fromWeights (center+z)).logits r (prompt x) (Fin.last T)
    law.real {z | StrictOn (score z) (fun x : free => x.val)} ≤ 1/(free.card.factorial : ℝ) := by
  classical
  intro law score
  apply symmetric_assignment_mass law free score (fun f =>
    ((strict_prompt_weights_open (H := H) (d := d) (L := L) r prompt f).preimage
      (continuous_const.add continuous_id)).measurableSet) (fun σ => relabelWeights (Equiv.Perm.ofSubtype σ))
    (fun σ => proposal_relabel_preimage μ (Equiv.Perm.ofSubtype σ))
  · intro σ z x a
    let e := Equiv.Perm.ofSubtype σ
    have pr : e ∘ prompt x = prompt x := funext fun j =>
      Equiv.Perm.ofSubtype_apply_of_not_mem σ (unused x j)
    have shift : center + relabelWeights e z = relabelWeights e (center+z) := by
      change center + relabelWeights e z = relabelWeights e center + relabelWeights e z
      rw [fixed σ]
    unfold score
    rw [shift,fromWeights_relabel]
    simpa only [pr] using
      (Model.logits_relabel e (Model.fromWeights (center+z)) r (prompt x) (Fin.last T) a)

/-- Residues 10 through 82 have token IDs 32 through 104. -/
def divisionProbeTokens : Finset (Fin 239) := Finset.Icc 32 104

def divisionProbeIndex (x : divisionProbeTokens) : Fin 73 :=
  ⟨x.val.val-32,by
    obtain ⟨lo,hi⟩ := Finset.mem_Icc.mp x.property
    change 32 ≤ x.val.val at lo
    change x.val.val ≤ 104 at hi
    omega⟩

/-- Every perfect hidden table also solves the selected prompt for each
of the 73 different quotient tokens. -/
theorem division_probe_ceiling {D : Type*} [MeasurableSpace D] (law : Measure D) [IsProbabilityMeasure law]
    (score : D → DivisionInput → Fin 239 → ℝ) (hidden : Set DivisionInput) (probe : Fin 73 → DivisionInput)
    (inHidden : ∀ i, probe i ∈ hidden) (answers : ∀ i, (divisionAnswer (probe i)).val = 32+i.val)
    (bound : law.real {z | StrictOn (fun x => score z (probe (divisionProbeIndex x))) (fun x => x.val)} ≤
      1/(divisionProbeTokens.card.factorial : ℝ)) :
    law.real {z | ∀ x ∈ hidden, ∀ b, b ≠ divisionAnswer x → score z x b < score z x (divisionAnswer x)} ≤
      1/(Nat.factorial 73 : ℝ) := by
  have answer (x : divisionProbeTokens) : divisionAnswer (probe (divisionProbeIndex x)) = x.val := by
    apply Fin.ext
    rw [answers]
    have lo := (Finset.mem_Icc.mp x.property).1
    change 32 ≤ x.val.val at lo
    change 32+(x.val.val-32) = x.val.val
    omega
  refine (measureReal_mono (μ := law) (s₂ := {z | StrictOn
    (fun x => score z (probe (divisionProbeIndex x))) (fun x => x.val)}) ?_).trans ?_
  · intro z hz x b hb
    simpa only [answer] using
      (hz (probe (divisionProbeIndex x)) (inHidden _) b (fun eq => hb (eq.trans (answer x))))
  · simpa only [show divisionProbeTokens.card = 73 by decide] using bound

/-- A finite witness verifies the actual split without using a trained
checkpoint. Any uniform lower bound valid on a set containing the zero
center must also satisfy this upper bound. -/
theorem division_zero_proposal_ceiling (hidden : Set DivisionInput) (probe : Fin 73 → DivisionInput)
    (inHidden : ∀ i, probe i ∈ hidden) (answers : ∀ i, (divisionAnswer (probe i)).val = 32+i.val)
    (unused : ∀ i j, divisionPrompt (probe i) j ∉ divisionProbeTokens) (variance : ℝ≥0) :
    let law := Measure.pi (fun _ : Parameter (Fin 239) 4 16 6 => gaussianReal 0 variance)
    law.real {z | ∀ x ∈ hidden, ∀ b, b ≠ divisionAnswer x →
      (Model.fromWeights z).logits 1 (divisionPrompt x) (Fin.last 4) b <
        (Model.fromWeights z).logits 1 (divisionPrompt x) (Fin.last 4) (divisionAnswer x)} ≤
        1/(Nat.factorial 73 : ℝ) := by
  intro law
  apply division_probe_ceiling law (fun z x => (Model.fromWeights z).logits 1 (divisionPrompt x) (Fin.last 4))
    hidden probe inHidden answers
  simpa only [zero_add] using symmetric_proposal_mass (H := 4) (d := 16) (L := 6) (gaussianReal 0 variance) 1
    divisionProbeTokens (fun x => divisionPrompt (probe (divisionProbeIndex x)))
    (fun x => unused (divisionProbeIndex x)) 0 (fun _ => rfl)

/-- The original compact-ball proof supplies a strictly positive p. The
largest possible p for that proof is already bounded by the zero center,
at every nonnegative radius and every positive noise variance. -/
theorem division_uniform_proposal_range (hidden : Set DivisionInput) (probe : Fin 73 → DivisionInput)
    (inHidden : ∀ i, probe i ∈ hidden) (answers : ∀ i, (divisionAnswer (probe i)).val = 32+i.val)
    (unused : ∀ i j, divisionPrompt (probe i) j ∉ divisionProbeTokens)
    (v : ℝ≥0) (hv : 0 < v) (R : ℝ) (hR : 0 ≤ R) :
    let law := Measure.pi (fun _ : Parameter (Fin 239) 4 16 6 => gaussianReal 0 v)
    let U := {w : Weights (Fin 239) 4 16 6 | ∀ x ∈ hidden, ∀ b, b ≠ divisionAnswer x →
      (Model.fromWeights w).logits 1 (divisionPrompt x) (Fin.last 4) b <
        (Model.fromWeights w).logits 1 (divisionPrompt x) (Fin.last 4) (divisionAnswer x)}
    let p := uniformProposalMass law (closedBall 0 R) U
    0 < p ∧ p ≤ ENNReal.ofReal (1/(Nat.factorial 73 : ℝ)) := by
  intro law U p
  let : (gaussianReal 0 v).IsOpenPosMeasure := (gaussianReal_absolutelyContinuous' 0 hv.ne').isOpenPosMeasure
  constructor
  · apply uniformProposalMass_pos law (isCompact_closedBall 0 R)
    · let prompt (x : hidden) := divisionPrompt x.val
      simpa only [U,StrictOn,Subtype.forall,prompt] using
        (strict_prompt_weights_open (H := 4) (d := 16) (L := 6) 1 prompt (fun x => divisionAnswer x.val))
    · obtain ⟨w,hw⟩ := Division.division_native_realization
      exact ⟨sgdInitialWeights true w,fun x _ => hw x⟩
  · have atZero : p ≤ law U := by
      simpa only [p,zero_add,ofPred_mem_eq] using
        uniformProposalMass_le law (closedBall (0 : Weights (Fin 239) 4 16 6) R) U (mem_closedBall_self hR)
    rw [← ofReal_measureReal (μ := law) (s := U)] at atZero
    exact atZero.trans (ENNReal.ofReal_le_ofReal
      (division_zero_proposal_ceiling hidden probe inHidden answers unused v))

end Transformer.Grokking.FiniteRecovery.Argand
