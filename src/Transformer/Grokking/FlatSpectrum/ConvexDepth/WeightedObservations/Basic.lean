module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.CoordinatePrograms
public import Mathlib.Analysis.Calculus.FDeriv.Add

/-!
# Exact multiplicities of original observations

Source: `review/flat_spectrum.md`, CM60. Grouping equal observations is an
exact storage transformation only when every original record contributes its
multiplicity. A minibatch uses its own counts and its actual size, including
the final short batch. This does not justify grouping unequal feature/label
pairs or treating a noisy observed label as an unknown clean target.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.WeightedObservations

open scoped BigOperators

variable {I K : Type*} [DecidableEq K] [Fintype K]

/-- Number of original record IDs in this batch assigned to a group (CM60). -/
def multiplicity (batch : Finset I) (key : I → K) (k : K) : ℕ :=
  (batch.filter (fun i => key i = k)).card

/-- Original empirical average, normalized by the actual record count
(CM60). The zero-inverse convention agrees with CM38 on the empty batch. -/
noncomputable def recordMean (batch : Finset I) (values : I → ℝ) : ℝ :=
  (batch.card : ℝ)⁻¹ * ∑ i ∈ batch, values i

/-- Grouped empirical average with batch-specific multiplicities (CM60).
This uses the original batch size, never the number of distinct groups. -/
noncomputable def groupedMean (batch : Finset I) (key : I → K) (values : K → ℝ) : ℝ :=
  (batch.card : ℝ)⁻¹ * ∑ k, (multiplicity batch key k : ℝ) * values k

/-- Partitioning original IDs into fibers preserves a sum in any additive
commutative monoid, including the space of derivative maps (CM60). -/
theorem sum_by_multiplicity {M : Type*} [AddCommMonoid M]
    (batch : Finset I) (key : I → K) (values : K → M) :
    (∑ i ∈ batch, values (key i)) = ∑ k, multiplicity batch key k • values k := by
  calc
    _ = ∑ k, ∑ i ∈ batch.filter (fun i => key i = k), values k :=
      (Finset.sum_fiberwise' batch key values).symm
    _ = _ := Finset.sum_congr rfl (fun k _ => Finset.sum_const (values k))

/-- Group masses sum to the full number of original records (CM60). -/
theorem multiplicities_sum (batch : Finset I) (key : I → K) :
    ∑ k, multiplicity batch key k = batch.card := by
  have h := sum_by_multiplicity batch key (fun _ => (1 : ℕ))
  simpa only [Finset.sum_const, Nat.nsmul_eq_mul, mul_one] using h.symm

omit [Fintype K] in
/-- An active group contains an actual batch record, and conversely (CM60). -/
theorem multiplicity_pos_iff (batch : Finset I) (key : I → K) (k : K) :
    0 < multiplicity batch key k ↔ ∃ i ∈ batch, key i = k := by
  rw [multiplicity, Finset.card_pos]
  constructor
  · rintro ⟨i, hi⟩
    exact ⟨i, (Finset.mem_filter.mp hi).1, (Finset.mem_filter.mp hi).2⟩
  · rintro ⟨i, hi, hk⟩
    exact ⟨i, Finset.mem_filter.mpr ⟨hi, hk⟩⟩

/-- Exact grouping retains the normalized original objective (CM60). -/
theorem groupedMean_eq (batch : Finset I) (key : I → K) (values : K → ℝ) :
    groupedMean batch key values = recordMean batch (fun i => values (key i)) := by
  unfold groupedMean recordMean
  congr 1
  have h := sum_by_multiplicity batch key values
  simpa only [nsmul_eq_mul] using h.symm

/-- Only rows in the current batch must agree with their representatives.
This is the explicit data-preservation premise for grouping (CM60). -/
theorem groupedMean_eq_of_records (batch : Finset I) (key : I → K)
    (values : K → ℝ) (original : I → ℝ)
    (hrows : ∀ i ∈ batch, original i = values (key i)) :
    groupedMean batch key values = recordMean batch original := by
  rw [groupedMean_eq]
  unfold recordMean
  congr 1
  exact Finset.sum_congr rfl (fun i hi => (hrows i hi).symm)

omit [Fintype K] in
/-- Repeating an identical constraint does not strengthen it; dropping an
active group can. This equivalence keeps every active group (CM60). -/
theorem constraints_iff (batch : Finset I) (key : I → K) (condition : K → Prop) :
    (∀ i ∈ batch, condition (key i)) ↔
      ∀ k, 0 < multiplicity batch key k → condition k := by
  constructor
  · intro h k hk
    obtain ⟨i, hi, rfl⟩ := (multiplicity_pos_iff batch key k).mp hk
    exact h i hi
  · intro h i hi
    exact h (key i) ((multiplicity_pos_iff batch key (key i)).mpr ⟨i, hi, rfl⟩)

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- The actual derivative is the same multiplicity-weighted sum of row
derivatives, divided by the actual batch size (CM60). -/
theorem groupedMean_hasFDerivAt (batch : Finset I) (key : I → K)
    (values : K → E → ℝ) (derivatives : K → E →L[ℝ] ℝ) (x : E)
    (hderiv : ∀ i ∈ batch, HasFDerivAt (values (key i)) (derivatives (key i)) x) :
    HasFDerivAt (fun θ => groupedMean batch key (fun k => values k θ))
      ((batch.card : ℝ)⁻¹ • ∑ k, multiplicity batch key k • derivatives k) x := by
  have heq : (fun θ => groupedMean batch key (fun k => values k θ)) =
      (fun θ => recordMean batch (fun i => values (key i) θ)) := by
    funext θ
    exact groupedMean_eq batch key (fun k => values k θ)
  rw [heq, ← sum_by_multiplicity batch key derivatives]
  have hf : (fun θ => recordMean batch (fun i => values (key i) θ)) =
      (batch.card : ℝ)⁻¹ • ∑ i ∈ batch, values (key i) := by
    funext θ
    simp only [recordMean, Pi.smul_apply, Finset.sum_apply, smul_eq_mul]
  rw [hf]
  exact (HasFDerivAt.sum hderiv).const_smul (batch.card : ℝ)⁻¹

/-- Three distinct IDs, with a repeated group; multiplicity is not a set
of distinct feature/label pairs (CM60's concrete counterexample). -/
def toyKey (i : Fin 3) : Bool := decide (i.val = 2)

/-- The two groups have different frequencies in the original pool (CM60). -/
theorem toy_multiplicities :
    multiplicity Finset.univ toyKey false = 2 ∧
      multiplicity Finset.univ toyKey true = 1 := by
  decide

/-- Unweighted deduplication changes an actual nonconstant empirical mean
from one third to one half (CM60). -/
theorem unweighted_changes_mean :
    groupedMean Finset.univ toyKey (fun k => if k then (1 : ℝ) else 0) ≠
      (2 : ℝ)⁻¹ * ∑ k : Bool, (if k then (1 : ℝ) else 0) := by
  unfold groupedMean
  rw [Fintype.sum_bool, Fintype.sum_bool, toy_multiplicities.1, toy_multiplicities.2]
  norm_num

/-- A one-record tail is normalized by one and has its own group counts;
global counts would give the wrong value (CM60). -/
theorem tail_mean :
    groupedMean ({2} : Finset (Fin 3)) toyKey (fun k => if k then (1 : ℝ) else 0) = 1 := by
  rw [groupedMean_eq]
  norm_num [recordMean, toyKey]
  decide

example : groupedMean Finset.univ toyKey (fun k => if k then (3 : ℝ) else 0) =
    recordMean Finset.univ (fun i => if toyKey i then (3 : ℝ) else 0) :=
  groupedMean_eq_of_records _ _ _ _ (fun _ _ => rfl)

example : (∀ i ∈ (Finset.univ : Finset (Fin 3)), toyKey i = true ∨ toyKey i = false) ↔
    ∀ k, 0 < multiplicity Finset.univ toyKey k → k = true ∨ k = false :=
  constraints_iff (Finset.univ : Finset (Fin 3)) toyKey
    (fun k : Bool => k = true ∨ k = false)

example (x : ℝ) :
    HasFDerivAt (fun θ => groupedMean Finset.univ toyKey (fun _ => θ))
      (((Finset.univ : Finset (Fin 3)).card : ℝ)⁻¹ •
        ∑ k : Bool, multiplicity Finset.univ toyKey k • ContinuousLinearMap.id ℝ ℝ) x :=
  groupedMean_hasFDerivAt _ _ (fun _ θ => θ) _ x (fun _ _ => hasFDerivAt_id x)

end Transformer.Grokking.FlatSpectrum.ConvexDepth.WeightedObservations
