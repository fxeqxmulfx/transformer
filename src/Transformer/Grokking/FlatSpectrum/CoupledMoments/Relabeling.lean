module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.PredictionHull

/-!
# Output relabeling and a convex class independent of label order

Source: `review/flat_spectrum.md`, CM12. Ordinary CE is unchanged when both
class coordinates and target labels are renamed. A fixed complex code instead
changes by composition with that permutation. Take the convex hull of every
renamed feasible family in direct prediction coordinates. This removes the
arbitrary label-order restriction without training a product of free factors.
It neither supplies an efficient permutation search nor identifies missing labels.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments

open scoped BigOperators
open Landscape.Neural (crossEntropy)

variable {m n C S : Type*}

/-- Rename output classes, leaving each visible input cell fixed (CM12). -/
def relabel (σ : Equiv.Perm C) (P : Table m n C) : Table m n C :=
  fun i j c => P i j (σ.symm c)

/-- Class reindexing is linear in the actual prediction coordinates (CM12). -/
theorem relabel_linear (σ : Equiv.Perm C) (P Q : Table m n C) (a b : ℝ) :
    relabel σ (a • P + b • Q) = a • relabel σ P + b • relabel σ Q := by
  funext i j c
  rfl

/-- Sequential class renamings compose in the corresponding order (CM12). -/
theorem relabel_comp (σ τ : Equiv.Perm C) (P : Table m n C) :
    relabel τ (relabel σ P) = relabel (σ.trans τ) P := by
  funext i j c
  rfl

/-- All output codings of a fixed family, convexified in direct prediction
coordinates. There are |C|! possible codings for finite C (CM12). -/
def relabelHull (A : Set (Table m n C)) : Set (Table m n C) :=
  predictionHull (fun σ : Equiv.Perm C => relabel σ '' A)

/-- The enlarged domain is convex even if searching its vertices is difficult
(CM12); this is not a convex parametrization of a learned permutation. -/
theorem relabelHull_convex (A : Set (Table m n C)) : Convex ℝ (relabelHull A) :=
  predictionHull_convex _

/-- Every renamed feasible predictor is retained, at the parent's fixed bounds
in its own coding. A hull mixture need not obey those bounds in one fixed code (CM12). -/
theorem relabel_mem_hull (σ : Equiv.Perm C) {A : Set (Table m n C)} {P : Table m n C}
    (hP : P ∈ A) : relabel σ P ∈ relabelHull A :=
  mem_predictionHull _ ⟨P, hP, rfl⟩

/-- The enlarged class includes every original feasible predictor (CM12). -/
theorem mem_relabelHull {A : Set (Table m n C)} {P : Table m n C} (hP : P ∈ A) :
    P ∈ relabelHull A := by
  have he : relabel (Equiv.refl C) P = P := by funext i j c; rfl
  rw [← he]
  exact relabel_mem_hull (Equiv.refl C) hP

/-- Renaming any mixture in the hull remains in that same hull (CM12). -/
theorem relabel_hull_closed (σ : Equiv.Perm C) {A : Set (Table m n C)}
    {P : Table m n C} (hP : P ∈ relabelHull A) : relabel σ P ∈ relabelHull A := by
  have hc : Convex ℝ {Q : Table m n C | relabel σ Q ∈ relabelHull A} := by
    intro Q hQ R hR a b ha hb hab
    change relabel σ (a • Q + b • R) ∈ relabelHull A
    rw [relabel_linear]
    exact relabelHull_convex A hQ hR ha hb hab
  apply convexHull_min (t := {Q : Table m n C | relabel σ Q ∈ relabelHull A}) ?_ hc hP
  intro Q hQ
  obtain ⟨τ, R, hR, rfl⟩ := Set.mem_iUnion.mp hQ
  change relabel σ (relabel τ R) ∈ relabelHull A
  rw [relabel_comp]
  exact relabel_mem_hull (τ.trans σ) hR

/-- The complete convex family, not just its generators, is invariant under
every output relabeling (CM12). -/
theorem relabelHull_invariant (σ : Equiv.Perm C) (A : Set (Table m n C)) :
    relabel σ '' relabelHull A = relabelHull A := by
  ext P
  constructor
  · rintro ⟨Q, hQ, rfl⟩
    exact relabel_hull_closed σ hQ
  · intro hP
    refine ⟨relabel σ.symm P, relabel_hull_closed σ.symm hP, ?_⟩
    funext i j c
    simp [relabel]

variable [Fintype C]

/-- Reindexing retains nonnegativity and unit row mass (CM12). -/
theorem rows_relabel (σ : Equiv.Perm C) {P : Table m n C} (hP : Rows P) :
    Rows (relabel σ P) := by
  refine ⟨fun i j c => hP.1 i j (σ.symm c), ?_⟩
  intro i j
  exact (σ.symm.sum_comp (P i j)).trans (hP.2 i j)

/-- The code changes with the output labels: it is φ∘σ, not merely a row or
column permutation of the input matrix (CM12). -/
theorem moment_relabel (σ : Equiv.Perm C) (φ : C → ℂ) (P : Table m n C) :
    moment φ (relabel σ P) = moment (φ ∘ σ) P := by
  funext i j
  simpa [moment, relabel] using
    (Equiv.sum_comp σ (fun c => (P i j (σ.symm c) : ℂ) * φ c)).symm

/-- Renaming logits and the actual target preserves ordinary CE exactly
(CM12), without changing the vocabulary or softmax denominator. -/
theorem crossEntropy_relabel (σ : Equiv.Perm C) (z : C → ℝ) (t : C) :
    crossEntropy (fun c => z (σ.symm c)) (σ t) = crossEntropy z t := by
  rw [ConvexRecall.crossEntropy_relative, ConvexRecall.crossEntropy_relative]
  congr 1
  simpa using σ.symm.sum_comp (fun c => Real.exp (z c - z t))

/-- Every hull prediction remains a full class distribution (CM12). -/
theorem relabelHull_rows (A : Set (Table m n C)) (hA : ∀ P ∈ A, Rows P)
    {P : Table m n C} (hP : P ∈ relabelHull A) : Rows P := by
  apply predictionHull_rows _ ?_ hP
  rintro σ Q ⟨R, hR, rfl⟩
  exact rows_relabel σ (hA R hR)

variable [Fintype S]

/-- The complete original observed CE sum, including repeated observations,
is invariant under renaming the targets together with the predictor (CM12). -/
theorem scaledLoss_relabel (σ : Equiv.Perm C) (scale : ℝ) (cell : S → m × n)
    (labels : S → C) (P : Table m n C) :
    scaledLoss scale cell (σ ∘ labels) (relabel σ P) = scaledLoss scale cell labels P := by
  apply Finset.sum_congr rfl
  intro s hs
  exact crossEntropy_relabel σ (scale • P (cell s).1 (cell s).2) (labels s)

/-- Ordinary CE is convex on the invariant prediction family (CM12). -/
theorem relabelHull_loss_convex (A : Set (Table m n C)) (scale : ℝ)
    (cell : S → m × n) (labels : S → C) :
    ConvexOn ℝ (relabelHull A) (scaledLoss scale cell labels) :=
  predictionHull_loss_convex _ scale cell labels

private def labelWitness : Table Bool Unit Bool := pointMass (fun x _ => x)
private def labelFamily : Set (Table Bool Unit Bool) := {labelWitness}

example := relabel_mem_hull (Equiv.swap true false)
  (show labelWitness ∈ labelFamily from Set.mem_singleton _)
example := mem_relabelHull (show labelWitness ∈ labelFamily from Set.mem_singleton _)
example := relabel_hull_closed (Equiv.swap true false)
  (mem_relabelHull (show labelWitness ∈ labelFamily from Set.mem_singleton _))
example := rows_relabel (Equiv.swap true false) (rows_pointMass (fun (x : Bool) (_ : Unit) => x))
example := relabelHull_rows labelFamily
  (by intro P hP; simp only [labelFamily, Set.mem_singleton_iff] at hP
      subst P; exact rows_pointMass _)
  (mem_relabelHull (show labelWitness ∈ labelFamily from Set.mem_singleton _))
example := scaledLoss_relabel (Equiv.swap true false) 24 (fun s : Bool => (s, ())) id labelWitness

end Transformer.Grokking.FlatSpectrum.CoupledMoments
