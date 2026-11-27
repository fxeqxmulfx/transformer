module

public import Transformer.Grokking.FiniteRecovery.FlatSpectrum
public import Mathlib.Combinatorics.Hypergraph.Basic

/-! Coordinate observations for regularized completion. Hyperedges are the
supports of differences from distinct competitors of no larger cost.
For nuclear completion these are precisely the supports of nonzero
descent directions (Amelunxen, Lotz, McCoy and Tropp, Definition 2.7
and Fact 2.8). Coordinates are the cells of the cut in Flat Spectrum, §D. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery

open Set FlatSpectrum

variable {U V Y : Type*}

/-- The vertices are cells. A hyperedge records all cells distinguishing
one target from one admissible competitor whose cost is at most its own.
The construction precedes the choice of observation mask. -/
def recoveryHypergraph (targets candidates : Set (Matrix U V Y))
    (cost : Matrix U V Y → ℝ) : Hypergraph (U × V) where
  vertexSet := univ
  edgeSet := {S | ∃ K ∈ targets, ∃ X ∈ candidates, X ≠ K ∧ cost X ≤ cost K ∧
    S = {c | X c.1 c.2 ≠ K c.1 c.2}}
  subset_vertexSet_of_mem_edgeSet' := by intro e _; exact subset_univ e

/-- A transversal meets every hyperedge. Our recovery hypergraphs have
the full coordinate domain as their vertex set. -/
def IsTransversal {B : Type*} (G : Hypergraph B) (mask : Set B) : Prop :=
  mask ⊆ G.vertexSet ∧ ∀ S ∈ G.edgeSet, (mask ∩ S).Nonempty

/-- Transversals are exactly the masks that distinguish every target
from all competitors of equal or lower cost. -/
theorem isTransversal_iff_unique_completion (targets candidates : Set (Matrix U V Y))
    (cost : Matrix U V Y → ℝ) (mask : Set (U × V)) :
    IsTransversal (recoveryHypergraph targets candidates cost) mask ↔
      ∀ K ∈ targets, ∀ X ∈ candidates, (∀ c ∈ mask, X c.1 c.2 = K c.1 c.2) →
        cost X ≤ cost K → X = K := by
  constructor
  · rintro ⟨_, hh⟩ K hK X hX hfit hcost
    by_contra hne
    obtain ⟨c, hm, hc⟩ := hh _ ⟨K, hK, X, hX, hne, hcost, rfl⟩
    exact hc (hfit c hm)
  · intro h
    refine ⟨subset_univ _, ?_⟩
    rintro _ ⟨K, hK, X, hX, hne, hcost, rfl⟩
    by_contra hempty
    apply hne (h K hK X hX (fun c hm => ?_) hcost)
    by_contra hc
    exact hempty ⟨c, hm, hc⟩

variable [Fintype U] [Fintype V] [DecidableEq V]

/-- The closed sublevel through the truth contains no other completion
exactly when the truth is the unique nuclear-norm minimizer. -/
theorem nuclearRecovers_iff_sublevel {mask : Set (U × V)} {K : Matrix U V ℂ} :
    NuclearRecovers mask K ↔ ∀ X ∈ completions mask K, nuclearNorm X ≤ nuclearNorm K → X = K := by
  have hK : K ∈ completions mask K := fun _ _ => rfl
  constructor
  · intro hr X hX hle
    have hmin := (hr K hK).2 rfl
    exact (hr X hX).1 (fun Z hZ => hle.trans (hmin hZ))
  · intro h
    have hmin : IsMinOn nuclearNorm (completions mask K) K := by
      intro X hX
      change nuclearNorm K ≤ nuclearNorm X
      by_contra! hlt
      exact hlt.ne (congrArg nuclearNorm (h X hX hlt.le))
    intro X hX
    exact ⟨fun hx => h X hX (hx hK), fun he => he ▸ hmin⟩

/-- Uniform recovery of an entire family by the Flat Spectrum program is
equivalent to a single transversal. Competitors range over all complex tables. -/
theorem isTransversal_iff_nuclearRecovers (targets : Set (Matrix U V ℂ)) (mask : Set (U × V)) :
    IsTransversal (recoveryHypergraph targets univ nuclearNorm) mask ↔
      ∀ K ∈ targets, NuclearRecovers mask K := by
  rw [isTransversal_iff_unique_completion]
  simp only [mem_univ, forall_const, nuclearRecovers_iff_sublevel, completions, mem_ofPred_eq]

/-- The real descent cone at K, with the positive step kept explicit,
as in ALMT Definition 2.7. -/
def nuclearDescentCone (K : Matrix U V ℂ) : Set (Matrix U V ℂ) :=
  {D | ∃ t : ℝ, 0 < t ∧ nuclearNorm (K + t • D) ≤ nuclearNorm K}

/-- In the coordinate form of ALMT Fact 2.8, a direction both in the descent
cone and invisible to the observations must vanish. -/
theorem nuclearRecovers_iff_descentCone {K : Matrix U V ℂ} {mask : Set (U × V)} :
    NuclearRecovers mask K ↔
      ∀ D ∈ nuclearDescentCone K, (∀ c ∈ mask, D c.1 c.2 = 0) → D = 0 := by
  rw [nuclearRecovers_iff_sublevel]
  constructor
  · rintro hr D ⟨t, ht, hn⟩ hz
    have hfit : K + t • D ∈ completions mask K := fun c hc => by simp [hz c hc]
    have heq := hr _ hfit hn
    have hs : t • D = 0 := add_left_cancel (heq.trans (add_zero K).symm)
    exact (smul_eq_zero.mp hs).resolve_left ht.ne'
  · intro h X hfit hn
    have hd : X - K ∈ nuclearDescentCone K := ⟨1, zero_lt_one, by simpa using hn⟩
    exact sub_eq_zero.mp (h _ hd (fun c hc => sub_eq_zero.mpr (hfit c hc)))

/-- Minimum transversal size for a finite hypergraph admitting a transversal.
Every recovery hypergraph has this property because its tables differ on a cell. -/
noncomputable def transversalNumber {B : Type*} (G : Hypergraph B) : ℕ :=
  sInf {n : ℕ | ∃ mask : Finset B, mask.card = n ∧ IsTransversal G mask}

omit [DecidableEq V] in
/-- Every finite recovery hypergraph admits a smallest transversal,
including when competitors are restricted to a model class. -/
theorem isLeast_recovery_transversal (targets candidates : Set (Matrix U V Y))
    (cost : Matrix U V Y → ℝ) :
    let G := recoveryHypergraph targets candidates cost
    IsLeast {n : ℕ | ∃ mask : Finset (U × V), mask.card = n ∧ IsTransversal G mask}
      (transversalNumber G) := by
  classical
  let G := recoveryHypergraph targets candidates cost
  let counts := {n : ℕ | ∃ mask : Finset (U × V), mask.card = n ∧ IsTransversal G mask}
  have hn : counts.Nonempty := by
    refine ⟨_, Finset.univ, rfl, (isTransversal_iff_unique_completion _ _ _ _).2 ?_⟩
    intro K _ X _ hfit _
    exact Matrix.ext fun i j => hfit (i,j) (Finset.mem_univ _)
  exact ⟨Nat.sInf_mem hn, fun _ h => Nat.sInf_le h⟩

/-- The minimum number of cells for nuclear recovery of a whole class.
It counts distinct observed inputs. -/
theorem isLeast_nuclear_observations (targets : Set (Matrix U V ℂ)) :
    IsLeast {n : ℕ | ∃ mask : Finset (U × V), mask.card = n ∧
      ∀ K ∈ targets, NuclearRecovers mask K}
      (transversalNumber (recoveryHypergraph targets univ nuclearNorm)) := by
  classical
  let G := recoveryHypergraph targets univ nuclearNorm
  let counts := {n : ℕ | ∃ mask : Finset (U × V), mask.card = n ∧ IsTransversal G mask}
  have heq : counts = {n : ℕ | ∃ mask : Finset (U × V), mask.card = n ∧
      ∀ K ∈ targets, NuclearRecovers mask K} := by
    ext n
    simp only [counts, G, mem_ofPred_eq, isTransversal_iff_nuclearRecovers]
  rw [← heq]
  exact isLeast_recovery_transversal targets univ nuclearNorm

example (K : Matrix (Fin 2) (Fin 2) ℂ) :
    IsTransversal (recoveryHypergraph {K} univ nuclearNorm) univ :=
  (isTransversal_iff_nuclearRecovers _ _).2 fun X _ => nuclearRecovers_univ X

end Transformer.Grokking.FiniteRecovery
