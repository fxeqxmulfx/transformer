module

public import Transformer.Clusters.Section4_SquareDynamics
public import Transformer.Clusters.Section4_Hyperplanes

/-!
# Four limiting corners contradict the codimension conjecture

Source: arXiv:2305.05465v6, §4, Conjecture `c:codim`.
The source proposes at most three parallel affine subspaces of
codimension `k`, where `k` counts the eigenvalues of `V` with
positive real part. For `V = I₂`, this means at most three points.
Both positive eigenvalues count with multiplicity, so `k = 2`
although their numerical values coincide. Unlike the preceding
leading-eigenvalue theorem, this conjecture has no simplicity
assumption on the eigenvalues of `V`.

The actual four-token trajectory constructed in `Section4_SquareDynamics`
starts at the corners `(±2,±2)` with `Q = K = V = I₂`. Its four limits
are `(±q,±q)` for one `q ≥ 1`. This file proves the finite obstruction:
any candidate family of codimension-two affine subspaces to which
the four tokens approach must contain all four distinct limits.

The argument first shows that a codimension-two linear subspace
of the plane is zero, so each of its affine translates is a singleton.
Distance convergence to such a singleton is ordinary convergence
to its point. Uniqueness of limits then forces the four distinct
corners to belong to the candidate finite set.

The final existence theorem includes the actual rescaled dynamics.
It refutes the source's universal conjecture; it does not substitute
an arbitrary curve for a solution or alter the proposed bound.
-/

@[expose] public section

open Real Filter Topology
namespace Transformer.Clusters

/-- Translating the zero subspace gives precisely one point.
Source: arXiv:2305.05465v6, §4 `c:codim`, the codimension-two planar case. -/
theorem affineShift_bot {d : ℕ} (p : EucSpace d) :
    affineShift (⊥ : Submodule ℝ (EucSpace d)) p = {p} := by
  ext x
  simp [affineShift, sub_eq_zero]

/-- A codimension-two linear subspace of the plane is the zero subspace.
Source: arXiv:2305.05465v6, §4 `c:codim`, the identity matrix's two unstable directions. -/
theorem codim_two_eq_bot (W : Submodule ℝ (EucSpace 2))
    (hW : Module.finrank ℝ W + 2 = 2) : W = ⊥ := by
  apply Submodule.finrank_eq_zero.mp
  omega

example : Module.finrank ℝ (⊥ : Submodule ℝ (EucSpace 2)) + 2 = 2 := by simp

/-- Approaching a translate of the zero subspace is convergence to its unique point.
Source: arXiv:2305.05465v6, §4 `c:codim`, its distance conclusion at codimension two. -/
theorem tendsto_infDist_affineShift_bot_iff {d : ℕ} (Z : ℝ → EucSpace d) (p : EucSpace d) :
    Tendsto (fun t => Metric.infDist (Z t) (affineShift (⊥ : Submodule ℝ (EucSpace d)) p))
      atTop (nhds 0) ↔ Tendsto Z atTop (nhds p) := by
  simp only [affineShift_bot, Metric.infDist_singleton]
  exact tendsto_iff_dist_tendsto_zero.symm

/-- A candidate codimension-two affine limit must equal the actual corner limit.
Source: arXiv:2305.05465v6, §4 `c:codim`, uniqueness of limits in the square test. -/
theorem squareCorner_eq_of_infDist_tendsto_zero (Z : ℝ → Idx 4 → EucSpace 2)
    (q : ℝ) (i : Idx 4) (W : Submodule ℝ (EucSpace 2)) (p : EucSpace 2)
    (hW : Module.finrank ℝ W + 2 = 2)
    (hlim : Tendsto (fun t => Z t i) atTop (nhds (squareCorner q i)))
    (hp : Tendsto (fun t => Metric.infDist (Z t i) (affineShift W p)) atTop (nhds 0)) :
    squareCorner q i = p := by
  rw [codim_two_eq_bot W hW] at hp
  exact tendsto_nhds_unique hlim ((tendsto_infDist_affineShift_bot_iff _ p).mp hp)

example (i : Idx 4) :
    Module.finrank ℝ (⊥ : Submodule ℝ (EucSpace 2)) + 2 = 2 ∧
      Tendsto (fun _ : ℝ => squareCorner 1 i) atTop (nhds (squareCorner 1 i)) ∧
      Tendsto (fun _ : ℝ => Metric.infDist (squareCorner 1 i)
        (affineShift (⊥ : Submodule ℝ (EucSpace 2)) (squareCorner 1 i))) atTop (nhds 0) := by
  refine ⟨by simp, tendsto_const_nhds, ?_⟩
  exact (tendsto_infDist_affineShift_bot_iff _ _).mpr tendsto_const_nhds

/-- A finite set containing the four nonzero-amplitude corners has at least four elements.
Source: arXiv:2305.05465v6, §4 `c:codim`, the obstruction to its bound of three. -/
theorem four_le_card_of_squareCorner_subset (q : ℝ) (S : Finset (EucSpace 2))
    (hq : q ≠ 0) (hS : ∀ i : Idx 4, squareCorner q i ∈ S) : 4 ≤ S.card := by
  classical
  have hsub : Finset.univ.image (squareCorner q) ⊆ S := by
    intro x hx
    obtain ⟨i, _, rfl⟩ := Finset.mem_image.mp hx
    exact hS i
  simpa only [card_squareCorner_image hq] using Finset.card_le_card hsub

example : ∃ S : Finset (EucSpace 2), (1 : ℝ) ≠ 0 ∧ ∀ i : Idx 4, squareCorner 1 i ∈ S := by
  classical
  refine ⟨Finset.univ.image (squareCorner 1), one_ne_zero, ?_⟩
  intro i
  exact Finset.mem_image.mpr ⟨i, Finset.mem_univ i, rfl⟩

/-- Any finite family of codimension-two limits of the four actual corner limits needs four points.
Source: arXiv:2305.05465v6, §4 `c:codim`, the square test's necessary cardinality. -/
theorem four_le_card_of_square_limits (Z : ℝ → Idx 4 → EucSpace 2) (q : ℝ)
    (hq : q ≠ 0) (hlim : ∀ i : Idx 4, Tendsto (fun t => Z t i) atTop (nhds (squareCorner q i)))
    (W : Submodule ℝ (EucSpace 2)) (S : Finset (EucSpace 2))
    (hW : Module.finrank ℝ W + 2 = 2)
    (hS : ∀ i : Idx 4, ∃ p ∈ S,
      Tendsto (fun t => Metric.infDist (Z t i) (affineShift W p)) atTop (nhds 0)) :
    4 ≤ S.card := by
  apply four_le_card_of_squareCorner_subset q S hq
  intro i
  obtain ⟨p, hpS, hp⟩ := hS i
  rw [squareCorner_eq_of_infDist_tendsto_zero Z q i W p hW (hlim i) hp]
  exact hpS

example : ∃ (Z : ℝ → Idx 4 → EucSpace 2) (q : ℝ)
    (W : Submodule ℝ (EucSpace 2)) (S : Finset (EucSpace 2)),
    q ≠ 0 ∧ (∀ i : Idx 4, Tendsto (fun t => Z t i) atTop (nhds (squareCorner q i))) ∧
      Module.finrank ℝ W + 2 = 2 ∧ ∀ i : Idx 4, ∃ p ∈ S,
        Tendsto (fun t => Metric.infDist (Z t i) (affineShift W p)) atTop (nhds 0) := by
  classical
  refine ⟨fun _ => squareCorner 1, 1, ⊥, Finset.univ.image (squareCorner 1), one_ne_zero,
    fun _ => tendsto_const_nhds, by simp, ?_⟩
  intro i
  refine ⟨squareCorner 1 i, Finset.mem_image.mpr ⟨i, Finset.mem_univ i, rfl⟩, ?_⟩
  exact (tendsto_infDist_affineShift_bot_iff _ _).mpr tendsto_const_nhds

/-- Four distinct square limits cannot approach at most three codimension-two affine subspaces.
Source: arXiv:2305.05465v6, §4, the exact distance conclusion of Conjecture `c:codim`. -/
theorem not_three_codim_two_square_limits (Z : ℝ → Idx 4 → EucSpace 2) (q : ℝ)
    (hq : q ≠ 0) (hlim : ∀ i : Idx 4, Tendsto (fun t => Z t i) atTop (nhds (squareCorner q i))) :
    ¬∃ (W : Submodule ℝ (EucSpace 2)) (S : Finset (EucSpace 2)),
      Module.finrank ℝ W + 2 = 2 ∧ S.card ≤ 3 ∧ ∀ i : Idx 4, ∃ p ∈ S,
        Tendsto (fun t => Metric.infDist (Z t i) (affineShift W p)) atTop (nhds 0) := by
  rintro ⟨W, S, hW, hcard, hS⟩
  have hfour := four_le_card_of_square_limits Z q hq hlim W S hW hS
  omega

example : ∃ (Z : ℝ → Idx 4 → EucSpace 2) (q : ℝ),
    q ≠ 0 ∧ ∀ i : Idx 4, Tendsto (fun t => Z t i) atTop (nhds (squareCorner q i)) :=
  ⟨fun _ => squareCorner 1, 1, one_ne_zero, fun _ => tendsto_const_nhds⟩

/-- A genuine attention trajectory refutes the three-subspace conclusion of the codimension conjecture.
Source: arXiv:2305.05465v6, §4, Conjecture `c:codim`.
Here `Q = K = V = I₂`, there are four tokens, and `k = 2`.
The four distinct positive-amplitude corner limits require four points. -/
theorem exists_square_counterexample :
    ∃ Z : ℝ → Idx 4 → EucSpace 2,
      RescaledDynamics (1 : ParamMatrix 2) 1 1 Z ∧ Z 0 = squareCorner 2 ∧
        ¬∃ (W : Submodule ℝ (EucSpace 2)) (S : Finset (EucSpace 2)),
          Module.finrank ℝ W + 2 = 2 ∧ S.card ≤ 3 ∧ ∀ i : Idx 4, ∃ p ∈ S,
            Tendsto (fun t => Metric.infDist (Z t i) (affineShift W p)) atTop (nhds 0) := by
  obtain ⟨Z, q, hZ, h0, hq, hlim⟩ := exists_rescaled_square
  exact ⟨Z, hZ, h0, not_three_codim_two_square_limits Z q (by linarith) hlim⟩

end Transformer.Clusters
