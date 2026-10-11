/-
# Attention's forward pass and Frank-Wolfe — observable bad steps

The bad-step event is evaluated on the actual sampled trajectory, and is
counted only when all particles are still near their assigned vertices.
For a particle, the transition kernel charges only indices outside its
group when the next position is outside the same-group convex combinations.
This turns the score estimate into a conditional probability estimate.
Source: arXiv:2508.09628v1, §5, `cl: group.cluster.bound` and the proof of
`lem: metastab.1`, `eq: union.bound`.
-/

import Transformer.FrankWolfe.Section5_MetaLevel
import Transformer.FrankWolfe.Section5_MetaWeights
import Transformer.FrankWolfe.Section5_TransitionAS

open scoped ENNReal BigOperators
open MeasureTheory

namespace Transformer.FrankWolfe

variable {d n κ : ℕ}

/-- Configurations lying within radius `r` of their assigned vertices. -/
def vertexNeighborhood (v : Idx κ → EucSpace d) (σ : Idx n → Idx κ) (r : ℝ) :
    Set (Idx n → EucSpace d) := {z | ∀ i, ‖z i - v (σ i)‖ ≤ r}

/-- The configuration-dependent target of a particle that leaves the
same-group update positions. -/
def badTarget (σ : Idx n → Idx κ) (γ : ℝ) (i : Idx n) :
    Set ((Idx n → EucSpace d) × EucSpace d) := {p | p.2 ∉ goodSet σ γ p.1 i}

/-- Bad steps observed while the current configuration is near its vertices.
Stopping here keeps the score estimate valid until the first exit. -/
def stoppedBadStep (v : Idx κ → EucSpace d) (σ : Idx n → Idx κ) (γ r : ℝ)
    (s : ℕ) : Set (ℕ → Idx n → EucSpace d) :=
  {x | x s ∈ vertexNeighborhood v σ r ∧ BadStep σ γ s x}

/-- The vertex neighborhood is measurable. Source: arXiv:2508.09628v1, §5,
auxiliary to the conditioning in `cl: group.cluster.bound`. -/
theorem measurableSet_vertexNeighborhood (v : Idx κ → EucSpace d)
    (σ : Idx n → Idx κ) (r : ℝ) : MeasurableSet (vertexNeighborhood v σ r) := by
  have he : vertexNeighborhood v σ r = ⋂ i, {z | ‖z i - v (σ i)‖ ≤ r} := by
    ext z
    simp [vertexNeighborhood]
  rw [he]
  exact MeasurableSet.iInter fun i => measurableSet_le (by fun_prop) measurable_const

/-- The bad target is measurable, since the group has finitely many indices.
Source: arXiv:2508.09628v1, §5, auxiliary to `eq: union.bound`. -/
theorem measurableSet_badTarget (σ : Idx n → Idx κ) (γ : ℝ) (i : Idx n) :
    MeasurableSet (badTarget (d := d) σ γ i) := by
  have he : badTarget (d := d) σ γ i =
      (⋃ j, {p | σ j = σ i ∧ p.2 = (1 - γ) • p.1 i + γ • p.1 j})ᶜ := by
    ext p
    simp [badTarget, goodSet]
  rw [he]
  refine (MeasurableSet.iUnion fun j => ?_).compl
  by_cases hj : σ j = σ i
  · simp only [hj, true_and]
    exact measurableSet_eq_fun measurable_snd (by fun_prop)
  · simp [hj]

/-- A stopped bad step depends only on the trajectory through the next time.
Source: arXiv:2508.09628v1, §5, auxiliary to `eq: union.bound`. -/
theorem pastDet_stoppedBadStep (v : Idx κ → EucSpace d) (σ : Idx n → Idx κ)
    (γ r : ℝ) (s : ℕ) : PastDet (s + 1) (stoppedBadStep v σ γ r s) := by
  have he : stoppedBadStep v σ γ r s =
      {x | x s ∈ vertexNeighborhood v σ r} ∩
        ⋃ i, {x | (x s, x (s + 1) i) ∈ badTarget σ γ i} := by
    ext x
    simp [stoppedBadStep, BadStep, badTarget]
  refine ⟨?_, ?_⟩
  · rw [he]
    exact (measurable_pi_apply s (measurableSet_vertexNeighborhood v σ r)).inter
      (MeasurableSet.iUnion fun i => (measurableSet_badTarget σ γ i).preimage (by fun_prop))
  · intro x y hxy
    simp only [stoppedBadStep, Set.mem_ofPred_eq, BadStep, hxy s (by omega), hxy (s + 1) le_rfl]

/-- Only indices of another group can charge the bad target. The total is
at most `n exp(-β c₀/2)` near the vertices. Source: arXiv:2508.09628v1, §5,
`cl: group.cluster.bound`; the self-score comparison gives the stated factor. -/
theorem sum_badTarget_weight_le {v : Idx κ → EucSpace d} {σ : Idx n → Idx κ}
    {c₀ R r β γ : ℝ}
    (hsep : ∀ a b, a ≠ b →
      inner (𝕜 := ℝ) (v a) (v b) + c₀ ≤ inner (𝕜 := ℝ) (v a) (v a))
    (hR : ∀ a, ‖v a‖ ≤ R) (hsmall : 4 * R * r + 2 * r ^ 2 ≤ c₀ / 2)
    (hβ : 0 ≤ β) (z : Idx n → EucSpace d) (hz : z ∈ vertexNeighborhood v σ r)
    (i : Idx n) :
    ∑ j, ENNReal.ofReal (attWeight β z i j) *
      (badTarget σ γ i).indicator (fun _ => (1 : ℝ≥0∞))
        (z, (1 - γ) • z i + γ • z j) ≤
      ENNReal.ofReal (n * Real.exp (-(β * (c₀ / 2)))) := by
  classical
  calc (∑ j, ENNReal.ofReal (attWeight β z i j) *
        (badTarget σ γ i).indicator (fun _ => (1 : ℝ≥0∞))
          (z, (1 - γ) • z i + γ • z j))
      ≤ ∑ j ∈ Finset.univ.filter (fun j => σ j ≠ σ i),
          ENNReal.ofReal (attWeight β z i j) := by
        rw [Finset.sum_filter]
        apply Finset.sum_le_sum
        intro j _
        by_cases hj : σ j = σ i
        · have hn : (z, (1 - γ) • z i + γ • z j) ∉ badTarget σ γ i :=
            fun h => h ⟨j, hj, rfl⟩
          simp [hj, Set.indicator_of_notMem hn]
        · by_cases hb : (z, (1 - γ) • z i + γ • z j) ∈ badTarget σ γ i
          · simp [hj, Set.indicator_of_mem hb]
          · simp [hj, Set.indicator_of_notMem hb]
    _ = ENNReal.ofReal (∑ j ∈ Finset.univ.filter (fun j => σ j ≠ σ i),
        attWeight β z i j) :=
      (ENNReal.ofReal_sum_of_nonneg fun j _ => (attWeight_pos β z i j).le).symm
    _ ≤ _ := ENNReal.ofReal_le_ofReal (sum_attWeight_other_le hsep hR hsmall hβ z hz i)

/-- A bad step conditional on any event of the past has probability at most
`n² exp(-β c₀/2)` while the particles are near their assigned vertices.
Source: arXiv:2508.09628v1, §5, `cl: group.cluster.bound`, `eq: union.bound`;
the union is over actual particle updates and does not assume independence. -/
theorem IsSAProcess.measure_stoppedBadStep_le {v : Idx κ → EucSpace d}
    {σ : Idx n → Idx κ} {c₀ R r β γ : ℝ}
    (hsep : ∀ a b, a ≠ b →
      inner (𝕜 := ℝ) (v a) (v b) + c₀ ≤ inner (𝕜 := ℝ) (v a) (v a))
    (hR : ∀ a, ‖v a‖ ≤ R) (hsmall : 4 * R * r + 2 * r ^ 2 ≤ c₀ / 2)
    (hβ : 0 ≤ β) {X₀ : Idx n → EucSpace d} {P : Measure (ℕ → Idx n → EucSpace d)}
    (hP : IsSAProcess β γ X₀ P) (s : ℕ) {S : Set (ℕ → Idx n → EucSpace d)}
    (hS : PastDet s S) : P (S ∩ stoppedBadStep v σ γ r s) ≤
      ENNReal.ofReal (n ^ 2 * Real.exp (-(β * (c₀ / 2)))) * P S := by
  let c := ENNReal.ofReal (n * Real.exp (-(β * (c₀ / 2))))
  have hi : ∀ i, P (S ∩ {x | x s ∈ vertexNeighborhood v σ r} ∩
      {x | (x s, x (s + 1) i) ∈ badTarget σ γ i}) ≤ c * P S := by
    intro i
    refine (hP.measure_inter_prod_le s i hS (measurableSet_badTarget σ γ i)
      (measurableSet_vertexNeighborhood v σ r)
      (sum_badTarget_weight_le hsep hR hsmall hβ · · i)).trans ?_
    exact mul_le_mul_of_nonneg_left (measure_mono Set.inter_subset_left) zero_le
  have he : S ∩ stoppedBadStep v σ γ r s =
      ⋃ i, S ∩ {x | x s ∈ vertexNeighborhood v σ r} ∩
        {x | (x s, x (s + 1) i) ∈ badTarget σ γ i} := by
    ext x
    simp [stoppedBadStep, BadStep, badTarget, and_assoc]
  rw [he]
  calc P (⋃ i, S ∩ {x | x s ∈ vertexNeighborhood v σ r} ∩
        {x | (x s, x (s + 1) i) ∈ badTarget σ γ i})
      ≤ ∑ i, P (S ∩ {x | x s ∈ vertexNeighborhood v σ r} ∩
          {x | (x s, x (s + 1) i) ∈ badTarget σ γ i}) := measure_iUnion_fintype_le _ _
    _ ≤ ∑ i : Idx n, c * P S := Finset.sum_le_sum fun i _ => hi i
    _ = (n : ℝ≥0∞) * c * P S := by simp [nsmul_eq_mul, mul_assoc]
    _ = _ := by
      rw [← ENNReal.ofReal_natCast n, ← ENNReal.ofReal_mul (Nat.cast_nonneg n)]
      congr 2
      ring

/-- The geometric and score hypotheses have a one-particle realization, with
the particle and its vertex at the origin. -/
example :
    (∀ a b : Idx 1, a ≠ b → inner (𝕜 := ℝ) (0 : EucSpace 1) 0 + 1 ≤
      inner (𝕜 := ℝ) (0 : EucSpace 1) 0) ∧
      (∀ a : Idx 1, ‖(fun _ : Idx 1 => (0 : EucSpace 1)) a‖ ≤ (0 : ℝ)) ∧
      4 * (0 : ℝ) * 0 + 2 * 0 ^ 2 ≤ 1 / 2 ∧ 0 ≤ (0 : ℝ) ∧
      (fun _ : Idx 1 => (0 : EucSpace 1)) ∈
        vertexNeighborhood (fun _ : Idx 1 => (0 : EucSpace 1)) (fun i => i) 0 ∧
      IsSAProcess 0 (1 / 2) (fun _ : Idx 1 => (0 : EucSpace 1))
        (Measure.dirac (fun _ : ℕ => fun _ : Idx 1 => (0 : EucSpace 1))) ∧
      PastDet 0 (Set.univ : Set (ℕ → Idx 1 → EucSpace 1)) := by
  refine ⟨fun a b h => False.elim (h (Subsingleton.elim a b)), by simp, by norm_num,
    le_rfl, ?_, isSAProcess_single 0 (1 / 2) _, pastDet_univ 0⟩
  intro i
  simp

end Transformer.FrankWolfe
