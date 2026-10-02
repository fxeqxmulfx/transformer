/-
# Attention's forward pass and Frank-Wolfe — null sets of score ties

For a linear branch fixing consensus tuples, a common translation leaves
token differences unchanged. If a difference is nonzero and B is invertible,
the translations giving equal attention scores form an affine hyperplane.
Fubini's theorem and translation invariance of Lebesgue measure then exclude
such ties for almost every initial tuple, even for singular branches.

Source: arXiv:2508.09628v1, §2.1, lem:singleLeader.
-/

import Transformer.FrankWolfe.Section2_LinearBranches
import Mathlib.MeasureTheory.Measure.Haar.InnerProductSpace
import Mathlib.MeasureTheory.Measure.Lebesgue.EqHaar
import Mathlib.MeasureTheory.Constructions.Pi

open MeasureTheory

namespace Transformer.FrankWolfe

variable {d n : ℕ}

/-- The score against a nonzero token difference vanishes only on a null
hyperplane. Surjectivity of B suffices.
Source: arXiv:2508.09628v1, §2.1, the hyperplanes in lem:singleLeader. -/
theorem ae_inner_comp_ne_zero (B : ParamMatrix d) (hB : Function.Surjective B)
    (v : EucSpace d) (hv : v ≠ 0) :
    ∀ᵐ z : EucSpace d, inner (𝕜 := ℝ) (B z) v ≠ 0 := by
  let f : EucSpace d →ₗ[ℝ] ℝ := (innerSL ℝ v).toLinearMap.comp B.toLinearMap
  have hker : f.ker ≠ ⊤ := by
    intro htop
    obtain ⟨z, hz⟩ := hB v
    have hzmem : z ∈ f.ker := by rw [htop]; trivial
    have hz0 : f z = 0 := LinearMap.mem_ker.mp hzmem
    have hv0 : inner (𝕜 := ℝ) v v = 0 := by simpa [f, hz] using hz0
    exact hv (inner_self_eq_zero.mp hv0)
  have hnull := Measure.addHaar_submodule volume f.ker hker
  have hae : ∀ᵐ z : EucSpace d, f z ≠ 0 := by
    have hset : {z : EucSpace d | ¬ f z ≠ 0} = (f.ker : Set (EucSpace d)) := by
      ext z
      simp
    rw [ae_iff, hset]
    exact hnull
  filter_upwards [hae] with z hz
  simpa [f, real_inner_comm] using hz

/-- The hyperplane hypotheses hold for the identity and a unit vector.
Source: arXiv:2508.09628v1, §2.1. -/
example : Function.Surjective (ContinuousLinearMap.id ℝ (EucSpace 1)) ∧
    EuclideanSpace.single (0 : Fin 1) (1 : ℝ) ≠ 0 :=
  ⟨Function.surjective_id, by simp⟩

/-- Distinct images under a consensus-preserving linear branch almost surely
have different scores at any token. The branch need not be invertible.
Source: arXiv:2508.09628v1, §2.1, lem:singleLeader. -/
theorem ae_linearBranch_scores_ne
    (L : (Idx n → EucSpace d) →ₗ[ℝ] (Idx n → EucSpace d))
    (hL : ∀ v : EucSpace d, L (fun _ => v) = fun _ => v)
    (B : ParamMatrix d) (hB : Function.Surjective B) (i j k : Idx n) :
    ∀ᵐ X : Idx n → EucSpace d, L X j ≠ L X k →
      inner (𝕜 := ℝ) (B (L X i)) (L X j) ≠
        inner (𝕜 := ℝ) (B (L X i)) (L X k) := by
  have hshift (X : Idx n → EucSpace d) (z : EucSpace d) (l : Idx n) :
      L (X + fun _ => z) l = L X l + z := by
    rw [L.map_add, hL]
    rfl
  have hslice (X : Idx n → EucSpace d) :
      ∀ᵐ z : EucSpace d, L (X + fun _ => z) j ≠ L (X + fun _ => z) k →
        inner (𝕜 := ℝ) (B (L (X + fun _ => z) i)) (L (X + fun _ => z) j) ≠
          inner (𝕜 := ℝ) (B (L (X + fun _ => z) i)) (L (X + fun _ => z) k) := by
    by_cases hv : L X j - L X k = 0
    · exact ae_of_all _ fun z hdist =>
        (hdist (by rw [hshift, hshift, sub_eq_zero.mp hv])).elim
    · have hz := (measurePreserving_add_left volume (L X i)).quasiMeasurePreserving.ae
        (ae_inner_comp_ne_zero B hB (L X j - L X k) hv)
      filter_upwards [hz] with z hz
      intro _
      apply sub_ne_zero.mp
      rw [hshift, hshift, hshift, ← inner_sub_right, add_sub_add_right_eq_sub]
      exact hz
  let Q (p : (Idx n → EucSpace d) × EucSpace d) : Prop :=
    L (p.1 + fun _ => p.2) j ≠ L (p.1 + fun _ => p.2) k →
      inner (𝕜 := ℝ) (B (L (p.1 + fun _ => p.2) i)) (L (p.1 + fun _ => p.2) j) ≠
        inner (𝕜 := ℝ) (B (L (p.1 + fun _ => p.2) i)) (L (p.1 + fun _ => p.2) k)
  have hc : Continuous (fun p : (Idx n → EucSpace d) × EucSpace d =>
      L (p.1 + fun _ => p.2)) :=
    L.continuous_of_finiteDimensional.comp
      (continuous_fst.add (continuous_pi fun _ => continuous_snd))
  have hr (l : Idx n) : Continuous (fun p : (Idx n → EucSpace d) × EucSpace d =>
      L (p.1 + fun _ => p.2) l) := (continuous_apply l).comp hc
  have hs (l : Idx n) : Continuous (fun p : (Idx n → EucSpace d) × EucSpace d =>
      inner (𝕜 := ℝ) (B (L (p.1 + fun _ => p.2) i)) (L (p.1 + fun _ => p.2) l)) :=
    (B.continuous.comp (hr i)).inner (hr l)
  have hmeas : MeasurableSet {p | Q p} := by
    convert (isClosed_eq (hr j) (hr k)).measurableSet.union
      (isClosed_eq (hs j) (hs k)).measurableSet.compl using 1
    ext p
    simp only [Set.mem_ofPred_eq, Set.mem_union, Set.mem_compl_iff, Q]
    tauto
  have hprod : ∀ᵐ p ∂(volume : Measure (Idx n → EucSpace d)).prod
      (volume : Measure (EucSpace d)), Q p :=
    (Measure.ae_prod_iff_ae_ae hmeas).mpr (ae_of_all _ hslice)
  have hswap := (Measure.measurePreserving_swap
    (μ := (volume : Measure (EucSpace d)))
    (ν := (volume : Measure (Idx n → EucSpace d)))).quasiMeasurePreserving.ae hprod
  obtain ⟨z, hz⟩ := (Measure.ae_ae_of_ae_prod hswap).exists
  have hback := (measurePreserving_add_right
      (volume : Measure (Idx n → EucSpace d)) (fun _ => -z)).quasiMeasurePreserving.ae hz
  simpa [Q, hL] using hback

/-- The branch hypotheses hold for the identity on two one-dimensional tokens.
Source: arXiv:2508.09628v1, §2.1. -/
example : (∀ v : EucSpace 1,
    (LinearMap.id : (Idx 2 → EucSpace 1) →ₗ[ℝ] (Idx 2 → EucSpace 1))
      (fun _ => v) = fun _ => v) ∧
    Function.Surjective (ContinuousLinearMap.id ℝ (EucSpace 1)) :=
  ⟨fun _ => rfl, Function.surjective_id⟩

/-- Every token has a leader, since the configuration is finite and nonempty.
Source: arXiv:2508.09628v1, §1.2, the definition of the leader set. -/
theorem leaderSet_nonempty (B : ParamMatrix d) (X : Idx n → EucSpace d) (i : Idx n) :
    (leaderSet B X i).Nonempty := by
  classical
  obtain ⟨j, _, hj⟩ := Finset.exists_max_image Finset.univ
    (fun j : Idx n => inner (𝕜 := ℝ) (B (X i)) (X j)) ⟨i, Finset.mem_univ i⟩
  exact ⟨X j, Finset.mem_filter.mpr ⟨Finset.mem_image.mpr ⟨j, Finset.mem_univ _, rfl⟩,
    fun k => hj k (Finset.mem_univ _)⟩⟩

/-- Excluding score ties between distinct points makes the leader set a
singleton; coinciding indices still represent only one point.
Source: arXiv:2508.09628v1, §2.1, lem:singleLeader. -/
theorem leaderSet_card_eq_one_of_scores_ne (B : ParamMatrix d)
    (X : Idx n → EucSpace d) (i : Idx n)
    (hsep : ∀ j k : Idx n, X j ≠ X k →
      inner (𝕜 := ℝ) (B (X i)) (X j) ≠ inner (𝕜 := ℝ) (B (X i)) (X k)) :
    (leaderSet B X i).card = 1 := by
  classical
  obtain ⟨a, ha⟩ := leaderSet_nonempty B X i
  refine Finset.card_eq_one.mpr ⟨a, Finset.eq_singleton_iff_unique_mem.mpr ⟨ha, ?_⟩⟩
  intro b hb
  obtain ⟨haX, hmaxa⟩ := Finset.mem_filter.mp ha
  obtain ⟨hbX, hmaxb⟩ := Finset.mem_filter.mp hb
  obtain ⟨j, _, haeq⟩ := Finset.mem_image.mp haX
  obtain ⟨k, _, hbeq⟩ := Finset.mem_image.mp hbX
  by_contra hba
  have hscore : inner (𝕜 := ℝ) (B (X i)) (X k) =
      inner (𝕜 := ℝ) (B (X i)) (X j) := by
    rw [hbeq, haeq]
    exact le_antisymm (by simpa [hbeq] using hmaxa k) (by simpa [haeq] using hmaxb j)
  exact hsep k j (by rwa [hbeq, haeq]) hscore

/-- A nontrivial separated-score configuration: the points 1 and 0, queried
from 1 with the identity key. Source: arXiv:2508.09628v1, §2.1. -/
example :
    let X : Idx 2 → EucSpace 1 := ![EuclideanSpace.single (0 : Fin 1) (1 : ℝ), 0]
    ∀ j k : Idx 2, X j ≠ X k →
      inner (𝕜 := ℝ) (X 0) (X j) ≠ inner (𝕜 := ℝ) (X 0) (X k) := by
  intro X j k
  fin_cases j <;> fin_cases k <;> simp [X]

end Transformer.FrankWolfe
