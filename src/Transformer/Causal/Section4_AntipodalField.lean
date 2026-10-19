module

public import Transformer.Causal.Section4_WeightSensitivity
public import Transformer.Causal.Section4_FollowerDrift

/-!
# Expansion of the actual conditional follower field near its antipode

ArXiv:2411.04990v2, §4, proof of `thm1`, antipodal instability.
Two original followers share their actual preceding tokens. The
projection identity gives a positive squared-separation contribution
from each common predecessor near the original leader. Changes of
the original normalized weights contribute an error bounded by
their proved original Lipschitz constants times the small projected
predecessor force. The actual last-token self terms vanish while
their original exponentials remain in the original partitions.

Summing all original interactions gives a lower bound with coefficient
`floor/2 - 2*epsilon*sumLipschitz`. It is positive for a sufficiently
small actual neighborhood. This proves the quantitative expansion
needed for uniqueness of an antipodal follower with a fixed prefix,
using the actual masked field and arbitrary original query and key
matrices.
-/

@[expose] public section

open scoped BigOperators
open Real

namespace Transformer.Causal

variable {d n : ℕ}

/-- An actual weighted predecessor expands two nearby antipodal followers up to its original weight-sensitivity error.
Source: arXiv:2411.04990v2, §4, the spherical antipodal linearization in `thm1`. -/
theorem csaWeightedProjected_separation_lower (x y p a : SSphere d) (u v L ε : ℝ)
    (hu : 0 ≤ u) (hL : 0 ≤ L) (hε : ε ≤ 1 / 4)
    (hx : ‖(x : EucSpace d) + (a : EucSpace d)‖ ≤ ε)
    (hy : ‖(y : EucSpace d) + (a : EucSpace d)‖ ≤ ε)
    (hp : ‖(p : EucSpace d) - (a : EucSpace d)‖ ≤ ε)
    (hw : |u - v| ≤ L * ‖(x : EucSpace d) - (y : EucSpace d)‖) :
    (u / 2 - 2 * ε * L) * ‖(x : EucSpace d) - (y : EucSpace d)‖ ^ 2 ≤
      inner ℝ ((x : EucSpace d) - (y : EucSpace d))
        (u • proj d (x : EucSpace d) (p : EucSpace d) -
          v • proj d (y : EucSpace d) (p : EucSpace d)) := by
  let z := (x : EucSpace d) - (y : EucSpace d)
  have he0 : 0 ≤ ε := (norm_nonneg _).trans hx
  have hxp : inner ℝ (x : EucSpace d) (p : EucSpace d) ≤ -1 / 2 := by
    have h := inner_csa_near_antipode x p a
    linarith
  have hyp : inner ℝ (y : EucSpace d) (p : EucSpace d) ≤ -1 / 2 := by
    have h := inner_csa_near_antipode y p a
    linarith
  have hmain : ‖z‖ ^ 2 / 2 ≤ inner ℝ z
      (proj d (x : EucSpace d) (p : EucSpace d) - proj d (y : EucSpace d) (p : EucSpace d)) := by
    rw [csaProjected_separation]
    have hs := mul_le_mul_of_nonneg_right (show
      (1 / 2 : ℝ) ≤ -((inner ℝ (x : EucSpace d) (p : EucSpace d) +
        inner ℝ (y : EucSpace d) (p : EucSpace d)) / 2) by linarith) (sq_nonneg ‖z‖)
    linarith
  have hproj : ‖proj d (y : EucSpace d) (p : EucSpace d)‖ ≤ 2 * ε := by
    have h := norm_csaProj_near_antipode y (p : EucSpace d) (a : EucSpace d)
    linarith
  have herr : |(u - v) * inner ℝ z (proj d (y : EucSpace d) (p : EucSpace d))| ≤
      2 * ε * L * ‖z‖ ^ 2 := by
    rw [abs_mul]
    calc
      _ ≤ (L * ‖z‖) * (‖z‖ * (2 * ε)) :=
        mul_le_mul hw ((abs_real_inner_le_norm _ _).trans
          (mul_le_mul_of_nonneg_left hproj (norm_nonneg z)))
          (abs_nonneg _) (mul_nonneg hL (norm_nonneg z))
      _ = _ := by ring
  have hdecomp : u • proj d (x : EucSpace d) (p : EucSpace d) -
      v • proj d (y : EucSpace d) (p : EucSpace d) =
      u • (proj d (x : EucSpace d) (p : EucSpace d) - proj d (y : EucSpace d) (p : EucSpace d)) +
        (u - v) • proj d (y : EucSpace d) (p : EucSpace d) := by module
  rw [hdecomp, inner_add_right, real_inner_smul_right, real_inner_smul_right]
  have hm := mul_le_mul_of_nonneg_left hmain hu
  have he := (abs_le.mp herr).1
  change (u / 2 - 2 * ε * L) * ‖z‖ ^ 2 ≤ _
  nlinarith

/-- An original last row with a genuine predecessor has total predecessor weight at least the original floor.
Source: arXiv:2411.04990v2, §4, the strictly positive predecessor coefficient in `thm1`. -/
theorem csaPredecessorWeight_floor (β : ℝ) (Q K : ParamMatrix d) (Y : SphereTuple d (n + 1))
    (hn : 1 ≤ n) : csaWeightFloor β Q K (n + 1) ≤
      ∑ j : Idx n, csaTokenWeight β Q K (fun l => (Y l : EucSpace d)) (Fin.last n) j.castSucc := by
  have hn' : (1 : ℝ) ≤ n := by exact_mod_cast hn
  have hf := csaWeightFloor_pos β Q K (Nat.zero_lt_succ n)
  refine le_trans (show csaWeightFloor β Q K (n + 1) ≤
    (n : ℝ) * csaWeightFloor β Q K (n + 1) by nlinarith) ?_
  exact csaPredecessorWeight_lower β Q K _ (fun j => mem_sphere_zero_iff_norm.mp (Y j).2)

/-- The actual original conditional follower field expands squared separation near its actual antipode.
Source: arXiv:2411.04990v2, §4, the antipodal instability step in `thm1`.
The error uses only the proved sensitivity of the original normalized row weights. -/
theorem csaLastField_separation_lower (β : ℝ) (Q K : ParamMatrix d)
    (Y Z : SphereTuple d (n + 1)) (a : SSphere d) (hn : 1 ≤ n) (ε : ℝ) (hε : ε ≤ 1 / 4)
    (hYZ : ∀ j : Idx n, Y j.castSucc = Z j.castSucc)
    (hp : ∀ j : Idx n, ‖(Y j.castSucc : EucSpace d) - (a : EucSpace d)‖ ≤ ε)
    (hY : ‖(Y (Fin.last n) : EucSpace d) + (a : EucSpace d)‖ ≤ ε)
    (hZ : ‖(Z (Fin.last n) : EucSpace d) + (a : EucSpace d)‖ ≤ ε) :
    (csaWeightFloor β Q K (n + 1) / 2 - 2 * ε * csaPredecessorLipschitz β Q K n) *
      ‖(Y (Fin.last n) : EucSpace d) - (Z (Fin.last n) : EucSpace d)‖ ^ 2 ≤
        inner ℝ ((Y (Fin.last n) : EucSpace d) - (Z (Fin.last n) : EucSpace d))
          (csaAmbientField β Q K (ContinuousLinearMap.id ℝ _) (fun j => (Y j : EucSpace d)) (Fin.last n) -
            csaAmbientField β Q K (ContinuousLinearMap.id ℝ _) (fun j => (Z j : EucSpace d)) (Fin.last n)) := by
  let D := ‖(Y (Fin.last n) : EucSpace d) - (Z (Fin.last n) : EucSpace d)‖ ^ 2
  calc
    _ ≤ ((∑ j : Idx n, csaTokenWeight β Q K (fun l => (Y l : EucSpace d)) (Fin.last n) j.castSucc) / 2 -
        2 * ε * csaPredecessorLipschitz β Q K n) * D := by
      apply mul_le_mul_of_nonneg_right _ (sq_nonneg _)
      linarith [csaPredecessorWeight_floor β Q K Y hn]
    _ = ∑ j : Idx n, (csaTokenWeight β Q K (fun l => (Y l : EucSpace d)) (Fin.last n) j.castSucc / 2 -
        2 * ε * (csaWeightLipschitz β Q K (Fin.last n) j.castSucc : ℝ)) * D := by
      rw [← Finset.sum_mul, Finset.sum_sub_distrib, ← Finset.sum_div, ← Finset.mul_sum]
      rfl
    _ ≤ _ := by
      rw [csaLastField_id_eq_sum β Q K _ (fun j => mem_sphere_zero_iff_norm.mp (Y j).2),
        csaLastField_id_eq_sum β Q K _ (fun j => mem_sphere_zero_iff_norm.mp (Z j).2),
        ← Finset.sum_sub_distrib, inner_sum]
      apply Finset.sum_le_sum
      intro j hj
      have h := csaWeightedProjected_separation_lower (Y (Fin.last n)) (Z (Fin.last n)) (Y j.castSucc) a
        (csaTokenWeight β Q K (fun l => (Y l : EucSpace d)) (Fin.last n) j.castSucc)
        (csaTokenWeight β Q K (fun l => (Z l : EucSpace d)) (Fin.last n) j.castSucc)
        (csaWeightLipschitz β Q K (Fin.last n) j.castSucc : ℝ) ε
        (csaTokenWeight_nonneg β Q K _ _ _) (NNReal.coe_nonneg _) hε hY hZ (hp j)
        (abs_csaWeight_sub_le_of_prefix_eq β Q K Y Z hYZ _ _)
      simpa only [hYZ j] using h

/-- Actual original antipodal configurations jointly realize the shared-prefix and neighborhood premises.
Source: arXiv:2411.04990v2, §4, the antipodal equilibrium in `thm1`. -/
example (a : SSphere 2) :
    let Y : SphereTuple 2 2 := Fin.snoc (fun _ : Idx 1 => a) (Perspective.antipode 2 a)
    (∀ j : Idx 1, Y j.castSucc = Y j.castSucc) ∧
      (∀ j : Idx 1, ‖(Y j.castSucc : EucSpace 2) - (a : EucSpace 2)‖ ≤ 0) ∧
      ‖(Y (Fin.last 1) : EucSpace 2) + (a : EucSpace 2)‖ ≤ 0 ∧ (0 : ℝ) ≤ 1 / 4 ∧ (1 : ℕ) ≤ 1 := by
  dsimp only
  simp only [Fin.snoc_castSucc, Fin.snoc_last, Perspective.antipode, sub_self,
    norm_zero, neg_add_cancel, le_refl]
  norm_num

/-- The same actual antipodal geometry jointly realizes nonnegative interaction weight and zero sensitivity error.
Source: arXiv:2411.04990v2, §4, the coincident antipodal followers in the instability argument for `thm1`. -/
example (a : SSphere 2) :
    let x := Perspective.antipode 2 a
    (0 : ℝ) ≤ 1 / 2 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 / 4 ∧
      ‖(x : EucSpace 2) + (a : EucSpace 2)‖ ≤ 0 ∧
      ‖(a : EucSpace 2) - (a : EucSpace 2)‖ ≤ 0 ∧
      |(1 / 2 : ℝ) - 1 / 2| ≤ 0 * ‖(x : EucSpace 2) - (x : EucSpace 2)‖ := by
  dsimp only [Perspective.antipode]
  simp only [neg_add_cancel, sub_self, norm_zero, mul_zero, abs_zero, le_refl]
  norm_num

end Transformer.Causal
