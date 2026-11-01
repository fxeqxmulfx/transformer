module

public import Transformer.SequenceModel.Attention.Concentration

/-!
# Centered values preserve exact XSA and avoid constant-value cancellation

Source: `review/residual_connections.md`, *Repeated-input numerical audit*;
the supplied `diff_gpt/domain/model/mha.py` at b98c665b. Its projection
subtracts two equal nonzero vectors on constant values. Peri-RMSNorm can
amplify floating-point residuals across depth. Centering values before the
attention average avoids that subtraction without changing the real model.
The clipped own-value remainder is retained; omitting it below the L2
epsilon would change the function. No optimizer or roundoff bound follows
from these equalities. Query/key normalization and arbitrary RoPE are kept.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention

open GPTMini
open scoped BigOperators RealInnerProductSpace

/-- The part of an own value left by the source's clipped projection. -/
def ownRemainder {d : ℕ} (eps : ℝ) (v : EucSpace d) : EucSpace d :=
  if eps ≤ ‖v‖ then 0 else (1 - (‖v‖ / eps) ^ 2) • v

/-- Center before averaging; correct the own value when L2 clipping is active. -/
def centeredXSA {T d : ℕ} (eps : ℝ) (weight : Fin T → ℝ)
    (value : Fin T → EucSpace d) (anchor own : EucSpace d) : EucSpace d :=
  let delta := (∑ j, weight j • (value j - anchor)) - (own - anchor)
  let unit := normL2 eps own
  delta - (inner (𝕜 := ℝ) delta unit) • unit + ownRemainder eps own

/-- Repeated-input audit, C1: normalized weighted sums commute with centering.
No positivity or particular query/key scores are needed for this identity. -/
theorem weighted_centering {T d : ℕ} (weight : Fin T → ℝ)
    (value : Fin T → EucSpace d) (anchor : EucSpace d)
    (hweight : ∑ j, weight j = 1) :
    ∑ j, weight j • (value j - anchor) = (∑ j, weight j • value j) - anchor := by
  simp only [smul_sub, Finset.sum_sub_distrib]
  rw [← Finset.sum_smul, hweight, one_smul]

/-- C2: exact source projection of the own value, including clipped values. -/
theorem own_projection_coefficient {d : ℕ} (eps : ℝ) (v : EucSpace d) :
    v - (inner (𝕜 := ℝ) v (normL2 eps v)) • normL2 eps v =
      (1 - (‖v‖ / max ‖v‖ eps) ^ 2) • v := by
  simp only [normL2, real_inner_smul_right, real_inner_self_eq_norm_sq, smul_smul]
  module

/-- C2: the branchwise correction avoids subtracting equal large terms when
the own value is above the clipping threshold. Zero values are included. -/
theorem own_projection_remainder {d : ℕ} (eps : ℝ) (v : EucSpace d) :
    v - (inner (𝕜 := ℝ) v (normL2 eps v)) • normL2 eps v = ownRemainder eps v := by
  rw [own_projection_coefficient]
  unfold ownRemainder
  split_ifs with h
  · rw [max_eq_left h]
    by_cases hv : ‖v‖ = 0
    · have hv0 := norm_eq_zero.mp hv
      subst v
      simp
    · rw [div_self hv]
      simp
  · rw [max_eq_right (le_of_not_ge h)]

/-- C3: centered computation equals the original XSA for every normalized
row and every anchor, including the clipped own-value region. -/
theorem centeredXSA_eq {T d : ℕ} (eps : ℝ) (weight : Fin T → ℝ)
    (value : Fin T → EucSpace d) (anchor own : EucSpace d)
    (hweight : ∑ j, weight j = 1) :
    centeredXSA eps weight value anchor own =
      (∑ j, weight j • value j) -
        (inner (𝕜 := ℝ) (∑ j, weight j • value j) (normL2 eps own)) • normL2 eps own := by
  unfold centeredXSA
  rw [weighted_centering weight value anchor hweight, ← own_projection_remainder]
  simp only [inner_sub_left]
  module

/-- C4: the actual normalized rotary causal head admits the centered
evaluation, retaining its explicit first-position zero. -/
theorem headOutput_centered {T d : ℕ} (eps xeps : ℝ)
    (angle : Fin T → Fin (d / 2) → ℝ) (q k v : Fin T → EucSpace d)
    (i : Fin T) (anchor : EucSpace d) :
    headOutput eps xeps angle q k v i =
      if i.val = 0 then 0 else
        centeredXSA xeps
          (causalWeight (fun a b => headScore d eps (angle a) (angle b) (q a) (k b)) i)
          v anchor (v i) := by
  unfold headOutput
  split_ifs with h
  · rfl
  · exact (centeredXSA_eq xeps _ v anchor (v i) (causalWeight_sum _ i)).symm

/-- C5: constant values vanish for arbitrary source scores and rotary phases
when their norm reaches the L2 clipping threshold. -/
theorem headOutput_constant_values {T d : ℕ} (eps xeps : ℝ)
    (angle : Fin T → Fin (d / 2) → ℝ) (q k : Fin T → EucSpace d)
    (v : EucSpace d) (i : Fin T) (hclip : xeps ≤ ‖v‖) :
    headOutput eps xeps angle q k (fun _ => v) i = 0 := by
  rw [headOutput_centered eps xeps angle q k (fun _ => v) i v]
  split_ifs with h
  · rfl
  · simp [centeredXSA, ownRemainder, hclip]

example := weighted_centering (T := 2) (d := 2) (fun _ => 1 / 2)
  (fun j => EuclideanSpace.single j 3) (EuclideanSpace.single 0 7)
  (by norm_num [Fin.sum_univ_two])

example := centeredXSA_eq (T := 2) (d := 2) (1 / 1000) (fun _ => 1 / 2)
  (fun j => EuclideanSpace.single j 3) (EuclideanSpace.single 0 7)
  (EuclideanSpace.single 1 (1 / 10000)) (by norm_num [Fin.sum_univ_two])

example := headOutput_constant_values (T := 3) (d := 2) (1 / 8388608)
  (1 / 1000000000000) (fun j _ => (j.val : ℝ) / 7)
  (fun j => EuclideanSpace.single 0 j.val) (fun j => EuclideanSpace.single 1 j.val)
  (EuclideanSpace.single 0 1) 2 (by norm_num)

end Transformer.SequenceModel.Attention
