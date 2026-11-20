module

public import Transformer.RoPE.ReRoPE2D.Basic
public import Transformer.AttentionResiduals.Section4_OnlineMerge

/-!
# 2D ReRoPE: nine attention passes and their exact merge

2D ReRoPE is in no paper; see `Transformer.RoPE.ReRoPE2D.Defs`. Su computes ReRoPE with two
attention matrices: a RoPE pass for the keys inside the window, and a pass with the query at `w`
and the keys at `0` for the others (kexue.fm/archives/9708, 计算 (Computation);
`Transformer.RoPE.ReRoPE.merge_rectified`). With the two-sided clamp each axis has three cases,
and the two axes give `3 × 3 = 9` regions.

* `caseOf w d`: `pos` from `d ≥ w` on, `neg` up to `d ≤ −w`, `zero` in between; `region w x y`:
  the cases of the two axes' offsets;
* `queryPosition`, `keyPosition`, `regionScore`: the pass of a region places the query at its own
  coordinate on a near axis and at `±w` on a far one, the key at its own coordinate on a near axis
  and at `0` on a far one. Each rotation depends on one position and the region, so each pass is
  an ordinary attention pass over rotated queries and keys, the keys of the other regions masked.
  The key rotation depends only on which axes are near: the nine passes read four key caches;
* `score_eq_regionScore`: in its own region the 2D ReRoPE score is that pass's score;
* `attend_eq_merge`: softmax attention with 2D ReRoPE's scores is `(∑_r Z_r)⁻¹ ∑_r N_r`, from the
  nine passes' masses `Z_r` and unnormalized outputs `N_r`; `attend_eq_sum_smul`: it is the
  average of their outputs `o_r = Z_r⁻¹ N_r` weighted by `Z_r / ∑ Z`, the log-sum-exp merge of
  separately computed passes. Attention is `AttentionResiduals.attend` and its statistics are
  `AttentionResiduals.attend_from_statistics`, with exact reals.

In one dimension `Transformer.RoPE.ReRoPE.rerope_attend_eq_onlineMerge` merges Su's two regions,
the window and the keys beyond it, in the same way.
-/

@[expose] public section

open scoped BigOperators
open Matrix Transformer.RoPE.RoFormer Transformer.AttentionResiduals

namespace Transformer.RoPE.ReRoPE2D

variable {p : ℕ}

/-- The case of the offset `d` at the window `w`: `pos` from `w` on, `neg` up to `−w`, `zero`
inside. -/
noncomputable def caseOf (w d : ℝ) : SignType :=
  if w ≤ d then .pos else if d ≤ -w then .neg else .zero

/-- The region of a query at `x` and a key at `y`: the case of each axis's offset. -/
noncomputable def region (w x y : Fin 2 → ℝ) (a : Fin 2) : SignType :=
  caseOf (w a) (x a - y a)

/-- A query's coordinate in the pass of a case: its own `x` inside the window, `±w` outside. -/
def queryPosition (w x : ℝ) : SignType → ℝ
  | .zero => x
  | .pos => w
  | .neg => -w

/-- A key's coordinate in the pass of a case: its own `y` inside the window, `0` outside. -/
def keyPosition (y : ℝ) : SignType → ℝ
  | .zero => y
  | .pos => 0
  | .neg => 0

/-- The score of the pass of the region `r`: the query at `x` and the key at `y` rotated by axial
RoPE to their coordinates in that pass. -/
noncomputable def regionScore (θ : Fin p → ℝ) (ax : Fin p → Fin 2) (w : Fin 2 → ℝ)
    (r : Fin 2 → SignType) (x y : Fin 2 → ℝ) (q k : Fin 2 × Fin p → ℝ) : ℝ :=
  (axialRotary θ ax (fun a => queryPosition (w a) (x a) (r a)) *ᵥ q) ⬝ᵥ
    (axialRotary θ ax (fun a => keyPosition (y a) (r a)) *ᵥ k)

/-- Three cases on each of the two axes: nine regions. -/
example : Fintype.card (Fin 2 → SignType) = 9 := by
  simp only [Fintype.card_fun, Fintype.card_fin]
  rfl

/-- From the window on, the clamp is `w`. -/
theorem clampPosition_of_le {w d : ℝ} (hw : 0 ≤ w) (h : w ≤ d) : clampPosition w d = w := by
  rw [clampPosition, min_eq_right h, max_eq_right (by linarith)]

/-- The hypotheses of `clampPosition_of_le` hold for the offset `1000` and the window `256`. -/
example := clampPosition_of_le (w := 256) (d := 1000) (by norm_num) (by norm_num)

/-- Up to `−w`, the clamp is `−w`. -/
theorem clampPosition_of_le_neg {w d : ℝ} (hw : 0 ≤ w) (h : d ≤ -w) :
    clampPosition w d = -w := by
  rw [clampPosition, min_eq_left (by linarith), max_eq_left h]

/-- The hypotheses of `clampPosition_of_le_neg` hold for the offset `−1000` and the window
`256`. -/
example := clampPosition_of_le_neg (w := 256) (d := -1000) (by norm_num) (by norm_num)

/-- **Each region is one pass**: with nonnegative windows, the 2D ReRoPE score of a query and a key
is the score of the pass of their region. -/
theorem score_eq_regionScore (θ : Fin p → ℝ) (ax : Fin p → Fin 2) {w : Fin 2 → ℝ}
    (hw : ∀ a, 0 ≤ w a) (x y : Fin 2 → ℝ) (q k : Fin 2 × Fin p → ℝ) :
    score θ ax w x y q k = regionScore θ ax w (region w x y) x y q k := by
  rw [regionScore, axialRotary_dotProduct, score]
  congr 3
  funext a
  simp only [Pi.neg_apply, Pi.sub_apply, clampOffset, region, caseOf]
  split_ifs with h1 h2
  · rw [clampPosition_of_le (hw a) h1]
    simp [queryPosition, keyPosition]
  · rw [clampPosition_of_le_neg (hw a) h2]
    simp [queryPosition, keyPosition]
  · rw [clampPosition_of_mem_Icc ⟨(not_le.mp h2).le, (not_le.mp h1).le⟩]
    simp [queryPosition, keyPosition]

/-- The hypothesis of `score_eq_regionScore` holds in MoonViT's layout for the windows `256`. -/
example (q k : Fin 2 × Fin 32 → ℝ) := score_eq_regionScore (kimiFreq 10000) kimiAxis
  (w := fun _ => 256) (fun _ => by norm_num) ![1000, 3] ![0, 7] q k

variable {κ : Type*} [Fintype κ] {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- The mass `Z_r` of the pass of the region `r` for the query at `x`: `∑ exp` of its scores over
the keys of `r`. -/
noncomputable def regionMass (θ : Fin p → ℝ) (ax : Fin p → Fin 2) (w : Fin 2 → ℝ)
    (r : Fin 2 → SignType) (x : Fin 2 → ℝ) (y : κ → Fin 2 → ℝ) (q : Fin 2 × Fin p → ℝ)
    (k : κ → Fin 2 × Fin p → ℝ) : ℝ :=
  ∑ j with region w x (y j) = r, Real.exp (regionScore θ ax w r x (y j) q (k j))

/-- The unnormalized output `N_r` of the pass of the region `r` for the query at `x`. -/
noncomputable def regionNumerator (θ : Fin p → ℝ) (ax : Fin p → Fin 2) (w : Fin 2 → ℝ)
    (r : Fin 2 → SignType) (x : Fin 2 → ℝ) (y : κ → Fin 2 → ℝ) (q : Fin 2 × Fin p → ℝ)
    (k : κ → Fin 2 × Fin p → ℝ) (v : κ → E) : E :=
  ∑ j with region w x (y j) = r, Real.exp (regionScore θ ax w r x (y j) q (k j)) • v j

/-- The output `o_r = Z_r⁻¹ N_r` of the pass of the region `r` alone. -/
noncomputable def regionOutput (θ : Fin p → ℝ) (ax : Fin p → Fin 2) (w : Fin 2 → ℝ)
    (r : Fin 2 → SignType) (x : Fin 2 → ℝ) (y : κ → Fin 2 → ℝ) (q : Fin 2 × Fin p → ℝ)
    (k : κ → Fin 2 × Fin p → ℝ) (v : κ → E) : E :=
  (regionMass θ ax w r x y q k)⁻¹ • regionNumerator θ ax w r x y q k v

/-- **Nine passes, merged exactly**: softmax attention of the query at `x` over the keys at `y j`
with 2D ReRoPE's scores is `(∑_r Z_r)⁻¹ ∑_r N_r`, from the masses and unnormalized outputs of the
nine passes, each masked to its region. -/
theorem attend_eq_merge [Nonempty κ] (θ : Fin p → ℝ) (ax : Fin p → Fin 2) {w : Fin 2 → ℝ}
    (hw : ∀ a, 0 ≤ w a) (x : Fin 2 → ℝ) (y : κ → Fin 2 → ℝ) (q : Fin 2 × Fin p → ℝ)
    (k : κ → Fin 2 × Fin p → ℝ) (v : κ → E) :
    attend (fun j => score θ ax w x (y j) q (k j)) v =
      (∑ r, regionMass θ ax w r x y q k)⁻¹ • ∑ r, regionNumerator θ ax w r x y q k v := by
  have hZ : shiftedMass (fun j => score θ ax w x (y j) q (k j)) 0 =
      ∑ r, regionMass θ ax w r x y q k := by
    simp only [shiftedMass, regionMass]
    rw [← Finset.sum_fiberwise Finset.univ fun j => region w x (y j)]
    refine Finset.sum_congr rfl fun r _ => Finset.sum_congr rfl fun j hj => ?_
    rw [sub_zero, score_eq_regionScore θ ax hw, (Finset.mem_filter.mp hj).2]
  have hN : shiftedNumerator (fun j => score θ ax w x (y j) q (k j)) v 0 =
      ∑ r, regionNumerator θ ax w r x y q k v := by
    simp only [shiftedNumerator, Residual.weightedSum, regionNumerator]
    rw [← Finset.sum_fiberwise Finset.univ fun j => region w x (y j)]
    refine Finset.sum_congr rfl fun r _ => Finset.sum_congr rfl fun j hj => ?_
    rw [sub_zero, score_eq_regionScore θ ax hw, (Finset.mem_filter.mp hj).2]
  rw [attend_from_statistics _ _ 0, hZ, hN]

/-- The hypotheses of `attend_eq_merge` hold in GLM-5.3-Flash's layout for sixteen keys and the
windows `2`. -/
example (y : Fin 16 → Fin 2 → ℝ) (q : Fin 2 × Fin 32 → ℝ) (k : Fin 16 → Fin 2 × Fin 32 → ℝ)
    (v : Fin 16 → Fin 8 → ℝ) := attend_eq_merge (glmFreq 10000) glmAxis (w := fun _ => 2)
  (fun _ => by norm_num) ![1, 3] y q k v

/-- **The log-sum-exp merge**: softmax attention with 2D ReRoPE's scores is the average of the
nine passes' outputs `o_r`, weighted by `Z_r / ∑ Z`. A pass whose region holds no key has
`Z_r = 0` and weight `0`. -/
theorem attend_eq_sum_smul [Nonempty κ] (θ : Fin p → ℝ) (ax : Fin p → Fin 2) {w : Fin 2 → ℝ}
    (hw : ∀ a, 0 ≤ w a) (x : Fin 2 → ℝ) (y : κ → Fin 2 → ℝ) (q : Fin 2 × Fin p → ℝ)
    (k : κ → Fin 2 × Fin p → ℝ) (v : κ → E) :
    attend (fun j => score θ ax w x (y j) q (k j)) v =
      ∑ r, (regionMass θ ax w r x y q k / ∑ r', regionMass θ ax w r' x y q k) •
        regionOutput θ ax w r x y q k v := by
  rw [attend_eq_merge θ ax hw, Finset.smul_sum]
  refine Finset.sum_congr rfl fun r _ => ?_
  rw [regionOutput, smul_smul]
  by_cases h : regionMass θ ax w r x y q k = 0
  · have hN : regionNumerator θ ax w r x y q k v = 0 :=
      Finset.sum_eq_zero fun j hj => by
        rw [(Finset.sum_eq_zero_iff_of_nonneg fun j _ => (Real.exp_pos _).le).mp h j hj,
          zero_smul]
    rw [hN, smul_zero, smul_zero]
  · congr 1
    field_simp

/-- The hypotheses of `attend_eq_sum_smul` hold in MoonViT's layout for sixteen keys and the
windows `2`. -/
example (y : Fin 16 → Fin 2 → ℝ) (q : Fin 2 × Fin 32 → ℝ) (k : Fin 16 → Fin 2 × Fin 32 → ℝ)
    (v : Fin 16 → Fin 8 → ℝ) := attend_eq_sum_smul (kimiFreq 10000) kimiAxis (w := fun _ => 2)
  (fun _ => by norm_num) ![1, 3] y q k v

end Transformer.RoPE.ReRoPE2D
