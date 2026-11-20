module

public import Transformer.AttentionResiduals.Section3_Attention

/-!
# Softmax attention: renamed keys, convexity, moved scores

Tools for `Transformer.ReRoPEBlock.Clamp`, over any finite set of keys, for the attention
`attend s u = ∑_j (e^{s_j} / ∑_k e^{s_k}) u_j` of `Transformer.AttentionResiduals`:

* `attend_comp_equiv`: renaming the keys does not change an attention;
* `attend_mem_convexHull`: an attention lies in the convex hull of its values;
* `depthWeight_le_exp_mul`: scores within `ε` of each other change a weight by a factor of at
  most `e^{2ε}`;
* `norm_attend_sub_attend_le`: so they move the attention by at most `(e^{2ε} − 1) R` when its
  values have norms at most `R`.
-/

@[expose] public section

open Transformer.AttentionResiduals Transformer.Residual

namespace Transformer.ReRoPEBlock

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- **Renaming the keys** does not change an attention. -/
theorem attend_comp_equiv {ι κ : Type*} [Fintype ι] [Fintype κ] (σ : ι ≃ κ) (s : κ → ℝ)
    (u : κ → E) : attend (fun j => s (σ j)) (fun j => u (σ j)) = attend s u := by
  simp only [attend, weightedSum, depthWeight, depthPartition]
  rw [Equiv.sum_comp σ fun k => Real.exp (s k)]
  exact Equiv.sum_comp σ fun k => (Real.exp (s k) / ∑ k, Real.exp (s k)) • u k

/-- Swapping the scores and the values of two keys. -/
example (s : Fin 2 → ℝ) (u : Fin 2 → ℝ) :
    attend (fun j => s (Equiv.swap 0 1 j)) (fun j => u (Equiv.swap 0 1 j)) = attend s u :=
  attend_comp_equiv (Equiv.swap 0 1) s u

/-- **An attention lies in the convex hull of its values.** -/
theorem attend_mem_convexHull {ι : Type*} [Fintype ι] [Nonempty ι] (s : ι → ℝ) (u : ι → E) :
    attend s u ∈ convexHull ℝ (Set.range u) :=
  (convex_convexHull ℝ _).sum_mem (fun j _ => (depthWeight_pos s j).le) (depthWeight_sum s)
    fun j _ => subset_convexHull ℝ _ (Set.mem_range_self j)

/-- Scores within `ε` of each other change a weight by a factor of at most `e^{2ε}`: one `e^ε`
from the key's own score, one from the normalizer. -/
theorem depthWeight_le_exp_mul {ι : Type*} [Fintype ι] [Nonempty ι] {s s' : ι → ℝ} {ε : ℝ}
    (hs : ∀ j, |s j - s' j| ≤ ε) (j : ι) :
    depthWeight s' j ≤ Real.exp (2 * ε) * depthWeight s j := by
  have hZ := depthPartition_pos s
  have hZ' := depthPartition_pos s'
  have hup : ∀ k, Real.exp (s' k) ≤ Real.exp ε * Real.exp (s k) := fun k => by
    rw [← Real.exp_add, Real.exp_le_exp]
    linarith [(abs_le.1 (hs k)).1]
  have hdown : ∀ k, Real.exp (s k) ≤ Real.exp ε * Real.exp (s' k) := fun k => by
    rw [← Real.exp_add, Real.exp_le_exp]
    linarith [(abs_le.1 (hs k)).2]
  have hZle : depthPartition s ≤ Real.exp ε * depthPartition s' := by
    simp only [depthPartition, Finset.mul_sum]
    exact Finset.sum_le_sum fun k _ => hdown k
  simp only [depthWeight]
  rw [← mul_div_assoc, div_le_div_iff₀ hZ' hZ, show 2 * ε = ε + ε by ring, Real.exp_add]
  calc Real.exp (s' j) * depthPartition s
      ≤ (Real.exp ε * Real.exp (s j)) * (Real.exp ε * depthPartition s') :=
        mul_le_mul (hup j) hZle hZ.le (by positivity)
    _ = Real.exp ε * Real.exp ε * Real.exp (s j) * depthPartition s' := by ring

/-- Two keys whose scores differ by at most `1`. -/
example : depthWeight (fun j : Fin 2 => (j : ℝ)) 0 ≤
    Real.exp (2 * 1) * depthWeight (fun _ : Fin 2 => (0 : ℝ)) 0 :=
  depthWeight_le_exp_mul (fun j => by fin_cases j <;> norm_num) 0

/-- **Moved scores move an attention little**: scores within `ε` of each other move the attention
of values of norms at most `R` by at most `(e^{2ε} − 1) R`. -/
theorem norm_attend_sub_attend_le {ι : Type*} [Fintype ι] [Nonempty ι] {s s' : ι → ℝ} {ε : ℝ}
    (hs : ∀ j, |s j - s' j| ≤ ε) (u : ι → E) {R : ℝ} (hu : ∀ j, ‖u j‖ ≤ R) :
    ‖attend s u - attend s' u‖ ≤ (Real.exp (2 * ε) - 1) * R := by
  have hε : 0 ≤ ε := (abs_nonneg _).trans (hs (Classical.arbitrary ι))
  have hE : 1 ≤ Real.exp (2 * ε) := Real.one_le_exp (by linarith)
  have hs' : ∀ j, |s' j - s j| ≤ ε := fun j => by rw [abs_sub_comm]; exact hs j
  have hw : ∀ j, |depthWeight s j - depthWeight s' j| ≤ (Real.exp (2 * ε) - 1) * depthWeight s j :=
    fun j => by
      have h1 := depthWeight_le_exp_mul hs j
      have h2 := depthWeight_le_exp_mul hs' j
      have hp := (depthWeight_pos s j).le
      have hp' := (depthWeight_pos s' j).le
      rw [abs_le]
      constructor
      · linarith
      · nlinarith [mul_nonneg hp (sq_nonneg (Real.exp (2 * ε) - 1))]
  simp only [attend, weightedSum, ← Finset.sum_sub_distrib, ← sub_smul]
  calc ‖∑ j, (depthWeight s j - depthWeight s' j) • u j‖
      ≤ ∑ j, ‖(depthWeight s j - depthWeight s' j) • u j‖ := norm_sum_le _ _
    _ ≤ ∑ j, (Real.exp (2 * ε) - 1) * depthWeight s j * R :=
        Finset.sum_le_sum fun j _ => by
          rw [norm_smul, Real.norm_eq_abs]
          exact mul_le_mul (hw j) (hu j) (norm_nonneg _)
            (mul_nonneg (by linarith) (depthWeight_pos s j).le)
    _ = (Real.exp (2 * ε) - 1) * R := by
        rw [← Finset.sum_mul, ← Finset.mul_sum, depthWeight_sum, mul_one]

/-- Two keys, scores `0, 1` against `0, 0`, values of norm at most `1`. -/
example (u : Fin 2 → ℝ) (hu : ∀ j, ‖u j‖ ≤ 1) :
    ‖attend (fun _ => (0 : ℝ)) u - attend (fun j : Fin 2 => (j : ℝ)) u‖ ≤
      (Real.exp (2 * 1) - 1) * 1 :=
  norm_attend_sub_attend_le (fun j => by fin_cases j <;> norm_num) u hu

end Transformer.ReRoPEBlock
