import Transformer.Grokking.Geometry.Decisions

/-!
# Raw reference margins and row-aligned cleanup errors

Source: Nanda et al., arXiv:2301.05217v1, section 5.1; the fixed-orbit
observer at 4436290 and geometry_certificates.py in this study. The
reference is the actual cell mean of raw logits, retaining class bias.
Each observed row is shifted to the reference's class mean before its
error is measured. This removes only a common offset from that row.

The equations justify computing an error from row-centered logits while
testing the correct margin of raw cell means. They do not assume that
discarding global class bias preserves answers. Class bias cancels from
the residual but changes the reference; its restoration is essential.
No floating-point implementation or optimization trajectory is identified
with these exact-real functional identities.
-/

namespace Transformer.Grokking.Geometry

open scoped BigOperators
open Transformer.Grokking.NaiveLoss

variable {I C : Type*}

/-- Raw class-coordinate cell mean, including any global class bias.
Source: restricted-logit averaging at 4436290, adapting
arXiv:2301.05217v1, section 5.1. -/
noncomputable def columnMean (s : Finset I) (x : I → C → ℝ) (c : C) : ℝ :=
  meanOver s (fun i => x i c)

/-- Shift one raw row to the reference's average row mean. Source:
geometry_certificates.py, derived from restricted logits in
arXiv:2301.05217v1, section 5.1; no class-dependent shift is applied. -/
noncomputable def alignedLogits (s : Finset I) (cs : Finset C) (x : I → C → ℝ)
    (d : I) (c : C) : ℝ :=
  x d c - meanOver cs (x d) + meanOver s (fun i => meanOver cs (x i))

/-- Averaging commutes with subtraction, including empty cells. Source:
the fixed-cell projection adaptation of arXiv:2301.05217v1, section 5.1. -/
theorem meanOver_sub (s : Finset I) (f g : I → ℝ) :
    meanOver s (fun i => f i - g i) = meanOver s f - meanOver s g := by
  unfold meanOver
  rw [Finset.sum_sub_distrib, div_sub_div_same]

/-- The two finite averages commute. Source: row and cell centering in
the orbit adaptation of arXiv:2301.05217v1, section 5.1, at 4436290. -/
theorem meanOver_comm (s : Finset I) (cs : Finset C) (x : I → C → ℝ) :
    meanOver s (fun i => meanOver cs (x i)) =
      meanOver cs (fun c => meanOver s (fun i => x i c)) := by
  unfold meanOver
  rw [← Finset.sum_div, ← Finset.sum_div, Finset.sum_comm]
  ring

/-- Row alignment leaves its strict decision unchanged. Source: the
harmless-row-offset policy of the adaptation of arXiv:2301.05217v1,
section 5.1, at 4436290 and geometry_certificates.py. -/
theorem aligned_decisions_iff (s : Finset I) (cs : Finset C) (x : I → C → ℝ)
    (d : I) (y : C) : StrictCorrect (alignedLogits s cs x d) y ↔ StrictCorrect (x d) y := by
  have hf : alignedLogits s cs x d = fun c => x d c +
      (meanOver s (fun i => meanOver cs (x i)) - meanOver cs (x d)) := by
    funext c
    unfold alignedLogits
    ring
  rw [hf]
  exact strictCorrect_shift_iff _ _ _

/-- The error against the raw reference is exactly the centered-logit
cell residual. Source: the arithmetic of geometry_certificates.py,
adapting restricted logits in arXiv:2301.05217v1, section 5.1. -/
theorem aligned_error_eq_centered_residual (s : Finset I) (cs : Finset C)
    (x : I → C → ℝ) (d : I) (c : C) :
    alignedLogits s cs x d c - columnMean s x c =
      (x d c - meanOver cs (x d)) -
        meanOver s (fun i => x i c - meanOver cs (x i)) := by
  unfold alignedLogits columnMean
  rw [meanOver_sub]
  ring

/-- A common row shift is added to its mean on nonempty class sets.
Source: row centering at 4436290, adapting arXiv:2301.05217v1, section 5.1. -/
theorem row_mean_after_shift (cs : Finset C) (x : C → ℝ) (b : ℝ) (hc : cs.Nonempty) :
    meanOver cs (fun c => x c + b) = meanOver cs x + b := by
  rw [meanOver_add, meanOver_const cs b hc]

example : ({false, true} : Finset Bool).Nonempty := by
  exact ⟨false, by norm_num⟩

/-- Arbitrary row offsets do not change the aligned error. Source:
the centering policy of the adaptation of arXiv:2301.05217v1, section 5.1,
at 4436290; each row may have a different shift. -/
theorem aligned_error_row_shift_invariant (s : Finset I) (cs : Finset C)
    (x : I → C → ℝ) (b : I → ℝ) (d : I) (c : C) (hc : cs.Nonempty) :
    alignedLogits s cs (fun i k => x i k + b i) d c -
      columnMean s (fun i k => x i k + b i) c =
        alignedLogits s cs x d c - columnMean s x c := by
  rw [aligned_error_eq_centered_residual, aligned_error_eq_centered_residual]
  simp_rw [row_mean_after_shift cs _ _ hc]
  have hf : (fun i => x i c + b i - (meanOver cs (x i) + b i)) =
      fun i => x i c - meanOver cs (x i) := by
    funext i
    ring
  rw [hf]
  ring

example : ({0, 1} : Finset ℕ).Nonempty := by
  exact ⟨0, by norm_num⟩

/-- Class bias remains in the raw reference rather than being discarded.
Source: the correctness policy in the adaptation of arXiv:2301.05217v1,
section 5.1, at 4436290 and geometry_certificates.py. -/
theorem column_mean_after_class_bias (s : Finset I) (x : I → C → ℝ)
    (g : C → ℝ) (c : C) (hs : s.Nonempty) :
    columnMean s (fun i k => x i k + g k) c = columnMean s x c + g c := by
  unfold columnMean
  rw [meanOver_add, meanOver_const s (g c) hs]

example : ({0, 1} : Finset ℕ).Nonempty := by
  exact ⟨1, by norm_num⟩

/-- Class bias cancels from the aligned error on a nonempty cell, even
though it changes reference margins. Source: the centering policy of the
adaptation of arXiv:2301.05217v1, section 5.1, at 4436290. -/
theorem aligned_error_class_bias_invariant (s : Finset I) (cs : Finset C)
    (x : I → C → ℝ) (g : C → ℝ) (d : I) (c : C) (hs : s.Nonempty) :
    alignedLogits s cs (fun i k => x i k + g k) d c -
      columnMean s (fun i k => x i k + g k) c =
        alignedLogits s cs x d c - columnMean s x c := by
  rw [aligned_error_eq_centered_residual, aligned_error_eq_centered_residual]
  simp_rw [meanOver_add]
  have hf : (fun i => x i c + g c - (meanOver cs (x i) + meanOver cs g)) =
      fun i => (x i c - meanOver cs (x i)) + (g c - meanOver cs g) := by
    funext i
    ring
  rw [hf, meanOver_add, meanOver_const s _ hs]
  ring

example : ({false, true} : Finset Bool).Nonempty := by
  exact ⟨true, by norm_num⟩

/-- The exact row-aligned error and raw cell margin certify the original
decision. Source: Geometry.strictCorrect_of_margin_and_energy at 3661a44,
derived from the interpretation of arXiv:2301.05217v1, section 5.1.
This is the current-logit condition evaluated by geometry_certificates.py. -/
theorem row_aligned_cleanup_certifies [Fintype C] (s : Finset I) (x : I → C → ℝ)
    (d : I) (y : C) (margin : ℝ) (hm : MarginAtLeast (columnMean s x) y margin)
    (hp : 0 < margin) (he : 2 * energyOver Finset.univ
      (fun c => alignedLogits s Finset.univ x d c - columnMean s x c) < margin ^ 2) :
    StrictCorrect (x d) y := by
  apply (aligned_decisions_iff s Finset.univ x d y).mp
  exact strictCorrect_of_margin_and_energy _ _ y margin hm hp he

example : MarginAtLeast
    (columnMean ({0, 1} : Finset ℕ) (fun _ (b : Bool) => if b then (1 : ℝ) else -1)) true 2 ∧
    0 < (2 : ℝ) ∧ 2 * energyOver Finset.univ (fun b : Bool =>
      alignedLogits ({0, 1} : Finset ℕ) Finset.univ
        (fun _ (b : Bool) => if b then (1 : ℝ) else -1) 0 b -
      columnMean ({0, 1} : Finset ℕ) (fun _ (b : Bool) => if b then (1 : ℝ) else -1) b) < 2 ^ 2 := by
  refine ⟨?_, by norm_num, ?_⟩
  · intro b hb
    cases b
    · norm_num [columnMean, meanOver]
    · exact False.elim (hb rfl)
  · norm_num [alignedLogits, columnMean, meanOver, energyOver, Fintype.sum_bool]

end Transformer.Grokking.Geometry
