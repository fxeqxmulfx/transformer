/-
# The reshapes on differences

`GPTMini.Reshape` shows that `qkvSlice`, `headSlice` and `headMerge` do not
increase the Euclidean norm.  All three are linear, so the same bounds hold
for the *displacement* between two inputs, which is what the Lipschitz
estimates of the attention sub-layer need.  Each proof is the corresponding
norm bound after pushing the subtraction through the reshape.

Source: `reference/model.py` (`CausalMHA.forward`), the `chunk` and `view`
calls around the heads.
-/

import Transformer.GPTMini.Reshape

open scoped BigOperators
open Real

namespace Transformer
namespace GPTMini

/-- `qkvSlice` is norm-decreasing on differences. -/
theorem qkvSlice_dist_le (cfg : Config)
    (e : Fin cfg.d_model → Fin (3 * cfg.d_model)) (he : Function.Injective e)
    (u u' : EucSpace (3 * cfg.d_model)) :
    ‖qkvSlice cfg e u - qkvSlice cfg e u'‖ ≤ ‖u - u'‖ := by
  have hsub : qkvSlice cfg e u - qkvSlice cfg e u' = qkvSlice cfg e (u - u') := by
    ext i; simp
  rw [hsub]
  exact qkvSlice_norm_le cfg e he _

/-- `headSlice` is norm-decreasing on differences. -/
theorem headSlice_dist_le (cfg : Config) (u u' : EucSpace cfg.d_model)
    (h : Fin cfg.n_heads) :
    ‖headSlice cfg u h - headSlice cfg u' h‖ ≤ ‖u - u'‖ := by
  have hsub : headSlice cfg u h - headSlice cfg u' h = headSlice cfg (u - u') h := by
    ext c; simp
  rw [hsub]
  exact headSlice_norm_le cfg _ h

/-- `headMerge` of `n_heads` displacements of size `B` is a displacement of
size `√n_heads · B`. -/
theorem headMerge_dist_le (cfg : Config)
    (y y' : Fin cfg.n_heads → EucSpace cfg.head_dim) (B : ℝ)
    (hB : ∀ h, ‖y h - y' h‖ ≤ B) :
    ‖headMerge cfg y - headMerge cfg y'‖ ≤ Real.sqrt (cfg.n_heads : ℝ) * B := by
  have hsub : headMerge cfg y - headMerge cfg y'
      = headMerge cfg (fun h => y h - y' h) := by
    ext i; simp
  rw [hsub]
  exact headMerge_norm_le cfg _ B hB

/-- The hypotheses are satisfiable: a vector is `0` away from itself. -/
example (cfg : Config) (y : Fin cfg.n_heads → EucSpace cfg.head_dim) :
    ‖headMerge cfg y - headMerge cfg y‖ ≤ Real.sqrt (cfg.n_heads : ℝ) * 0 :=
  headMerge_dist_le cfg y y 0 (fun _ => by simp)

end GPTMini
end Transformer
