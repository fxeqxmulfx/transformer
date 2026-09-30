/-
# AdaFisher: the visualization grid and projection limitation

arXiv:2405.16397v3, Appendix C, the 200×200 first-layer-weight mesh.
An interpolant of logged losses is not generally a single objective as a
function of projected first-layer weights: other layers also change.
-/

import Transformer.AdaFisher.Section3_ConvexFalse
import Transformer.AdaFisher.SectionA_Descent

noncomputable section

namespace Transformer.AdaFisher

/-- The 200 equally spaced coordinates on each grid axis, Appendix C. -/
def gridCoordinate (lo hi : ℝ) (i : Fin 200) : ℝ :=
  lo + (hi - lo) * (i : ℝ) / 199

/-- The visualization grid contains both parameter-range endpoints,
Appendix C, the `meshgrid` formula. -/
theorem gridCoordinate_endpoints (lo hi : ℝ) :
    gridCoordinate lo hi 0 = lo ∧ gridCoordinate lo hi 199 = hi := by
  norm_num [gridCoordinate]

/-- All 200 mesh coordinates lie in their selected parameter range,
Appendix C. -/
theorem gridCoordinate_bounds (lo hi : ℝ) (h : lo ≤ hi) (i : Fin 200) :
    lo ≤ gridCoordinate lo hi i ∧ gridCoordinate lo hi i ≤ hi := by
  have hi0 : (0 : ℝ) ≤ i := by positivity
  have hi199 : (i : ℝ) ≤ 199 := by exact_mod_cast (by omega : i.val ≤ 199)
  unfold gridCoordinate
  constructor <;> nlinarith [mul_nonneg (sub_nonneg.mpr h) hi0,
    mul_nonneg (sub_nonneg.mpr h) (sub_nonneg.mpr hi199)]

example : (0 : ℝ) ≤ 1 := by norm_num

/-- A two-layer linear network's squared-error loss, Appendix C's
first-layer projection discussion. Both layer weights affect the loss. -/
def twoLayerToyLoss (first second : Fin 2 → ℝ) : ℝ := quadratic (pairing first second)

/-- Keeping the first layer fixed does not fix the loss, Appendix C.
This records a limitation on interpreting `griddata(W,L)` as the actual
loss landscape, not a claim that the recorded empirical trajectories fail. -/
theorem projected_weights_do_not_determine_loss :
    ¬ ∃ landscape : (Fin 2 → ℝ) → ℝ,
      ∀ first second, landscape first = twoLayerToyLoss first second := by
  intro h
  obtain ⟨landscape, heq⟩ := h
  have h0 := heq (![1, 0]) (![0, 0])
  have h1 := heq (![1, 0]) (![1, 0])
  norm_num [twoLayerToyLoss, quadratic, pairing, Fin.sum_univ_two] at h0 h1
  linarith

end Transformer.AdaFisher
