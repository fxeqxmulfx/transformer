/-
# What the concentration plot measures

arXiv:2601.22813v2, "Quartet II: Accurate LLM Pre-Training in NVFP4 by
Improved Unbiased Gradient Estimation" (ICML 2026), Appendix A.

The paper checks unbiasedness of a quantized backward pass by averaging `B`
independent quantized gradients and plotting the error against the exact one:
"If `Ĝ` is unbiased, i.e. `E_ω Ĝ(ω) = G`, the error will decrease to
arbitrarily small values as `~1/B` asymptotically, from the Central Limit
Theorem" — while a biased estimator plateaus, which is how Figure 5 separates
`MS-EDEN` and the NVIDIA recipe from Four Over Six on the backward pass.

Both halves of that reading are stated below for one coordinate of the
gradient: a draw is a uniform seed out of `N`, `g` is the quantized value it
produces, and `G` is the exact one.  The slope and the plateau are the two
statements; the figure is the evidence the paper offers for them.
-/

import Mathlib.Topology.UniformSpace.Real
import Mathlib.Algebra.BigOperators.Fin

namespace Transformer
namespace Quartet

variable {N B : ℕ}

/-- The expectation of a quantized coordinate over a uniform seed. -/
noncomputable def seedMean (g : Fin N → ℝ) : ℝ := (N : ℝ)⁻¹ * ∑ ω, g ω

/-- Its variance. -/
noncomputable def seedVar (g : Fin N → ℝ) : ℝ := (N : ℝ)⁻¹ * ∑ ω, (g ω - seedMean g) ^ 2

/-- The expected squared error of the average of `B` independent draws against
the exact value `G`: the quantity Appendix A plots against `B`. -/
noncomputable def meanSqErr (B : ℕ) (g : Fin N → ℝ) (G : ℝ) : ℝ :=
  ((N : ℝ) ^ B)⁻¹ * ∑ ω : Fin B → Fin N, ((B : ℝ)⁻¹ * ∑ b, g (ω b) - G) ^ 2

/-- **An unbiased estimator concentrates like `1/B`** (Appendix A, "the error
will decrease … as `~1/B` asymptotically"): the expected squared error of the
`B`-fold average is exactly the variance of one draw over `B`. -/
theorem meanSqErr_of_seedMean_eq {g : Fin N → ℝ} {G : ℝ} (hN : 0 < N) (hB : 0 < B)
    (h : seedMean g = G) : meanSqErr B g G = seedVar g / B :=
  sorry

/-- The hypotheses are satisfiable: a single seed that returns the exact value
is unbiased, with one draw. -/
example : 0 < 1 ∧ 0 < 1 ∧ seedMean (fun _ : Fin 1 => (2 : ℝ)) = 2 := by
  refine ⟨Nat.one_pos, Nat.one_pos, ?_⟩
  simp [seedMean]

/-- **A biased estimator plateaus** at the square of its bias, however many
draws are averaged (Appendix A, "Plateauing methods (NVIDIA+4/6) introduce
bias").  Together with the statement above, this is what makes the plot a
test: the slope, not the height, is what unbiasedness shows up as. -/
theorem tendsto_meanSqErr {g : Fin N → ℝ} {G : ℝ} (hN : 0 < N) :
    Filter.Tendsto (fun B => meanSqErr B g G) Filter.atTop
      (nhds ((seedMean g - G) ^ 2)) :=
  sorry

/-- Its hypothesis is satisfiable, and the limit is not always `0`: two seeds
that both overshoot by `1` leave a plateau of `1`. -/
example : 0 < 2 ∧ (seedMean (fun _ : Fin 2 => (3 : ℝ)) - 2) ^ 2 = 1 := by
  have h : seedMean (fun _ : Fin 2 => (3 : ℝ)) = 3 := by norm_num [seedMean]
  exact ⟨Nat.succ_pos 1, by rw [h]; norm_num⟩

end Quartet
end Transformer
