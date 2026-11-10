module

public import Mathlib.Analysis.SpecificLimits.Normed
public import Transformer.Grokking.FlatSpectrum.Training.Continuity

/-!
# A run in closed form: the linear loss

Source: `review/flat_spectrum.md`, *Training reaches the optimal fit* (added), item WD5, its
example. For the linear loss `L X = −Re tr X` of a square matrix the gradient is the constant
`−1`, and with weight decay `λ = 1` a step is

    U ← U − η (U − V),    V ← V − η (V − U).

The sum `U + V` is kept and the difference `U − V` is multiplied by `1 − 2η`, so the run is
`½ (U₀ + V₀) ± ½ (1 − 2η)^t (U₀ − V₀)` (`decayRun_linear`) and converges to
`U = V = ½ (U₀ + V₀)` for `0 < η < 1` (`tendsto_decayRun_linear`): a nonzero limit from the
full-rank start `U₀ = [1 0]`, `V₀ = [0 1]`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix Filter Topology

variable {m k : Type*} [Fintype m] [Fintype k] [DecidableEq m]

/-- The run of the linear loss `L X = −Re tr X` with weight decay `1`, in closed form. -/
theorem decayRun_linear (η : ℝ) (U₀ V₀ : Matrix m k ℂ) (t : ℕ) :
    decayRun (fun _ => (-1 : Matrix m m ℂ)) η 1 U₀ V₀ t =
      ((1 / 2 : ℝ) • (U₀ + V₀) + ((1 - 2 * η) ^ t / 2) • (U₀ - V₀),
        (1 / 2 : ℝ) • (U₀ + V₀) - ((1 - 2 * η) ^ t / 2) • (U₀ - V₀)) := by
  induction t with
  | zero =>
    simp only [decayRun, Function.iterate_zero, id_eq, pow_zero]
    ext : 1 <;> module
  | succ t ih =>
    rw [decayRun, Function.iterate_succ_apply', ← decayRun, ih, decayStep]
    simp only [conjTranspose_neg, conjTranspose_one, Matrix.neg_mul, Matrix.one_mul, one_smul,
      pow_succ]
    ext : 1 <;> module

/-- The run of the linear loss converges to `U = V = ½ (U₀ + V₀)` for `0 < η < 1`. -/
theorem tendsto_decayRun_linear {η : ℝ} (h0 : 0 < η) (h1 : η < 1) (U₀ V₀ : Matrix m k ℂ) :
    Tendsto (decayRun (fun _ => (-1 : Matrix m m ℂ)) η 1 U₀ V₀) atTop
      (𝓝 ((1 / 2 : ℝ) • (U₀ + V₀), (1 / 2 : ℝ) • (U₀ + V₀))) := by
  have hpow : Tendsto (fun t : ℕ => ((1 - 2 * η) ^ t / 2) • (U₀ - V₀)) atTop (𝓝 0) := by
    have h := ((tendsto_pow_atTop_nhds_zero_of_abs_lt_one (r := 1 - 2 * η)
      (abs_lt.mpr ⟨by linarith, by linarith⟩)).div_const 2).smul_const (U₀ - V₀)
    rwa [zero_div, zero_smul] at h
  have h := (tendsto_const_nhds (x := (1 / 2 : ℝ) • (U₀ + V₀))).add hpow
  have h' := (tendsto_const_nhds (x := (1 / 2 : ℝ) • (U₀ + V₀))).sub hpow
  rw [add_zero] at h
  rw [sub_zero] at h'
  rw [funext (decayRun_linear η U₀ V₀)]
  exact h.prodMk_nhds h'

example := tendsto_decayRun_linear (η := 1 / 4) (by norm_num) (by norm_num)
  (1 : Matrix (Fin 1) (Fin 1) ℂ) 0

end Transformer.Grokking.FlatSpectrum.Training
