/-
# Adapted Euler chains for the actual Brownian driver

arXiv:2506.12543v1, Section 4.3, equations (2)--(3), Theorem 1.
These are finite stochastic approximations to the continuous diffusion.
Every update uses a fresh Brownian increment and the current random state.
-/

import Transformer.BatchSize.Section4_VectorItoSums
import Transformer.BatchSize.Section4_NoiseLipschitz

open MeasureTheory
open scoped NNReal ENNReal

noncomputable section

namespace Transformer.BatchSize

/-- Lipschitz coefficient composition preserves finite-measure Lp even
when the coefficient does not vanish at zero; Section 4.3, Theorem 1. -/
theorem lipschitz_comp_memLp_finite {Ω E F : Type*} [MeasurableSpace Ω]
    [NormedAddCommGroup E] [NormedAddCommGroup F] {P : Measure Ω} [IsFiniteMeasure P]
    {p : ℝ≥0∞} {X : Ω → E} {g : E → F} {K : ℝ≥0}
    (hg : LipschitzWith K g) (hX : MemLp X p P) : MemLp (fun ω => g (X ω)) p P := by
  have hc : LipschitzWith K (fun x => g x - g 0) :=
    LipschitzWith.of_dist_le_mul fun x y => by
      simpa only [dist_eq_norm, sub_sub_sub_cancel_right] using hg.norm_sub_le x y
  have h := (hc.comp_memLp (by simp) hX).add (memLp_const (g 0))
  convert h using 1
  ext ω
  simp only [Function.comp_def, Pi.add_apply, sub_add_cancel]

/-- Joint nonvacuity of coefficient composition, Section 4.3:
the identity applied to an actual Brownian coordinate. -/
example : LipschitzWith 1 (id : ℝ → ℝ) ∧
    MemLp (coordinateBrownian (0 : Fin 1) 1) 2 (brownianNoiseLaw 1) :=
  ⟨LipschitzWith.id, ((coordinateBrownian_isBrownian (0 : Fin 1)).isGaussianProcess.hasGaussianLaw_eval 1).memLp_two⟩

/-- A diagonal-noise Euler chain for Section 4.3 (2)--(3).
Time t(n+1) uses coefficients at the current state at t(n). -/
def eulerChain {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (t : ℕ → ℝ≥0) (x₀ : EucSpace d) :
    ℕ → BrownianSample d → EucSpace d
  | 0, _ => x₀
  | n + 1, ω =>
    let x := eulerChain b a t x₀ n ω
    x + ((t (n + 1) : ℝ) - t n) • b x + WithLp.toLp 2
      (fun k => a x k * brownianIncrement k (t n) (t (n + 1)) ω)

/-- Each Euler state is measurable in the actual joint past,
Section 4.3 (2)--(3). -/
theorem eulerChain_adapted {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (hb : Continuous b)
    (ha : ∀ k, Continuous (fun x => a x k)) (t : ℕ → ℝ≥0) (hmono : Monotone t)
    (x₀ : EucSpace d) (n : ℕ) :
    StronglyMeasurable[brownianFiltration d (t n)] (eulerChain b a t x₀ n) := by
  induction n with
  | zero => exact stronglyMeasurable_const
  | succ n ih =>
    have ht := (brownianFiltration d).mono (hmono (Nat.le_succ n))
    have hX := ih.mono ht
    have hnoise : StronglyMeasurable[brownianFiltration d (t (n + 1))]
        (fun ω => WithLp.toLp 2 (fun k => a (eulerChain b a t x₀ n ω) k *
          brownianIncrement k (t n) (t (n + 1)) ω)) := by
      let : MeasurableSpace (BrownianSample d) := brownianFiltration d (t (n + 1))
      apply (PiLp.continuousLinearEquiv 2 ℝ (fun _ : Fin d => ℝ)).symm.continuous.comp_stronglyMeasurable
      apply Measurable.stronglyMeasurable
      apply Measurable.of_eval
      intro k
      exact ((ha k).comp_stronglyMeasurable hX).measurable.mul
        (((coordinateBrownian_filtered k).stronglyAdapted (t (n + 1))).sub
          (((coordinateBrownian_filtered k).stronglyAdapted (t n)).mono ht)).measurable
    have hdrift : StronglyMeasurable[brownianFiltration d (t (n + 1))]
        (fun ω => ((t (n + 1) : ℝ) - t n) • b (eulerChain b a t x₀ n ω)) :=
      (hb.comp_stronglyMeasurable hX).const_smul ((t (n + 1) : ℝ) - t n)
    exact (hX.add hdrift).add hnoise

/-- Euler states remain square integrable for globally Lipschitz
coefficients, Section 4.3 (2)--(3). The noise-product step uses independence
from the joint past, rather than an incorrect L2-by-L2 product estimate. -/
theorem eulerChain_memLp {d : ℕ} (b : EucSpace d → EucSpace d)
    (a : EucSpace d → Fin d → ℝ) (Kb Ka : ℝ≥0)
    (hb : LipschitzWith Kb b) (ha : ∀ k, LipschitzWith Ka (fun x => a x k))
    (t : ℕ → ℝ≥0) (hmono : Monotone t) (x₀ : EucSpace d) (n : ℕ) :
    MemLp (eulerChain b a t x₀ n) 2 (brownianNoiseLaw d) := by
  induction n with
  | zero => exact memLp_const x₀
  | succ n ih =>
    have hX := eulerChain_adapted b a hb.continuous (fun k => (ha k).continuous) t hmono x₀ n
    have hnoise : MemLp
        (fun ω => WithLp.toLp 2 (fun k => a (eulerChain b a t x₀ n ω) k *
          brownianIncrement k (t n) (t (n + 1)) ω)) 2 (brownianNoiseLaw d) := by
      apply MemLp.of_eval_piLp
      intro k
      exact brownianIncrement_adapted_memLp k (t n) (t (n + 1)) (hmono (Nat.le_succ n))
        ((ha k).continuous.comp_stronglyMeasurable hX) (lipschitz_comp_memLp_finite (ha k) ih)
    have hdrift : MemLp
        (fun ω => ((t (n + 1) : ℝ) - t n) • b (eulerChain b a t x₀ n ω)) 2 (brownianNoiseLaw d) :=
      (lipschitz_comp_memLp_finite hb ih).const_smul ((t (n + 1) : ℝ) - t n)
    exact (ih.add hdrift).add hnoise

/-- Joint nonvacuity of the Euler hypotheses, Section 4.3:
linear drift and positive constant diagonal amplitudes in two dimensions. -/
example : LipschitzWith 1 (id : EucSpace 2 → EucSpace 2) ∧
    (∀ k : Fin 2, LipschitzWith 0 (fun _ : EucSpace 2 => (k : ℝ) + 1)) ∧
    Monotone (fun n : ℕ => (n : ℝ≥0)) :=
  ⟨LipschitzWith.id, fun _ => LipschitzWith.const _,
    fun i j h => by change (i : ℝ≥0) ≤ (j : ℝ≥0); exact_mod_cast h⟩

/-- The Euler chain with exactly the optimizer diffusion coefficients
of Section 4.3 (2)--(3). -/
def optimizerEulerChain {d : ℕ} (method : UpdateKind) (η : ℝ) (B : ℕ)
    (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (t : ℕ → ℝ≥0) (x₀ : EucSpace d) : ℕ → BrownianSample d → EucSpace d :=
  eulerChain (diffusionDrift method B f σ) (diffusionNoiseScale method η B f σ) t x₀

/-- The model hypotheses establish adaptedness and L2 membership of
every actual optimizer Euler state, Section 4.3 (2)--(3), Theorem 1. -/
theorem optimizerEulerChain_wellFormed {d : ℕ} (method : UpdateKind) (η : ℝ) (B : ℕ)
    (f : EucSpace d → ℝ) (σ : EucSpace d → EucSpace d)
    (hη : 0 ≤ η) (hmodel : RegularGaussianModel f σ)
    (t : ℕ → ℝ≥0) (hmono : Monotone t) (x₀ : EucSpace d) (n : ℕ) :
    StronglyMeasurable[brownianFiltration d (t n)] (optimizerEulerChain method η B f σ t x₀ n) ∧
      MemLp (optimizerEulerChain method η B f σ t x₀ n) 2 (brownianNoiseLaw d) := by
  obtain ⟨Kb, hb⟩ := diffusionDrift_lipschitz method B f σ hmodel
  obtain ⟨Ka, ha⟩ := diffusionNoiseScale_lipschitz method B f σ hmodel
  exact ⟨eulerChain_adapted _ _ hb.continuous (fun k => (ha η hη k).continuous) t hmono x₀ n,
    eulerChain_memLp _ _ Kb (‖Real.sqrt η‖₊ * Ka) hb (ha η hη) t hmono x₀ n⟩

/-- Joint nonvacuity of the optimizer Euler hypotheses, Section 4.3. -/
example : (0 : ℝ) ≤ 1 / 1000 ∧
    RegularGaussianModel (fun _ : EucSpace 2 => (0 : ℝ))
      (fun _ => WithLp.toLp 2 (fun _ : Fin 2 => (1 : ℝ))) ∧
    Monotone (fun n : ℕ => (n : ℝ≥0)) :=
  ⟨by norm_num, regularGaussianModel_flat 2,
    fun i j h => by change (i : ℝ≥0) ≤ (j : ℝ≥0); exact_mod_cast h⟩

end Transformer.BatchSize
