module

public import Mathlib.Analysis.SpecialFunctions.ExpDeriv
public import Transformer.Grokking.FlatSpectrum.Training.SmallSteps

/-!
# The exponential loss: a loss that is not quadratic

Source: `review/flat_spectrum.md`, *Training reaches the optimal fit* (added), item WD9, its
example. The exponential loss `expLoss X = Σ_ij exp (Re X_ij)` has the gradient
`expGrad X = (exp (Re X_ij))_ij`, with `⟨expGrad X, H⟩ = Σ_ij exp (Re X_ij) Re H_ij`
(`pairing_expGrad`). It satisfies the first-order inequality, by `1 + t ≤ exp t`
(`expLoss_first_order`); its gradient is continuous (`continuous_expGrad`); it is real-analytic
in the entries (`analyticAt_expLoss`) and nonnegative (`expLoss_nonneg`). So WD9 applies to
it: small steps converge, to a decayed minimum from full row rank
(`exists_pos_tendsto_decayRun_expLoss_isDecayedMin`), and decayed minima exist for every
`λ > 0` (`exists_isDecayedMin_expLoss`). The constants of WD8 are not verified for this loss;
WD9 needs none.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix Filter Topology

variable {m n k : Type*} [Fintype m] [Fintype n] [Fintype k]

/-- The exponential loss `Σ_ij exp (Re X_ij)`. -/
noncomputable def expLoss (X : Matrix m n ℂ) : ℝ := ∑ i, ∑ j, Real.exp (X i j).re

/-- The gradient of the exponential loss: the real matrix with entries `exp (Re X_ij)`. -/
noncomputable def expGrad (X : Matrix m n ℂ) : Matrix m n ℂ :=
  of fun i j => (Real.exp (X i j).re : ℂ)

/-- `⟨expGrad X, H⟩ = Σ_ij exp (Re X_ij) Re H_ij`. -/
lemma pairing_expGrad (X H : Matrix m n ℂ) :
    pairing (expGrad X) H = ∑ i, ∑ j, Real.exp (X i j).re * (H i j).re := by
  rw [pairing, trace_conjTranspose_mul, Complex.re_sum]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [Complex.re_sum]
  refine Finset.sum_congr rfl fun j _ => ?_
  simp only [expGrad, of_apply, Complex.star_def, Complex.conj_ofReal, Complex.re_ofReal_mul]

/-- **The exponential loss satisfies the first-order inequality** with the gradient `expGrad`:
entrywise `exp x + exp x (y − x) ≤ exp y`. -/
theorem expLoss_first_order (X Y : Matrix m n ℂ) :
    expLoss X + pairing (expGrad X) (Y - X) ≤ expLoss Y := by
  rw [pairing_expGrad, expLoss, expLoss, ← Finset.sum_add_distrib]
  refine Finset.sum_le_sum fun i _ => ?_
  rw [← Finset.sum_add_distrib]
  refine Finset.sum_le_sum fun j _ => ?_
  have h := Real.add_one_le_exp ((Y i j).re - (X i j).re)
  have hpos := Real.exp_pos (X i j).re
  have e : Real.exp (Y i j).re = Real.exp (X i j).re * Real.exp ((Y i j).re - (X i j).re) := by
    rw [← Real.exp_add]
    congr 1
    ring
  rw [Matrix.sub_apply, Complex.sub_re, e]
  nlinarith [mul_le_mul_of_nonneg_left h hpos.le]

omit [Fintype m] [Fintype n] in
lemma continuous_expGrad : Continuous (expGrad : Matrix m n ℂ → Matrix m n ℂ) :=
  continuous_matrix fun i j => Complex.continuous_ofReal.comp
    (Real.continuous_exp.comp (Complex.continuous_re.comp (continuous_id.matrix_elem i j)))

/-- The exponential loss is real-analytic in the entries. -/
theorem analyticAt_expLoss (y₀ : EuclideanSpace ℂ (m × n)) :
    AnalyticAt ℝ (fun y : EuclideanSpace ℂ (m × n) =>
      expLoss (of fun i j => y (i, j) : Matrix m n ℂ)) y₀ := by
  have hre (i : m) (j : n) :
      AnalyticAt ℝ (fun y : EuclideanSpace ℂ (m × n) => (y (i, j)).re) y₀ :=
    (Complex.reCLM.analyticAt _).comp
      (((EuclideanSpace.proj (i, j)).restrictScalars ℝ).analyticAt y₀)
  unfold expLoss
  exact Finset.analyticAt_fun_sum _ fun i _ => Finset.analyticAt_fun_sum _ fun j _ =>
    (hre i j).rexp'

lemma expLoss_nonneg (X : Matrix m n ℂ) : 0 ≤ expLoss X :=
  Finset.sum_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => (Real.exp_pos _).le

variable [DecidableEq m] [DecidableEq n] {lam : ℝ}

/-- **Small steps on the exponential loss converge to a decayed minimum**
(`review/flat_spectrum.md`, item WD9): for a weight decay `λ > 0` and a level `c`, some
`η₀ > 0` has: for every `0 < η < η₀` the run from every start of full row rank with
`F(U₀, V₀) ≤ c` converges, and its limit has the least weight-decayed loss among the
factorizations of every width. -/
theorem exists_pos_tendsto_decayRun_expLoss_isDecayedMin (hlam : 0 < lam) (c : ℝ) :
    ∃ η₀ > 0, ∀ η, 0 < η → η < η₀ → ∀ (U₀ : Matrix m k ℂ) (V₀ : Matrix n k ℂ),
      decayedLoss expLoss lam U₀ V₀ ≤ c → Function.Surjective (stack U₀ V₀).mulVec →
      ∃ (U : Matrix m k ℂ) (V : Matrix n k ℂ),
        Tendsto (decayRun expGrad η lam U₀ V₀) atTop (𝓝 (U, V)) ∧
          IsDecayedMin expLoss lam U V :=
  exists_pos_tendsto_decayRun_isDecayedMin expLoss_first_order continuous_expGrad
    analyticAt_expLoss hlam expLoss_nonneg c

/-- **The exponential loss has decayed minima** (`review/flat_spectrum.md`, item WD9), of
width `m + n`, for every weight decay `λ > 0`. -/
theorem exists_isDecayedMin_expLoss (hlam : 0 < lam) :
    ∃ (U : Matrix m (m ⊕ n) ℂ) (V : Matrix n (m ⊕ n) ℂ), IsDecayedMin expLoss lam U V :=
  exists_isDecayedMin expLoss_first_order continuous_expGrad analyticAt_expLoss hlam
    expLoss_nonneg

example := exists_pos_tendsto_decayRun_expLoss_isDecayedMin (m := Fin 2) (n := Fin 2)
  (k := Fin 2 ⊕ Fin 2) one_pos 5

example := exists_isDecayedMin_expLoss (m := Fin 2) (n := Fin 2) one_pos

/-- The run from a start of width one converges as well, to a stationary point. -/
example := exists_pos_tendsto_decayRun (m := Fin 2) (n := Fin 2) (k := Fin 1)
  expLoss_first_order continuous_expGrad analyticAt_expLoss one_pos expLoss_nonneg 5

end Transformer.Grokking.FlatSpectrum.Training
