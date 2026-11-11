module

public import Transformer.Grokking.FlatSpectrum.Training.Convergence
public import Transformer.Grokking.FlatSpectrum.Training.LinearRun

/-!
# A loss that is not bounded below: the run diverges

Source: `review/flat_spectrum.md`, *Training reaches the optimal fit* (added), item WD11, the
bullet "Losses outside WD8 and WD9": for a loss "that is not bounded below, convergence stays
the hypothesis of WD5". This file shows that the hypothesis cannot be dropped (item WD12).

The linear loss `L X = −Re tr X` of a square matrix is convex with the constant gradient `−1`
(`linear_first_order`), real-analytic (`analyticAt_linear`), and not bounded below
(`exists_linear_lt`). With a weight decay `λ` a step is

    U ← (1 − η λ) U + η V,    V ← (1 − η λ) V + η U,

so `U + V` is multiplied by `1 + η (1 − λ)` and `U − V` by `1 − η (1 + λ)`
(`decayRun_linear_decay`, `decayRun_linear_add`). For `λ < 1` and `η > 0` the first factor
exceeds one. So from every start with `U₀ + V₀ ≠ 0` the norm of the factors tends to infinity
(`tendsto_frobeniusSq_decayRun_linear_atTop`) and the run has no limit
(`not_tendsto_decayRun_linear`), at every step size and every width. Along `(s U, s U)` the
weight-decayed loss is `−(1 − λ) s² ‖U‖²` (`decayedLoss_linear_smul`), so it is not bounded
below and no factorization is a decayed minimum (`not_isDecayedMin_linear`).

The examples are the `1 × 1` loss `−Re X` with `λ = 1/2` and `η = 1/4`, from `U₀ = [1 0]`,
`V₀ = [0 1]`: the start has full row rank and the step matrix is positive definite, the two
other hypotheses of WD5, and the run has no limit. At `λ = 1` the same run converges
(`tendsto_decayRun_linear`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training.Scope

open Matrix Filter Topology
open scoped ComplexOrder

variable {m k : Type*} [Fintype m] [Fintype k] [DecidableEq m]

/-- The linear loss `−Re tr X` has the first-order inequality with the gradient `−1`. -/
lemma linear_first_order (X Y : Matrix m m ℂ) :
    pairing (-1) X + pairing (-1 : Matrix m m ℂ) (Y - X) ≤ pairing (-1) Y := by
  rw [pairing_sub_right]
  exact (add_sub_cancel _ _).le

/-- The linear loss on a multiple of the identity: `−s |m|`. -/
lemma pairing_neg_one_smul_one (s : ℝ) :
    pairing (-1 : Matrix m m ℂ) (s • 1) = -(s * frobeniusSq (1 : Matrix m m ℂ)) := by
  rw [pairing_smul_right, show (-1 : Matrix m m ℂ) = (-1 : ℝ) • 1 by simp, pairing_smul_left,
    pairing_self]
  ring

/-- **The linear loss is not bounded below.** -/
theorem exists_linear_lt [Nonempty m] (c : ℝ) : ∃ X : Matrix m m ℂ, pairing (-1) X < c := by
  have hpos : 0 < frobeniusSq (1 : Matrix m m ℂ) :=
    lt_of_le_of_ne (frobeniusSq_nonneg _) fun h =>
      one_ne_zero (eq_zero_of_frobeniusSq_eq_zero h.symm)
  refine ⟨((|c| + 1) / frobeniusSq (1 : Matrix m m ℂ)) • 1, ?_⟩
  rw [pairing_neg_one_smul_one, div_mul_cancel₀ _ hpos.ne']
  linarith [neg_abs_le c]

/-- The linear loss is real-analytic in the entries, in the form of WD8 and WD9. -/
lemma analyticAt_linear (y : EuclideanSpace ℂ (m × m)) :
    AnalyticAt ℝ (fun y : EuclideanSpace ℂ (m × m) =>
      pairing (-1 : Matrix m m ℂ) (of fun i j => y (i, j))) y := by
  have e : (fun y : EuclideanSpace ℂ (m × m) =>
      pairing (-1 : Matrix m m ℂ) (of fun i j => y (i, j))) = fun y => -∑ i, (y (i, i)).re := by
    funext y
    simp [pairing, Matrix.trace]
  rw [e]
  exact (Finset.analyticAt_fun_sum _ fun i _ => (Complex.reCLM.analyticAt _).comp
    (((EuclideanSpace.proj (i, i)).restrictScalars ℝ).analyticAt y)).neg

/-- **The run of the linear loss with a weight decay `λ`, in closed form.** -/
theorem decayRun_linear_decay (η lam : ℝ) (U₀ V₀ : Matrix m k ℂ) (t : ℕ) :
    decayRun (fun _ => (-1 : Matrix m m ℂ)) η lam U₀ V₀ t =
      (((1 + η * (1 - lam)) ^ t / 2) • (U₀ + V₀) + ((1 - η * (1 + lam)) ^ t / 2) • (U₀ - V₀),
        ((1 + η * (1 - lam)) ^ t / 2) • (U₀ + V₀) -
          ((1 - η * (1 + lam)) ^ t / 2) • (U₀ - V₀)) := by
  induction t with
  | zero =>
    simp only [decayRun, Function.iterate_zero, id_eq, pow_zero]
    ext : 1 <;> module
  | succ t ih =>
    rw [decayRun, Function.iterate_succ_apply', ← decayRun, ih, decayStep]
    simp only [conjTranspose_neg, conjTranspose_one, Matrix.neg_mul, Matrix.one_mul, pow_succ]
    ext : 1 <;> module

/-- The sum of the factors is multiplied by `1 + η (1 − λ)` at every step. -/
lemma decayRun_linear_add (η lam : ℝ) (U₀ V₀ : Matrix m k ℂ) (t : ℕ) :
    (decayRun (fun _ => (-1 : Matrix m m ℂ)) η lam U₀ V₀ t).1 +
      (decayRun (fun _ => (-1 : Matrix m m ℂ)) η lam U₀ V₀ t).2 =
        (1 + η * (1 - lam)) ^ t • (U₀ + V₀) := by
  rw [decayRun_linear_decay]
  module

/-- **The norm of the factors tends to infinity** for a weight decay `λ < 1`, at every step
`η > 0`, from every start with `U₀ + V₀ ≠ 0`. -/
theorem tendsto_frobeniusSq_decayRun_linear_atTop {η lam : ℝ} (hη : 0 < η) (hlam : lam < 1)
    {U₀ V₀ : Matrix m k ℂ} (h0 : U₀ + V₀ ≠ 0) :
    Tendsto (fun t => frobeniusSq (decayRun (fun _ => (-1 : Matrix m m ℂ)) η lam U₀ V₀ t).1 +
      frobeniusSq (decayRun (fun _ => (-1 : Matrix m m ℂ)) η lam U₀ V₀ t).2) atTop atTop := by
  have ha : 1 < (1 + η * (1 - lam)) ^ 2 := by nlinarith [mul_pos hη (sub_pos.2 hlam)]
  have hpos : 0 < frobeniusSq (U₀ + V₀) :=
    lt_of_le_of_ne (frobeniusSq_nonneg _) fun h => h0 (eq_zero_of_frobeniusSq_eq_zero h.symm)
  have hpow : Tendsto (fun t : ℕ => ((1 + η * (1 - lam)) ^ 2) ^ t * frobeniusSq (U₀ + V₀) / 2)
      atTop atTop :=
    ((tendsto_pow_atTop_atTop_of_one_lt ha).atTop_mul_const hpos).atTop_div_const two_pos
  refine tendsto_atTop_mono (fun t => ?_) hpow
  have h := frobeniusSq_add_le (decayRun (fun _ => (-1 : Matrix m m ℂ)) η lam U₀ V₀ t).1
    (decayRun (fun _ => (-1 : Matrix m m ℂ)) η lam U₀ V₀ t).2
  rw [decayRun_linear_add, frobeniusSq_smul, pow_right_comm] at h
  linarith

/-- **The run has no limit** for a weight decay `λ < 1`. -/
theorem not_tendsto_decayRun_linear {η lam : ℝ} (hη : 0 < η) (hlam : lam < 1)
    {U₀ V₀ : Matrix m k ℂ} (h0 : U₀ + V₀ ≠ 0) (W : Matrix m k ℂ × Matrix m k ℂ) :
    ¬ Tendsto (decayRun (fun _ => (-1 : Matrix m m ℂ)) η lam U₀ V₀) atTop (𝓝 W) := fun h => by
  have hc : Continuous fun p : Matrix m k ℂ × Matrix m k ℂ =>
      frobeniusSq p.1 + frobeniusSq p.2 :=
    (continuous_frobeniusSq.comp continuous_fst).add (continuous_frobeniusSq.comp continuous_snd)
  exact not_tendsto_nhds_of_tendsto_atTop
    (tendsto_frobeniusSq_decayRun_linear_atTop hη hlam h0) _ ((hc.tendsto W).comp h)

/-- The weight-decayed linear loss along `(s U, s U)`: `−(1 − λ) s² ‖U‖²`. -/
lemma decayedLoss_linear_smul (lam s : ℝ) (U : Matrix m k ℂ) :
    decayedLoss (fun X : Matrix m m ℂ => pairing (-1) X) lam (s • U) (s • U) =
      -((1 - lam) * s ^ 2 * frobeniusSq U) := by
  have h1 : pairing (-1 : Matrix m m ℂ) (U * Uᴴ) = -frobeniusSq U := by
    rw [show (-1 : Matrix m m ℂ) = (-1 : ℝ) • 1 by simp, pairing_smul_left,
      ← pairing_mul_right, Matrix.one_mul, pairing_self]
    ring
  rw [decayedLoss, conjTranspose_smul, star_trivial, Matrix.smul_mul, Matrix.mul_smul,
    pairing_smul_right, pairing_smul_right, h1, frobeniusSq_smul]
  ring

/-- **No factorization is a decayed minimum** of the linear loss for a weight decay `λ < 1`:
the weight-decayed loss is not bounded below. -/
theorem not_isDecayedMin_linear [Nonempty m] {lam : ℝ} (hlam : lam < 1) (U V : Matrix m k ℂ) :
    ¬ IsDecayedMin (fun X : Matrix m m ℂ => pairing (-1) X) lam U V := fun hmin => by
  set d := decayedLoss (fun X : Matrix m m ℂ => pairing (-1) X) lam U V with hd
  set E : Matrix m (Fin 1) ℂ := fun _ _ => 1 with hE
  have hEpos : 0 < frobeniusSq E :=
    lt_of_le_of_ne (frobeniusSq_nonneg _) fun h => by
      have h0 := congrFun (congrFun (eq_zero_of_frobeniusSq_eq_zero h.symm)
        (Classical.arbitrary m)) 0
      exact one_ne_zero h0
  have h1 : 0 < 1 - lam := sub_pos.2 hlam
  have h := hmin 1 (√((|d| + 1) / ((1 - lam) * frobeniusSq E)) • E)
    (√((|d| + 1) / ((1 - lam) * frobeniusSq E)) • E)
  rw [decayedLoss_linear_smul, Real.sq_sqrt (by positivity), ← hd] at h
  have e : (1 - lam) * ((|d| + 1) / ((1 - lam) * frobeniusSq E)) * frobeniusSq E = |d| + 1 := by
    field_simp
  rw [e] at h
  linarith [neg_abs_le d]

/-! The witnesses. `M₁` is the type of the complex `1 × 1` matrices. -/

local notation "M₁" => Matrix (Fin 1) (Fin 1) ℂ

example : ∃ X : M₁, pairing (-1) X < -5 := exists_linear_lt _

/-- The `1 × 1` loss `−Re X` with `λ = 1/2` and `η = 1/4`, from `U₀ = [1 0]`, `V₀ = [0 1]`:
the run has no limit. -/
example (W : Matrix (Fin 1) (Fin 1 ⊕ Fin 1) ℂ × Matrix (Fin 1) (Fin 1 ⊕ Fin 1) ℂ) :
    ¬ Tendsto (decayRun (fun _ => (-1 : M₁)) (1 / 4) (1 / 2) (fromCols (1 : M₁) (0 : M₁))
      (fromCols (0 : M₁) (1 : M₁))) atTop (𝓝 W) :=
  not_tendsto_decayRun_linear (by norm_num) (by norm_num) (fun h => by
    have h0 := congrFun (congrFun h 0) (Sum.inl 0)
    simp at h0) W

/-- Its start has full row rank, a hypothesis of WD5. -/
example : Function.Surjective
    (stack (fromCols (1 : M₁) (0 : M₁)) (fromCols (0 : M₁) (1 : M₁))).mulVec :=
  fun v => ⟨v, by rw [stack_fromCols_one, one_mulVec]⟩

/-- Its step matrix, the same at every step, is positive definite, a hypothesis of WD5. -/
example : (1 - (1 / 4 : ℝ) • lift (1 / 2) (-1 : M₁)).PosDef :=
  one_sub_smul_lift_posDef (g := 1) (by norm_num) zero_le_one (by simp [frobeniusSq])
    (by norm_num [abs_of_pos])

example := not_isDecayedMin_linear (m := Fin 1) (lam := 1 / 2) (by norm_num) (1 : M₁) 1

end Transformer.Grokking.FlatSpectrum.Training.Scope
