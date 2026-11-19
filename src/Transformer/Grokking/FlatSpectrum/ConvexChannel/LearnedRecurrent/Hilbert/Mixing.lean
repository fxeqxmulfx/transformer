module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Hilbert.Basic

/-!
# Actual norm-preserving complex mixing

Source: `review/convex/moments.md`, CM107. Full complex execution of real mixing
coefficients is an exact isometry when their squared sum is one. This supplies
an inhabited Hilbert component, not a benchmark answer rule or initialization.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Hilbert

/-- CM107: a mixing subfamily of the complete complex matrix family. -/
noncomputable def mixing (a b : ℝ) : Pair.Matrix :=
  (((a : ℂ), -(b : ℂ)), ((b : ℂ), (a : ℂ)))

/-- CM107: exact energy of every real/imaginary channel under actual mixing. -/
theorem mixing_energy (a b : ℝ) (v : State) :
    ‖operator (mixing a b) v‖ ^ 2 = (a ^ 2 + b ^ 2) * ‖v‖ ^ 2 := by
  rw [norm_squared, norm_squared v, operator_apply]
  simp only [mixing, Complex.sq_norm, Complex.normSq_apply,
    Complex.add_re, Complex.add_im, Complex.mul_re, Complex.mul_im,
    Complex.neg_re, Complex.neg_im, Complex.ofReal_re, Complex.ofReal_im]
  ring

/-- CM107: normalized mixing preserves the complete Hilbert state norm. -/
theorem mixing_norm (a b : ℝ) (h : a ^ 2 + b ^ 2 = 1) (v : State) :
    ‖operator (mixing a b) v‖ = ‖v‖ := by
  have he := mixing_energy a b v
  rw [h, one_mul] at he
  nlinarith [norm_nonneg (operator (mixing a b) v), norm_nonneg v]

/-- CM107: the explicit energy condition implies the actual operator budget. -/
theorem mixing_bound (a b : ℝ) (h : a ^ 2 + b ^ 2 ≤ 1) :
    ‖operator (mixing a b)‖ ≤ 1 := by
  apply ContinuousLinearMap.opNorm_le_bound _ (by norm_num)
  intro v
  have he := mixing_energy a b v
  have hb := mul_le_mul_of_nonneg_right h (sq_nonneg ‖v‖)
  nlinarith [norm_nonneg (operator (mixing a b) v), norm_nonneg v]

/-- CM107: nonzero complex parameters for the rational mixing witness. -/
noncomputable def witness (a b : ℝ) : Parameters Bool State :=
  ⟨WithLp.toLp 2 (Complex.I, 0), fun _ => operator (mixing a b),
    fun c => row (if c then (-Complex.I, 0) else (0, -Complex.I))⟩

/-- CM107: the rational unitary witness is inside the complete Hilbert family. -/
theorem witness_unit : Unit (witness (3 / 5) (4 / 5)) := by
  refine ⟨?_, fun _ => mixing_bound _ _ (by norm_num), ?_⟩
  · norm_num [witness]
  · intro c
    apply row_norm_bound
    cases c <;> simp

/-- CM107: the unitary witness actually preserves norm at every sequence length. -/
theorem witness_flat (v : State) (xs : List Bool) :
    ‖runFrom (witness (3 / 5) (4 / 5)) v xs‖ = ‖v‖ := by
  apply flat_run
  intro o z
  exact mixing_norm _ _ (by norm_num) z

/-- CM107: constant one-step energy factors multiply in actual ordered execution. -/
theorem run_squared_scale {O : Type*} (f : Parameters O State) (q : ℝ)
    (h : ∀ o v, ‖f.transition o v‖ ^ 2 = q * ‖v‖ ^ 2) (v : State) (xs : List O) :
    ‖runFrom f v xs‖ ^ 2 = q ^ xs.length * ‖v‖ ^ 2 := by
  induction xs generalizing v with
  | nil => simp [runFrom]
  | cons o xs ih =>
    change ‖runFrom f (f.transition o v) xs‖ ^ 2 = _
    rw [ih, h, List.length_cons, pow_succ]
    ring

example : ‖operator (mixing (3 / 5) (4 / 5)) (WithLp.toLp 2 (Complex.I, 1))‖ =
    ‖(WithLp.toLp 2 (Complex.I, (1 : ℂ)) : State)‖ :=
  mixing_norm _ _ (by norm_num) _

example : ‖operator (mixing (3 / 5) (4 / 5))‖ ≤ 1 :=
  mixing_bound _ _ (by norm_num)

example : ‖runFrom (witness (3 / 5) (4 / 5)) (WithLp.toLp 2 (Complex.I, 0))
    [true, false, true]‖ ^ 2 = 1 ^ ([true, false, true] : List Bool).length *
      ‖(WithLp.toLp 2 (Complex.I, (0 : ℂ)) : State)‖ ^ 2 :=
  run_squared_scale _ 1 (by
    intro o v
    have he := mixing_energy (3 / 5) (4 / 5) v
    norm_num at he
    simpa [witness] using he) _ _

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Hilbert
