module

public import Transformer.Homogenized.Section7_SimplexMomentBounds
public import Mathlib.Analysis.Calculus.MeanValue

/-!
# Scalar barriers for the simplex ODE

ArXiv:2604.01978v1, §7, eq:gamma_ODE_selfcontained and the
reference R(γ) used in thm:clustering_random_init.
The moment bounds imply 0≤b(c)≤2(1-c) on the feasible interval.
Here strict exponential barriers prove that a solution starting
in its interior stays there at every finite time.

The argument only evaluates b at the barrier contact points,
which are explicitly inside the feasible interval. It needs no
extension, continuity or Lipschitz bound for b outside that interval.
The trajectory is then nondecreasing by its actual derivative.
The subsequent application derives all drift premises from the
original attention moments and explicit simplex realization.
-/

@[expose] public section

open Set

namespace Transformer.Homogenized

/-- Strict exponential barriers keep the scalar solution away from both
endpoints at finite times. Source: arXiv:2604.01978v1, §7,
eq:gamma_ODE_selfcontained. Drift bounds are explicit inputs here;
they were derived from actual attention moments in SimplexMomentBounds. -/
theorem scalar_ode_exp_bounds (L γ₀ : ℝ) (hγ₀ : γ₀ ∈ Ioo L 1)
    (b : ℝ → ℝ) (hb : ∀ c ∈ Icc L 1, b c ∈ Icc (0 : ℝ) (2 * (1 - c)))
    (γ : ℝ → ℝ) (hinit : γ 0 = γ₀)
    (hd : ∀ t ∈ Ici (0 : ℝ), HasDerivWithinAt γ (b (γ t)) (Ici 0) t)
    (T : ℝ) (hT : 0 ≤ T) :
    L + (γ₀ - L) * Real.exp (-T) ≤ γ T ∧
      γ T ≤ 1 - (1 - γ₀) * Real.exp (-3 * T) := by
  have hc : ContinuousOn γ (Icc (0 : ℝ) T) := fun t ht =>
    (hd t ht.1).continuousWithinAt.mono (fun s hs => hs.1)
  have hdr (t : ℝ) (ht : t ∈ Ico (0 : ℝ) T) :
      HasDerivWithinAt γ (b (γ t)) (Ici t) t :=
    (hd t ht.1).mono (fun s hs => ht.1.trans hs)
  have hu (t : ℝ) : HasDerivAt (fun s : ℝ => 1 - (1 - γ₀) * Real.exp (-3 * s))
      (3 * (1 - γ₀) * Real.exp (-3 * t)) t := by
    convert ((((hasDerivAt_id t).const_mul (-3)).exp).const_mul (1 - γ₀)).const_sub 1 using 1
    · simp only [id_eq]
    · simp only [id_eq]; ring
  have hl (t : ℝ) : HasDerivAt (fun s : ℝ => -(L + (γ₀ - L) * Real.exp (-s)))
      ((γ₀ - L) * Real.exp (-t)) t := by
    convert (((hasDerivAt_neg t).exp.const_mul (γ₀ - L)).const_add L).neg using 1
    ring
  have huI (t : ℝ) (ht : 0 ≤ t) :
      1 - (1 - γ₀) * Real.exp (-3 * t) ∈ Icc L 1 := by
    have he : Real.exp (-3 * t) ≤ 1 := Real.exp_le_one_iff.mpr (by linarith)
    have hp := Real.exp_pos (-3 * t)
    have hm := mul_le_mul_of_nonneg_left he (sub_nonneg.mpr hγ₀.2.le)
    constructor <;> nlinarith [hγ₀.1, hγ₀.2]
  have hlI (t : ℝ) (ht : 0 ≤ t) :
      L + (γ₀ - L) * Real.exp (-t) ∈ Icc L 1 := by
    have he : Real.exp (-t) ≤ 1 := Real.exp_le_one_iff.mpr (by linarith)
    have hp := Real.exp_pos (-t)
    have hm := mul_le_mul_of_nonneg_left he (sub_nonneg.mpr hγ₀.1.le)
    constructor <;> nlinarith [hγ₀.1, hγ₀.2]
  constructor
  · have hneg : -γ T ≤ -(L + (γ₀ - L) * Real.exp (-T)) := by
      apply image_le_of_deriv_right_lt_deriv_boundary hc.neg (fun t ht => (hdr t ht).neg)
        (B := fun t => -(L + (γ₀ - L) * Real.exp (-t)))
        (B' := fun t => (γ₀ - L) * Real.exp (-t))
        (by simp [hinit]) hl (fun t ht hEq => ?_) ⟨hT, le_rfl⟩
      change -γ t = -(L + (γ₀ - L) * Real.exp (-t)) at hEq
      have heq : γ t = L + (γ₀ - L) * Real.exp (-t) := by linarith
      have hnonneg := (hb _ (hlI t ht.1)).1
      rw [heq]
      nlinarith [mul_pos (sub_pos.mpr hγ₀.1) (Real.exp_pos (-t))]
    linarith
  · apply image_le_of_deriv_right_lt_deriv_boundary hc hdr
      (B := fun t => 1 - (1 - γ₀) * Real.exp (-3 * t))
      (B' := fun t => 3 * (1 - γ₀) * Real.exp (-3 * t))
      (by simp [hinit]) hu (fun t ht heq => ?_) ⟨hT, le_rfl⟩
    have hbound := (hb _ (huI t ht.1)).2
    rw [heq]
    nlinarith [mul_pos (sub_pos.mpr hγ₀.2) (Real.exp_pos (-3 * t))]

/-- The strict initial interval is invariant for all finite nonnegative
times. Source: arXiv:2604.01978v1, §7, eq:gamma_ODE_selfcontained. -/
theorem scalar_ode_mem_Ioo (L γ₀ : ℝ) (hγ₀ : γ₀ ∈ Ioo L 1)
    (b : ℝ → ℝ) (hb : ∀ c ∈ Icc L 1, b c ∈ Icc (0 : ℝ) (2 * (1 - c)))
    (γ : ℝ → ℝ) (hinit : γ 0 = γ₀)
    (hd : ∀ t ∈ Ici (0 : ℝ), HasDerivWithinAt γ (b (γ t)) (Ici 0) t)
    (t : ℝ) (ht : 0 ≤ t) : γ t ∈ Ioo L 1 := by
  have h := scalar_ode_exp_bounds L γ₀ hγ₀ b hb γ hinit hd t ht
  have hlow := mul_pos (sub_pos.mpr hγ₀.1) (Real.exp_pos (-t))
  have hhigh := mul_pos (sub_pos.mpr hγ₀.2) (Real.exp_pos (-3 * t))
  constructor <;> linarith [h.1, h.2]

/-- The same solution is nondecreasing, now that its actual values are
known to remain in the domain of the drift bounds.
Source: arXiv:2604.01978v1, §7, eq:gamma_ODE_selfcontained. -/
theorem scalar_ode_monotoneOn (L γ₀ : ℝ) (hγ₀ : γ₀ ∈ Ioo L 1)
    (b : ℝ → ℝ) (hb : ∀ c ∈ Icc L 1, b c ∈ Icc (0 : ℝ) (2 * (1 - c)))
    (γ : ℝ → ℝ) (hinit : γ 0 = γ₀)
    (hd : ∀ t ∈ Ici (0 : ℝ), HasDerivWithinAt γ (b (γ t)) (Ici 0) t) :
    MonotoneOn γ (Ici (0 : ℝ)) := by
  apply monotoneOn_of_hasDerivWithinAt_nonneg (convex_Ici 0)
    (fun t ht => (hd t ht).continuousWithinAt)
  · intro t ht
    rw [interior_Ici] at ht ⊢
    have ht0 : 0 < t := ht
    exact (hd t ht0.le).mono (fun s hs => le_of_lt (show 0 < s from hs))
  · intro t ht
    rw [interior_Ici] at ht
    have ht0 : 0 < t := ht
    have hγt := scalar_ode_mem_Ioo L γ₀ hγ₀ b hb γ hinit hd t ht0.le
    exact (hb _ ⟨hγt.1.le, hγt.2.le⟩).1

/-- Every later value is at least the actual initial overlap.
Source: arXiv:2604.01978v1, §7, eq:gamma_ODE_selfcontained.
This follows from the proved monotonicity, without a separate
monotonicity assumption on the solution. -/
theorem scalar_ode_initial_le (L γ₀ : ℝ) (hγ₀ : γ₀ ∈ Ioo L 1)
    (b : ℝ → ℝ) (hb : ∀ c ∈ Icc L 1, b c ∈ Icc (0 : ℝ) (2 * (1 - c)))
    (γ : ℝ → ℝ) (hinit : γ 0 = γ₀)
    (hd : ∀ t ∈ Ici (0 : ℝ), HasDerivWithinAt γ (b (γ t)) (Ici 0) t)
    (t : ℝ) (ht : 0 ≤ t) : γ₀ ≤ γ t := by
  rw [← hinit]
  exact scalar_ode_monotoneOn L γ₀ hγ₀ b hb γ hinit hd (by simp) ht ht

/-- A nonconstant trajectory and nonzero drift satisfy every barrier
premise: γ(t)=1-exp(-t), b(c)=1-c, L=-1 and γ₀=0.
Source: arXiv:2604.01978v1, §7, the scalar comparison step. -/
example : ∃ b γ : ℝ → ℝ,
    (0 : ℝ) ∈ Ioo (-1 : ℝ) 1 ∧
    (∀ c ∈ Icc (-1 : ℝ) 1, b c ∈ Icc (0 : ℝ) (2 * (1 - c))) ∧
    γ 0 = 0 ∧
    (∀ t ∈ Ici (0 : ℝ), HasDerivWithinAt γ (b (γ t)) (Ici 0) t) ∧
    (0 : ℝ) ≤ 1 ∧ γ 1 = 1 - Real.exp (-1) ∧
    MonotoneOn γ (Ici (0 : ℝ)) ∧ ∀ t ∈ Ici (0 : ℝ), γ t ∈ Ioo (-1 : ℝ) 1 := by
  let b : ℝ → ℝ := fun c => 1 - c
  let γ : ℝ → ℝ := fun t => 1 - Real.exp (-t)
  have hzero : (0 : ℝ) ∈ Ioo (-1 : ℝ) 1 := by norm_num
  have hb : ∀ c ∈ Icc (-1 : ℝ) 1, b c ∈ Icc (0 : ℝ) (2 * (1 - c)) := by
    intro c hc
    dsimp [b]
    constructor <;> linarith [hc.2]
  have hinit : γ 0 = 0 := by simp [γ]
  have hd : ∀ t ∈ Ici (0 : ℝ), HasDerivWithinAt γ (b (γ t)) (Ici 0) t := by
    intro t ht
    convert (((hasDerivAt_neg t).exp).const_sub 1).hasDerivWithinAt using 1
    simp [γ, b]
  exact ⟨b, γ, hzero, hb, hinit, hd, by norm_num, rfl,
    scalar_ode_monotoneOn (-1) 0 hzero b hb γ hinit hd,
    fun t ht => scalar_ode_mem_Ioo (-1) 0 hzero b hb γ hinit hd t ht⟩

end Transformer.Homogenized
