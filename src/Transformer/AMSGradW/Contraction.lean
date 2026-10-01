/-
# Frozen-metric equilibrium and momentum contraction

arXiv:1904.03590v4, Algorithm 1 and §4, AMSGradW extension.
The limiting equation is `gradient f x + decay*D*x = 0`, not
stationarity of the unregularized loss.
-/

import Transformer.AMSGradW.Bounds

noncomputable section

namespace Transformer.AMSGradW

variable {d : ℕ}

/-- Fixed-point map for a positive frozen denominator. Source:
arXiv:1904.03590v4, Algorithm 1, AMSGradW extension. -/
def frozenMap (wd : ℝ) (D : Fin d → ℝ) (f : TrainingSpace d → ℝ)
    (x : Fin d → ℝ) : Fin d → ℝ := fun i => -coordinateGradient f x i / (wd * D i)

/-- Lipschitz continuity of the frozen map follows from the genuine
objective gradient and denominator lower bound. Source:
arXiv:1904.03590v4, §4, AMSGradW extension. -/
theorem frozenMap_lipschitz (ε wd L : ℝ) (D : Fin d → ℝ)
    (f : TrainingSpace d → ℝ) (hε : 0 < ε) (hwd : 0 < wd) (hL : 0 ≤ L)
    (hD : ∀ i, ε ≤ D i) (hgrad : LipschitzGradient f L) :
    LipschitzWith ⟨L / (wd * ε), div_nonneg hL (mul_pos hwd hε).le⟩
      (frozenMap wd D f) := by
  apply LipschitzWith.of_dist_le_mul
  intro x y
  rw [dist_eq_norm, dist_eq_norm]
  have heq : frozenMap wd D f x - frozenMap wd D f y =
      fun i => -(coordinateGradient f x - coordinateGradient f y) i / (wd * D i) := by
    ext i
    simp only [frozenMap, Pi.sub_apply]
    ring
  rw [heq]
  have hdiv := division_norm_bound (wd * ε) (-(coordinateGradient f x - coordinateGradient f y))
    (fun i => wd * D i) (mul_pos hwd hε) (fun i => mul_le_mul_of_nonneg_left (hD i) hwd.le)
  rw [norm_neg] at hdiv
  have h := div_le_div_of_nonneg_right (hgrad x y) (mul_pos hwd hε).le
  change ‖fun i => -(coordinateGradient f x - coordinateGradient f y) i / (wd * D i)‖ ≤
    L / (wd * ε) * ‖x - y‖
  simpa only [Pi.neg_apply, div_mul_eq_mul_div] using hdiv.trans h

/-- Frozen-map assumptions hold for a nonconstant objective and positive
decay. Source: arXiv:1904.03590v4, §4, AMSGradW extension. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) < 2 ∧ (0 : ℝ) ≤ 1 ∧
    (∀ i : Fin 1, (1 : ℝ) ≤ (fun _ => 2) i) ∧
    LipschitzGradient (Optimization.energy : TrainingSpace 1 → ℝ) 1 := by
  exact ⟨by norm_num, by norm_num, by norm_num, fun i => by norm_num, energy_lipschitz⟩

/-- A positive denominator has a unique equilibrium in the strict
decay-dominated regime. Existence follows from Banach's theorem; an
existing minimizer or convergence of weights is not assumed.
Source: arXiv:1904.03590v4, Algorithm 1 and §4, AMSGradW extension. -/
theorem exists_unique_equilibrium (ε wd L : ℝ) (D : Fin d → ℝ)
    (f : TrainingSpace d → ℝ) (hε : 0 < ε) (hwd : 0 < wd) (hL : 0 ≤ L)
    (hdom : L < wd * ε) (hD : ∀ i, ε ≤ D i) (hgrad : LipschitzGradient f L) :
    ∃! x : Fin d → ℝ, ∀ i, coordinateGradient f x i + wd * D i * x i = 0 := by
  have hf : ContractingWith ⟨L / (wd * ε), div_nonneg hL (mul_pos hwd hε).le⟩
      (frozenMap wd D f) :=
    ⟨(div_lt_one (mul_pos hwd hε)).mpr hdom,
      frozenMap_lipschitz ε wd L D f hε hwd hL hD hgrad⟩
  let star := hf.fixedPoint (frozenMap wd D f)
  have hfixed := hf.fixedPoint_isFixedPt
  have hequiv : ∀ x : Fin d → ℝ, Function.IsFixedPt (frozenMap wd D f) x ↔
      ∀ i, coordinateGradient f x i + wd * D i * x i = 0 := by
    intro x
    constructor
    · intro h i
      have hi := congrArg (fun y : Fin d → ℝ => y i) h
      have hd : wd * D i ≠ 0 := (mul_pos hwd (hε.trans_le (hD i))).ne'
      change -coordinateGradient f x i / (wd * D i) = x i at hi
      have hh := (div_eq_iff hd).mp hi
      nlinarith
    · intro h
      ext i
      apply (div_eq_iff (mul_pos hwd (hε.trans_le (hD i))).ne').mpr
      have hi := h i
      nlinarith
  refine ⟨star, (hequiv star).mp hfixed, fun x hx => ?_⟩
  exact hf.fixedPoint_unique' ((hequiv x).mpr hx) hfixed

/-- Strict contraction assumptions admit positive decay and a nonconstant
loss. Source: arXiv:1904.03590v4, §4, AMSGradW extension. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) < 2 ∧ (0 : ℝ) ≤ 1 ∧ (1 : ℝ) < 2 * 1 ∧
    (∀ i : Fin 1, (1 : ℝ) ≤ (fun _ => 2) i) ∧
    LipschitzGradient (Optimization.energy : TrainingSpace 1 → ℝ) 1 := by
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num,
    fun i => by norm_num, energy_lipschitz⟩

/-- Momentum contraction coefficient in the weighted state norm.
Source: arXiv:1904.03590v4, Algorithm 1, AMSGradW extension. -/
def momentumFactor (ε wd β L : ℝ) : ℝ := β + (1 - β) * (L / (wd * ε))

/-- The larger of the weight and momentum contraction factors.
Source: arXiv:1904.03590v4, §4, AMSGradW convergence extension. -/
def contractionFactor (η ε wd β L : ℝ) : ℝ :=
  max (momentumFactor ε wd β L)
    (1 - η * wd + η * wd * momentumFactor ε wd β L)

/-- The coefficient is nonnegative and strictly below one for positive
momentum below one and strict decay domination. The separate state-error
estimate additionally requires a non-overshooting decay step.
Source: arXiv:1904.03590v4, Algorithm 1 and §4, AMSGradW extension. -/
theorem contractionFactor_bounds (η ε wd β L : ℝ)
    (hη : 0 < η) (hε : 0 < ε) (hwd : 0 < wd) (hL : 0 ≤ L)
    (hβ : 0 ≤ β) (hβ' : β < 1) (hdom : L < wd * ε) :
    0 ≤ contractionFactor η ε wd β L ∧ contractionFactor η ε wd β L < 1 := by
  have hr0 : 0 ≤ L / (wd * ε) := div_nonneg hL (mul_pos hwd hε).le
  have hr : L / (wd * ε) < 1 := (div_lt_one (mul_pos hwd hε)).mpr hdom
  have hb : 0 < 1 - β := sub_pos.mpr hβ'
  have ha : 0 < η * wd := mul_pos hη hwd
  have hm0 : 0 ≤ momentumFactor ε wd β L := add_nonneg hβ (mul_nonneg hb.le hr0)
  have hm : momentumFactor ε wd β L < 1 := by
    dsimp only [momentumFactor]
    nlinarith [mul_pos hb (sub_pos.mpr hr)]
  have hx : 1 - η * wd + η * wd * momentumFactor ε wd β L < 1 := by
    nlinarith [mul_pos ha (sub_pos.mpr hm)]
  exact ⟨hm0.trans (le_max_left _ _), max_lt hm hx⟩

/-- Actual nonzero-momentum contraction parameters exist.
Source: arXiv:1904.03590v4, §4, AMSGradW extension. -/
example : (0 : ℝ) < 1 / 4 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 2 ∧ (0 : ℝ) ≤ 1 ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (1 / 4 : ℝ) * 2 ≤ 1 ∧
    (1 : ℝ) < 2 * 1 := by norm_num

end Transformer.AMSGradW
