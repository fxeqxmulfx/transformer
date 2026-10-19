module

public import Transformer.Metastability.Section4_UniformProjectionTail
public import Mathlib.Algebra.Order.Archimedean.Real.Basic

/-!
# The original radius and dimension rate for uniform projection

ArXiv:2410.06833v1, §4, `prop: concentration unif`, and
arXiv:2312.10794v5, Appendix D, Step 2, almost orthogonality.
The actual projection radius is `sqrt(log(d)/d)` and the actual moment
half-degree is the natural floor of `log(d)/16`. Above `log(d)=32`,
this degree lies between `log(d)/32` and `log(d)/16`, exactly the
range needed by the finite-direction estimate at half the radius.

An explicit integer dimension threshold, depending only on the
maximum frame length, absorbs its finite net budget. Every actual
previous orthonormal frame then has projection failure probability
at most `d^(-1/64)` at the original radius, uniformly in its values
and in the original invariant probability law. The radius is below
one, so a successful projection has a genuine nonzero residual.
The final independent-sample union bound remains a separate step.
-/

@[expose] public section

open Real MeasureTheory

namespace Transformer.Metastability

/-- The actual projection radius of the original construction.
Source: arXiv:2410.06833v1, §4, `prop: concentration unif`;
arXiv:2312.10794v5, Appendix D, Step 2, the projection threshold. -/
noncomputable def uniformProjectionRadius (d : ℕ) : ℝ := Real.sqrt (Real.log d / (d : ℝ))

/-- The actual integer half-degree used to prove the original tail rate.
Source: arXiv:2410.06833v1, §4, the projection concentration input. -/
noncomputable def uniformProjectionDegree (d : ℕ) : ℕ := ⌊Real.log (d : ℝ) / 16⌋₊

/-- The actual radius is positive in every dimension above one.
Source: arXiv:2410.06833v1, §4, the original normalization scale. -/
theorem uniformProjectionRadius_pos {d : ℕ} (hd : 1 < d) : 0 < uniformProjectionRadius d := by
  have hdreal : (1 : ℝ) < d := by exact_mod_cast hd
  exact Real.sqrt_pos.mpr (div_pos (Real.log_pos hdreal) (by linarith))

/-- The actual squared radius equals the original logarithmic ratio.
Source: arXiv:2410.06833v1, §4, the original projection threshold. -/
theorem uniformProjectionRadius_sq {d : ℕ} (hd : 1 ≤ d) :
    uniformProjectionRadius d ^ 2 = Real.log d / (d : ℝ) := by
  unfold uniformProjectionRadius
  exact Real.sq_sqrt (div_nonneg (Real.log_nonneg (by exact_mod_cast hd)) (by positivity))

/-- The original projection radius is strictly below one.
Source: arXiv:2410.06833v1, §4, the nonzero residual in the actual
inductive construction; arXiv:2312.10794v5, Appendix D, Step 2. -/
theorem uniformProjectionRadius_lt_one {d : ℕ} (hd : 1 < d) : uniformProjectionRadius d < 1 := by
  have hdreal : (1 : ℝ) < d := by exact_mod_cast hd
  have hdpos : (0 : ℝ) < d := by linarith
  have hlog := Real.log_le_sub_one_of_pos hdpos
  have hratio : Real.log (d : ℝ) / (d : ℝ) < 1 := by
    apply (div_lt_iff₀ hdpos).mpr
    linarith
  apply (Real.sqrt_lt (div_nonneg (Real.log_nonneg hdreal.le) hdpos.le) zero_le_one).mpr
  simpa only [one_pow] using hratio

/-- Twice the actual projection radius is exactly the distance radius
in the original statement. Source: arXiv:2410.06833v1, §4,
`eq: upto-t`, including its original factor four under the square root. -/
theorem two_uniformProjectionRadius (d : ℕ) :
    2 * uniformProjectionRadius d = Real.sqrt (4 * Real.log d / (d : ℝ)) := by
  have h : 4 * Real.log d / (d : ℝ) = (2 : ℝ) ^ 2 * (Real.log d / (d : ℝ)) := by ring
  rw [h, Real.sqrt_mul (by positivity), Real.sqrt_sq (by norm_num)]
  rfl

/-- The actual logarithmic half-degree fits the original squared
threshold, and is large enough for the original dimension power.
Source: arXiv:2410.06833v1, §4, the actual moment estimate proving
the projection concentration input for `prop: concentration unif`. -/
theorem uniformProjectionDegree_bounds {d : ℕ} (hd : 1 ≤ d)
    (hlog : 32 ≤ Real.log (d : ℝ)) :
    4 * (uniformProjectionDegree d : ℝ) ≤ (d : ℝ) * (uniformProjectionRadius d / 2) ^ 2 ∧
      Real.log (d : ℝ) / 32 ≤ uniformProjectionDegree d := by
  have hdpos : (0 : ℝ) < d := by exact_mod_cast (show 0 < d by omega)
  have hupper := Nat.floor_le (by linarith : 0 ≤ Real.log (d : ℝ) / 16)
  have hlower := Nat.sub_one_lt_floor (Real.log (d : ℝ) / 16)
  change (uniformProjectionDegree d : ℝ) ≤ Real.log (d : ℝ) / 16 at hupper
  change Real.log (d : ℝ) / 16 - 1 < (uniformProjectionDegree d : ℝ) at hlower
  have hsq : (d : ℝ) * (uniformProjectionRadius d / 2) ^ 2 = Real.log (d : ℝ) / 4 := by
    rw [div_pow, uniformProjectionRadius_sq hd]
    field_simp
    norm_num
  constructor <;> linarith [hsq]

/-- An actual fixed finite factor is absorbed into the original
dimension rate. Source: arXiv:2410.06833v1, §4, the dimension threshold
depending only on `n` in `prop: concentration unif`. -/
theorem projection_budget_absorbed (c : ℝ) (hc : 0 ≤ c) {d : ℕ}
    (hd : 0 < d) (hcd : c ^ 64 ≤ (d : ℝ)) :
    c * (d : ℝ) ^ (-(1 : ℝ) / 32) ≤ (d : ℝ) ^ (-(1 : ℝ) / 64) := by
  have hdpos : (0 : ℝ) < d := by exact_mod_cast hd
  have hroot : (c ^ 64) ^ ((1 : ℝ) / 64) = c := by
    rw [← Real.rpow_natCast c 64, ← Real.rpow_mul hc]
    norm_num
  have hcroot := Real.rpow_le_rpow (pow_nonneg hc 64) hcd (by norm_num : (0 : ℝ) ≤ 1 / 64)
  rw [hroot] at hcroot
  calc
    _ ≤ (d : ℝ) ^ ((1 : ℝ) / 64) * (d : ℝ) ^ (-(1 : ℝ) / 32) :=
      mul_le_mul_of_nonneg_right hcroot (Real.rpow_nonneg hdpos.le _)
    _ = _ := by rw [← Real.rpow_add hdpos]; congr 1; norm_num

/-- A concrete integer dimension threshold for all previous frame
lengths at most `n`. Source: arXiv:2410.06833v1, §4, the threshold
`d⋆(n)` in the original concentration statement. No optimal size is claimed. -/
noncomputable def uniformProjectionThreshold (n : ℕ) : ℕ :=
  max (n + 3) (max ⌈Real.exp 32⌉₊ (sphericalNetBudget n ^ 64 + 1))

/-- The actual threshold exceeds the original sample size, gives the
original logarithmic degree scale, and absorbs the actual finite net.
Source: arXiv:2410.06833v1, §4, the dimension choice in the original
projection concentration estimate. -/
theorem uniformProjectionThreshold_spec (n : ℕ) :
    n < uniformProjectionThreshold n ∧ ∀ d : ℕ, uniformProjectionThreshold n ≤ d →
      1 < d ∧ 32 ≤ Real.log (d : ℝ) ∧ (sphericalNetBudget n : ℝ) ^ 64 ≤ (d : ℝ) := by
  have hn := le_max_left (n + 3) (max ⌈Real.exp 32⌉₊ (sphericalNetBudget n ^ 64 + 1))
  refine ⟨by unfold uniformProjectionThreshold; omega, fun d hd => ?_⟩
  have hnd : n + 3 ≤ d := hn.trans hd
  have hceil : ⌈Real.exp 32⌉₊ ≤ d :=
    (le_max_left _ _).trans ((le_max_right _ _).trans hd)
  have hbudget : sphericalNetBudget n ^ 64 + 1 ≤ d :=
    (le_max_right _ _).trans ((le_max_right _ _).trans hd)
  have hexp : Real.exp 32 ≤ (d : ℝ) :=
    (Nat.le_ceil _).trans (by exact_mod_cast hceil)
  have hlog := Real.log_le_log (Real.exp_pos 32) hexp
  rw [Real.log_exp] at hlog
  exact ⟨by omega, hlog, by exact_mod_cast (show sphericalNetBudget n ^ 64 ≤ d by omega)⟩

/-- The original projection failure probability is at most the original
dimension rate, uniformly over every earlier orthonormal frame and
every original uniform law. Source: arXiv:2410.06833v1, §4,
`prop: concentration unif`; arXiv:2312.10794v5, Appendix D, Step 2. -/
theorem uniform_frameProjection_tail_original_scale {d m n : ℕ}
    (hd : uniformProjectionThreshold n ≤ d) (hmn : m ≤ n)
    (ν : Measure (SSphere d)) (hν : IsUniformOn d ν)
    (v : Fin m → EucSpace d) (hv : Orthonormal ℝ v) :
    ν.real {x : SSphere d | uniformProjectionRadius d ≤ ‖frameProjection v (x : EucSpace d)‖} ≤
      (d : ℝ) ^ (-(1 : ℝ) / 64) := by
  obtain ⟨hdone, hlog, hbudget⟩ := (uniformProjectionThreshold_spec n).2 d hd
  obtain ⟨hdegree, hdegreeLog⟩ := uniformProjectionDegree_bounds hdone.le hlog
  apply (uniform_frameProjection_tail_le_budget (by omega) hmn ν hν v hv
    (uniformProjectionRadius d) (uniformProjectionRadius_pos hdone)
    (uniformProjectionDegree d) hdegree hdegreeLog).trans
  exact projection_budget_absorbed (sphericalNetBudget n) (by positivity) (by omega) hbudget

/-- The actual threshold in the two-point initialization problem gives
a positive dimension with its actual uniform law, an allowable frame
length, both original radius conditions, and both scale conditions. -/
example : ∃ d : ℕ, ∃ v : Fin 1 → EucSpace d,
    uniformProjectionThreshold 2 ≤ d ∧ (1 : ℕ) ≤ 2 ∧
    IsUniformOn d (MeanField.uniformLaw d) ∧ Orthonormal ℝ v ∧ 1 < d ∧
    0 < uniformProjectionRadius d ∧ uniformProjectionRadius d < 1 ∧
    32 ≤ Real.log (d : ℝ) ∧ (sphericalNetBudget 2 : ℝ) ^ 64 ≤ (d : ℝ) := by
  let d := uniformProjectionThreshold 2
  obtain ⟨hdone, hlog, hbudget⟩ := (uniformProjectionThreshold_spec 2).2 d le_rfl
  let e : Fin d := ⟨0, by omega⟩
  let v : Fin 1 → EucSpace d := fun _ => EuclideanSpace.single e 1
  refine ⟨d, v, le_rfl, by decide, isUniformOn_uniformLaw (by omega), ?_, hdone,
    uniformProjectionRadius_pos hdone, uniformProjectionRadius_lt_one hdone, hlog, hbudget⟩
  rw [orthonormal_iff_ite]
  intro i j
  have hij : i = j := Subsingleton.elim _ _
  simp [v, hij]

end Transformer.Metastability
