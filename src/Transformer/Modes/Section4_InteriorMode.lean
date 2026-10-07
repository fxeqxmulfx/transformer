import Transformer.Modes.Section2_Field
import Mathlib.Analysis.Calculus.Deriv.Slope
import Mathlib.Topology.Order.Compact

/-
# The number of modes of a Gaussian KDE — an interior maximum from endpoint signs

The counterexample to `lem:scale-space`, arXiv:2412.09080v3, §4.2, needs
actual local maxima of the KDE, rather than zeros of its derivative alone.
These elementary calculus lemmas produce an interior local maximum from
continuity and opposite derivative signs at the ends of a closed interval.

The extreme value theorem gives a maximum on the compact interval. At the
left endpoint, such a maximum forces a nonpositive right derivative; at
the right endpoint it forces a nonnegative left derivative. Positive and
negative derivatives, respectively, therefore exclude both endpoints.
An interior maximum on the interval is a local maximum on the real line.

For the KDE we express the same test using `F_n` from `eq:Fn`: it is a
strictly negative multiple of the derivative when `β > 0`, `n > 0`.
Thus `F_n(a) < 0 < F_n(b)` implies an actual mode in `(a,b)`. The test
does not assert that every critical point is a mode or is nondegenerate.

Source: arXiv:2412.09080v3, §2.1, `eq:Fn`; §4.2, `lem:scale-space`.
-/

open Real Filter
open scoped Topology

namespace Transformer.Modes

/-- A maximum on a closed interval has nonpositive derivative at its left
endpoint. Source: arXiv:2412.09080v3, §4.2, the counterexample to `lem:scale-space`. -/
theorem deriv_nonpos_of_isMaxOn_Icc_left {f : ℝ → ℝ} {a b d : ℝ}
    (hab : a < b) (hmax : IsMaxOn f (Set.Icc a b) a) (hd : HasDerivAt f d a) : d ≤ 0 := by
  have hlim := hd.tendsto_slope.mono_left (nhdsGT_le_nhdsNE a)
  have hineq : ∀ᶠ x in 𝓝[>] a, slope f a x ≤ 0 := by
    filter_upwards [Icc_mem_nhdsGT hab, self_mem_nhdsWithin] with x hx hx'
    rw [slope_def_field]
    exact div_nonpos_of_nonpos_of_nonneg (sub_nonpos.mpr (hmax hx))
      (sub_nonneg.mpr hx'.le)
  exact le_of_tendsto_of_tendsto hlim tendsto_const_nhds hineq

/-- A constant function satisfies all left-endpoint maximum hypotheses. -/
example : (0 : ℝ) < 1 ∧ IsMaxOn (fun _ : ℝ => (0 : ℝ)) (Set.Icc 0 1) 0 ∧
    HasDerivAt (fun _ : ℝ => (0 : ℝ)) 0 0 := by
  refine ⟨one_pos, ?_, hasDerivAt_const _ _⟩
  intro _ _
  change (0 : ℝ) ≤ 0
  exact le_rfl

/-- A maximum on a closed interval has nonnegative derivative at its right
endpoint. Source: arXiv:2412.09080v3, §4.2, the counterexample to `lem:scale-space`. -/
theorem deriv_nonneg_of_isMaxOn_Icc_right {f : ℝ → ℝ} {a b d : ℝ}
    (hab : a < b) (hmax : IsMaxOn f (Set.Icc a b) b) (hd : HasDerivAt f d b) : 0 ≤ d := by
  have hlim := hd.tendsto_slope.mono_left (nhdsLT_le_nhdsNE b)
  have hineq : ∀ᶠ x in 𝓝[<] b, 0 ≤ slope f b x := by
    filter_upwards [Icc_mem_nhdsLT hab, self_mem_nhdsWithin] with x hx hx'
    rw [slope_def_field]
    exact div_nonneg_of_nonpos (sub_nonpos.mpr (hmax hx)) (sub_nonpos.mpr hx'.le)
  exact le_of_tendsto_of_tendsto tendsto_const_nhds hlim hineq

/-- The same constant function satisfies the right-endpoint hypotheses. -/
example : (0 : ℝ) < 1 ∧ IsMaxOn (fun _ : ℝ => (0 : ℝ)) (Set.Icc 0 1) 1 ∧
    HasDerivAt (fun _ : ℝ => (0 : ℝ)) 0 1 := by
  refine ⟨one_pos, ?_, hasDerivAt_const _ _⟩
  intro _ _
  change (0 : ℝ) ≤ 0
  exact le_rfl

/-- A positive left derivative excludes the left endpoint as an interval
maximum. Source: arXiv:2412.09080v3, §4.2, counterexample to `lem:scale-space`. -/
theorem not_isMaxOn_Icc_left_of_deriv_pos {f : ℝ → ℝ} {a b d : ℝ}
    (hab : a < b) (hd : HasDerivAt f d a) (hdpos : 0 < d) :
    ¬ IsMaxOn f (Set.Icc a b) a := by
  intro hmax
  exact hdpos.not_ge (deriv_nonpos_of_isMaxOn_Icc_left hab hmax hd)

/-- The identity has positive derivative at the left endpoint of `[0,1]`. -/
example : (0 : ℝ) < 1 ∧ HasDerivAt id 1 (0 : ℝ) ∧ (0 : ℝ) < 1 :=
  ⟨one_pos, hasDerivAt_id 0, one_pos⟩

/-- A negative right derivative excludes the right endpoint as an interval
maximum. Source: arXiv:2412.09080v3, §4.2, counterexample to `lem:scale-space`. -/
theorem not_isMaxOn_Icc_right_of_deriv_neg {f : ℝ → ℝ} {a b d : ℝ}
    (hab : a < b) (hd : HasDerivAt f d b) (hdneg : d < 0) :
    ¬ IsMaxOn f (Set.Icc a b) b := by
  intro hmax
  exact hdneg.not_ge (deriv_nonneg_of_isMaxOn_Icc_right hab hmax hd)

/-- The decreasing identity has negative derivative at the right endpoint. -/
example : (0 : ℝ) < 1 ∧ HasDerivAt (fun x : ℝ => -x) (-1) 1 ∧ (-1 : ℝ) < 0 := by
  refine ⟨one_pos, ?_, by norm_num⟩
  convert (hasDerivAt_id (1 : ℝ)).neg using 1
  rfl

/-- Opposite endpoint derivative signs force an interior local maximum.
The proof uses the maximum of the function on the compact interval, not an
unproved implication from critical points to modes.
Source: arXiv:2412.09080v3, §4.2, the counterexample to `lem:scale-space`. -/
theorem exists_isLocalMax_Ioo_of_deriv_sign {f : ℝ → ℝ} {a b d₀ d₁ : ℝ}
    (hab : a < b) (hc : ContinuousOn f (Set.Icc a b))
    (hda : HasDerivAt f d₀ a) (hdb : HasDerivAt f d₁ b) (hd₀ : 0 < d₀) (hd₁ : d₁ < 0) :
    ∃ t ∈ Set.Ioo a b, IsLocalMax f t := by
  obtain ⟨t, ht, hmax⟩ := isCompact_Icc.exists_isMaxOn ⟨a, le_rfl, hab.le⟩ hc
  have hta : t ≠ a := by
    intro h
    subst t
    exact not_isMaxOn_Icc_left_of_deriv_pos hab hda hd₀ hmax
  have htb : t ≠ b := by
    intro h
    subst t
    exact not_isMaxOn_Icc_right_of_deriv_neg hab hdb hd₁ hmax
  have hat : a < t := lt_of_le_of_ne ht.1 hta.symm
  have htb' : t < b := lt_of_le_of_ne ht.2 htb
  exact ⟨t, ⟨hat, htb'⟩, hmax.isLocalMax (Icc_mem_nhds hat htb')⟩

/-- The concave parabola has the required continuity, endpoint derivatives,
and signs simultaneously; in particular the existence test is nonvacuous. -/
example : (-1 : ℝ) < 1 ∧ ContinuousOn (fun x : ℝ => -(x ^ 2)) (Set.Icc (-1) 1) ∧
    HasDerivAt (fun x : ℝ => -(x ^ 2)) 2 (-1) ∧
    HasDerivAt (fun x : ℝ => -(x ^ 2)) (-2) 1 ∧ (0 : ℝ) < 2 ∧ (-2 : ℝ) < 0 := by
  refine ⟨by norm_num, by fun_prop, ?_, ?_, by norm_num, by norm_num⟩
  · convert ((hasDerivAt_id (-1 : ℝ)).pow 2).neg using 1 <;> first | rfl | norm_num
  · convert ((hasDerivAt_id (1 : ℝ)).pow 2).neg using 1 <;> first | rfl | norm_num

/-- The field signs in `eq:Fn` force an actual mode between the endpoints.
Positive bandwidth and nonempty sample make its derivative factor nonzero.
Source: arXiv:2412.09080v3, §2.1, `eq:Fn`; §4.2, `lem:scale-space`. -/
theorem exists_isLocalMax_kde_Ioo_of_field_sign {β : ℝ} (hβ : 0 < β) {n : ℕ}
    (hn : 0 < n) (X : Fin n → ℝ) {a b : ℝ} (hab : a < b)
    (ha : fieldF β X a < 0) (hb : 0 < fieldF β X b) :
    ∃ t ∈ Set.Ioo a b, IsLocalMax (kde β X) t := by
  have hC : 0 < Real.sqrt (2 * π * n / β ^ 3) := Real.sqrt_pos.mpr (by positivity)
  have ha' : 0 < deriv (kde β X) a := by
    apply (mul_pos_iff_of_pos_left hC).mp
    nlinarith [fieldF_eq hβ hn X a]
  have hb' : deriv (kde β X) b < 0 := by
    have hneg : 0 < -deriv (kde β X) b := by
      apply (mul_pos_iff_of_pos_left hC).mp
      nlinarith [fieldF_eq hβ hn X b]
    linarith
  exact exists_isLocalMax_Ioo_of_deriv_sign hab
    (contDiff_kde (k := 0) β X).continuous.continuousOn
    (hasDerivAt_kde β X a).differentiableAt.hasDerivAt
    (hasDerivAt_kde β X b).differentiableAt.hasDerivAt ha' hb'

/-- A one-sample KDE centered at zero satisfies every field-sign hypothesis
on `[-1,1]`, with positive bandwidth and nonempty sample. -/
example : (0 : ℝ) < 1 ∧ 0 < (1 : ℕ) ∧ (-1 : ℝ) < 1 ∧
    fieldF 1 (fun _ : Fin 1 => 0) (-1) < 0 ∧ 0 < fieldF 1 (fun _ : Fin 1 => 0) 1 := by
  refine ⟨one_pos, one_pos, by norm_num, ?_, ?_⟩ <;> norm_num [fieldF] <;> positivity

end Transformer.Modes
