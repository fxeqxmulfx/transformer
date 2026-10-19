module

public import Transformer.Interpolation.Basic
public import Mathlib.MeasureTheory.Integral.Bochner.Set

/-!
# Joining the integrated equations at one control switch

ArXiv:2411.04551v3, §1, `eq: cauchy.pb`, and §3.1,
the two phases of `lem: first.quadrant`. The vector primitive of a
flux can be joined to a second primitive after a positive switching
time, provided their endpoint values match. The flux may jump at
the switch; its single value there does not change its time integral.
The argument uses interval integrability and the actual primitive
identities, including for negative observation times.
-/

@[expose] public section

open Real MeasureTheory

namespace Transformer.Interpolation

/-- The first original phase before `S`, followed by the second phase
with its clock starting at zero. Source: arXiv:2411.04551v3, §1,
piecewise constant controls and the two phases of §3.1. -/
noncomputable def concatenate {E : Type*} (S : ℝ) (f g : ℝ → E) (t : ℝ) : E :=
  if t < S then f t else g (t - S)

/-- Before the original switch the first phase is retained exactly.
Source: arXiv:2411.04551v3, §1, piecewise controls in `eq: cauchy.pb`. -/
theorem concatenate_of_lt {E : Type*} (S : ℝ) (f g : ℝ → E) (t : ℝ) (ht : t < S) :
    concatenate S f g t = f t := ite_eq_left ht

/-- At the switch the second phase has its original zero-time value.
Source: arXiv:2411.04551v3, §1, piecewise controls in `eq: cauchy.pb`. -/
theorem concatenate_switch {E : Type*} (S : ℝ) (f g : ℝ → E) :
    concatenate S f g S = g 0 := by
  simp only [concatenate, lt_self_iff_false, ite_false, sub_self]

/-- The second clock measures actual elapsed time after the original
switch. Source: arXiv:2411.04551v3, §1 and §3.1, the joined phases. -/
theorem concatenate_elapsed {E : Type*} (S : ℝ) (f g : ℝ → E) (t : ℝ) (ht : 0 ≤ t) :
    concatenate S f g (t + S) = g t := by
  have hnot : ¬t + S < S := by linarith
  simp only [concatenate, ite_eq_right hnot, add_sub_cancel_right]

/-- At a positive switch time the original initial value comes from
the first phase. Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem concatenate_zero {E : Type*} (S : ℝ) (hS : 0 < S) (f g : ℝ → E) :
    concatenate S f g 0 = f 0 := by
  simp only [concatenate, ite_eq_left hS]

/-- After the switch the second original phase uses its own elapsed time.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem concatenate_of_ge {E : Type*} (S : ℝ) (f g : ℝ → E) (t : ℝ) (ht : S ≤ t) :
    concatenate S f g t = g (t - S) := by
  simp only [concatenate, ite_eq_right (not_lt_of_ge ht)]

/-- Matching endpoint values retain the first-phase value up to and
including the switch. Source: arXiv:2411.04551v3, §1,
`eq: cauchy.pb`, used at the switch of §3.1. -/
theorem concatenate_of_le {E : Type*} (S : ℝ) (f g : ℝ → E) (t : ℝ)
    (ht : t ≤ S) (hmatch : g 0 = f S) : concatenate S f g t = f t := by
  by_cases hlt : t < S
  · exact ite_eq_left hlt
  · have heq : t = S := le_antisymm ht (le_of_not_gt hlt)
    subst t
    simp only [concatenate, lt_self_iff_false, ite_false, sub_self, hmatch]

section Primitives

variable {E : Type*} [NormedAddCommGroup E]

/-- Joining two original locally integrable fluxes preserves interval
integrability. Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`. -/
theorem intervalIntegrable_concatenate (S : ℝ) (f g : ℝ → E) (a b : ℝ)
    (hf : IntervalIntegrable f volume a b)
    (hg : IntervalIntegrable g volume (a - S) (b - S)) :
    IntervalIntegrable (concatenate S f g) volume a b := by
  have hg' : IntervalIntegrable (fun x => g (x - S)) volume a b := by
    simpa only [sub_add_cancel] using hg.comp_sub_right S
  constructor
  · exact Integrable.piecewise (s := Set.Iio S) measurableSet_Iio.nullMeasurableSet
      hf.1.integrableOn hg'.1.integrableOn
  · exact Integrable.piecewise (s := Set.Iio S) measurableSet_Iio.nullMeasurableSet
      hf.2.integrableOn hg'.2.integrableOn

variable [NormedSpace ℝ E]

/-- Before the switch the actual time integral is the original first
integral. The endpoint value of the flux at `S` is immaterial.
Source: arXiv:2411.04551v3, §1 and §3.1, `eq: cauchy.pb`. -/
theorem integral_concatenate_before (S : ℝ) (hS : 0 < S) (f g : ℝ → E)
    (t : ℝ) (ht : t ≤ S) :
    (∫ s in (0 : ℝ)..t, concatenate S f g s) = ∫ s in (0 : ℝ)..t, f s := by
  have hne : ∀ᵐ x : ℝ ∂volume, x ≠ S := by
    rw [ae_iff]
    simp
  apply intervalIntegral.integral_congr_ae
  filter_upwards [hne] with x hx hxI
  have hle : x ≤ S := hxI.2.trans (max_le hS.le ht)
  exact ite_eq_left (lt_of_le_of_ne hle hx)

/-- After the switch the actual time integral uses the original second
clock starting at zero. Source: arXiv:2411.04551v3, §1 and §3.1,
`eq: cauchy.pb`. -/
theorem integral_concatenate_after (S : ℝ) (f g : ℝ → E) (t : ℝ) (ht : S ≤ t) :
    (∫ s in S..t, concatenate S f g s) = ∫ s in (0 : ℝ)..(t - S), g s := by
  calc
    _ = ∫ s in S..t, g (s - S) := by
      apply intervalIntegral.integral_congr_ae
      exact ae_of_all volume fun x hx =>
        ite_eq_right (not_lt_of_ge (le_of_lt (lt_of_eq_of_lt (min_eq_left ht).symm hx.1)))
    _ = _ := by rw [intervalIntegral.integral_comp_sub_right]; simp only [sub_self]

/-- Matching actual primitive values gives the integrated equation for
the joined curve, with no derivative at the switch assumed.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`, and §3.1,
`lem: first.quadrant`. -/
theorem concatenate_primitives (S : ℝ) (hS : 0 < S) (F₁ F₂ g₁ g₂ : ℝ → E)
    (hI₁ : ∀ a b, IntervalIntegrable g₁ volume a b)
    (hI₂ : ∀ a b, IntervalIntegrable g₂ volume a b)
    (hF₁ : ∀ t, F₁ t = F₁ 0 + ∫ s in (0 : ℝ)..t, g₁ s)
    (hF₂ : ∀ t, F₂ t = F₂ 0 + ∫ s in (0 : ℝ)..t, g₂ s)
    (hmatch : F₂ 0 = F₁ S) (t : ℝ) :
    IntervalIntegrable (concatenate S g₁ g₂) volume 0 t ∧
      concatenate S F₁ F₂ t = concatenate S F₁ F₂ 0 +
        ∫ s in (0 : ℝ)..t, concatenate S g₁ g₂ s := by
  have hi (a b : ℝ) : IntervalIntegrable (concatenate S g₁ g₂) volume a b :=
    intervalIntegrable_concatenate S g₁ g₂ a b (hI₁ a b) (hI₂ (a - S) (b - S))
  refine ⟨hi 0 t, ?_⟩
  by_cases ht : t ≤ S
  · rw [concatenate_of_le S F₁ F₂ t ht hmatch, concatenate_zero S hS,
      integral_concatenate_before S hS g₁ g₂ t ht]
    exact hF₁ t
  · have hSt : S ≤ t := (lt_of_not_ge ht).le
    rw [concatenate_of_ge S F₁ F₂ t hSt, concatenate_zero S hS,
      ← intervalIntegral.integral_add_adjacent_intervals (hi 0 S) (hi S t),
      integral_concatenate_before S hS g₁ g₂ S le_rfl,
      integral_concatenate_after S g₁ g₂ t hSt, hF₂ (t - S), hmatch, hF₁ S]
    abel

end Primitives

/-- Every primitive-joining input is realized by nonconstant primitives
with genuinely different nonzero fluxes and a positive switch time. -/
example : (0 : ℝ) < 1 ∧
    (∀ a b : ℝ, IntervalIntegrable (fun _ : ℝ => (1 : ℝ)) volume a b) ∧
    (∀ a b : ℝ, IntervalIntegrable (fun _ : ℝ => (2 : ℝ)) volume a b) ∧
    (∀ t : ℝ, t = (0 : ℝ) + ∫ _s in (0 : ℝ)..t, (1 : ℝ)) ∧
    (∀ t : ℝ, 1 + 2 * t = (1 + 2 * (0 : ℝ)) + ∫ _s in (0 : ℝ)..t, (2 : ℝ)) ∧
    1 + 2 * (0 : ℝ) = (1 : ℝ) := by
  refine ⟨one_pos, fun _ _ => intervalIntegrable_const,
    fun _ _ => intervalIntegrable_const, ?_, ?_, by norm_num⟩
  · intro t
    simp
  · intro t
    simp
    ring

/-- Constant vector velocities give matching vector primitives in an
actual two-dimensional space. These are the Bochner-valued inputs needed
for the original particle equation at a control switch. -/
example (u v : EucSpace 2) :
    (0 : ℝ) < 1 ∧
    (∀ a b : ℝ, IntervalIntegrable (fun _ : ℝ => u) volume a b) ∧
    (∀ a b : ℝ, IntervalIntegrable (fun _ : ℝ => v) volume a b) ∧
    (∀ t : ℝ, t • u = (0 : ℝ) • u + ∫ _s in (0 : ℝ)..t, u) ∧
    (∀ t : ℝ, u + t • v = u + (0 : ℝ) • v + ∫ _s in (0 : ℝ)..t, v) ∧
    u + (0 : ℝ) • v = (1 : ℝ) • u := by
  refine ⟨one_pos, fun _ _ => intervalIntegrable_const,
    fun _ _ => intervalIntegrable_const, ?_, ?_, by simp⟩
  · intro t
    simp
  · intro t
    simp

end Transformer.Interpolation
