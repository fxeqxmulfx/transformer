/-
# The number of modes of a Gaussian KDE — the field `F_n` and its upcrossings

§2.1 of arXiv:2412.09080v3, `eq:Fn`: the random function whose upcrossings of
level `0` the Kac-Rice formula is applied to, and its relation to the modes of
the KDE.

**What the source says and what is carried here.**

* `eq:Fn` states both a formula and an identity:
  `F_n(t) = n^{-1/2} Σ_i (t - X_i) e^{-β(t-X_i)²/2} = -√(2πn/β³) P̂_n'(t)`.
  `fieldF` is the first expression and `fieldF_eq` proves the identity, for
  `β > 0` and `n ≥ 1`.

* "`t` is an upcrossing of `F_n` at level `0` … is equivalent to `P̂_n'(t) = 0`
  and `P̂_n''(t) < 0`" is `isUpcrossing_fieldF_iff`, proved.

* "i.e. `t` is a mode of `P̂_n`" is the second-derivative test, and it goes one
  way only: an upcrossing is a mode, but a mode need not be an upcrossing, the
  second derivative being allowed to vanish there.  `Section2_Degenerate.lean`
  exhibits a KDE with a degenerate mode and refutes the pathwise equality; the
  identity the paper uses holds in expectation and is carried there.

Source: arXiv:2412.09080v3, `eq:Fn`, `sec:kac-rice`.
-/

import Transformer.Modes.Section2_KacRice

open scoped BigOperators NNReal ENNReal
open Real MeasureTheory Filter

namespace Transformer
namespace Modes

variable {n : ℕ}

/-! ### The derivatives of the KDE -/

/-- `HasDerivAt.sum` in the `fun`-form the goals below are in. -/
theorem hasDerivAt_fun_sum {ι : Type*} (u : Finset ι) (A : ι → ℝ → ℝ) (A' : ι → ℝ) (x : ℝ)
    (h : ∀ i ∈ u, HasDerivAt (A i) (A' i) x) :
    HasDerivAt (fun y => ∑ i ∈ u, A i y) (∑ i ∈ u, A' i) x := by
  have heq : (fun y => ∑ i ∈ u, A i y) = ∑ i ∈ u, A i :=
    funext fun y => (Finset.sum_apply y u A).symm
  rw [heq]
  exact HasDerivAt.sum h

/-- The hypothesis of `hasDerivAt_fun_sum` is satisfiable: the empty family. -/
example (x : ℝ) : ∀ i ∈ (∅ : Finset ℕ), HasDerivAt (fun _ : ℝ => (0 : ℝ)) (0 : ℝ) x :=
  fun i hi => absurd hi (Finset.notMem_empty i)

/-- The derivative of one Gaussian bump of `eq:gkde`. -/
theorem hasDerivAt_bump (β : ℝ) (c t : ℝ) :
    HasDerivAt (fun s : ℝ => Real.exp (-(β / 2) * (s - c) ^ 2))
      (-(β * (t - c)) * Real.exp (-(β / 2) * (t - c) ^ 2)) t := by
  have hbase : HasDerivAt (fun s : ℝ => s - c) 1 t := (hasDerivAt_id t).sub_const c
  have hsq : HasDerivAt (fun s : ℝ => (s - c) ^ 2) (2 * (t - c)) t := by
    have h := hbase.fun_pow 2
    rwa [show ((2 : ℕ) : ℝ) * (t - c) ^ (2 - 1) * 1 = 2 * (t - c) from by push_cast; ring] at h
  have h1 : HasDerivAt (fun s : ℝ => -(β / 2) * (s - c) ^ 2) (-(β * (t - c))) t := by
    have h := hsq.const_mul (-(β / 2))
    rwa [show -(β / 2) * (2 * (t - c)) = -(β * (t - c)) from by ring] at h
  have h := h1.exp
  rwa [show Real.exp (-(β / 2) * (t - c) ^ 2) * -(β * (t - c))
    = -(β * (t - c)) * Real.exp (-(β / 2) * (t - c) ^ 2) from by ring] at h

/-- The derivative of one term of `P̂_n'`. -/
theorem hasDerivAt_bump_deriv (β : ℝ) (c t : ℝ) :
    HasDerivAt (fun s : ℝ => -(β * (s - c)) * Real.exp (-(β / 2) * (s - c) ^ 2))
      ((β ^ 2 * (t - c) ^ 2 - β) * Real.exp (-(β / 2) * (t - c) ^ 2)) t := by
  have hbase : HasDerivAt (fun s : ℝ => s - c) 1 t := (hasDerivAt_id t).sub_const c
  have ha : HasDerivAt (fun s : ℝ => -(β * (s - c))) (-β) t := by
    have h := hbase.const_mul (-β)
    rw [show (fun s : ℝ => -β * (s - c)) = fun s : ℝ => -(β * (s - c)) from
      funext fun s => by ring] at h
    rwa [show -β * (1 : ℝ) = -β from by ring] at h
  have h := ha.mul (hasDerivAt_bump β c t)
  rwa [show -β * Real.exp (-(β / 2) * (t - c) ^ 2) + -(β * (t - c)) *
    (-(β * (t - c)) * Real.exp (-(β / 2) * (t - c) ^ 2))
      = (β ^ 2 * (t - c) ^ 2 - β) * Real.exp (-(β / 2) * (t - c) ^ 2) from by ring] at h

/-- The first derivative of `eq:gkde`:
`P̂_n'(t) = √β/(n√(2π)) Σ_i (-β(t - X_i)) e^{-β(t-X_i)²/2}`. -/
theorem hasDerivAt_kde (β : ℝ) (X : Idx n → ℝ) (t : ℝ) :
    HasDerivAt (kde β X)
      (Real.sqrt β / (n * Real.sqrt (2 * π)) *
        ∑ i : Idx n, -(β * (t - X i)) * Real.exp (-(β / 2) * (t - X i) ^ 2)) t :=
  (hasDerivAt_fun_sum _ _ _ _ fun i _ => hasDerivAt_bump β (X i) t).const_mul _

/-- The second derivative of `eq:gkde`:
`P̂_n''(t) = √β/(n√(2π)) Σ_i (β²(t - X_i)² - β) e^{-β(t-X_i)²/2}`. -/
theorem hasDerivAt_deriv_kde (β : ℝ) (X : Idx n → ℝ) (t : ℝ) :
    HasDerivAt (deriv (kde β X))
      (Real.sqrt β / (n * Real.sqrt (2 * π)) *
        ∑ i : Idx n, (β ^ 2 * (t - X i) ^ 2 - β) * Real.exp (-(β / 2) * (t - X i) ^ 2)) t := by
  rw [show deriv (kde β X) = fun s => Real.sqrt β / (n * Real.sqrt (2 * π)) *
      ∑ i : Idx n, -(β * (s - X i)) * Real.exp (-(β / 2) * (s - X i) ^ 2) from
    funext fun s => (hasDerivAt_kde β X s).deriv]
  exact (hasDerivAt_fun_sum _ _ _ _ fun i _ => hasDerivAt_bump_deriv β (X i) t).const_mul _

/-! ### `eq:Fn` -/

/-- **Equation (eq:Fn).**  The random function whose upcrossings of level `0`
are counted:

  `F_n(t) = n^{-1/2} Σ_i (t - X_i) e^{-β(t-X_i)²/2}`.

Source: arXiv:2412.09080v3, `eq:Fn`. -/
noncomputable def fieldF (β : ℝ) (X : Idx n → ℝ) (t : ℝ) : ℝ :=
  (1 / Real.sqrt n) * ∑ i : Idx n, (t - X i) * Real.exp (-(β / 2) * (t - X i) ^ 2)

/-- **The second half of `eq:Fn`:** `F_n(t) = -√(2πn/β³) P̂_n'(t)`.

Source: arXiv:2412.09080v3, `eq:Fn`. -/
theorem fieldF_eq {β : ℝ} (hβ : 0 < β) (hn : 0 < n) (X : Idx n → ℝ) (t : ℝ) :
    fieldF β X t = -Real.sqrt (2 * π * n / β ^ 3) * deriv (kde β X) t := by
  have hn' : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  have hsβ : 0 < Real.sqrt β := Real.sqrt_pos.mpr hβ
  have hsn : 0 < Real.sqrt n := Real.sqrt_pos.mpr hn'
  have hsp : 0 < Real.sqrt (2 * π) := Real.sqrt_pos.mpr (by positivity)
  have hb : Real.sqrt (β ^ 3) = β * Real.sqrt β := by
    rw [show β ^ 3 = β ^ 2 * β by ring, Real.sqrt_mul (sq_nonneg β), Real.sqrt_sq hβ.le]
  have h2 : Real.sqrt (2 * π * n / β ^ 3)
      = Real.sqrt (2 * π) * Real.sqrt n / (β * Real.sqrt β) := by
    rw [Real.sqrt_div (by positivity), Real.sqrt_mul (by positivity), hb]
  have hsum : ∑ i : Idx n, -(β * (t - X i)) * Real.exp (-(β / 2) * (t - X i) ^ 2)
      = -β * ∑ i : Idx n, (t - X i) * Real.exp (-(β / 2) * (t - X i) ^ 2) := by
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun i _ => by ring
  rw [(hasDerivAt_kde β X t).deriv, hsum, fieldF, h2]
  have e2 : Real.sqrt n ^ 2 = (n : ℝ) := Real.sq_sqrt hn'.le
  set S := ∑ i : Idx n, (t - X i) * Real.exp (-(β / 2) * (t - X i) ^ 2) with hSdef
  field_simp
  rw [e2]
  ring

/-! ### Upcrossings of `F_n` are modes of `P̂_n` -/

/-- **"`t` is an upcrossing of `F_n` at level `0` … is equivalent to
`P̂_n'(t) = 0` and `P̂_n''(t) < 0`".**

Source: arXiv:2412.09080v3, `sec:kac-rice`, after `eq:Fn`. -/
theorem isUpcrossing_fieldF_iff {β : ℝ} (hβ : 0 < β) (hn : 0 < n) (X : Idx n → ℝ) (t : ℝ) :
    IsUpcrossing (fieldF β X) 0 t ↔
      deriv (kde β X) t = 0 ∧ deriv (deriv (kde β X)) t < 0 := by
  have hn' : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  have hc : 0 < Real.sqrt (2 * π * n / β ^ 3) := Real.sqrt_pos.mpr (by positivity)
  have hfun : fieldF β X = fun s => -Real.sqrt (2 * π * n / β ^ 3) * deriv (kde β X) s :=
    funext fun s => fieldF_eq hβ hn X s
  have hderiv : deriv (fieldF β X) t
      = -Real.sqrt (2 * π * n / β ^ 3) * deriv (deriv (kde β X)) t := by
    rw [hfun, deriv_const_mul_field]
  rw [IsUpcrossing, hderiv, fieldF_eq hβ hn X t]
  constructor
  · rintro ⟨h1, h2⟩
    refine ⟨by simpa [hc.ne'] using h1, ?_⟩
    nlinarith
  · rintro ⟨h1, h2⟩
    exact ⟨by rw [h1, mul_zero], by nlinarith⟩

/-- **An upcrossing of `F_n` is a mode of `P̂_n`.**  This is the second
derivative test, and it is the direction of the source's "i.e." that always
holds.

Source: arXiv:2412.09080v3, `sec:kac-rice`, after `eq:Fn`. -/
theorem upcrossingSet_subset_modeSet {β : ℝ} (hβ : 0 < β) (hn : 0 < n) (X : Idx n → ℝ)
    (T : Set ℝ) : upcrossingSet (fieldF β X) 0 T ⊆ modeSet (kde β X) T := by
  rintro t ⟨htT, ht⟩
  refine ⟨htT, ?_⟩
  obtain ⟨h1, h2⟩ := (isUpcrossing_fieldF_iff hβ hn X t).mp ht
  exact isLocalMax_of_deriv_deriv_neg h2 h1
    ((contDiff_kde (k := 1) β X).continuous.continuousAt)

/-- Hence the number of upcrossings never exceeds the number of modes. -/
theorem upcrossingCount_le_modeCount {β : ℝ} (hβ : 0 < β) (hn : 0 < n) (X : Idx n → ℝ)
    (T : Set ℝ) : upcrossingCount (fieldF β X) 0 T ≤ modeCount (kde β X) T := by
  simpa [upcrossingCount, modeCount] using
    (ENat.toENNReal_le.mpr (Set.encard_le_encard (upcrossingSet_subset_modeSet hβ hn X T)))

/-- The hypotheses of the results above are satisfiable. -/
example : (0 : ℝ) < 1 ∧ 0 < 1 := ⟨one_pos, one_pos⟩

end Modes
end Transformer
