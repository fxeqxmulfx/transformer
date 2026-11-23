module

public import Transformer.Phasor.TripleTransport.OrbitInterpolation
public import Transformer.Phasor.TripleTransport.DenseInduction
public import Mathlib.Data.Int.GCD
public import Mathlib.Data.Nat.Prime.Basic

/-!
# Sample and coordinate requirements for the cyclic fit

Source: the Phasor experiment section “Interpolation in a learned cyclic
order”. Newton corrections select observed mismatches in any order. An
inverse table for every nonzero cyclic coordinate exists exactly at prime
orbit lengths.
-/

@[expose] public section

namespace Transformer.Phasor.TripleTransport.OrbitInterpolation.Newton
open Polynomial

variable {F : Type*} [Field F] [DecidableEq F]

/-- The last mismatching coordinate is at the front of the list.
This is the native adaptive loop in reverse chronological order. -/
noncomputable def fit (measure : F → F) : List F → F[X]
  | [] => 0
  | x :: previous =>
    let p := fit measure previous
    let basis := Lagrange.nodal previous.toFinset id
    p + C ((measure x - p.eval x) / basis.eval x) * basis

/-- Each Newton correction raises the degree at most to the number of
already chosen coordinates, including a zero correction. -/
theorem degree_step (s : Finset F) (p : F[X]) (c : F) :
    (p + C c * Lagrange.nodal s id).degree ≤ max p.degree s.card := by
  have upper : (C c * Lagrange.nodal s id).degree ≤ ↑s.card := by
    by_cases zero : c=0
    · simp [zero]
    · rw [degree_C_mul zero, Lagrange.degree_nodal]
  exact (degree_add_le ..).trans (max_le_max le_rfl upper)

/-- Distinct adaptive choices need no predetermined order. After k
corrections the fit has degree below k and agrees on all k coordinates. -/
theorem invariant (y : F → F) (choices : List F) (distinct : choices.Nodup) :
    (fit y choices).degree < choices.length ∧ ∀ x ∈ choices, (fit y choices).eval x = y x := by
  induction choices with
  | nil => simp [fit]
  | cons x xs ih =>
    obtain ⟨fresh,prior⟩ := List.nodup_cons.mp distinct
    obtain ⟨degree,agrees⟩ := ih prior
    have card := List.toFinset_card_of_nodup prior
    have next := update xs.toFinset (fit y xs) x (y x) (by simpa using fresh)
    constructor
    · have bound := degree_step xs.toFinset (fit y xs)
        ((y x - (fit y xs).eval x) / (Lagrange.nodal xs.toFinset id).eval x)
      rw [card, max_eq_right degree.le] at bound
      exact bound.trans_lt (by exact_mod_cast Nat.lt_succ_self xs.length)
    · intro i member
      rcases List.mem_cons.mp member with rfl | old
      · exact next.1
      · exact (next.2 i (List.mem_toFinset.mpr old)).trans (agrees i old)

/-- Selecting a mismatching sample always extends a history of distinct
coordinates. Thus the search may select any mismatch in native iteration order. -/
theorem mismatch_fresh (y : F → F) (choices : List F) (distinct : choices.Nodup)
    (x : F) (mismatch : (fit y choices).eval x ≠ y x) : (x::choices).Nodup :=
  List.nodup_cons.mpr ⟨fun old => mismatch ((invariant y choices distinct).2 x old), distinct⟩

/-- After degree plus one distinct corrections, agreement with the learned
anchors recovers the entire hidden polynomial. Extra validation samples can
reject a fit outside the class. -/
theorem complete (p : F[X]) (y : F → F) (choices : List F) (distinct : choices.Nodup)
    (degree : p.degree < choices.length) (exact_on : ∀ x ∈ choices, y x = p.eval x) :
    fit y choices = p := by
  have card := List.toFinset_card_of_nodup distinct
  have learned := invariant y choices distinct
  apply Polynomial.eq_of_degrees_lt_of_eval_finset_eq choices.toFinset
  · simpa [card] using learned.1
  · simpa [card] using degree
  · intro x hx
    have mem := List.mem_toFinset.mp hx
    exact (learned.2 x mem).trans (exact_on x mem)

/-- A nonempty native row fit has at most eight coefficients and strictly
more than two observations per coefficient. -/
theorem budget (samples k : ℕ) (nonempty : 0 < k) :
    k ≤ min 8 ((samples-1)/2) ↔ k ≤ 8 ∧ 2*k < samples := by omega

example : fit (fun x : ℚ => 2*x^2+3) [5,0,2] = (C 2 * X^2 + C (3 : ℚ)) := by
  apply complete
  · decide
  · apply lt_of_le_of_lt (Polynomial.degree_add_le ..)
    norm_num [degree_C_mul, degree_X_pow, degree_C]
  · simp

example : ((1 : ℚ)::[]).Nodup :=
  mismatch_fresh (fun _ => (2 : ℚ)) [] (by simp) 1 (by simp [fit])

example : 4 ≤ min 8 ((9-1)/2) := (budget 9 4 (by decide)).2 (by decide)

end Transformer.Phasor.TripleTransport.OrbitInterpolation.Newton

namespace Transformer.Phasor.TripleTransport.OrbitCapacity

/-- A complete cyclic chart assigns an inverse to every nonzero coordinate
in its orbit. The implementation accepts at least two prototypes. -/
def UnitsAvailable (n : ℕ) : Prop :=
  1 < n ∧ ∀ i, 0 < i → i < n → ∃ j < n, i*j % n = 1

/-- The inverse-table check in the native chart is equivalent to primality
of the learned orbit length. A composite prefix dictionary can skip chart
construction without inspecting any hidden partner maps. -/
theorem units_iff_prime (n : ℕ) : UnitsAvailable n ↔ Nat.Prime n := by
  constructor
  · rintro ⟨large,inverses⟩
    apply Nat.prime_def_lt.mpr
    refine ⟨by omega, ?_⟩
    intro i small divisor
    have pos : 0 < i := by
      by_contra! zero
      have z : i=0 := by omega
      subst i
      simp at divisor
      omega
    obtain ⟨j,_,inv⟩ := inverses i pos small
    have h : i ∣ i*j % n := (Nat.dvd_mod_iff divisor).2 (Nat.dvd_mul_right i j)
    rw [inv] at h
    exact Nat.dvd_one.mp h
  · intro prime
    refine ⟨prime.one_lt,fun i hi bound => ?_⟩
    exact Nat.exists_mul_mod_eq_one_of_coprime
      ((prime.coprime_iff_not_dvd.mpr (Nat.not_dvd_of_pos_of_lt hi bound)).symm) prime.one_lt

/-- A chart-only fallback cannot alter the dense read when every complete
chart requires this inverse table and its capacity is composite. The equality
holds on arbitrary sequences and learned residuals and gains. -/
theorem composite_fallback {A : Type*} (n : ℕ) (base chart : Option A)
    (no : ¬Nat.Prime n) (required : ∀ y, chart = some y → UnitsAvailable n) :
    base.or chart = base := by
  cases h : chart with
  | none => simp
  | some y => exact False.elim (no ((units_iff_prime n).1 (required y h)))

open DenseInduction

theorem composite_feature {d : ℕ} (n : ℕ) (xs residual : ℕ → Fin d → ℝ)
    (base chart : ℕ → Option (Fin d → ℝ)) (gate : ℕ → ℝ) (t : ℕ) (c : Fin d)
    (no : ¬Nat.Prime n) (required : ∀ y, chart t = some y → UnitsAvailable n) :
    feature xs residual (fun i => (base i).or (chart i)) gate t c =
      feature xs residual base gate t c := by
  simp only [feature,composite_fallback n (base t) (chart t) no required]

example : UnitsAvailable 17 := (units_iff_prime 17).2 (by decide)
example : ¬ UnitsAvailable 128 := fun h => (by decide : ¬Nat.Prime 128) ((units_iff_prime 128).1 h)

example (base : Option ℕ) :
    base.or (if Nat.Prime 128 then some 7 else none) = base := by
  simp [show ¬Nat.Prime 128 by decide]

example (stream : ℕ → Fin 2 → ℝ) : feature stream stream
    (fun t => (some (stream t)).or none) (fun _ => 1) 3 0 =
      feature stream stream (fun t => some (stream t)) (fun _ => 1) 3 0 :=
  composite_feature 128 _ _ _ (fun _ => none) _ _ _ (by decide) (by simp)

end Transformer.Phasor.TripleTransport.OrbitCapacity
