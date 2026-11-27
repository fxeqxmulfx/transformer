module

public import Transformer.Grokking.FiniteRecovery.SeriesGrid

/-! Additive periodic forecasts. One period suffices to specify the whole
future. The observation count comes from the row-time graph of Flat Spectrum,
and the same cross gives an explicit prediction rule. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery

open Transformer.Grokking.FlatSpectrum.TablePart
open Filter
open scoped Topology

variable {R G : Type*} {period : ℕ}

/-- The phase at an absolute time, including times outside the observed period. -/
def periodIndex (period : ℕ) [NeZero period] (time : ℕ) : Fin period :=
  ⟨time % period, Nat.mod_lt _ (NeZero.pos period)⟩

@[simp] theorem periodIndex_val [NeZero period] (t : Fin period) : periodIndex period t.val = t :=
  Fin.ext (Nat.mod_eq_of_lt t.isLt)

/-- Extend one period on a finite grid to every natural time. -/
def periodicExtend [NeZero period] (f : R × Fin period → G) : R × ℕ → G :=
  fun p => f (p.1, periodIndex period p.2)

theorem periodicExtend_injective [NeZero period] :
    Function.Injective (periodicExtend : (R × Fin period → G) → R × ℕ → G) := by
  intro f g heq
  funext p
  rcases p with ⟨r, t⟩
  simpa [periodicExtend] using congrFun heq (r, t.val)

/-- The admissible series have individual offsets and one common periodic shape. -/
def additivePeriodicFamily (R G : Type*) (period : ℕ) [NeZero period] [Add G] :
    Set (R × ℕ → G) :=
  periodicExtend '' additiveSeriesFamily R (Fin period) G

/-- A fixed mask in the first period determines all future values exactly
when it determines the finite row-time table. -/
theorem determines_periodic_iff [NeZero period] [Add G] (sites : Set (R × Fin period)) :
    Determines (additivePeriodicFamily R G period) ((fun p => (p.1, p.2.val)) '' sites) ↔
      Determines (additiveSeriesFamily R (Fin period) G) sites := by
  constructor
  · intro h f hf g hg hagree
    apply periodicExtend_injective
    refine h _ (Set.mem_image_of_mem _ hf) _ (Set.mem_image_of_mem _ hg) ?_
    rintro _ ⟨⟨r, t⟩, hmem, rfl⟩
    simpa [periodicExtend] using hagree (r, t) hmem
  · intro h
    rintro _ ⟨f, hf, rfl⟩ _ ⟨g, hg, rfl⟩ hagree
    apply congrArg periodicExtend
    refine h _ hf _ hg fun p hp => ?_
    have heq := hagree (p.1, p.2.val) (Set.mem_image_of_mem _ hp)
    simpa [periodicExtend] using heq

/-- For period length P and R nonempty series the exact minimum is
R + P - 1, with no bound on the forecast horizon. -/
theorem isLeast_periodic_observations [NeZero period] [Fintype R] [Nonempty R]
    (G : Type*) [AddCommGroup G] [Nontrivial G] :
    IsLeast {n : ℕ | ∃ sites : Finset (R × Fin period), sites.card = n ∧
      Determines (additivePeriodicFamily R G period)
        ((fun p : R × Fin period => (p.1, p.2.val)) '' (sites : Set (R × Fin period)))}
      (Fintype.card R + period - 1) := by
  simpa only [determines_periodic_iff, Fintype.card_fin] using
    isLeast_additiveSeries_observations (R := R) (T := Fin period) G

/-- Copy the observed level at time zero for this series and add the
offset of the current phase measured on the reference series. -/
def periodicForecast [NeZero period] [Add G] [Sub G] (initial : R → G) (reference : Fin period → G)
    (row : R) (time : ℕ) : G :=
  initial row + reference (periodIndex period time) - reference (periodIndex period 0)

theorem periodicForecast_eq [NeZero period] [AddCommGroup G] (offset : R → G) (shape : Fin period → G)
    (referenceRow : R) (row : R) (time : ℕ) :
    periodicForecast (fun r => offset r + shape (periodIndex period 0))
      (fun t => offset referenceRow + shape t) row time =
      offset row + shape (periodIndex period time) := by
  unfold periodicForecast
  apply sub_eq_iff_eq_add.mpr
  dsimp only
  ac_rfl

/-- A uniform error bound on the cross gives a uniform bound at every
forecast horizon. The prediction uses three observed entries. -/
theorem periodicForecast_error_le [NeZero period] (initial initial' : R → ℝ)
    (reference reference' : Fin period → ℝ) (ε : ℝ)
    (hrow : ∀ r, |initial' r - initial r| ≤ ε) (htime : ∀ t, |reference' t - reference t| ≤ ε)
    (r : R) (time : ℕ) :
    |periodicForecast initial' reference' r time - periodicForecast initial reference r time| ≤
      3 * ε := by
  unfold periodicForecast
  have e : initial' r + reference' (periodIndex period time) - reference' (periodIndex period 0) -
      (initial r + reference (periodIndex period time) - reference (periodIndex period 0)) =
      (initial' r - initial r) + (reference' (periodIndex period time) - reference (periodIndex period time)) -
        (reference' (periodIndex period 0) - reference (periodIndex period 0)) := by ring
  rw [e]
  have h₁ := abs_sub_le
    ((initial' r - initial r) + (reference' (periodIndex period time) - reference (periodIndex period time)))
    0 (reference' (periodIndex period 0) - reference (periodIndex period 0))
  have h₂ := abs_add_le (initial' r - initial r)
    (reference' (periodIndex period time) - reference (periodIndex period time))
  simp only [sub_zero, zero_sub, abs_neg] at h₁
  linarith [hrow r, htime (periodIndex period time), htime (periodIndex period 0)]

/-- When the estimates of the cross converge, forecasts converge at
every time by the same reconstruction formula. -/
theorem periodicForecast_tendsto [NeZero period] (initial : R → ℝ) (reference : Fin period → ℝ)
    (initial' : ℕ → R → ℝ) (reference' : ℕ → Fin period → ℝ)
    (hrow : ∀ r, Tendsto (fun k => initial' k r) atTop (𝓝 (initial r)))
    (htime : ∀ t, Tendsto (fun k => reference' k t) atTop (𝓝 (reference t)))
    (r : R) (time : ℕ) :
    Tendsto (fun k => periodicForecast (initial' k) (reference' k) r time)
      atTop (𝓝 (periodicForecast initial reference r time)) :=
  ((hrow r).add (htime (periodIndex period time))).sub (htime (periodIndex period 0))

example :
    IsLeast {n : ℕ | ∃ sites : Finset (Fin 1024 × Fin 4096), sites.card = n ∧
        Determines (additivePeriodicFamily (Fin 1024) ℤ 4096)
          ((fun p : Fin 1024 × Fin 4096 => (p.1, p.2.val)) '' (sites : Set (Fin 1024 × Fin 4096)))} 5119 := by
  simpa using isLeast_periodic_observations (R := Fin 1024) (period := 4096) ℤ

example : Fintype.card (Fin 1024 × Fin 4096) = 4194304 := by
  simp [Fintype.card_prod]

example (a : Fin 1024 → ℤ) (b : Fin 4096 → ℤ) (r : Fin 1024) :=
  periodicForecast_eq a b 0 r 1000000

example := periodicForecast_error_le (R := Fin 2) (period := 4)
  (fun _ => 0) (fun _ => 1) (fun _ => 0) (fun _ => 1) 1
  (by intro r; norm_num) (by intro t; norm_num) 0 1000000

example (initial : Fin 2 → ℝ) (reference : Fin 4 → ℝ) :=
  periodicForecast_tendsto initial reference (fun _ => initial) (fun _ => reference)
    (fun _ => tendsto_const_nhds) (fun _ => tendsto_const_nhds) 0 1000000

end Transformer.Grokking.FiniteRecovery
