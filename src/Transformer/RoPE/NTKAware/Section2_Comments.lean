module

public import Transformer.RoPE.NTKAware.Section1_Post

/-!
# NTK-aware scaled RoPE: the comments

The thread under bloc97's post r/LocalLLaMA 14lz7j5 (2023-06-29).

* `last_frequency_adjusted`: pseudonerv: "The angle theta now have an additional factor of
  `8**(-2(i-1)/126)`. For `i=1`, the factor is `1`. for `i=d/2=64`, the factor is `1/8`." The factor
  of every pair is `YaRN.coordinate_frequency_adjusted`, that of the first
  `YaRN.highest_frequency_preserved`; the last pair's frequency is divided by `α` exactly;
* `scaled_base_slow_phase`, `scaled_base_slow_phase_gt`: Alternative_World936: "why do we need this
  `\alpha^{dim/dim-2}`, can we simply scale the base up with alpha?" With the base `base·α` the
  slowest pair's phase over the stretched context `αT` is `α^{2/d}` times its phase over `T`, and
  exceeds it for `α > 1`; `α^{d/(d−2)}` keeps it (`YaRN.slow_phase_preserved`);
* `clockReading`, `wallClock_measures`, `slowedClock_measures`: bloc97's "ELI14". A clock whose
  hands turn every `60`, `60·60` and `12·60·60` seconds tells apart the times below `43.2k` seconds
  and repeats after them. Keeping the seconds, slowing the minutes by `1.5` ("you can fit `90`
  minutes in a hour") and the hours by `2` against the new minutes ("and fit `24` hours in half a
  day") gives the periods `60`, `90·60` and `24·90·60 = 3·43200`: "a clock that can measure `129.6k`
  seconds instead of `43.2k` seconds". Slowed by `2` against the seconds, the hours would give
  `86.4k`. The factors `1.5` and `2` illustrate ("let's say"): the NTK-aware base with `α = 3`
  and `d = 6` slows three hands by `1`, `√3` and `3` (`YaRN.coordinate_frequency_adjusted`). The
  clock is "a RoPE … with a base of `60`" only up to the hour hand, which turns once per `12`
  turns of the minute hand.
-/

@[expose] public section

namespace Transformer.RoPE.NTKAware

/-- pseudonerv: the NTK-aware base divides the last pair's frequency, `i = d/2` counted from `1`, by
`α` exactly; with `α = 8` and `d = 128` the factor is `1/8`. -/
theorem last_frequency_adjusted {base α : ℝ} (hbase : 0 < base) (hα : 0 < α) {p : ℕ}
    (hp : 1 ≤ p) :
    frequency (YaRN.adjustedBase base α ((2 * (p + 1) : ℕ) : ℝ)) (2 * (p + 1)) p =
      frequency base (2 * (p + 1)) p / α := by
  rw [YaRN.coordinate_frequency_adjusted hbase hα (by omega)]
  have hp0 : (p : ℝ) ≠ 0 := by exact_mod_cast (by omega : p ≠ 0)
  rw [show ((2 * (p + 1) : ℕ) : ℝ) - 2 = 2 * p by push_cast; ring,
    div_self (mul_ne_zero two_ne_zero hp0), Real.rpow_one]

example : frequency (YaRN.adjustedBase 10000 8 ((2 * (63 + 1) : ℕ) : ℝ)) (2 * (63 + 1)) 63 =
    frequency 10000 (2 * (63 + 1)) 63 / 8 :=
  last_frequency_adjusted (by norm_num) (by norm_num) (by norm_num)

/-- Alternative_World936: with the base `base·α`, the slowest pair's phase over the stretched
context `αT` is `α^{2/d}` times its phase over `T`. -/
theorem scaled_base_slow_phase {base α d : ℝ} (hbase : 0 < base) (hα : 0 < α) (hd : d ≠ 0)
    (T : ℝ) :
    slowFrequency (base * α) d * (α * T) = α ^ (2 / d) * (slowFrequency base d * T) := by
  unfold slowFrequency
  rw [Real.mul_rpow hbase.le hα.le]
  have h : α ^ (-(d - 2) / d) * α = α ^ (2 / d) := by
    rw [← Real.rpow_add_one hα.ne']
    congr 1
    field_simp
    ring
  rw [← h]
  ring

example : slowFrequency (10000 * 8) 128 * (8 * 2048) =
    8 ^ (2 / (128 : ℝ)) * (slowFrequency 10000 128 * 2048) :=
  scaled_base_slow_phase (by norm_num) (by norm_num) (by norm_num) 2048

/-- Alternative_World936: for `α > 1` the base `base·α` takes the slowest pair beyond the phase
range it was trained on. -/
theorem scaled_base_slow_phase_gt {base α d T : ℝ} (hbase : 0 < base) (hα : 1 < α) (hd : 0 < d)
    (hT : 0 < T) :
    slowFrequency base d * T < slowFrequency (base * α) d * (α * T) := by
  rw [scaled_base_slow_phase hbase (by linarith) hd.ne']
  have h0 : 0 < slowFrequency base d * T := mul_pos (Real.rpow_pos_of_pos hbase _) hT
  have h1 : 1 < α ^ (2 / d) := Real.one_lt_rpow hα (by positivity)
  nlinarith

example : slowFrequency 10000 128 * 2048 < slowFrequency (10000 * 8) 128 * (8 * 2048) :=
  scaled_base_slow_phase_gt (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- bloc97's clock: hand `i` turns once every `P i` seconds and reads `t mod P i` at `t` seconds. -/
def clockReading {h : ℕ} (P : Fin h → ℕ) (t : ℕ) : Fin h → ℕ :=
  fun i => t % P i

/-- A clock tells apart all the times below the period of any of its hands. -/
theorem clockReading_injOn {h : ℕ} (P : Fin h → ℕ) (i : Fin h) :
    Set.InjOn (clockReading P) (Set.Iio (P i)) := by
  intro t ht u hu htu
  have hi := congr_fun htu i
  simp only [clockReading] at hi
  rwa [Nat.mod_eq_of_lt ht, Nat.mod_eq_of_lt hu] at hi

/-- A clock repeats after a time that every hand's period divides. -/
theorem clockReading_add {h : ℕ} (P : Fin h → ℕ) {T : ℕ} (hT : ∀ i, P i ∣ T) (t : ℕ) :
    clockReading P (t + T) = clockReading P t := by
  funext i
  obtain ⟨c, hc⟩ := hT i
  simp only [clockReading, hc, Nat.add_mul_mod_self_left]

example : clockReading ![60, 60 * 60] (5 + 3600) = clockReading ![60, 60 * 60] 5 :=
  clockReading_add _ (by intro i; fin_cases i <;> norm_num) 5

/-- bloc97: the twelve-hour wall clock, hands of periods `60`, `60·60` and `12·60·60` seconds,
measures `43.2k` seconds: it tells apart the times below `43200` and reads the same at `0` and
`43200`. -/
theorem wallClock_measures :
    Set.InjOn (clockReading ![60, 60 * 60, 12 * (60 * 60)]) (Set.Iio 43200) ∧
      clockReading ![60, 60 * 60, 12 * (60 * 60)] 43200 =
        clockReading ![60, 60 * 60, 12 * (60 * 60)] 0 := by
  refine ⟨clockReading_injOn _ 2, ?_⟩
  simpa using clockReading_add ![60, 60 * 60, 12 * (60 * 60)] (T := 43200)
    (by intro i; fin_cases i <;> norm_num) 0

/-- bloc97: with `90` minutes in an hour and `24` hours in half a day, the periods `60`, `90·60` and
`24·(90·60)` seconds, the clock measures `129.6k = 3 · 43.2k` seconds. -/
theorem slowedClock_measures :
    Set.InjOn (clockReading ![60, 90 * 60, 24 * (90 * 60)]) (Set.Iio 129600) ∧
      clockReading ![60, 90 * 60, 24 * (90 * 60)] 129600 =
        clockReading ![60, 90 * 60, 24 * (90 * 60)] 0 ∧ 129600 = 3 * 43200 := by
  refine ⟨clockReading_injOn _ 2, ?_, by norm_num⟩
  simpa using clockReading_add ![60, 90 * 60, 24 * (90 * 60)] (T := 129600)
    (by intro i; fin_cases i <;> norm_num) 0

end Transformer.RoPE.NTKAware
