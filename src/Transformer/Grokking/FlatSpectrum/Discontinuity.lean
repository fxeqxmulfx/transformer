module

public import Mathlib.NumberTheory.LSeries.PrimesInAP
public import Mathlib.MeasureTheory.Measure.Lebesgue.Basic
public import Mathlib.Analysis.Normed.Group.Uniform

/-!
# Where `frac(p x y)` has a limit as `p` runs through the primes

Source: the course's home page (`~/rust_projects/grokking`, `#/en/home/full`, the table of a
task and the card *A function with discontinuities of the second kind*), added to
`review/flat_spectrum.md` as *The discontinuity of `ab`*; the note itself does not state it.
The table of `f(a, b) = a b mod p`, divided by `p` and read at `x = a/p`, `y = b/p`, is
`frac(p x y)`.

If `2 α` is an integer, `frac(p α) = frac α` for every odd `p` (`fract_mul_of_odd`), so
`frac(p α)` tends to `frac α` as `p → ∞` through the primes
(`tendsto_fract_mul_of_two_mul_eq`). For rational `α` with denominator `q` in lowest terms and
every `k < q` prime to `q`, Dirichlet's theorem on primes in arithmetic progressions gives
infinitely many primes `p` with `frac(p α) = k/q` (`infinite_setOf_prime_fract_mul_eq`). For
`q ≥ 3` both `1/q` and `(q − 1)/q` are such values, so for rational `α`, `frac(p α)` converges
through the primes exactly when `2 α` is an integer (`exists_tendsto_fract_mul_iff`). On
`[0, 1)²`, `2 x y` is an integer exactly on the axes and the hyperbola `x y = ½`
(`two_mul_mul_mem_range_intCast_iff`), and in the plane the points where `2 x y` is an integer
form a set of area zero (`volume_setOf_two_mul_mul_mem_range_intCast`).

Not formalized: for irrational `α` the values `frac(p α)` are equidistributed in `[0, 1)`
(Vinogradov), so they have no limit either. Mathlib does not have this theorem.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Filter Topology MeasureTheory

/-- If `2 α` is an integer, `frac(p α) = frac α` for every odd `p`. -/
theorem fract_mul_of_odd {α : ℝ} {m : ℤ} (h : 2 * α = m) {p : ℕ} (hp : Odd p) :
    Int.fract ((p : ℝ) * α) = Int.fract α := by
  obtain ⟨j, rfl⟩ := hp
  have hj : ((2 * j + 1 : ℕ) : ℝ) * α = ((j * m : ℤ) : ℝ) + α := by
    push_cast
    rw [← h]
    ring
  rw [hj, Int.fract_intCast_add]

example : Int.fract ((3 : ℕ) * (1 / 2 : ℝ)) = Int.fract (1 / 2 : ℝ) :=
  fract_mul_of_odd (m := 1) (by norm_num) (by decide)

/-- **The limit at a half-integer**: if `2 α` is an integer, `frac(p α) → frac α` as `p → ∞`
through the primes. -/
theorem tendsto_fract_mul_of_two_mul_eq {α : ℝ} {m : ℤ} (h : 2 * α = m) :
    Tendsto (fun p : ℕ => Int.fract ((p : ℝ) * α)) (atTop ⊓ 𝓟 {p | p.Prime})
      (𝓝 (Int.fract α)) := by
  refine tendsto_const_nhds.congr' ?_
  rw [EventuallyEq, eventually_inf_principal, eventually_atTop]
  exact ⟨3, fun p hp3 hp => (fract_mul_of_odd h (Nat.Prime.odd_of_ne_two hp (by omega))).symm⟩

example : Tendsto (fun p : ℕ => Int.fract ((p : ℝ) * (1 / 2))) (atTop ⊓ 𝓟 {p | p.Prime})
    (𝓝 (Int.fract (1 / 2 : ℝ))) :=
  tendsto_fract_mul_of_two_mul_eq (m := 1) (by norm_num)

/-- **Each value prime to the denominator, infinitely often** (Dirichlet): for rational `α`
with denominator `q` in lowest terms and every `k < q` prime to `q`, infinitely many primes `p`
have `frac(p α) = k/q`. -/
theorem infinite_setOf_prime_fract_mul_eq (α : ℚ) {k : ℕ} (hk : k < α.den)
    (hkq : k.Coprime α.den) :
    {p : ℕ | p.Prime ∧ Int.fract ((p : ℝ) * α) = k / α.den}.Infinite := by
  have : NeZero α.den := ⟨α.den_nz⟩
  have hn : IsUnit (α.num : ZMod α.den) := by
    rcases Int.natAbs_eq α.num with h | h
    · rw [h, Int.cast_natCast]
      exact (ZMod.isUnit_iff_coprime _ _).mpr α.reduced
    · rw [h, Int.cast_neg, Int.cast_natCast]
      exact ((ZMod.isUnit_iff_coprime _ _).mpr α.reduced).neg
  have hinv : IsUnit (α.num : ZMod α.den)⁻¹ := by
    obtain ⟨u, hu⟩ := hn
    rw [← hu, ZMod.inv_coe_unit]
    exact (u⁻¹).isUnit
  have hu := ((ZMod.isUnit_iff_coprime k α.den).mpr hkq).mul hinv
  refine Set.infinite_of_forall_exists_gt fun M => ?_
  obtain ⟨p, hpM, hp, hpq⟩ := Nat.forall_exists_prime_gt_and_eq_mod hu M
  refine ⟨p, ⟨hp, ?_⟩, hpM⟩
  have hmod : ((p * α.num : ℤ) : ZMod α.den) = ((k : ℤ) : ZMod α.den) := by
    push_cast
    rw [hpq, mul_assoc, ZMod.inv_mul_of_unit _ hn, mul_one]
  obtain ⟨t, ht⟩ := (ZMod.intCast_eq_intCast_iff_dvd_sub _ _ _).mp hmod
  have hq : (0 : ℝ) < α.den := by exact_mod_cast α.den_pos
  have hpα : (p : ℝ) * α = (k : ℝ) / α.den + ((-t : ℤ) : ℝ) := by
    have hr := congrArg (fun z : ℤ => (z : ℝ)) ht
    push_cast at hr
    rw [Rat.cast_def]
    field_simp
    push_cast
    linarith
  rw [hpα, Int.fract_add_intCast, Int.fract_eq_self]
  exact ⟨by positivity, (div_lt_one hq).mpr (by exact_mod_cast hk)⟩

example : {p : ℕ | p.Prime ∧ Int.fract ((p : ℝ) * ((2 : ℚ) / 5 : ℚ)) =
    (3 : ℕ) / ((2 : ℚ) / 5).den}.Infinite :=
  infinite_setOf_prime_fract_mul_eq ((2 : ℚ) / 5) (k := 3) (by norm_num) (by norm_num)

/-- **The limit exists only at half-integers, for rational `α`**: as `p → ∞` through the primes,
`frac(p α)` converges exactly when `2 α` is an integer. -/
theorem exists_tendsto_fract_mul_iff (α : ℚ) :
    (∃ L : ℝ, Tendsto (fun p : ℕ => Int.fract ((p : ℝ) * α)) (atTop ⊓ 𝓟 {p | p.Prime})
      (𝓝 L)) ↔ ∃ m : ℤ, 2 * (α : ℝ) = m := by
  refine ⟨fun ⟨L, hL⟩ => ?_, fun ⟨m, hm⟩ => ⟨_, tendsto_fract_mul_of_two_mul_eq hm⟩⟩
  by_contra hm
  have hcast : (α : ℝ) = α.num / α.den := Rat.cast_def α
  have hq : 3 ≤ α.den := by
    by_contra h3
    have hpos := α.den_pos
    interval_cases hd : α.den
    · exact hm ⟨2 * α.num, by rw [hcast]; push_cast; ring⟩
    · exact hm ⟨α.num, by rw [hcast]; push_cast; ring⟩
  have key (k : ℕ) (hk : k < α.den) (hkq : k.Coprime α.den) : L = k / α.den := by
    refine tendsto_nhds_unique_of_frequently_eq hL tendsto_const_nhds ?_
    rw [frequently_inf_principal, Nat.frequently_atTop_iff_infinite]
    exact infinite_setOf_prime_fract_mul_eq α hk hkq
  have h1 := key 1 (by omega) (Nat.coprime_one_left _)
  have h2 := key (α.den - 1) (by omega)
    ((Nat.coprime_self_sub_left (by omega)).mpr (Nat.coprime_one_left _))
  have hq0 : (α.den : ℝ) ≠ 0 := by positivity
  rw [h1, div_left_inj' hq0, Nat.cast_sub (by omega)] at h2
  have : (α.den : ℝ) = 2 := by push_cast at h2; linarith
  norm_cast at this
  omega

example : ¬ ∃ L : ℝ, Tendsto (fun p : ℕ => Int.fract ((p : ℝ) * ((1 : ℚ) / 3 : ℚ)))
    (atTop ⊓ 𝓟 {p | p.Prime}) (𝓝 L) := by
  rw [exists_tendsto_fract_mul_iff]
  rintro ⟨m, hm⟩
  push_cast at hm
  have h1 : (0 : ℝ) < m := by linarith
  have h2 : (m : ℝ) < 1 := by linarith
  norm_cast at h1 h2
  omega

/-- **On the unit square**: for `x, y ∈ [0, 1)`, `2 x y` is an integer exactly when `x = 0`,
`y = 0` or `x y = ½`. -/
theorem two_mul_mul_mem_range_intCast_iff {x y : ℝ} (hx : x ∈ Set.Ico 0 1)
    (hy : y ∈ Set.Ico 0 1) :
    2 * x * y ∈ Set.range ((↑) : ℤ → ℝ) ↔ x = 0 ∨ y = 0 ∨ x * y = 1 / 2 := by
  constructor
  · rintro ⟨m, hm⟩
    have h0 : 0 ≤ x * y := mul_nonneg hx.1 hy.1
    have h1 : x * y < 1 := by nlinarith [hx.1, hx.2, hy.1, hy.2]
    have hm0 : (0 : ℝ) ≤ m := by linarith
    have hm2 : (m : ℝ) < 2 := by linarith
    norm_cast at hm0 hm2
    obtain rfl | rfl : m = 0 ∨ m = 1 := by omega
    · push_cast at hm
      rcases mul_eq_zero.mp (show x * y = 0 by linarith) with h | h
      · exact Or.inl h
      · exact Or.inr (Or.inl h)
    · push_cast at hm
      exact Or.inr (Or.inr (by linarith))
  · rintro (rfl | rfl | h)
    · exact ⟨0, by simp⟩
    · exact ⟨0, by simp⟩
    · exact ⟨1, by rw [mul_assoc, h]; norm_num⟩

example : 2 * (3 / 4 : ℝ) * (2 / 3) ∈ Set.range ((↑) : ℤ → ℝ) :=
  (two_mul_mul_mem_range_intCast_iff (x := 3 / 4) (y := 2 / 3) ⟨by norm_num, by norm_num⟩
    ⟨by norm_num, by norm_num⟩).mpr (Or.inr (Or.inr (by norm_num)))

/-- **Area zero**: the points of the plane where `2 x y` is an integer form a null set. -/
theorem volume_setOf_two_mul_mul_mem_range_intCast :
    volume {z : ℝ × ℝ | 2 * z.1 * z.2 ∈ Set.range ((↑) : ℤ → ℝ)} = 0 := by
  have hs : MeasurableSet {z : ℝ × ℝ | 2 * z.1 * z.2 ∈ Set.range ((↑) : ℤ → ℝ)} :=
    (Real.isClosed_range_intCast.preimage (by fun_prop)).measurableSet
  rw [Measure.volume_eq_prod]
  refine Measure.measure_prod_null_of_ae_null hs ?_
  have h0 : ∀ᵐ x : ℝ, x ≠ 0 := by
    rw [ae_iff]
    simp
  filter_upwards [h0] with x hx
  refine Set.Countable.measure_zero ((Set.countable_range fun m : ℤ => (m : ℝ) / (2 * x)).mono ?_) _
  rintro y ⟨m, hm⟩
  refine ⟨m, ?_⟩
  simp only at hm
  field_simp
  linarith

end Transformer.Grokking.FlatSpectrum
