module

public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds
public import Mathlib.Combinatorics.Pigeonhole

/-!
# A RoPE score of the distance comes back near its value

In no paper. Rotary embeddings, as in arXiv:2309.00071v3, §2.1, Eq. (5), rotate the token at the
position `m` by `e^{i m θ_i}` in the plane `i`, so that the score of a query and a key depends on
their distance only. A head whose query and key do not depend on the tokens, a positional head,
scores the token at the distance `t` by the trigonometric polynomial

    ropeScore θ a b t = ∑_i (a_i cos (θ_i t) + b_i sin (θ_i t))

(`ropeScore`), with `K` frequencies and the coefficients set by its query and key
(`re_sum_rotary_eq_ropeScore`).

Such a score is almost periodic. A distance `d` that brings every phase `θ_i d` within `η` of a
multiple of `2π` moves the score by at most `η ∑_i (|a_i| + |b_i|)`, forward and backward
(`abs_ropeScore_add_sub_le`, `abs_ropeScore_sub_sub_le`). Cut the torus of the phases into
`(N + 1)^K` boxes (`phaseBox`): some box holds at least `H / (N + 1)^K` of the positions
`0, …, H − 1`, and their distances to the first of them are returns within `2π / (N + 1)`
(`exists_returns`). This is Dirichlet's pigeonhole argument for simultaneous approximation; the
number of returns grows linearly in `H` for every fixed set of frequencies.
-/

@[expose] public section

namespace Transformer.MRCR

open Finset

variable {K : ℕ}

/-- **A RoPE score of the distance** (arXiv:2309.00071v3, §2.1, Eq. (5)): a positional head with
rotary embeddings of the frequencies `θ i` scores the token at the distance `t` by
`∑_i (a_i cos (θ_i t) + b_i sin (θ_i t))`, the coefficients set by its query and key. -/
noncomputable def ropeScore (θ a b : Fin K → ℝ) (t : ℝ) : ℝ :=
  ∑ i, (a i * Real.cos (θ i * t) + b i * Real.sin (θ i * t))

/-- The size `∑_i (|a_i| + |b_i|)` of the coefficients of a RoPE score. -/
noncomputable def ropeNorm (a b : Fin K → ℝ) : ℝ := ∑ i, (|a i| + |b i|)

/-- **Rotary embeddings score by the distance** (arXiv:2309.00071v3, §2.1, Eqs. (3)–(6)): the
query `u` at the position `m` and the key `v` at the position `n`, rotated by `e^{i m θ}` and
`e^{i n θ}`, have the score `Re (q* k) = ropeScore θ a b (m − n)`, `a_i + i b_i = conj(u_i) v_i`. -/
theorem re_sum_rotary_eq_ropeScore (θ : Fin K → ℝ) (u v : Fin K → ℂ) (m n : ℝ) :
    (∑ i, starRingEnd ℂ (Complex.exp ((m * θ i : ℝ) * Complex.I) * u i) *
      (Complex.exp ((n * θ i : ℝ) * Complex.I) * v i)).re =
      ropeScore θ (fun i => (starRingEnd ℂ (u i) * v i).re)
        (fun i => (starRingEnd ℂ (u i) * v i).im) (m - n) := by
  simp only [ropeScore, Complex.re_sum]
  refine sum_congr rfl fun i _ => ?_
  rw [Complex.exp_mul_I, Complex.exp_mul_I, ← Complex.ofReal_cos, ← Complex.ofReal_sin,
    ← Complex.ofReal_cos, ← Complex.ofReal_sin]
  simp only [map_mul, map_add, Complex.conj_ofReal, Complex.conj_I, Complex.mul_re,
    Complex.mul_im, Complex.add_re, Complex.add_im, Complex.ofReal_re, Complex.ofReal_im,
    Complex.I_re, Complex.I_im, Complex.conj_re, Complex.conj_im, Complex.neg_re, Complex.neg_im]
  rw [show θ i * (m - n) = m * θ i - n * θ i by ring, Real.cos_sub, Real.sin_sub]
  ring

/-- **A near period moves the score little**: if every phase `θ_i d` is within `η` of a multiple
of `2π`, the score moves by at most `η ∑_i (|a_i| + |b_i|)` from `t` to `t + d`. -/
theorem abs_ropeScore_add_sub_le {θ a b : Fin K → ℝ} {d η : ℝ}
    (hd : ∀ i, ∃ m : ℤ, |θ i * d - m * (2 * Real.pi)| ≤ η) (t : ℝ) :
    |ropeScore θ a b (t + d) - ropeScore θ a b t| ≤ η * ropeNorm a b := by
  simp only [ropeScore, ropeNorm]
  rw [← sum_sub_distrib, mul_sum]
  refine (abs_sum_le_sum_abs _ _).trans (sum_le_sum fun i _ => ?_)
  obtain ⟨m, hm⟩ := hd i
  rw [show θ i * (t + d) = θ i * t + (θ i * d - m * (2 * Real.pi)) + m * (2 * Real.pi) by ring,
    Real.cos_add_int_mul_two_pi, Real.sin_add_int_mul_two_pi]
  generalize θ i * d - m * (2 * Real.pi) = ρ at hm ⊢
  generalize θ i * t = x
  have h1 := Real.abs_cos_sub_cos_le (x + ρ) x
  have h2 := Real.abs_sin_sub_sin_le (x + ρ) x
  rw [add_sub_cancel_left] at h1 h2
  rw [show a i * Real.cos (x + ρ) + b i * Real.sin (x + ρ) - (a i * Real.cos x + b i * Real.sin x) =
      a i * (Real.cos (x + ρ) - Real.cos x) + b i * (Real.sin (x + ρ) - Real.sin x) by ring]
  refine (abs_add_le _ _).trans ?_
  rw [abs_mul, abs_mul, mul_add, mul_comm η, mul_comm η]
  exact add_le_add (mul_le_mul_of_nonneg_left (h1.trans hm) (abs_nonneg _))
    (mul_le_mul_of_nonneg_left (h2.trans hm) (abs_nonneg _))

/-- One frequency `2π`: the distance `1` is an exact period. -/
example := abs_ropeScore_add_sub_le (θ := fun _ : Fin 1 => 2 * Real.pi) (a := fun _ => 1)
  (b := fun _ => 0) (d := 1) (η := 0) (fun _ => ⟨1, by simp⟩) 0

/-- The same backward: from `t` to `t − d`. -/
theorem abs_ropeScore_sub_sub_le {θ a b : Fin K → ℝ} {d η : ℝ}
    (hd : ∀ i, ∃ m : ℤ, |θ i * d - m * (2 * Real.pi)| ≤ η) (t : ℝ) :
    |ropeScore θ a b (t - d) - ropeScore θ a b t| ≤ η * ropeNorm a b := by
  rw [sub_eq_add_neg]
  refine abs_ropeScore_add_sub_le (fun i => ?_) t
  obtain ⟨m, hm⟩ := hd i
  refine ⟨-m, ?_⟩
  rw [show θ i * -d - ((-m : ℤ) : ℝ) * (2 * Real.pi) = -(θ i * d - m * (2 * Real.pi)) by
    push_cast; ring, abs_neg]
  exact hm

/-- One frequency `2π`, backward. -/
example := abs_ropeScore_sub_sub_le (θ := fun _ : Fin 1 => 2 * Real.pi) (a := fun _ => 1)
  (b := fun _ => 0) (d := 1) (η := 0) (fun _ => ⟨1, by simp⟩) 0

/-- The box of the position `x` among `(N + 1)^K`: each phase `x θ_i / 2π`, read modulo `1`, falls
into one of `N + 1` equal parts. -/
noncomputable def phaseBox (N : ℕ) (θ : Fin K → ℝ) (x : ℕ) : Fin K → Fin (N + 1) := fun i =>
  ⟨⌊((N : ℝ) + 1) * Int.fract (x * θ i / (2 * Real.pi))⌋₊, by
    rw [Nat.floor_lt (mul_nonneg (by positivity) (Int.fract_nonneg _))]
    have := mul_lt_mul_of_pos_left (Int.fract_lt_one (x * θ i / (2 * Real.pi)))
      (by positivity : (0 : ℝ) < (N : ℝ) + 1)
    push_cast
    linarith⟩

/-- Two positions in one box have phases within `1 / (N + 1)` of each other, modulo `1`. -/
lemma abs_fract_sub_lt_of_phaseBox_eq {N : ℕ} {θ : Fin K → ℝ} {x y : ℕ}
    (h : phaseBox N θ x = phaseBox N θ y) (i : Fin K) :
    |Int.fract (x * θ i / (2 * Real.pi)) - Int.fract (y * θ i / (2 * Real.pi))| <
      1 / ((N : ℝ) + 1) := by
  have hi : ⌊((N : ℝ) + 1) * Int.fract (x * θ i / (2 * Real.pi))⌋₊ =
      ⌊((N : ℝ) + 1) * Int.fract (y * θ i / (2 * Real.pi))⌋₊ := congrArg Fin.val (congrFun h i)
  have hN : (0 : ℝ) < (N : ℝ) + 1 := by positivity
  have hx := Nat.floor_le (mul_nonneg hN.le (Int.fract_nonneg (x * θ i / (2 * Real.pi))))
  have hx' := Nat.lt_floor_add_one (((N : ℝ) + 1) * Int.fract (x * θ i / (2 * Real.pi)))
  have hy := Nat.floor_le (mul_nonneg hN.le (Int.fract_nonneg (y * θ i / (2 * Real.pi))))
  have hy' := Nat.lt_floor_add_one (((N : ℝ) + 1) * Int.fract (y * θ i / (2 * Real.pi)))
  rw [hi] at hx hx'
  rw [abs_sub_lt_iff, lt_div_iff₀ hN, lt_div_iff₀ hN]
  constructor <;> linarith

/-- One position against itself. -/
example := abs_fract_sub_lt_of_phaseBox_eq (N := 3) (θ := fun _ : Fin 2 => 1) (x := 5) (y := 5)
  rfl 0

/-- **Returns of the phases** (Dirichlet's pigeonhole): among the distances `1, …, H − 1` at
least `H / (N + 1)^K − 1` bring every phase `θ_i d` within `2π / (N + 1)` of a multiple of
`2π`. -/
theorem exists_returns (θ : Fin K → ℝ) (N : ℕ) {H : ℕ} (hH : 0 < H) :
    ∃ R : Finset ℕ, (H : ℝ) / ((N : ℝ) + 1) ^ K - 1 ≤ R.card ∧ ∀ d ∈ R, 1 ≤ d ∧ d < H ∧
      ∀ i, ∃ m : ℤ, |θ i * d - m * (2 * Real.pi)| ≤ 2 * Real.pi / ((N : ℝ) + 1) := by
  have hP : (0 : ℝ) < ((N : ℝ) + 1) ^ K := by positivity
  obtain ⟨β, hβ⟩ := Fintype.exists_le_card_fiber_of_nsmul_le_card
    (fun x : Fin H => phaseBox N θ x) (b := (H : ℝ) / ((N : ℝ) + 1) ^ K) (by
      rw [nsmul_eq_mul, Fintype.card_fun, Fintype.card_fin, Fintype.card_fin, Fintype.card_fin]
      push_cast
      rw [mul_div_cancel₀ _ hP.ne'])
  set F := univ.filter fun x : Fin H => phaseBox N θ x = β
  have hF : F.Nonempty := by
    rw [← card_pos]
    have : (0 : ℝ) < F.card := lt_of_lt_of_le (by positivity) hβ
    exact_mod_cast this
  set x₀ := F.min' hF
  have hx₀ : x₀ ∈ F := min'_mem F hF
  refine ⟨(F.erase x₀).image fun x => x.val - x₀.val, ?_, fun d hd => ?_⟩
  · rw [card_image_of_injOn, card_erase_of_mem hx₀, Nat.cast_sub (card_pos.2 hF)]
    · push_cast
      linarith
    · intro x hx y hy hxy
      have hx1 : x₀ ≤ x := min'_le F x (mem_of_mem_erase hx)
      have hy1 : x₀ ≤ y := min'_le F y (mem_of_mem_erase hy)
      have h : x.val - x₀.val = y.val - x₀.val := hxy
      rw [Fin.le_def] at hx1 hy1
      exact Fin.ext (by omega)
  obtain ⟨x, hx, rfl⟩ := mem_image.1 hd
  have hle : x₀ ≤ x := min'_le F x (mem_of_mem_erase hx)
  have hlt : x₀.val < x.val :=
    lt_of_le_of_ne (Fin.le_def.1 hle) fun h => ne_of_mem_erase hx (Fin.ext h).symm
  refine ⟨by omega, by have := x.isLt; omega, fun i => ?_⟩
  have hbox : phaseBox N θ x = phaseBox N θ x₀ :=
    (mem_filter.1 (mem_of_mem_erase hx)).2.trans (mem_filter.1 hx₀).2.symm
  have hclose := abs_fract_sub_lt_of_phaseBox_eq hbox i
  refine ⟨⌊(x.val : ℝ) * θ i / (2 * Real.pi)⌋ - ⌊(x₀.val : ℝ) * θ i / (2 * Real.pi)⌋, ?_⟩
  have hpi : (0 : ℝ) < 2 * Real.pi := by positivity
  have e1 := Int.floor_add_fract ((x.val : ℝ) * θ i / (2 * Real.pi))
  have e2 := Int.floor_add_fract ((x₀.val : ℝ) * θ i / (2 * Real.pi))
  have h1 : 2 * Real.pi * ((x.val : ℝ) * θ i / (2 * Real.pi)) = x.val * θ i := by field_simp
  have h2 : 2 * Real.pi * ((x₀.val : ℝ) * θ i / (2 * Real.pi)) = x₀.val * θ i := by field_simp
  rw [Nat.cast_sub hlt.le, show θ i * ((x.val : ℝ) - x₀.val) -
      ((⌊(x.val : ℝ) * θ i / (2 * Real.pi)⌋ - ⌊(x₀.val : ℝ) * θ i / (2 * Real.pi)⌋ : ℤ) : ℝ) *
        (2 * Real.pi) = 2 * Real.pi * (Int.fract ((x.val : ℝ) * θ i / (2 * Real.pi)) -
          Int.fract ((x₀.val : ℝ) * θ i / (2 * Real.pi))) by
      push_cast
      linear_combination (-(2 * Real.pi)) * e1 + (2 * Real.pi) * e2 - h1 + h2,
    abs_mul, abs_of_pos hpi]
  rw [lt_div_iff₀ (by positivity)] at hclose
  rw [le_div_iff₀ (by positivity)]
  nlinarith [abs_nonneg (Int.fract ((x.val : ℝ) * θ i / (2 * Real.pi)) -
    Int.fract ((x₀.val : ℝ) * θ i / (2 * Real.pi)))]

/-- Two frequencies, eight boxes per frequency, the distances below `100`. -/
example := exists_returns (fun i : Fin 2 => (i : ℝ) + 1) 7 (H := 100) (by norm_num)

end Transformer.MRCR
