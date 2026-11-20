module

public import Transformer.ReRoPEBlock.Defs
public import Transformer.MRCR.ReRopeFading

/-!
# The main model's scores come back

Source: the score `headScore` of a head of the main model (`Transformer.ReRoPEBlock.Defs`, DiffGPT
`5a37c51`). Its two terms `Re Z` and `Im Z` are RoPE scores of the turn of the key
(`dotProduct_rotary_eq_ropeScore`, `Transformer.MRCR.ropeScore`), so a distance `n` that brings
every phase `n θ_l` near a multiple of `2π` moves the score little, however far it goes:

* `abs_headScore_add_sub_le`: if every `n θ_l` is within `η` of a multiple of `2π`, adding `n` to
  the position moves the score of the query `i` by at most `ln(i + 1)/d · η · C`, where
  `C = scoreBound` depends only on the query and key vectors, `β` and `d`;
* `exists_return_log`: for every `ε > 0` there are such distances `n` as far as wanted with
  `(c + ln(n + 2)) η ≤ ε`. SSMax's gain grows like `ln n`, while Dirichlet's pigeonhole
  (`Transformer.MRCR.exists_returns`) finds returns within `2π/(M + 1)` below `(N + 2)(M + 1)^K`.
-/

@[expose] public section

open Matrix Filter Topology
open Transformer.MRCR Transformer.RoPE.ReRoPE Transformer.RoPE.RoFormer

namespace Transformer.ReRoPEBlock

variable {X : Type*} {p : ℕ}

/-- The cosine coefficients of `x ⬝ᵥ R_t k`: `x_{0l} k_{0l} + x_{1l} k_{1l}`. -/
def pairRe (x k : Fin 2 × Fin p → ℝ) (l : Fin p) : ℝ := x (0, l) * k (0, l) + x (1, l) * k (1, l)

/-- The sine coefficients of `x ⬝ᵥ R_t k`: `x_{1l} k_{0l} − x_{0l} k_{1l}`. -/
def pairIm (x k : Fin 2 × Fin p → ℝ) (l : Fin p) : ℝ := x (1, l) * k (0, l) - x (0, l) * k (1, l)

/-- **A rotary product is a RoPE score** of the turn `t`. -/
theorem dotProduct_rotary_eq_ropeScore (θ : Fin p → ℝ) (t : ℝ) (x k : Fin 2 × Fin p → ℝ) :
    x ⬝ᵥ (rotary θ t *ᵥ k) = ropeScore θ (pairRe x k) (pairIm x k) t := by
  rw [rotary_mulVec]
  simp only [dotProduct, Fintype.sum_prod_type, Fin.sum_univ_two, rotateHalf, ropeScore, pairRe,
    pairIm, ← Finset.sum_add_distrib, mul_comm t]
  refine Finset.sum_congr rfl fun l _ => ?_
  simp
  ring

/-- The constant of `abs_headScore_add_sub_le`: `‖(a, b)‖ + 2 |β| ‖(a', b')‖² / d`, with `(a, b)`
the coefficients of `Re Z` and `(a', b')` those of `Im Z`. -/
noncomputable def scoreBound (β d : ℝ) (q k : Fin 2 × Fin p → ℝ) : ℝ :=
  ropeNorm (pairRe q k) (pairIm q k) +
    2 * |β| * ropeNorm (pairRe q (rotateHalf k)) (pairIm q (rotateHalf k)) ^ 2 / d

theorem scoreBound_nonneg (β : ℝ) {d : ℝ} (hd : 0 ≤ d) (q k : Fin 2 × Fin p → ℝ) :
    0 ≤ scoreBound β d q k :=
  add_nonneg (Finset.sum_nonneg fun l _ => by positivity) (div_nonneg (by positivity) hd)

example (q k : Fin 2 × Fin 3 → ℝ) : 0 ≤ scoreBound (1 / 2) 32 q k :=
  scoreBound_nonneg _ (by norm_num) q k

/-- **A near return moves the score little**: if every phase `n θ_l` is within `η` of a multiple
of `2π`, the score of the query `i` moves by at most `ln(i + 1)/d · η · C` from the position `t`
to `t + n`, `C = scoreBound β d (Q x) (K y)`. -/
theorem abs_headScore_add_sub_le {θ : Fin p → ℝ} {n η : ℝ}
    (hn : ∀ l, ∃ m : ℤ, |θ l * n - m * (2 * Real.pi)| ≤ η) (hη : 0 ≤ η) (β : ℝ) {d : ℝ}
    (hd : 0 < d) (Q K : X → Fin 2 × Fin p → ℝ) (i : ℕ) (x y : X) (t : ℝ) :
    |headScore θ β d Q K i x y (t + n) - headScore θ β d Q K i x y t| ≤
      Real.log ((i : ℝ) + 1) / d * (η * scoreBound β d (Q x) (K y)) := by
  have hn' : ∀ l, ∃ m : ℤ, |θ l * -n - m * (2 * Real.pi)| ≤ η := fun l => by
    obtain ⟨m, hm⟩ := hn l
    refine ⟨-m, ?_⟩
    rw [Int.cast_neg, show θ l * -n - -(m : ℝ) * (2 * Real.pi) =
      -(θ l * n - m * (2 * Real.pi)) by ring, abs_neg]
    exact hm
  have hg : 0 ≤ Real.log ((i : ℝ) + 1) / d :=
    div_nonneg (Real.log_nonneg (le_add_of_nonneg_left (Nat.cast_nonneg i))) hd.le
  have hA := abs_ropeScore_add_sub_le (a := pairRe (Q x) (K y)) (b := pairIm (Q x) (K y)) hn' (-t)
  have hB := abs_ropeScore_add_sub_le (a := pairRe (Q x) (rotateHalf (K y)))
    (b := pairIm (Q x) (rotateHalf (K y))) hn' (-t)
  have hB1 := abs_ropeScore_le θ (pairRe (Q x) (rotateHalf (K y)))
    (pairIm (Q x) (rotateHalf (K y))) (-t + -n)
  have hB0 := abs_ropeScore_le θ (pairRe (Q x) (rotateHalf (K y)))
    (pairIm (Q x) (rotateHalf (K y))) (-t)
  simp only [headScore, phaseScore, rotateHalf_rotary, dotProduct_rotary_eq_ropeScore, neg_add,
    scoreBound]
  generalize Real.log ((i : ℝ) + 1) / d = g at hg ⊢
  generalize ropeScore θ (pairRe (Q x) (K y)) (pairIm (Q x) (K y)) (-t + -n) = A1 at hA ⊢
  generalize ropeScore θ (pairRe (Q x) (K y)) (pairIm (Q x) (K y)) (-t) = A0 at hA ⊢
  generalize ropeScore θ (pairRe (Q x) (rotateHalf (K y))) (pairIm (Q x) (rotateHalf (K y)))
    (-t + -n) = B1 at hB hB1 ⊢
  generalize ropeScore θ (pairRe (Q x) (rotateHalf (K y))) (pairIm (Q x) (rotateHalf (K y)))
    (-t) = B0 at hB hB0 ⊢
  generalize ropeNorm (pairRe (Q x) (K y)) (pairIm (Q x) (K y)) = Na at hA ⊢
  generalize ropeNorm (pairRe (Q x) (rotateHalf (K y))) (pairIm (Q x) (rotateHalf (K y))) = Nb
    at hB hB1 hB0 ⊢
  have hNb : 0 ≤ Nb := (abs_nonneg _).trans hB0
  have hsq : |B1 ^ 2 - B0 ^ 2| ≤ 2 * η * Nb ^ 2 := by
    rw [sq, sq, show B1 * B1 - B0 * B0 = (B1 - B0) * (B1 + B0) by ring, abs_mul]
    calc |B1 - B0| * |B1 + B0| ≤ (η * Nb) * (Nb + Nb) :=
          mul_le_mul hB ((abs_add_le _ _).trans (add_le_add hB1 hB0)) (abs_nonneg _)
            (mul_nonneg hη hNb)
      _ = 2 * η * Nb ^ 2 := by ring
  rw [← mul_sub, abs_mul, abs_of_nonneg hg]
  refine mul_le_mul_of_nonneg_left ?_ hg
  calc |A1 + β * B1 ^ 2 / d - (A0 + β * B0 ^ 2 / d)| = |(A1 - A0) + β * (B1 ^ 2 - B0 ^ 2) / d| := by
        congr 1; ring
    _ ≤ |A1 - A0| + |β| * |B1 ^ 2 - B0 ^ 2| / d := by
        refine (abs_add_le _ _).trans (le_of_eq ?_)
        rw [abs_div, abs_mul, abs_of_pos hd]
    _ ≤ η * Na + |β| * (2 * η * Nb ^ 2) / d := by gcongr
    _ = η * (Na + 2 * |β| * Nb ^ 2 / d) := by ring

/-- The frequency `2π` comes back after `1`: the score does not move. -/
example (Q K : ℕ → Fin 2 × Fin 1 → ℝ) (i x y : ℕ) (t : ℝ) :
    |headScore (fun _ => 2 * Real.pi) (1 / 2) 32 Q K i x y (t + 1) -
        headScore (fun _ => 2 * Real.pi) (1 / 2) 32 Q K i x y t| ≤
      Real.log ((i : ℝ) + 1) / 32 * (0 * scoreBound (1 / 2) 32 (Q x) (K y)) :=
  abs_headScore_add_sub_le (fun _ => ⟨1, by simp⟩) le_rfl _ (by norm_num) Q K i x y t

/-- **Returns beat SSMax's gain**: for every `ε > 0`, `c` and `N` some distance `n ≥ N` brings
every phase `n θ_l` within `η` of a multiple of `2π` with `(c + ln(n + 2)) η ≤ ε`. -/
theorem exists_return_log {K : ℕ} (θ : Fin K → ℝ) (c : ℝ) {ε : ℝ} (hε : 0 < ε) (N : ℕ) :
    ∃ n : ℕ, ∃ η : ℝ, N ≤ n ∧ 0 ≤ η ∧ (c + Real.log ((n : ℝ) + 2)) * η ≤ ε ∧
      ∀ l, ∃ m : ℤ, |θ l * n - m * (2 * Real.pi)| ≤ η := by
  set A := |c| + Real.log ((N : ℝ) + 4)
  have hlim : Tendsto (fun M : ℕ =>
      (A + K * Real.log ((M : ℝ) + 1)) * (2 * Real.pi / ((M : ℝ) + 1))) atTop (𝓝 0) := by
    have h1 : Tendsto (fun M : ℕ => 1 / ((M : ℝ) + 1)) atTop (𝓝 0) :=
      tendsto_one_div_add_atTop_nhds_zero_nat
    have h2 : Tendsto (fun M : ℕ => Real.log ((M : ℝ) + 1) / ((M : ℝ) + 1)) atTop (𝓝 0) := by
      have := (Real.tendsto_pow_log_div_mul_add_atTop 1 0 1 one_ne_zero).comp
        (tendsto_natCast_atTop_atTop.atTop_add
          (tendsto_const_nhds : Tendsto (fun _ : ℕ => (1 : ℝ)) atTop (𝓝 1)))
      simpa [Function.comp_def] using this
    have := (h1.const_mul (2 * Real.pi * A)).add (h2.const_mul (2 * Real.pi * K))
    simp only [mul_zero, add_zero] at this
    refine this.congr fun M => ?_
    field_simp
  obtain ⟨M, hM⟩ := (hlim.eventually (Iic_mem_nhds hε)).exists
  have hP : (1 : ℝ) ≤ ((M : ℝ) + 1) ^ K :=
    one_le_pow₀ (le_add_of_nonneg_left (Nat.cast_nonneg M))
  obtain ⟨R, hcard, hR⟩ := exists_returns θ M (H := (N + 2) * (M + 1) ^ K) (by positivity)
  have e : (((N + 2) * (M + 1) ^ K : ℕ) : ℝ) / ((M : ℝ) + 1) ^ K - 1 = N + 1 := by
    push_cast
    rw [mul_div_cancel_right₀ _ (by positivity)]
    ring
  rw [e] at hcard
  obtain ⟨n, hnR, hNn⟩ : ∃ n ∈ R, N ≤ n := by
    by_contra hcon
    push Not at hcon
    have h := Finset.card_le_card fun n hn => Finset.mem_Ico.2 ⟨(hR n hn).1, hcon n hn⟩
    rw [Nat.card_Ico] at h
    have : (R.card : ℝ) ≤ N := by exact_mod_cast h.trans (Nat.sub_le N 1)
    linarith
  obtain ⟨-, hnH, hret⟩ := hR n hnR
  refine ⟨n, 2 * Real.pi / ((M : ℝ) + 1), hNn, by positivity,
    (mul_le_mul_of_nonneg_right ?_ (by positivity)).trans hM, hret⟩
  have hnH' : (n : ℝ) + 2 ≤ ((N : ℝ) + 4) * ((M : ℝ) + 1) ^ K := by
    have h : n + 1 ≤ (N + 2) * (M + 1) ^ K := hnH
    have : (n : ℝ) + 1 ≤ ((N : ℝ) + 2) * ((M : ℝ) + 1) ^ K := by exact_mod_cast h
    nlinarith
  calc c + Real.log ((n : ℝ) + 2) ≤ |c| + Real.log (((N : ℝ) + 4) * ((M : ℝ) + 1) ^ K) :=
        add_le_add (le_abs_self c) (Real.log_le_log (by positivity) hnH')
    _ = A + K * Real.log ((M : ℝ) + 1) := by
        rw [Real.log_mul (by positivity) (by positivity), Real.log_pow]
        ring

/-- Three frequencies, returns past `10` with `ln(n + 2) η ≤ 1`. -/
example (θ : Fin 3 → ℝ) :
    ∃ n : ℕ, ∃ η : ℝ, 10 ≤ n ∧ 0 ≤ η ∧ (0 + Real.log ((n : ℝ) + 2)) * η ≤ 1 ∧
      ∀ l, ∃ m : ℤ, |θ l * n - m * (2 * Real.pi)| ≤ η :=
  exists_return_log θ 0 one_pos 10

end Transformer.ReRoPEBlock
