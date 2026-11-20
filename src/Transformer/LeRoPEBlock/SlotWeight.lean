module

public import Transformer.LeRoPEBlock.SlotFejer

/-!
# A LeRoPE head keeps a bounded share of its weight on an inner key

Source: `review/lerope_block.md`, *Reading a slot* (added), RS1.

Softmax weighs the key at the distance `t` by `e^{ℓ(t)} / Z`, `Z = ∑ e^{ℓ}`. Take the window of
distances `t + s − s'`, `s, s' ≤ R`. The row `s` hits `R + 1` distinct keys, so its `R` pairs
`s' ≠ s` carry at most `Z − e^{ℓ(t)}`. If the margins `ℓ(t) − ℓ(t + s − s')` sum to at most
`(R + 1)² B`, they average to at most `B (1 + 1/R)` over the `R (R + 1)` pairs off the diagonal,
and Jensen's inequality for `exp` (`card_mul_exp_mean_le`) gives those pairs at least
`R (R + 1) e^{ℓ(t) − B (1 + 1/R)}`. So the key keeps at most `1 / (1 + R e^{−B (1 + 1/R)})` of the
weight (`depthWeight_le_of_sum_sum_le`).

For a head of the main model, Q/K RMS gives `|q|², |k|² ≤ d` and SSMax the gain `ln N / d`, so
`SlotFejer` gives `B = (1 + |β|) ln N`. A key whose `R` neighbours on each side carry its token
keeps at most `1 / (1 + R N^{−(1 + |β|)(1 + 1/R)})` of the weight (`depthWeight_leropeLogit_le`),
whatever the frequencies. At `β = 0` the middle of `N = 2R + 1` keys keeps at most `9/10`, and
the bound tends to `2/3` as `R` grows.
-/

@[expose] public section

open Matrix Finset
open Transformer.AttentionResiduals Transformer.RoPE.RoFormer

namespace Transformer.LeRoPEBlock

variable {p : ℕ}

/-- **Jensen's inequality for `exp`**: `N e^{mean f} ≤ ∑ e^{f}` over `N` points. -/
theorem card_mul_exp_mean_le {ι : Type*} (S : Finset ι) (hS : S.Nonempty) (f : ι → ℝ) :
    (S.card : ℝ) * Real.exp ((∑ i ∈ S, f i) / S.card) ≤ ∑ i ∈ S, Real.exp (f i) := by
  set m := (∑ i ∈ S, f i) / S.card
  have hN : (S.card : ℝ) ≠ 0 := Nat.cast_ne_zero.2 hS.card_pos.ne'
  have hm : (S.card : ℝ) * m = ∑ i ∈ S, f i := mul_div_cancel₀ _ hN
  calc (S.card : ℝ) * Real.exp m = ∑ i ∈ S, Real.exp m * (f i - m + 1) := by
        rw [← mul_sum, sum_add_distrib, sum_sub_distrib, sum_const, sum_const, nsmul_eq_mul,
          nsmul_eq_mul, ← hm]
        ring
    _ ≤ ∑ i ∈ S, Real.exp m * Real.exp (f i - m) := sum_le_sum fun i _ =>
        mul_le_mul_of_nonneg_left (Real.add_one_le_exp _) (Real.exp_pos _).le
    _ = ∑ i ∈ S, Real.exp (f i) := sum_congr rfl fun i _ => by rw [← Real.exp_add]; ring_nf

/-- Three points: `3 e^{(0 + 1 + 2)/3} ≤ e⁰ + e¹ + e²`. -/
example := card_mul_exp_mean_le (range 3) (by simp) fun i => (i : ℝ)

/-- **The row `s ≤ R` of the window hits `R + 1` distinct keys** among the distances `0, …, i`. -/
theorem sum_row_le {i t R s : ℕ} (hRt : R ≤ t) (hti : t + R ≤ i) (hs : s ≤ R) (f : ℕ → ℝ) :
    ∑ s' ∈ range (R + 1), Real.exp (f (t + s - s')) ≤ ∑ δ ∈ range (i + 1), Real.exp (f δ) := by
  rw [← sum_image (f := fun δ => Real.exp (f δ)) (g := fun s' => t + s - s') fun a ha b hb hab => by
    simp only [coe_range, Set.mem_Iio] at ha hb hab
    omega]
  refine sum_le_sum_of_subset_of_nonneg (fun δ hδ => ?_) fun _ _ _ => (Real.exp_pos _).le
  obtain ⟨s', hs', rfl⟩ := mem_image.1 hδ
  simp only [mem_range] at hs' ⊢
  omega

example (f : ℕ → ℝ) : ∑ s' ∈ range 3, Real.exp (f (4 + 1 - s')) ≤
    ∑ δ ∈ range 7, Real.exp (f δ) :=
  sum_row_le (i := 6) (t := 4) (R := 2) (by norm_num) (by norm_num) (by norm_num) f

/-- The partition function over the keys `j ≤ i` scored by the distance `i − j`. -/
theorem sum_fin_exp_sub (i : ℕ) (f : ℕ → ℝ) :
    ∑ j : Fin (i + 1), Real.exp (f (i - j)) = ∑ δ ∈ range (i + 1), Real.exp (f δ) := by
  rw [Fin.sum_univ_eq_sum_range (fun j => Real.exp (f (i - j))) (i + 1),
    ← sum_range_reflect (fun δ => Real.exp (f δ)) (i + 1)]
  simp

/-- **A key whose window margins sum to at most `(R + 1)² B` keeps at most
`1 / (1 + R e^{−B (1 + 1/R)})` of the softmax weight.** The key sits at the distance `t`, with `R`
distances on each side, and `f δ` scores the key at the distance `δ`. -/
theorem depthWeight_le_of_sum_sum_le {i t R : ℕ} (hR : 1 ≤ R) (hRt : R ≤ t) (hti : t + R ≤ i)
    (f : ℕ → ℝ) {B : ℝ}
    (hB : ∑ s ∈ range (R + 1), ∑ s' ∈ range (R + 1), (f t - f (t + s - s')) ≤
      ((R : ℝ) + 1) ^ 2 * B) :
    depthWeight (fun j : Fin (i + 1) => f (i - j)) ⟨i - t, by omega⟩ ≤
      1 / (1 + R * Real.exp (-(B * (1 + 1 / R)))) := by
  set Z := ∑ δ ∈ range (i + 1), Real.exp (f δ)
  set a := Real.exp (f t)
  set g : ℕ → ℕ → ℝ := fun s s' => f (t + s - s') - f t
  have hR0 : (0 : ℝ) < R := Nat.cast_pos.2 hR
  have ha : 0 < a := Real.exp_pos _
  have hw : depthWeight (fun j : Fin (i + 1) => f (i - j)) ⟨i - t, by omega⟩ = a / Z := by
    simp only [depthWeight, depthPartition, sum_fin_exp_sub, a, Z]
    rw [Nat.sub_sub_self (by omega)]
  have hcard (s : ℕ) (hs : s ∈ range (R + 1)) : (((range (R + 1)).erase s).card : ℝ) = R := by
    simp [card_erase_of_mem hs]
  have hne (s : ℕ) (hs : s ∈ range (R + 1)) : ((range (R + 1)).erase s).Nonempty :=
    card_pos.1 (by rw [card_erase_of_mem hs, card_range]; omega)
  -- each row: Jensen over its `R` pairs off the diagonal, against the row's share of `Z`
  have hrow (s : ℕ) (hs : s ∈ range (R + 1)) :
      R * Real.exp ((∑ s' ∈ range (R + 1), g s s') / R) ≤ (Z - a) / a := by
    have hJ := card_mul_exp_mean_le _ (hne s hs) (g s)
    rw [hcard s hs, sum_erase _ (by simp [g])] at hJ
    have hZs := sum_row_le hRt hti (Nat.lt_succ_iff.mp (mem_range.1 hs)) f
    refine hJ.trans ?_
    calc ∑ s' ∈ (range (R + 1)).erase s, Real.exp (g s s') =
        (∑ s' ∈ (range (R + 1)).erase s, Real.exp (f (t + s - s'))) / a := by
          rw [sum_div]
          exact sum_congr rfl fun s' _ => by simp only [g, a, Real.exp_sub]
      _ = (∑ s' ∈ range (R + 1), Real.exp (f (t + s - s')) - a) / a := by
          rw [sum_erase_eq_sub hs, Nat.add_sub_cancel]
      _ ≤ (Z - a) / a := div_le_div_of_nonneg_right (by linarith) ha.le
  -- Jensen over the rows
  have hmean : -(B * (1 + 1 / R)) ≤
      (∑ s ∈ range (R + 1), (∑ s' ∈ range (R + 1), g s s') / R) / (R + 1) := by
    have hg : ∑ s ∈ range (R + 1), ∑ s' ∈ range (R + 1), g s s' = -∑ s ∈ range (R + 1),
        ∑ s' ∈ range (R + 1), (f t - f (t + s - s')) := by
      simp only [g, ← sum_neg_distrib, neg_sub]
    rw [← sum_div, hg, div_div, le_div_iff₀ (by positivity)]
    have h : -(B * (1 + 1 / R)) * (R * (R + 1)) = -(((R : ℝ) + 1) ^ 2 * B) := by
      field_simp
    rw [h]
    linarith
  have hall : (R : ℝ) * (R + 1) * Real.exp (-(B * (1 + 1 / R))) ≤ (R + 1) * ((Z - a) / a) := by
    have hJ := card_mul_exp_mean_le (range (R + 1)) ⟨0, by simp⟩
      fun s => (∑ s' ∈ range (R + 1), g s s') / R
    simp only [card_range, Nat.cast_add, Nat.cast_one] at hJ
    calc (R : ℝ) * (R + 1) * Real.exp (-(B * (1 + 1 / R))) ≤
        R * ((R + 1) * Real.exp ((∑ s ∈ range (R + 1), (∑ s' ∈ range (R + 1), g s s') / R) /
          (R + 1))) := by
          rw [mul_assoc]
          gcongr
      _ ≤ R * ∑ s ∈ range (R + 1), Real.exp ((∑ s' ∈ range (R + 1), g s s') / R) := by gcongr
      _ = ∑ s ∈ range (R + 1), R * Real.exp ((∑ s' ∈ range (R + 1), g s s') / R) := mul_sum _ _ _
      _ ≤ ∑ _s ∈ range (R + 1), (Z - a) / a := sum_le_sum hrow
      _ = (R + 1) * ((Z - a) / a) := by simp
  have h1 : (R : ℝ) * Real.exp (-(B * (1 + 1 / R))) * a ≤ Z - a := by
    rw [← le_div_iff₀ ha]
    refine le_of_mul_le_mul_left ?_ (by positivity : (0 : ℝ) < R + 1)
    linarith
  have hZ : 0 < Z := by
    have : 0 ≤ (R : ℝ) * Real.exp (-(B * (1 + 1 / R))) * a := by positivity
    linarith
  rw [hw, div_le_div_iff₀ hZ (by positivity)]
  linarith

/-- Seven keys, the slot at the distance `3`, margins summing to at most `9 B`. -/
example (f : ℕ → ℝ) (B : ℝ)
    (hB : ∑ s ∈ range 3, ∑ s' ∈ range 3, (f 3 - f (3 + s - s')) ≤ ((2 : ℕ) + 1 : ℝ) ^ 2 * B) :
    depthWeight (fun j : Fin (6 + 1) => f (6 - j)) ⟨6 - 3, by omega⟩ ≤
      1 / (1 + (2 : ℕ) * Real.exp (-(B * (1 + 1 / (2 : ℕ))))) :=
  depthWeight_le_of_sum_sum_le (R := 2) (by norm_num) (by norm_num) (by norm_num) f hB

/-- **A head of the main model keeps at most `1 / (1 + R N^{−(1 + |β|)(1 + 1/R)})` of its weight on
a key whose `R` neighbours on each side carry its token**, `N = i + 1`, whatever the frequencies,
when the query and the key have squared length at most `d` (`review/lerope_block.md`, RS1). -/
theorem depthWeight_leropeLogit_le {X : Type*} (θ : Fin p → ℝ) {β d : ℝ} (hd : 0 < d)
    (Q K : X → Fin 2 × Fin p → ℝ) (x : ℕ → X) (y : X) {i t R : ℕ} (hR : 1 ≤ R) (hRt : R ≤ t)
    (hti : t + R ≤ i) (hx : ∀ δ, t ≤ δ + R → δ ≤ t + R → x (i - δ) = y)
    (hQ : Q (x i) ⬝ᵥ Q (x i) ≤ d) (hK : K y ⬝ᵥ K y ≤ d) :
    depthWeight (fun j : Fin (i + 1) => leropeLogit θ β d Q K i (x i) (x j) (i - j))
      ⟨i - t, by omega⟩ ≤ 1 / (1 + R * ((i : ℝ) + 1) ^ (-((1 + |β|) * (1 + 1 / R)))) := by
  set L := Real.log ((i : ℝ) + 1)
  have hfun : (fun j : Fin (i + 1) => leropeLogit θ β d Q K i (x i) (x j) (i - j)) =
      fun j : Fin (i + 1) => leropeLogit θ β d Q K i (x i) (x (i - (i - j))) (i - j) := by
    funext j
    rw [Nat.sub_sub_self j.is_le]
  have he : ((i : ℝ) + 1) ^ (-((1 + |β|) * (1 + 1 / R))) =
      Real.exp (-((1 + |β|) * L * (1 + 1 / R))) := by
    rw [Real.rpow_def_of_pos (by positivity)]; congr 1; ring
  rw [hfun, he]
  refine depthWeight_le_of_sum_sum_le hR hRt hti
    (fun δ => leropeLogit θ β d Q K i (x i) (x (i - δ)) δ) ?_
  have hwin (s : ℕ) (hs : s ∈ range (R + 1)) (s' : ℕ) (hs' : s' ∈ range (R + 1)) :
      leropeLogit θ β d Q K i (x i) (x (i - t)) t -
        leropeLogit θ β d Q K i (x i) (x (i - (t + s - s'))) (t + s - s') =
      phaseLogit (L / d) β d (Q (x i)) (rotary θ (-(t : ℝ)) *ᵥ K y) -
        phaseLogit (L / d) β d (Q (x i)) (rotary θ (-((t : ℝ) + s - s')) *ᵥ K y) := by
    rw [mem_range] at hs hs'
    rw [hx t (by omega) (by omega), hx (t + s - s') (by omega) (by omega), leropeLogit,
      leropeLogit, Nat.cast_sub (by omega), Nat.cast_add]
  have hL : 0 ≤ L := Real.log_nonneg (by linarith [(Nat.cast_nonneg i : (0 : ℝ) ≤ i)])
  have hP : (Q (x i) ⬝ᵥ Q (x i)) * (K y ⬝ᵥ K y) / d ≤ d :=
    (div_le_iff₀ hd).2 <| mul_le_mul hQ hK (sum_nonneg fun _ _ => mul_self_nonneg _) hd.le
  have hin : (Q (x i) ⬝ᵥ Q (x i) + K y ⬝ᵥ K y) / 2 + |β| * ((Q (x i) ⬝ᵥ Q (x i)) * (K y ⬝ᵥ K y)) / d
      ≤ d * (1 + |β|) := by
    rw [mul_div_assoc]
    nlinarith [abs_nonneg β]
  rw [sum_congr rfl fun s hs => sum_congr rfl fun s' hs' => hwin s hs s' hs']
  refine (sum_sum_phaseLogit_sub_le θ (div_nonneg hL hd.le) hd t (R + 1) (Q (x i)) (K y)).trans ?_
  calc _ ≤ ((R + 1 : ℕ) : ℝ) ^ 2 * (L / d * (d * (1 + |β|))) :=
        mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left hin (div_nonneg hL hd.le))
          (by positivity)
    _ = ((R : ℝ) + 1) ^ 2 * ((1 + |β|) * L) := by push_cast; field_simp

/-- **At `β = 0`, the middle of three keys of one token keeps at most `9/10`** of the weight. -/
example (θ : Fin 1 → ℝ) (Q K : Unit → Fin 2 × Fin 1 → ℝ) (hQ : Q () ⬝ᵥ Q () ≤ 2)
    (hK : K () ⬝ᵥ K () ≤ 2) :
    depthWeight (fun j : Fin (2 + 1) => leropeLogit θ 0 2 Q K 2 () () (2 - j)) ⟨2 - 1, by omega⟩ ≤
      9 / 10 := by
  refine (depthWeight_leropeLogit_le θ two_pos Q K (fun _ => ()) () (R := 1) le_rfl le_rfl le_rfl
    (fun _ _ _ => rfl) hQ hK).trans_eq ?_
  norm_num [Real.rpow_neg, Real.rpow_two]

end Transformer.LeRoPEBlock
