module

public import Transformer.ReRoPEBlock.Peak
public import Transformer.ReRoPEBlock.Softmax

/-!
# Blocks that cannot clamp

Source: the blocks of `Transformer.ReRoPEBlock.Defs` (DiffGPT `5a37c51`) and ReRoPE's peak head
(`Transformer.ReRoPEBlock.Peak`). At every query `D ≥ W` the peak head notices a swap of the keys
at the distances `1` and `D` (`le_norm_peakHead_sub`). A block whose scores give the distance `D`
nearly the score of the distance `1`, at queries as far as wanted, hardly notices it:

* `block_permute_swap`: swapping two keys before the query `D` swaps the positions that their
  scores read;
* `le_of_approx_peakHead`: if for every `ε > 0` there are queries `D` as far as wanted at which
  every head scores the tokens `a` and `b` at the distance `D` within `ε` of their scores at the
  distance `1`, and `Φ` is continuous, the block misses the peak head by at least
  `‖u a − u b‖ / (4(W + 1))` at queries as far as wanted. The swapped sequences hold only `a` and
  `b`, so the heads' outputs stay in a compact set where `Φ` is uniformly continuous.
-/

@[expose] public section

open Matrix
open Transformer.AttentionResiduals

namespace Transformer.ReRoPEBlock

variable {X E E' Y : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [NormedAddCommGroup E']
  [NormedSpace ℝ E'] {H p : ℕ}

/-- **Swapping two keys** `a, b ≠ D` before the query `D` keeps the query's token and gives every
key its token back with the position of its swapped place. -/
theorem block_permute_swap (ψ : ℕ → ℝ) (Φ : X → (Fin H → E') → Y)
    (F : Fin H → ℕ → X → X → ℝ → ℝ) (v : Fin H → X → E') (x : ℕ → X) {D : ℕ}
    {a b : Fin (D + 1)} (ha : (a : ℕ) ≠ D) (hb : (b : ℕ) ≠ D) :
    block ψ Φ F v (permute (Equiv.swap a b) x) D =
      Φ (x D) fun h => attend (fun j : Fin (D + 1) =>
        F h D (x D) (x j) (ψ (D - Equiv.swap a b j))) fun j => v h (x j) := by
  have hD : permute (Equiv.swap a b) x D = x D := by
    have h := permute_apply (Equiv.swap a b) x (Fin.last D)
    rw [Equiv.swap_apply_of_ne_of_ne (fun h => ha (by rw [← h, Fin.val_last]))
      (fun h => hb (by rw [← h, Fin.val_last]))] at h
    simpa using h
  simp only [block, hD, permute_apply]
  refine congrArg _ (funext fun h => ?_)
  rw [← attend_comp_equiv (Equiv.swap a b)
    (fun j => F h D (x D) (x j) (ψ (D - Equiv.swap a b j))) fun j => v h (x j)]
  simp only [Equiv.swap_apply_self]

/-- The keys `0` and `1` before the query `2`. -/
example (x : ℕ → ℝ) :
    block lePosition (fun _ a => a 0) (fun _ _ _ y t => y * t) (fun (_ : Fin 1) y => y)
        (permute (Equiv.swap (0 : Fin 3) 1) x) 2 =
      attend (fun j : Fin 3 => x j * lePosition (2 - Equiv.swap (0 : Fin 3) 1 j)) fun j => x j :=
  block_permute_swap _ _ _ _ x (by decide) (by decide)

/-- The peak head stays within the norms of its values. -/
theorem norm_peakHead_le (W : ℕ) (d : ℝ) (q : Fin 2 × Fin p → ℝ) {u : X → E} {R : ℝ}
    (hu : ∀ y, ‖u y‖ ≤ R) (x : ℕ → X) (i : ℕ) : ‖peakHead W d q u x i‖ ≤ R := by
  unfold peakHead block
  exact attend_norm_le _ _ R fun _ => hu _

example (x : ℕ → Bool) (i : ℕ) :
    ‖peakHead 2 2 (fun _ : Fin 2 × Fin 1 => (1 : ℝ)) (fun t : Bool => if t then (1 : ℝ) else 0)
      x i‖ ≤ 1 :=
  norm_peakHead_le 2 2 _ (fun t => by cases t <;> norm_num) x i

/-- **Blocks that cannot clamp miss the peak head**: if every head's scores of the tokens `a`
and `b` at the distance `D` come within any `ε > 0` of their scores at the distance `1`, at
queries `D` as far as wanted, and `Φ` of the query token `b` is continuous, a block within `δ`
of the peak head (window `W ≥ 2`, `0 < d ≤ 2 qᵀq`) at every query from `N` on has
`δ ≥ ‖u a − u b‖ / (4(W + 1))`. -/
theorem le_of_approx_peakHead {W N : ℕ} (hW : 2 ≤ W) {d : ℝ} (hd : 0 < d)
    (q : Fin 2 × Fin p → ℝ) (hq : d ≤ 2 * (q ⬝ᵥ q)) (u : X → E) (ψ : ℕ → ℝ)
    (Φ : X → (Fin H → E') → E) (F : Fin H → ℕ → X → X → ℝ → ℝ) (v : Fin H → X → E')
    {a b : X} (hΦ : Continuous (Φ b))
    (hF : ∀ M : ℕ, ∀ ε > 0, ∃ D, M ≤ D ∧ ∀ h, ∀ y ∈ ({a, b} : Set X),
      |F h D b y (ψ D) - F h D b y (ψ 1)| ≤ ε)
    {δ : ℝ} (hδ : ∀ x i, N ≤ i → ‖block ψ Φ F v x i - peakHead W d q u x i‖ ≤ δ) :
    ‖u a - u b‖ / (4 * ((W : ℝ) + 1)) ≤ δ := by
  suffices key : ∀ ε > 0, ‖u a - u b‖ / (2 * ((W : ℝ) + 1)) ≤ 2 * δ + ε by
    have := le_of_forall_pos_le_add key
    calc ‖u a - u b‖ / (4 * ((W : ℝ) + 1))
        = ‖u a - u b‖ / (2 * ((W : ℝ) + 1)) / 2 := by rw [div_div]; congr 1; ring
      _ ≤ δ := by linarith
  intro ε hε
  have hKc : IsCompact (Set.univ.pi fun h => convexHull ℝ {v h a, v h b}) :=
    isCompact_univ_pi fun h => Set.Finite.isCompact_convexHull (𝕜 := ℝ) (Set.toFinite _)
  obtain ⟨ρ, hρ, hΦρ⟩ := Metric.uniformContinuousOn_iff.1
    (hKc.uniformContinuousOn_of_continuous hΦ.continuousOn) ε hε
  obtain ⟨R, hR⟩ : ∃ R, R = ∑ h, (‖v h a‖ + ‖v h b‖) := ⟨_, rfl⟩
  have hR0 : 0 ≤ R := hR ▸ Finset.sum_nonneg fun h _ => by positivity
  have hρR : 0 < ρ / (2 * (R + 1)) := by positivity
  obtain ⟨εs, hεs⟩ : ∃ εs, εs = Real.log (1 + ρ / (2 * (R + 1))) / 2 := ⟨_, rfl⟩
  have hεs0 : 0 < εs := hεs ▸ div_pos (Real.log_pos (by linarith)) two_pos
  have hexp : (Real.exp (2 * εs) - 1) * R < ρ := by
    rw [hεs, show 2 * (Real.log (1 + ρ / (2 * (R + 1))) / 2) =
      Real.log (1 + ρ / (2 * (R + 1))) by ring, Real.exp_log (by linarith), add_sub_cancel_left,
      div_mul_eq_mul_div, div_lt_iff₀ (by positivity)]
    nlinarith
  obtain ⟨D, hD, hFD⟩ := hF (max N W) εs hεs0
  have hWD : W ≤ D := le_of_max_le_right hD
  obtain ⟨x, hx⟩ : ∃ x : ℕ → X, x = fun j => if j = D - 1 then a else b := ⟨_, rfl⟩
  have hxab : ∀ j, x j ∈ ({a, b} : Set X) := fun j => by
    rw [hx]
    dsimp only
    split_ifs
    exacts [Set.mem_insert _ _, Set.mem_insert_of_mem _ rfl]
  have hpeak := le_norm_peakHead_sub hW hWD hd q hq u x
  rw [show x (D - 1) = a by rw [hx]; exact ite_eq_left rfl,
    show x 0 = b by rw [hx]; exact ite_eq_right (by omega)] at hpeak
  have hblock : ‖block ψ Φ F v x D -
      block ψ Φ F v (permute (Equiv.swap (⟨D - 1, by omega⟩ : Fin (D + 1)) 0) x) D‖ < ε := by
    rw [block_permute_swap ψ Φ F v x (by simp; omega) (by simp; omega)]
    simp only [block]
    rw [show x D = b by rw [hx]; exact ite_eq_right (by omega), ← dist_eq_norm]
    have hmem : ∀ s : Fin H → Fin (D + 1) → ℝ,
        (fun h => attend (s h) fun j : Fin (D + 1) => v h (x j)) ∈
          Set.univ.pi fun h => convexHull ℝ {v h a, v h b} := fun s =>
      Set.mem_univ_pi.2 fun h => convexHull_mono (Set.range_subset_iff.2 fun j => by
        obtain h' | h' := hxab j
        · rw [h']; exact Set.mem_insert _ _
        · rw [Set.mem_singleton_iff.1 h']; exact Set.mem_insert_of_mem _ rfl)
        (attend_mem_convexHull _ _)
    refine hΦρ _ (hmem _) _ (hmem _) ((dist_pi_lt_iff hρ).2 fun h => ?_)
    rw [dist_eq_norm]
    refine (norm_attend_sub_attend_le (ε := εs) (fun j => ?_) _ (R := R) fun j => ?_).trans_lt hexp
    · rcases eq_or_ne j ⟨D - 1, by omega⟩ with rfl | hj1
      · simp only [Fin.val_zero, Nat.sub_zero, Equiv.swap_apply_left]
        rw [show D - (D - 1) = 1 by omega, abs_sub_comm]
        exact hFD h _ (hxab _)
      rcases eq_or_ne j 0 with rfl | hj0
      · simp only [Fin.val_mk, Fin.val_zero, Nat.sub_zero, Equiv.swap_apply_right]
        rw [show D - (D - 1) = 1 by omega]
        exact hFD h _ (hxab _)
      rw [Equiv.swap_apply_of_ne_of_ne hj1 hj0, sub_self, abs_zero]
      exact hεs0.le
    · have hle : ‖v h a‖ + ‖v h b‖ ≤ R := hR ▸ Finset.single_le_sum
        (f := fun h => ‖v h a‖ + ‖v h b‖) (fun h _ => by positivity) (Finset.mem_univ h)
      obtain h' | h' := hxab j
      · rw [h']; linarith [norm_nonneg (v h b)]
      · rw [Set.mem_singleton_iff.1 h']; linarith [norm_nonneg (v h a)]
  generalize permute (Equiv.swap (⟨D - 1, by omega⟩ : Fin (D + 1)) 0) x = x' at hpeak hblock
  have h1 := hδ x D (le_of_max_le_left hD)
  have h2 := hδ x' D (le_of_max_le_left hD)
  have e : peakHead W d q u x D - peakHead W d q u x' D =
      -(block ψ Φ F v x D - peakHead W d q u x D) + (block ψ Φ F v x D - block ψ Φ F v x' D) +
        (block ψ Φ F v x' D - peakHead W d q u x' D) := by abel
  rw [e] at hpeak
  have h3 := hpeak.trans norm_add₃_le
  rw [norm_neg] at h3
  linarith

/-- The zero block misses the peak head of the indicator of `true` by at most `1`. -/
example : ‖(if true then (1 : ℝ) else 0) - (if false then (1 : ℝ) else 0)‖ /
    (4 * (((2 : ℕ) : ℝ) + 1)) ≤ 1 :=
  le_of_approx_peakHead (W := 2) (N := 0) (δ := 1) (a := true) (b := false) le_rfl two_pos
    (fun _ : Fin 2 × Fin 1 => (1 : ℝ)) (by norm_num [dotProduct])
    (fun t : Bool => if t then (1 : ℝ) else 0) lePosition (fun _ (_ : Fin 1 → ℝ) => (0 : ℝ))
    (fun _ _ _ _ _ => 0) (fun _ _ => (0 : ℝ)) continuous_const
    (fun M ε hε => ⟨M, le_rfl, fun _ _ _ => by simp [hε.le]⟩) fun x i _ => by
      simp only [block, zero_sub, norm_neg]
      exact norm_peakHead_le 2 2 _ (fun t => by cases t <;> norm_num) x i

end Transformer.ReRoPEBlock
