module

public import Transformer.Grokking.Protocol.SolverParams

/-!
# The hidden layer of the solver marks the quotient

Source: openai/grok at 3d64b1d8, `grok/transformer.py`, `FFN`: two linear maps without bias
around `ReLU`; arXiv:2201.02177v1 (Power et al.), appendix *Additional experimental details*.
The weights are those of `SolverParams`.

The hidden unit `u` computes `max 0 (g - (u - 96))` of the gap `g`. Three consecutive units
with the weights `1, -2, 1` are `1` at one integer and `0` at the others (`hat_unit`), and two
with the weights `1, -1` are `1` from `101` on and `0` up to `100` (`ramp_unit`). So the
entry `o` of the layer's output, `hiddenOut g o`, is known at every integer gap
(`hiddenOut_eq`), and at the gap of an equation it is the indicator of the entry of its
quotient (`hiddenOut_logGap`): `287` gaps decide `9312` equations, with `195` of the `512`
units.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

/-- The entry `o` of the output of the hidden layer at the gap `g`, before the factor of the
normalization. -/
noncomputable def hiddenOut (g : ℝ) (o : Fin 128) : ℝ :=
  ∑ u : Fin 512, solverDown o u * max 0 (g - knot u)

/-- **Three consecutive units mark one integer gap.** -/
lemma hat_unit (g : ℤ) (j : ℕ) :
    max 0 ((g : ℝ) - ((j : ℝ) - 96)) - 2 * max 0 ((g : ℝ) - (((j + 1 : ℕ) : ℝ) - 96)) +
        max 0 ((g : ℝ) - (((j + 2 : ℕ) : ℝ) - 96)) =
      if g = (j : ℤ) - 95 then 1 else 0 := by
  rcases lt_trichotomy g ((j : ℤ) - 95) with h | h | h
  · have h' : (g : ℝ) ≤ (j : ℝ) - 96 := by exact_mod_cast (show g ≤ (j : ℤ) - 96 by omega)
    rw [ite_eq_right h.ne, max_eq_left (by linarith), max_eq_left (by push_cast; linarith),
      max_eq_left (by push_cast; linarith)]
    ring
  · have h' : (g : ℝ) = (j : ℝ) - 95 := by exact_mod_cast h
    rw [ite_eq_left h, max_eq_right (by linarith), max_eq_left (by push_cast; linarith),
      max_eq_left (by push_cast; linarith)]
    linarith
  · have h' : (j : ℝ) - 94 ≤ (g : ℝ) := by exact_mod_cast (show (j : ℤ) - 94 ≤ g by omega)
    rw [ite_eq_right h.ne', max_eq_right (by linarith), max_eq_right (by push_cast; linarith),
      max_eq_right (by push_cast; linarith)]
    push_cast
    ring

/-- **Two units mark the gaps above `100`.** -/
lemma ramp_unit (g : ℤ) :
    max 0 ((g : ℝ) - (((196 : ℕ) : ℝ) - 96)) - max 0 ((g : ℝ) - (((197 : ℕ) : ℝ) - 96)) =
      if 101 ≤ g then 1 else 0 := by
  by_cases h : 101 ≤ g
  · have h' : (101 : ℝ) ≤ (g : ℝ) := by exact_mod_cast h
    rw [ite_eq_left h, max_eq_right (by push_cast; linarith),
      max_eq_right (by push_cast; linarith)]
    push_cast
    ring
  · have h' : (g : ℝ) ≤ 100 := by exact_mod_cast (show g ≤ 100 by omega)
    rw [ite_eq_right h, max_eq_left (by push_cast; linarith),
      max_eq_left (by push_cast; linarith)]
    ring

lemma sum_hatCoef_mul (g : ℤ) (j : Fin 191) :
    ∑ u : Fin 512, hatCoef j u * max 0 ((g : ℝ) - knot u) =
      if g = ((j : ℕ) : ℤ) - 95 then 1 else 0 := by
  have hj := j.isLt
  have e : ∀ u : Fin 512, hatCoef j u * max 0 ((g : ℝ) - knot u) =
      ind j u * max 0 ((g : ℝ) - knot u) - 2 * (ind (j + 1) u * max 0 ((g : ℝ) - knot u)) +
        ind (j + 2) u * max 0 ((g : ℝ) - knot u) := fun u => by
    rw [hatCoef]
    ring
  rw [Finset.sum_congr rfl fun u _ => e u, Finset.sum_add_distrib, Finset.sum_sub_distrib,
    ← Finset.mul_sum, sum_ind_mul (by omega : (j : ℕ) < 512),
    sum_ind_mul (by omega : (j : ℕ) + 1 < 512), sum_ind_mul (by omega : (j : ℕ) + 2 < 512)]
  exact hat_unit g j

/-- **The output of the hidden layer at an integer gap.** -/
theorem hiddenOut_eq (g : ℤ) (o : Fin 128) :
    hiddenOut g o =
      (if -95 ≤ g ∧ g ≤ 95 ∧ (o : ℕ) = (powFive g).val + 5 then 1 else 0) +
        if (o : ℕ) = 5 ∧ 101 ≤ g then 1 else 0 := by
  have h1 : ∀ j : Fin 191, ∑ u : Fin 512,
      (if (o : ℕ) = (powFive (((j : ℕ) : ℤ) - 95)).val + 5 then hatCoef j u else 0) *
        max 0 ((g : ℝ) - knot u) =
      if (o : ℕ) = (powFive (((j : ℕ) : ℤ) - 95)).val + 5 ∧ g = ((j : ℕ) : ℤ) - 95 then 1
      else 0 := fun j => by
    by_cases hc : (o : ℕ) = (powFive (((j : ℕ) : ℤ) - 95)).val + 5
    · rw [Finset.sum_congr rfl fun u _ => by rw [ite_eq_left hc], sum_hatCoef_mul]
      exact if_congr ⟨fun h => ⟨hc, h⟩, fun h => h.2⟩ rfl rfl
    · rw [Finset.sum_congr rfl fun u _ => by rw [ite_eq_right hc, zero_mul],
        Finset.sum_const_zero, ite_eq_right fun h => hc h.1]
  have h2 : ∑ u : Fin 512, (if (o : ℕ) = 5 then ind 196 u - ind 197 u else 0) *
      max 0 ((g : ℝ) - knot u) = if (o : ℕ) = 5 ∧ 101 ≤ g then 1 else 0 := by
    by_cases hc : (o : ℕ) = 5
    · rw [Finset.sum_congr rfl fun u _ => by rw [ite_eq_left hc, sub_mul],
        Finset.sum_sub_distrib, sum_ind_mul (by norm_num : 196 < 512),
        sum_ind_mul (by norm_num : 197 < 512)]
      exact (ramp_unit g).trans (if_congr ⟨fun h => ⟨hc, h⟩, fun h => h.2⟩ rfl rfl)
    · rw [Finset.sum_congr rfl fun u _ => by rw [ite_eq_right hc, zero_mul],
        Finset.sum_const_zero, ite_eq_right fun h => hc h.1]
  have h3 : ∑ j : Fin 191,
      (if (o : ℕ) = (powFive (((j : ℕ) : ℤ) - 95)).val + 5 ∧ g = ((j : ℕ) : ℤ) - 95 then (1 : ℝ)
        else 0) = if -95 ≤ g ∧ g ≤ 95 ∧ (o : ℕ) = (powFive g).val + 5 then 1 else 0 := by
    by_cases hr : -95 ≤ g ∧ g ≤ 95
    · have hj : (g + 95).toNat < 191 := by omega
      have e : (((g + 95).toNat : ℕ) : ℤ) - 95 = g := by omega
      rw [Finset.sum_eq_single (⟨(g + 95).toNat, hj⟩ : Fin 191)]
      · rw [show ((⟨(g + 95).toNat, hj⟩ : Fin 191) : ℕ) = (g + 95).toNat from rfl, e]
        exact if_congr ⟨fun h => ⟨hr.1, hr.2, h.1⟩, fun h => ⟨h.2.2, rfl⟩⟩ rfl rfl
      · intro j _ hj'
        refine ite_eq_right fun h => hj' (Fin.ext ?_)
        have h4 := h.2
        change (j : ℕ) = (g + 95).toNat
        omega
      · intro h
        exact absurd (Finset.mem_univ _) h
    · rw [Finset.sum_eq_zero fun j _ => ite_eq_right fun h => hr (by have := j.isLt; omega),
        ite_eq_right fun h => hr ⟨h.1, h.2.1⟩]
  rw [hiddenOut, Finset.sum_congr rfl fun u _ => by rw [solverDown, add_mul, Finset.sum_mul],
    Finset.sum_add_distrib, Finset.sum_comm, Finset.sum_congr rfl fun j _ => h1 j, h2, h3]

/-- **At the gap of an equation the hidden layer marks the entry of the quotient.** -/
theorem hiddenOut_logGap (x : Equation) (o : Fin 128) :
    hiddenOut (logGap x) o = ind ((quotient x).val + 5) o := by
  rw [hiddenOut_eq]
  by_cases hx : quotient x = 0
  · obtain ⟨h0, h1⟩ := logGap_of_eq_zero hx
    have hz : ¬(-95 ≤ logGap x ∧ logGap x ≤ 95 ∧ (o : ℕ) = (powFive (logGap x)).val + 5) :=
      fun h => by omega
    rw [ite_eq_right hz, zero_add, hx, ZMod.val_zero, zero_add]
    exact if_congr (and_iff_left (by omega)) rfl rfl
  · obtain ⟨h0, h1, h2⟩ := logGap_of_ne_zero hx
    have hz : ¬((o : ℕ) = 5 ∧ 101 ≤ logGap x) := fun h => by omega
    rw [ite_eq_right hz, add_zero, h2]
    exact if_congr ⟨fun h => h.2.2, fun h => ⟨h0, h1, h⟩⟩ rfl rfl

/-! The witnesses: the gap `0`, where `5⁰ = 1` marks the entry `6`, and `3 / 5 = 20`. -/

example : hiddenOut ((0 : ℤ) : ℝ) 6 = 1 := by
  rw [hiddenOut_eq, ite_eq_left ⟨by norm_num, by norm_num, by decide⟩,
    ite_eq_right fun h => absurd h.2 (by norm_num)]
  norm_num

example := hiddenOut_logGap (20, 4) 25

end Transformer.Grokking.Protocol
