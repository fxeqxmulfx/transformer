module

public import Transformer.Grokking.Protocol.SolverAttention

/-!
# The first block of the solver reads the gap of the equation

Source: openai/grok at 3d64b1d8, `grok/transformer.py`, `MultiHeadAttention.forward`, as
`Model.attention`; arXiv:2201.02177v1 (Power et al.), appendix *Additional experimental
details*. The parameters are those of `SolverParams`.

Both heads at work read the entry `0` of the rows, where the numerator and the denominator
carry their features (`headRead`). The head with equal weights reads a fifth of their sum
(`headRead_uniform`), the keyed head reads them with the weights `keyExp 1` and `keyExp 3`
over `3 + keyExp 1 + keyExp 3` (`headRead_keyed`). The two pairs of weights are not
proportional, since `keyExp 3 < keyExp 1`, so a combination of the two heads has the
coefficient `1` on the numerator and `-1` on the denominator: with `mixUniform` and
`mixKeyed`, the attention at `=` is the difference of what the two carry, in the entry `2`
and nowhere else (`solver_attention`).
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

variable {l : Fin paperSizes.blocks} {h : Fin paperSizes.heads} {r : Fin paperSizes.headWidth}

lemma solver_proj (hl : (l : ℕ) = 0) (hr : (r : ℕ) = 0) (o : Fin 128) :
    solver (.proj l o h r) =
      (if (h : ℕ) = 0 then mixUniform else if (h : ℕ) = 1 then mixKeyed else 0) * ind 2 o :=
  ite_eq_left ⟨hl, hr⟩

lemma solver_proj_zero (hr : (r : ℕ) ≠ 0) (o : Fin 128) : solver (.proj l o h r) = 0 :=
  ite_eq_right fun hc => hr hc.2

/-- What the head `h` of the first block reads at `=`: the entries `0` of the rows, weighed. -/
noncomputable def headRead (l : Fin paperSizes.blocks) (h : Fin paperSizes.heads)
    (x : Equation) : ℝ :=
  ∑ j ∈ Finset.Iic (4 : Fin 6), weight solver (solverRows x) l h 4 j *
    ∑ i, solver (.value l h ⟨0, by decide⟩ i) * solverRows x j i

/-- The head with equal weights reads a fifth of the sum of the two features. -/
lemma headRead_uniform (hl : (l : ℕ) = 0) (hh : (h : ℕ) = 0) (x : Equation) :
    headRead l h x = 1 / 5 * ((logFeature (num x) + position paperSizes 1 (0 : Fin 128)) +
      (logFeature (den x) + position paperSizes 3 (0 : Fin 128))) := by
  obtain ⟨v0, v1, v2, v3, v4⟩ := solverRows_value x
  have hne : (h : ℕ) ≠ 1 := by omega
  have hw : ∀ j : Fin 6, j ≤ 4 → weight solver (solverRows x) l h 4 j = 1 / 5 := fun j hj =>
    solver_weight_of_ne hne _ hj
  have hv : ∀ j : Fin 6, ∑ i, solver (.value l h ⟨0, by decide⟩ i) * solverRows x j i =
      solverRows x j 0 := fun j => solver_value_sum hl (by omega) rfl _
  rw [headRead, sum_Iic_four, hv, hv, hv, hv, hv, hw 0 (by decide), hw 1 (by decide),
    hw 2 (by decide), hw 3 (by decide), hw 4 (by decide), v0, v1, v2, v3, v4]
  ring

/-- The keyed head reads the two features with the weights of their positions. -/
lemma headRead_keyed (hl : (l : ℕ) = 0) (hh : (h : ℕ) = 1) (x : Equation) :
    headRead l h x =
      (keyExp 1 * (logFeature (num x) + position paperSizes 1 (0 : Fin 128)) +
        keyExp 3 * (logFeature (den x) + position paperSizes 3 (0 : Fin 128))) /
          (3 + keyExp 1 + keyExp 3) := by
  obtain ⟨v0, v1, v2, v3, v4⟩ := solverRows_value x
  obtain ⟨r0, r1, r2, r3, r4, -⟩ := solverRows_key x
  have hw : ∀ j : Fin 6, j ≤ 4 → weight solver (solverRows x) l h 4 j =
      Real.exp (solverRows x j 1) / (3 + keyExp 1 + keyExp 3) := fun j hj =>
    solver_weight_keyed hl hh x hj
  have hv : ∀ j : Fin 6, ∑ i, solver (.value l h ⟨0, by decide⟩ i) * solverRows x j i =
      solverRows x j 0 := fun j => solver_value_sum hl (by omega) rfl _
  rw [headRead, sum_Iic_four, hv, hv, hv, hv, hv, hw 0 (by decide), hw 1 (by decide),
    hw 2 (by decide), hw 3 (by decide), hw 4 (by decide), v0, v1, v2, v3, v4, r1, r3, keyExp,
    keyExp]
  ring

/-- **The attention of the first block at `=`** is the difference of what the numerator and
the denominator carry in the entry `0`, written into the entry `2`. -/
theorem solver_attention (hl : (l : ℕ) = 0) (x : Equation) (o : Fin 128) :
    attention solver (solverRows x) l 4 o =
      ind 2 o * ((logFeature (num x) + position paperSizes 1 (0 : Fin 128)) -
        (logFeature (den x) + position paperSizes 3 (0 : Fin 128))) := by
  have hd : keyExp 1 - keyExp 3 ≠ 0 := (sub_pos.2 keyExp_three_lt).ne'
  have hZ : 3 + keyExp 1 + keyExp 3 ≠ 0 := by
    have h1 : 0 < keyExp 1 := Real.exp_pos _
    have h3 : 0 < keyExp 3 := Real.exp_pos _
    positivity
  have hr : ∀ h : Fin paperSizes.heads, ∑ r, solver (.proj l o h r) *
      ∑ j ∈ Finset.Iic (4 : Fin 6), weight solver (solverRows x) l h 4 j *
        ∑ i, solver (.value l h r i) * solverRows x j i =
      (if (h : ℕ) = 0 then mixUniform else if (h : ℕ) = 1 then mixKeyed else 0) * ind 2 o *
        headRead l h x := fun h => by
    rw [Finset.sum_eq_single (⟨0, by decide⟩ : Fin paperSizes.headWidth)
      (fun r _ hr => by rw [solver_proj_zero (fun hc => hr (Fin.ext hc)), zero_mul])
      (fun hc => absurd (Finset.mem_univ _) hc), solver_proj hl rfl]
    rfl
  rw [attention, Finset.sum_congr rfl fun h _ => hr h, Fin.sum_univ_four,
    headRead_uniform hl rfl, headRead_keyed hl rfl]
  simp only [Fin.val_zero, Fin.val_one, Fin.val_two, ite_true, mixUniform, mixKeyed]
  norm_num
  field_simp
  ring

/-! The witnesses: the first block, its two heads, and the equation `3 / 5 = 20`. -/

example := solver_proj (l := ⟨0, by decide⟩) (h := ⟨1, by decide⟩) (r := ⟨0, by decide⟩) rfl rfl
  2

example := solver_proj_zero (l := ⟨0, by decide⟩) (h := ⟨1, by decide⟩) (r := ⟨1, by decide⟩)
  (by decide) 2

example := headRead_uniform (l := ⟨0, by decide⟩) (h := ⟨0, by decide⟩) rfl rfl (20, 4)

example := headRead_keyed (l := ⟨0, by decide⟩) (h := ⟨1, by decide⟩) rfl rfl (20, 4)

example := solver_attention (l := ⟨0, by decide⟩) rfl (20, 4) 2

end Transformer.Grokking.Protocol
