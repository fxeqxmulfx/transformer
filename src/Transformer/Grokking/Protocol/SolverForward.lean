module

public import Transformer.Grokking.Protocol.SolverGap
public import Transformer.Grokking.Protocol.SolverLookup

/-!
# The forward pass of the solver

Source: openai/grok at 3d64b1d8, `grok/transformer.py`, `DecoderBlock.forward`, as
`Model.attended` and `Model.block`; arXiv:2201.02177v1 (Power et al.), appendix *Additional
experimental details*. The parameters are those of `SolverParams`.

At `=`, the row after the attention of the first block is `gapRow`: the ballast in the entry
`3`, the gap of the equation in the entry `2` and zero elsewhere (`solver_attended_input`).
Its normalization is a positive multiple of the centred row (`solver_attended`), and the mean
that the centring takes off cancels in the hidden unit `u`, whose three weights sum to zero:
the unit reads the gap less its knot, times the factor (`solver_preactivation`). `max 0`
commutes with the positive factor, so the hidden layer adds the factor times `hiddenOut` of
the gap (`solver_hidden`), and the first block ends with a positive multiple of the centred
`markedRow` (`solver_block_zero`).

The second block has no attention and no hidden layer: it normalizes twice
(`solver_block_of_ne`).
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

variable {l : Fin paperSizes.blocks}

/-- The row at `=` after the attention of the first block: the ballast in the entry `3` and
the gap of the equation in the entry `2`. -/
noncomputable def gapRow (x : Equation) (o : Fin 128) : ℝ :=
  ballast * ind 3 o + (logGap x : ℝ) * ind 2 o

lemma solver_attended_input (hl : (l : ℕ) = 0) (x : Equation) (o : Fin 128) :
    solverRows x 4 o + attention solver (solverRows x) l 4 o = gapRow x o := by
  rw [solver_attention hl, solverRows_four, solverTarget, gapRow, logGap]
  push_cast
  ring

/-- The first normalization of the first block, at `=`. -/
lemma solver_attended (hl : (l : ℕ) = 0) (x : Equation) (i : Fin 128) :
    attended solver (solverRows x) l 4 i =
      normScale 1 (gapRow x) * (gapRow x i - rowMean (gapRow x)) := by
  have e : (fun i => solverRows x 4 i + attention solver (solverRows x) l 4 i) = gapRow x :=
    funext (solver_attended_input hl x)
  rw [attended]
  change layerNorm (fun _ => 1) (fun _ => 0) _ i = _
  rw [e, layerNorm_one]

lemma gapRow_two (x : Equation) (h : 2 < 128) : gapRow x ⟨2, h⟩ = logGap x := by
  rw [gapRow, ind_of_ne (k := 3) (i := (⟨2, h⟩ : Fin 128)) (by norm_num),
    ind_of_eq (k := 2) (i := (⟨2, h⟩ : Fin 128)) rfl]
  ring

lemma gapRow_three (x : Equation) (h : 3 < 128) : gapRow x ⟨3, h⟩ = ballast := by
  rw [gapRow, ind_of_eq (k := 3) (i := (⟨3, h⟩ : Fin 128)) rfl,
    ind_of_ne (k := 2) (i := (⟨3, h⟩ : Fin 128)) (by norm_num)]
  ring

lemma gapRow_of_ne (x : Equation) {o : Fin 128} (h2 : (o : ℕ) ≠ 2) (h3 : (o : ℕ) ≠ 3) :
    gapRow x o = 0 := by
  rw [gapRow, ind_of_ne h3, ind_of_ne h2]
  ring

/-- **The hidden unit `u` reads the gap less its knot**, times the factor of the
normalization. -/
lemma solver_preactivation (hl : (l : ℕ) = 0) (x : Equation) (u : Fin 512) :
    preactivation solver (solverRows x) l 4 u =
      normScale 1 (gapRow x) * ((logGap x : ℝ) - knot u) := by
  have hb : ballast ≠ 0 := by norm_num [ballast]
  have e : ∀ i : Fin 128, solver (.up l u i) * attended solver (solverRows x) l 4 i =
      normScale 1 (gapRow x) * (ind 2 i * (gapRow x i - rowMean (gapRow x))) -
        normScale 1 (gapRow x) * (knot u / ballast) *
          (ind 3 i * (gapRow x i - rowMean (gapRow x))) +
        normScale 1 (gapRow x) * (knot u / ballast - 1) *
          (ind 4 i * (gapRow x i - rowMean (gapRow x))) := fun i => by
    rw [show solver (.up l u i) = solverUp u i from ite_eq_left hl, solver_attended hl, solverUp]
    ring
  rw [preactivation, Finset.sum_congr rfl fun i _ => e i, Finset.sum_add_distrib,
    Finset.sum_sub_distrib, ← Finset.mul_sum, ← Finset.mul_sum, ← Finset.mul_sum,
    sum_ind_mul (by norm_num : 2 < 128), sum_ind_mul (by norm_num : 3 < 128),
    sum_ind_mul (by norm_num : 4 < 128), gapRow_two, gapRow_three,
    gapRow_of_ne x (o := ⟨4, by norm_num⟩) (by decide) (by decide)]
  field_simp
  ring

/-- **The hidden layer of the first block at `=`**: the factor of the normalization times
the output at the gap. -/
lemma solver_hidden (hl : (l : ℕ) = 0) (x : Equation) (o : Fin 128) :
    ∑ u, solver (.down l o u) *
        relu paperSizes l 4 u (preactivation solver (solverRows x) l 4 u) =
      normScale 1 (gapRow x) * hiddenOut (logGap x) o := by
  have hτ : 0 ≤ normScale 1 (gapRow x) := (normScale_pos one_pos _).le
  rw [hiddenOut, Finset.mul_sum]
  refine Finset.sum_congr rfl fun u _ => ?_
  rw [show solver (.down l o u) = solverDown o u from ite_eq_left hl, solver_preactivation hl]
  change solverDown o u * max 0 (normScale 1 (gapRow x) * ((logGap x : ℝ) - knot u)) = _
  have hm : max 0 (normScale 1 (gapRow x) * ((logGap x : ℝ) - knot u)) =
      normScale 1 (gapRow x) * max 0 ((logGap x : ℝ) - knot u) := by
    rw [mul_max_of_nonneg _ _ hτ, mul_zero]
  rw [hm]
  ring

/-- The row at `=` before the second normalization of the first block, without the factor:
the ballast, the gap, and the mark of the quotient. -/
noncomputable def markedRow (x : Equation) (o : Fin 128) : ℝ :=
  gapRow x o + hiddenOut (logGap x) o

/-- **The first block of the solver at `=`.** -/
theorem solver_block_zero (hl : (l : ℕ) = 0) (x : Equation) (i : Fin 128) :
    block solver (relu paperSizes) (solverRows x) l 4 i =
      normScale (normScale 1 (gapRow x)) (markedRow x) *
        (markedRow x i - rowMean (markedRow x)) := by
  have e : (fun o => attended solver (solverRows x) l 4 o + ∑ u, solver (.down l o u) *
      relu paperSizes l 4 u (preactivation solver (solverRows x) l 4 u)) =
      fun o => normScale 1 (gapRow x) * (markedRow x o - rowMean (gapRow x)) :=
    funext fun o => by
      rw [solver_attended hl, solver_hidden hl, markedRow]
      ring
  rw [block]
  change layerNorm (fun _ => 1) (fun _ => 0) _ i = _
  rw [e, layerNorm_mul_sub (by norm_num)]

lemma solver_attention_of_ne (hl : (l : ℕ) ≠ 0) (X : Fin 6 → Fin 128 → ℝ) (t : Fin 6)
    (o : Fin 128) : attention solver X l t o = 0 := by
  rw [attention]
  exact Finset.sum_eq_zero fun h _ => Finset.sum_eq_zero fun r _ => by
    rw [show solver (.proj l o h r) = 0 from ite_eq_right fun hc => hl hc.1, zero_mul]

/-- **The second block of the solver** only normalizes, twice. -/
theorem solver_block_of_ne (hl : (l : ℕ) ≠ 0) (act : Activation paperSizes)
    (X : Fin 6 → Fin 128 → ℝ) (t : Fin 6) :
    block solver act X l t = layerNorm (fun _ => 1) (fun _ => 0)
      (layerNorm (fun _ => 1) (fun _ => 0) (X t)) := by
  have ha : attended solver X l t = layerNorm (fun _ => 1) (fun _ => 0) (X t) := by
    rw [attended]
    change layerNorm (fun _ => 1) (fun _ => 0) _ = _
    refine congrArg _ (funext fun i => ?_)
    rw [solver_attention_of_ne hl, add_zero]
  rw [block]
  change layerNorm (fun _ => 1) (fun _ => 0) _ = _
  refine congrArg _ (funext fun o => ?_)
  rw [ha, Finset.sum_eq_zero fun u _ => by
    rw [show solver (.down l o u) = 0 from ite_eq_right hl, zero_mul], add_zero]

/-! The witnesses: the two blocks and the equation `3 / 5 = 20`. -/

example := solver_attended_input (l := ⟨0, by decide⟩) rfl (20, 4) 2

example := solver_attended (l := ⟨0, by decide⟩) rfl (20, 4) 2

example := gapRow_two (20, 4) (by norm_num)

example := gapRow_three (20, 4) (by norm_num)

example := gapRow_of_ne (20, 4) (o := 4) (by decide) (by decide)

example := solver_preactivation (l := ⟨0, by decide⟩) rfl (20, 4) 0

example := solver_hidden (l := ⟨0, by decide⟩) rfl (20, 4) 25

example := solver_block_zero (l := ⟨0, by decide⟩) rfl (20, 4) 25

example := solver_attention_of_ne (l := ⟨1, by decide⟩) (by decide) (fun _ _ => 1) 4 2

example := solver_block_of_ne (l := ⟨1, by decide⟩) (by decide) (relu paperSizes)
  (fun _ _ => 1) 4

end Transformer.Grokking.Protocol
