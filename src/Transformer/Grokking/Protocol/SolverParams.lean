module

public import Transformer.Grokking.Protocol.DiscreteLog
public import Transformer.Grokking.Protocol.NormScale
public import Mathlib.Analysis.Real.Pi.Bounds

/-!
# Parameters of the paper's decoder that answer every division

Source: arXiv:2201.02177v1 (Power et al.), appendix *Additional experimental details*, the
decoder of two blocks, width 128 and four heads; openai/grok at 3d64b1d8,
`grok/transformer.py`, whose forward pass is `Model.logits`. The parameters are this
repository's construction, not the paper's and not a trained model's.

`solver` is a point of the parameter space of `paperSizes`:

* the embedding of a residue is its `logFeature` in the entry `0`; those of `<|eos|>`, `/` and
  `=` cancel the positions `0`, `2` and `4`, and leave the ballast `10000` in the entry `3` of
  the row at `=` (`solverEmb`, `solverTarget`);
* the first block has two heads at work. One has no query and weighs the five positions
  equally. The other reads the entry `1`, where the positions of the numerator and of the
  denominator differ (`keyExp_three_lt`). Two heads with different weights separate the
  numerator from the denominator, and the map after them writes the gap of the equation into
  the entry `2` (`mixUniform`, `mixKeyed`);
* the hidden unit `u` of the first block has its kink where the gap is `u - 96`
  (`solverUp`, `knot`), and the second map sums the units to the indicator of the quotient in
  the entry `5 + r` of the residue `r` (`solverDown`, `hatCoef`);
* the second block has no attention and no hidden layer, and the readout of the residue `r`
  is the entry `5 + r` less the entry `4` (`solverOut`).

All gains are `1` and all biases `0`. `SolverLookup`, `SolverAttention` and `SolverForward`
compute the forward pass.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

/-- The ballast: the constant entry of the row at `=`. It makes the bias of the hidden units,
and keeps the variance of the row nearly the same for every equation. -/
def ballast : ℝ := 10000

/-- The exponential of the entry `1` of the position `j`: the weight of the keyed head. -/
noncomputable def keyExp (j : ℕ) : ℝ := Real.exp (position paperSizes j (1 : Fin 128))

lemma position_one (j : ℕ) : position paperSizes j (1 : Fin 128) = Real.cos j := by
  simp [position]

/-- **The keyed head weighs the numerator more than the denominator**: `cos 3 < cos 1`. -/
lemma keyExp_three_lt : keyExp 3 < keyExp 1 := by
  have hπ := Real.pi_gt_three
  rw [keyExp, keyExp, position_one, position_one, Real.exp_lt_exp]
  exact Real.strictAntiOn_cos ⟨by norm_num, by push_cast; linarith⟩
    ⟨by norm_num, by push_cast; linarith⟩ (by norm_num)

/-- The coefficient of the head with equal weights in the entry `2`. -/
noncomputable def mixUniform : ℝ := -5 * (keyExp 1 + keyExp 3) / (keyExp 1 - keyExp 3)

/-- The coefficient of the keyed head in the entry `2`. -/
noncomputable def mixKeyed : ℝ := 2 * (3 + keyExp 1 + keyExp 3) / (keyExp 1 - keyExp 3)

/-- The row at `=` before the attention: the ballast in the entry `3`, and in the entry `2`
what cancels the positions of the numerator and the denominator. -/
noncomputable def solverTarget (i : Fin 128) : ℝ :=
  ballast * ind 3 i -
    (position paperSizes 1 (0 : Fin 128) - position paperSizes 3 (0 : Fin 128)) * ind 2 i

/-- The embedding. -/
noncomputable def solverEmb (tok : Fin 239) (i : Fin 128) : ℝ :=
  if tok = eosToken then -position paperSizes 0 i
  else if tok = divToken then -position paperSizes 2 i
  else if tok = eqToken then solverTarget i - position paperSizes 4 i
  else ind 0 i * logFeature (((tok : ℕ) - 22 : ℕ) : ZMod 97)

/-- The gap at which the hidden unit `u` has its kink. -/
def knot (u : Fin 512) : ℝ := ((u : ℕ) : ℝ) - 96

/-- The first map of the hidden layer: the unit `u` reads the gap less its knot. -/
noncomputable def solverUp (u : Fin 512) (i : Fin 128) : ℝ :=
  ind 2 i - knot u / ballast * ind 3 i + (knot u / ballast - 1) * ind 4 i

/-- The three units whose sum is `1` at the gap `j - 95` and `0` at the other integers. -/
def hatCoef (j : ℕ) (u : Fin 512) : ℝ := ind j u - 2 * ind (j + 1) u + ind (j + 2) u

/-- The second map of the hidden layer: the entry `5 + r` sums the units of the gaps whose
power of `5` is `r`, and the entry `5` also the two units that mark a gap above `100`. -/
noncomputable def solverDown (o : Fin 128) (u : Fin 512) : ℝ :=
  (∑ j : Fin 191,
      if (o : ℕ) = (powFive (((j : ℕ) : ℤ) - 95)).val + 5 then hatCoef j u else 0) +
    if (o : ℕ) = 5 then ind 196 u - ind 197 u else 0

/-- The readout: the token of the residue `r` reads the entry `5 + r` less the entry `4`. -/
noncomputable def solverOut (tok : Fin 239) (i : Fin 128) : ℝ :=
  if 22 ≤ (tok : ℕ) ∧ (tok : ℕ) < 119 then ind ((tok : ℕ) - 17) i - ind 4 i else 0

/-- **Parameters of the paper's decoder that answer every equation** (`SolverForward`). -/
noncomputable def solver : Param paperSizes → ℝ
  | .emb tok i => solverEmb tok i
  | .query l h r i =>
    if (l : ℕ) = 0 ∧ (h : ℕ) = 1 ∧ (r : ℕ) = 0 then
      √(paperSizes.headWidth : ℝ) / ballast * ind 3 i
    else 0
  | .key l h r i => if (l : ℕ) = 0 ∧ (h : ℕ) = 1 ∧ (r : ℕ) = 0 then ind 1 i else 0
  | .value l h r i => if (l : ℕ) = 0 ∧ (h : ℕ) ≤ 1 ∧ (r : ℕ) = 0 then ind 0 i else 0
  | .proj l o h r =>
    if (l : ℕ) = 0 ∧ (r : ℕ) = 0 then
      (if (h : ℕ) = 0 then mixUniform else if (h : ℕ) = 1 then mixKeyed else 0) * ind 2 o
    else 0
  | .gain1 _ _ => 1
  | .bias1 _ _ => 0
  | .up l u i => if (l : ℕ) = 0 then solverUp u i else 0
  | .down l o u => if (l : ℕ) = 0 then solverDown o u else 0
  | .gain2 _ _ => 1
  | .bias2 _ _ => 0
  | .out tok i => solverOut tok i

lemma residueToken_ne_eqToken (r : Fin 97) : residueToken r ≠ eqToken := fun h => by
  simp only [eqToken, residueToken, Fin.ext_iff] at h
  omega

lemma residueToken_ne_divToken (r : Fin 97) : residueToken r ≠ divToken := fun h => by
  simp only [divToken, residueToken, Fin.ext_iff] at h
  omega

/-- A residue as an element of the field. -/
def toField (r : Fin 97) : ZMod 97 := r

/-- The embedding of a residue is its feature in the entry `0`. -/
lemma solverEmb_residueToken (r : Fin 97) (i : Fin 128) :
    solverEmb (residueToken r) i = ind 0 i * logFeature (toField r) := by
  have e : ((((residueToken r : Fin 239) : ℕ) - 22 : ℕ) : ZMod 97) = toField r := by
    have h : ((residueToken r : Fin 239) : ℕ) - 22 = ZMod.val (toField r) := by
      change 22 + (r : ℕ) - 22 = _
      rw [Nat.add_sub_cancel_left]
      rfl
    rw [h, ZMod.natCast_zmod_val]
  rw [solverEmb, ite_eq_right (residueToken_ne_eosToken r),
    ite_eq_right (residueToken_ne_divToken r), ite_eq_right (residueToken_ne_eqToken r), e]

/-- The readout of a residue. -/
lemma solverOut_residueToken (r : Fin 97) (i : Fin 128) :
    solverOut (residueToken r) i = ind ((r : ℕ) + 5) i - ind 4 i := by
  have h : ((residueToken r : Fin 239) : ℕ) = 22 + (r : ℕ) := rfl
  have hr := r.isLt
  rw [solverOut, ite_eq_left ⟨by omega, by omega⟩, h]
  congr 2
  omega

/-- The readout of a token that is not a residue's is zero. -/
lemma solverOut_of_not_residue {tok : Fin 239} (h : ∀ r, residueToken r ≠ tok) (i : Fin 128) :
    solverOut tok i = 0 := by
  refine ite_eq_right fun hc => h ⟨(tok : ℕ) - 22, by omega⟩ (Fin.ext ?_)
  change 22 + ((tok : ℕ) - 22) = tok
  omega

example : solver (.gain1 0 0) = 1 := rfl

example : solverEmb (residueToken 3) 0 = logFeature (toField 3) :=
  (solverEmb_residueToken 3 0).trans (by rw [ind_of_eq (k := 0) (i := (0 : Fin 128)) rfl, one_mul])

example := solverOut_of_not_residue (tok := eosToken) residueToken_ne_eosToken 0

end Transformer.Grokking.Protocol
