/-
# The lookup head is an ordinary attention head

Every softmax bound in this development is stated of the sum written out by
hand,

    ∑ⱼ (exp (β · score q (K j)) / ∑ₖ exp (β · score q (K k))) • V j,

and every docstring calls it "a standard softmax head".  Nothing said so.
`Transformer.XSA.SAOutput` is the causal self-attention of the formalized
papers — `Q, K, V : ParamMatrix d` applied to a residual stream, masked to the
prefix — and the two were never identified.

They are the same operator, and this file exhibits the weights.  What the
theorem must not claim is that `K` is the paraboloid lift: the lift is
quadratic and `K` is linear.  The honest statement, and the one the machine
realizes, is that the lifted coordinates are already in the residual stream —
written there upstream, as `PersistDimension` writes them in the VM — and that
`Q` and `K` are coordinate projections reading two blocks of it.  The
nonlinearity of lookup lives above the head, not in it.

So the residual stream at a position carries `(q, 1)` in its first block and
`(2k, -‖k‖²)` in its second; `queryProj` reads the first, `keyProj` moves the
second into the first block's coordinates, and their inner product is exactly
`score`.  The causal mask is definitional: `Idx n` is the stored prefix, the
decoding position `i` is the last of it, and the query token's own slot is a
stored key like any other — the machine is append-only.

`head_output_at_index` (`Transformer.ALM.SoftmaxValue`) then becomes a bound on
standard attention rather than on a hand-written sum: a lookup head and a
trained head are one operator at different weights, which is what licenses a
hybrid layer at all.

Source: Percepta, *Can LLMs Be Computers?* (2026-03-11), §"the geometric fast
path is not limited to our executor construction"; the head itself is
`transformer_vm/attention/hull2d_cht.h`, lines 203-215.
-/

import Transformer.ALM.SoftmaxValue
import Transformer.XSA

open scoped BigOperators

namespace Transformer
namespace ALM

variable {m n a b : ℕ}

/-! ### The two blocks of the residual stream -/

/-- The concatenation of two blocks of the residual stream. -/
noncomputable def packBlocks (u : EucSpace a) (v : EucSpace b) : EucSpace (a + b) :=
  WithLp.toLp 2 (Fin.append (WithLp.ofLp u) (WithLp.ofLp v))

/-- Concatenation is an isometry of the inner product, block by block. -/
theorem inner_packBlocks (u u' : EucSpace a) (v v' : EucSpace b) :
    inner (𝕜 := ℝ) (packBlocks u v) (packBlocks u' v')
      = inner (𝕜 := ℝ) u u' + inner (𝕜 := ℝ) v v' := by
  simp [packBlocks, PiLp.inner_apply, Fin.sum_univ_add]

/-- Reading the first block: linear, being a choice of coordinates. -/
noncomputable def fstBlockLin : EucSpace (a + b) →ₗ[ℝ] EucSpace a where
  toFun x := WithLp.toLp 2 (fun i => x (Fin.castAdd b i))
  map_add' x y := by ext i; simp
  map_smul' c x := by ext i; simp

/-- Reading the second block. -/
noncomputable def sndBlockLin : EucSpace (a + b) →ₗ[ℝ] EucSpace b where
  toFun x := WithLp.toLp 2 (fun i => x (Fin.natAdd a i))
  map_add' x y := by ext i; simp
  map_smul' c x := by ext i; simp

/-- Writing a block back, into the first block's coordinates. -/
noncomputable def packLeftLin : EucSpace a →ₗ[ℝ] EucSpace (a + b) where
  toFun u := packBlocks u (0 : EucSpace b)
  map_add' u u' := by
    ext i
    refine Fin.addCases ?_ ?_ i <;> intro j <;> simp [packBlocks]
  map_smul' c u := by
    ext i
    refine Fin.addCases ?_ ?_ i <;> intro j <;> simp [packBlocks]

@[simp] theorem fstBlockLin_packBlocks (u : EucSpace a) (v : EucSpace b) :
    fstBlockLin (packBlocks u v) = u := by
  ext i; simp [fstBlockLin, packBlocks]

@[simp] theorem sndBlockLin_packBlocks (u : EucSpace a) (v : EucSpace b) :
    sndBlockLin (packBlocks u v) = v := by
  ext i; simp [sndBlockLin, packBlocks]

/-! ### The lift, in the residual stream rather than in the head -/

/-- The query lift `q ↦ (q, 1)`, as a vector of the residual stream. -/
noncomputable def liftQueryVec (q : EucSpace m) : EucSpace (m + 1) :=
  WithLp.toLp 2 (Fin.snoc (WithLp.ofLp q) 1)

/-- The paraboloid lift `k ↦ (2k, -‖k‖²)`, as a vector of the residual
stream.  It is quadratic in `k`; nothing below applies a head to it. -/
noncomputable def liftKeyVec (k : EucSpace m) : EucSpace (m + 1) :=
  WithLp.toLp 2 (Fin.snoc (fun i => 2 * k i) (-‖k‖ ^ 2))

/-- **The lift is what makes the score an inner product.**  This is
`Transformer.ALM.Defs` read in coordinates: the score of the paraboloid
embedding is the plain inner product of the two lifted vectors. -/
theorem inner_liftQueryVec_liftKeyVec (q k : EucSpace m) :
    inner (𝕜 := ℝ) (liftQueryVec q) (liftKeyVec k) = score q k := by
  rw [PiLp.inner_apply]
  simp [liftQueryVec, liftKeyVec, Fin.sum_univ_castSucc, score, PiLp.inner_apply, mul_comm]
  rw [Finset.mul_sum]
  ring_nf

/-- The residual stream at one position: the query lift in the first block,
the key lift in the second.  Both are written upstream of the head. -/
noncomputable def residual (q k : EucSpace m) : EucSpace ((m + 1) + (m + 1)) :=
  packBlocks (liftQueryVec q) (liftKeyVec k)

/-! ### And the head is then two coordinate projections -/

/-- The query matrix: read the first block.  A coordinate projection, and in
particular linear. -/
noncomputable def queryProj : ParamMatrix ((m + 1) + (m + 1)) :=
  LinearMap.toContinuousLinearMap (packLeftLin.comp fstBlockLin)

/-- The key matrix: read the second block, into the first block's coordinates
so that the two meet under the inner product. -/
noncomputable def keyProj : ParamMatrix ((m + 1) + (m + 1)) :=
  LinearMap.toContinuousLinearMap (packLeftLin.comp sndBlockLin)

/-- **The attention logit of these weights is the lookup score.**  No
hypothesis on the residual stream beyond its being the one the lifts write. -/
theorem inner_queryProj_keyProj (q k q' k' : EucSpace m) :
    inner (𝕜 := ℝ) (queryProj (residual q k)) (keyProj (residual q' k')) = score q k' := by
  show inner (𝕜 := ℝ) (packLeftLin (fstBlockLin (residual q k)))
    (packLeftLin (sndBlockLin (residual q' k'))) = _
  show inner (𝕜 := ℝ) (packBlocks (fstBlockLin (residual q k)) (0 : EucSpace (m + 1)))
    (packBlocks (sndBlockLin (residual q' k')) (0 : EucSpace (m + 1))) = _
  rw [inner_packBlocks]
  simp [residual, inner_liftQueryVec_liftKeyVec]

/-! ### So standard causal attention is the lookup head -/

/-- **The lookup head is an ordinary attention head.**  At the decoding
position `i` — the last of the stored prefix, so that the causal mask admits
every slot — causal self-attention with `Q = β · queryProj`, `K = keyProj` and
any value matrix returns exactly the softmax head this development bounds:

    ∑ⱼ (exp (β · score qᵢ kⱼ) / ∑ₗ exp (β · score qᵢ kₗ)) • V xⱼ.

The inverse temperature is the scale of `Q`, as it is in the machine. -/
theorem SAOutput_eq_softmax_head [Nonempty (Idx n)] (β : ℝ) (q k : Idx n → EucSpace m)
    (Vm : ParamMatrix ((m + 1) + (m + 1))) (i : Idx n) (hi : ∀ j : Idx n, (j : ℕ) ≤ (i : ℕ)) :
    XSA.SAOutput ((m + 1) + (m + 1)) n (β • queryProj) keyProj Vm
        (fun j => residual (q j) (k j)) i
      = ∑ j, (Real.exp (β * score (q i) (k j)) / ∑ l, Real.exp (β * score (q i) (k l)))
          • Vm (residual (q j) (k j)) := by
  have hlogit : ∀ j : Idx n,
      inner (𝕜 := ℝ) ((β • queryProj) (residual (q i) (k i)))
          (keyProj (residual (q j) (k j)))
        = β * score (q i) (k j) := by
    intro j
    rw [smul_apply, real_inner_smul_left, inner_queryProj_keyProj]
  have hZ : ∀ j : Idx n, (if (j : ℕ) ≤ (i : ℕ) then
      Real.exp (inner (𝕜 := ℝ) ((β • queryProj) (residual (q i) (k i)))
        (keyProj (residual (q j) (k j)))) else 0)
      = Real.exp (β * score (q i) (k j)) := fun j => by rw [if_pos (hi j), hlogit j]
  have hnum : ∀ j : Idx n, (if (j : ℕ) ≤ (i : ℕ) then
      Real.exp (inner (𝕜 := ℝ) ((β • queryProj) (residual (q i) (k i)))
        (keyProj (residual (q j) (k j)))) • Vm (residual (q j) (k j)) else 0)
      = Real.exp (β * score (q i) (k j)) • Vm (residual (q j) (k j)) :=
    fun j => by rw [if_pos (hi j), hlogit j]
  simp only [XSA.SAOutput]
  rw [Finset.sum_congr rfl fun j _ => hZ j, Finset.sum_congr rfl fun j _ => hnum j,
    Finset.smul_sum]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [smul_smul, div_eq_inv_mul]

/-- The hypothesis is satisfiable, and it is the decoding position: over a
prefix of one slot, that slot is the last one. -/
example : ∀ j : Idx 1, (j : ℕ) ≤ ((0 : Idx 1) : ℕ) := by decide

end ALM
end Transformer
