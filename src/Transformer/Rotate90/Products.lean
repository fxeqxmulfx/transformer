module

public import Transformer.Rotate90.Basic

/-!
# Explicit SwiGLU matrices for the division construction

Source: `rotate90-swiglu-model.md`, How W_gate | W_value works, and the
requested exact existence theorem. The seven products are r2*r3, r4*r5,
r4*r3, r2*r5, r0*r2, r0*r4 and r0*r6. Each uses two opposed SiLU gates,
so fourteen hidden units suffice. Padding units remain part of the model
with zero weights. Coefficients depend only on epsilon and width, never
on the input pair or its target.
-/

@[expose] public section

open scoped BigOperators

namespace Transformer.Rotate90

open Landscape.Neural

section Compiler

variable {V K T Z : Type*} [Fintype K] [Fintype T] [Fintype Z]

/-- Two opposed gates per product, with optional unused hidden units. All entries
are actual free coordinates of E, W_gate, W_value and W_out. Source: Model and
`Landscape.Neural.Products`' exact SiLU identity. -/
noncomputable def compiledWeights (E : V → Channels K → ℝ)
    (g u : Channels K → T → ℝ) (d : T → Channels K → ℝ) :
    GatedWeights V K ((T × Fin 2) ⊕ Z) :=
  WithLp.toLp 2 fun coordinate => match coordinate with
    | .inl (q, i) => E q i
    | .inr (.inl (i, .inl j)) => pairedGate g i j
    | .inr (.inl (_, .inr _)) => 0
    | .inr (.inr (.inl (i, .inl j))) => pairedValue u i j
    | .inr (.inr (.inl (_, .inr _))) => 0
    | .inr (.inr (.inr (.inl j, i))) => pairedOutput d j i
    | .inr (.inr (.inr (.inr _, _))) => 0

omit [Fintype K] [Fintype T] [Fintype Z] in
/-- The compiler retains the shared embedding, including every padding channel. -/
theorem embedding_compiledWeights (E : V → Channels K → ℝ)
    (g u : Channels K → T → ℝ) (d : T → Channels K → ℝ) :
    gatedEmbedding (compiledWeights (Z := Z) E g u d) = E := by
  rfl

/-- The actual current branch realizes the specified quadratic expression exactly.
Source: `Rotate90.forward`; `silu(t) - silu(-t) = t`, without an approximation. -/
theorem branch_compiledWeights (E : V → Channels K → ℝ)
    (g u : Channels K → T → ℝ) (d : T → Channels K → ℝ)
    (r : Channels K → ℝ) (i : Channels K) :
    branch (compiledWeights (Z := Z) E g u d) r i =
      ∑ t, dense g r t * dense u r t * d t i := by
  unfold branch
  change (∑ j : (T × Fin 2) ⊕ Z, _ * _) = _
  rw [Fintype.sum_sum_type]
  simp only [outputMatrix, compiledWeights, WithLp.ofLp_toLp,
    mul_zero, Finset.sum_const_zero, add_zero]
  exact pairedBranch_eq_products g u d r i

/-- The input, branch-input and branch-output normalizations all remain in the
compiled residual; `logit` applies the fourth, final normalization. -/
theorem state_compiledWeights (eps : ℝ) (E : V → Channels K → ℝ)
    (g u : Channels K → T → ℝ) (d : T → Channels K → ℝ) (a b : V) :
    state eps (compiledWeights (Z := Z) E g u d) a b =
      let x := rmsNorm eps (tiedInput E a b)
      fun i => x i + rmsNorm eps (fun j => ∑ t,
        dense g (rmsNorm eps x) t * dense u (rmsNorm eps x) t * d t j) i := by
  have hb (r : Channels K → ℝ) : branch (compiledWeights (Z := Z) E g u d) r =
      fun i => ∑ t, dense g r t * dense u r t * d t i :=
    funext (branch_compiledWeights E g u d r)
  unfold state input
  rw [embedding_compiledWeights]
  simp only [hb]

example : silu (2 : ℝ) - silu (-2) = 2 := silu_sub_neg 2

end Compiler

variable {K Z : Type*} [Fintype K] [Fintype Z]

/-- The gate selects one of the actual eight active input coordinates. -/
def gateCoord : Fin 7 → Channels (Fin 4) :=
  ![(1, 0), (2, 0), (2, 0), (1, 0), (0, 0), (0, 0), (0, 0)]

/-- The independently selected value coordinate for each product. -/
def valueCoord : Fin 7 → Channels (Fin 4) :=
  ![(1, 1), (2, 1), (1, 1), (2, 1), (1, 0), (2, 0), (3, 0)]

/-- A genuine sparse matrix selecting those coordinates, with no input dependence. -/
noncomputable def coordinateProjection (c : Fin 7 → Channels (Fin 4)) :
    Channels (PairIndex K) → Fin 7 → ℝ := by
  classical
  exact fun i t => if i = (.inl (c t).1, (c t).2) then 1 else 0

/-- Output coefficients of the seven products, in all actual residual coordinates. -/
def outputShape (t : Fin 7) : Channels (PairIndex K) → ℝ :=
  pack (![![![0, 0], ![1, 1], ![0, 0], ![0, 0]],
    ![![0, 0], ![1, 1], ![0, 0], ![0, 0]],
    ![![0, 0], ![0, 0], ![1, 1], ![0, 0]],
    ![![0, 0], ![0, 0], ![-1, -1], ![0, 0]],
    ![![0, 0], ![-1, 1], ![0, 0], ![0, 0]],
    ![![0, 0], ![0, 0], ![-1, 1], ![0, 0]],
    ![![0, 0], ![0, 0], ![0, 0], ![0, 2]]] t)

/-- Multiplying the sparse matrix selects exactly the prescribed channel. -/
theorem dense_coordinateProjection (c : Fin 7 → Channels (Fin 4))
    (r : Channels (PairIndex K) → ℝ) (t : Fin 7) :
    dense (coordinateProjection c) r t = r (.inl (c t).1, (c t).2) := by
  classical
  simp [dense, coordinateProjection]

variable {p : ℕ} [Fact p.Prime]

/-- All free parameters of the constructive solution. The single embedding
is shared; all three branch matrices are finite real matrices. -/
noncomputable def divisionWeights (eps : ℝ) (χ : MulChar (ZMod p) ℂ) :
    GatedWeights (ZMod p) (PairIndex K) ((Fin 7 × Fin 2) ⊕ Z) :=
  compiledWeights (embedding χ) (coordinateProjection gateCoord)
    (coordinateProjection valueCoord)
    (fun t i => ((branchScale (K := K) eps) ^ 2)⁻¹ * outputShape t i)

omit [Fintype Z] in
/-- The weights use the asserted single embedding for lookup and decoding. -/
theorem embedding_divisionWeights (eps : ℝ) (χ : MulChar (ZMod p) ℂ) :
    gatedEmbedding (divisionWeights (K := K) (Z := Z) eps χ) = embedding χ := by
  rfl

/-- Fourteen real SwiGLU channels compute the target branch exactly after
both input RMSNorms. All dependence on a/b is derived from character division. -/
theorem branch_divisionWeights {eps : ℝ} (heps : 0 < eps)
    (χ : MulChar (ZMod p) ℂ) (a b : ZMod p) :
    branch (divisionWeights (K := K) (Z := Z) eps χ)
      (fun i => branchScale (K := K) eps * raw χ a b i) = branchVector χ a b := by
  have hs : branchScale (K := K) eps ≠ 0 := (branchScale_pos heps).ne'
  funext ⟨k, j⟩
  rw [divisionWeights, branch_compiledWeights]
  simp only [dense_coordinateProjection]
  rcases k with k | k
  · fin_cases k <;> fin_cases j <;>
      simp [Fin.sum_univ_succ, gateCoord, valueCoord, outputShape, raw, pack,
        branchVector, character_div, RCLike.star_def] <;> field_simp <;> ring
  · simp [outputShape, pack, branchVector]

example := dense_coordinateProjection (K := Fin 0) gateCoord
  (fun i => if i.2 = 0 then 2 else 3) 0

example := embedding_divisionWeights (K := Fin 4) (Z := Fin 29) (p := 3) (1 / 8388608) 1

example := branch_divisionWeights (K := Fin 4) (Z := Fin 29) (p := 3)
  (eps := 1 / 8388608) (by norm_num) 1 0 2

end Transformer.Rotate90
