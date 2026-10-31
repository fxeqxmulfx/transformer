module

public import Transformer.Rotate90.Power.Geometry
public import Transformer.Rotate90.Products

/-!
# Ten explicit products in twenty genuine SwiGLU units

Source: `review/flat_spectrum.md`, *Actual network expressivity for every exponent*,
weight construction. Four products multiply the powered complex operand codes.
Six more multiply a's nonconstant coordinates by the constant input coordinate,
so they can cancel the original input and route the zero flag. Every product uses
two opposed SiLU gates through the existing exact compiler. No activation is
replaced by multiplication in the actual forward definition.

All gate, value and output entries are independent of the operand pair and target.
Only the shared embedding depends on the task exponent. The output coefficients
compensate the two input RMSNorm scales, including their positive epsilon terms.

Writing χ(a^k)=x+iy and χ(b^k)=u+iv, the first four products form
Re χ((a/b)^k)=xu+yv and Im χ((a/b)^k)=yu-xv. The final product
uses the stored zero flag of a^k, so the same matrices handle exponent zero
and zero numerators. Faithfulness is needed only for the final decoding theorem.
-/

@[expose] public section

namespace Transformer.Rotate90.Power

open Landscape.Neural
open scoped BigOperators

variable {K Z : Type*} [Fintype K] [Fintype Z]

/-- Input gate coordinates: four complex-product terms and six constant gates. -/
def gateCoord : Fin 10 → Channels (Fin 7) :=
  ![(4, 0), (5, 0), (5, 0), (4, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]

/-- Input value coordinates for those ten products. -/
def valueCoord : Fin 10 → Channels (Fin 7) :=
  ![(4, 1), (5, 1), (4, 1), (5, 1), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0)]

/-- A sparse trainable matrix selecting actual residual coordinates. -/
noncomputable def coordinateProjection (c : Fin 10 → Channels (Fin 7)) :
    Channels (PairIndex K) → Fin 10 → ℝ := by
  classical
  exact fun i t => if i = (.inl (c t).1, (c t).2) then 1 else 0

/-- An output direction supported on one active pair; padding remains zero. -/
def pairShape (i : Fin 7) (e o : ℝ) : Channels (PairIndex K) → ℝ :=
  pack fun j => if j = i then ![e, o] else 0

/-- Sparse coefficients for the ten products. The last product both supplies the
target zero flag and cancels the powered-code zero coordinate in the readout. -/
def outputShape (t : Fin 10) : Channels (PairIndex K) → ℝ :=
  ![pairShape 1 1 1, pairShape 1 1 1, pairShape 2 1 1, pairShape 2 (-1) (-1),
    pairShape 1 (-1) 1, pairShape 2 (-1) 1, pairShape 3 (-1) 1,
    pairShape 4 (-1) 1, pairShape 5 (-1) 1,
    fun i => pairShape 3 1 1 i + pairShape 6 (-1) 1 i] t

omit [Fintype Z] in
/-- Multiplying by the sparse projection returns exactly its selected coordinate. -/
theorem dense_coordinateProjection (c : Fin 10 → Channels (Fin 7))
    (r : Channels (PairIndex K) → ℝ) (t : Fin 10) :
    dense (coordinateProjection c) r t = r (.inl (c t).1, (c t).2) := by
  classical
  simp [dense, coordinateProjection]

variable {p : ℕ} [Fact p.Prime]

/-- **Explicit actual weights** for a fixed task exponent. The embedding is tied,
the branch matrices are ordinary real matrices, and all optional padding is zero. -/
noncomputable def weights (eps : ℝ) (χ : MulChar (ZMod p) ℂ) (k : ℕ) :
    GatedWeights (ZMod p) (PairIndex K) ((Fin 10 × Fin 2) ⊕ Z) :=
  compiledWeights (embedding χ k) (coordinateProjection gateCoord)
    (coordinateProjection valueCoord)
    (fun t i => ((branchScale (K := K) eps) ^ 2)⁻¹ * outputShape t i)

omit [Fintype Z] in
/-- The compiled weight vector uses precisely the prescribed single embedding. -/
theorem embedding_weights (eps : ℝ) (χ : MulChar (ZMod p) ℂ) (k : ℕ) :
    gatedEmbedding (weights (K := K) (Z := Z) eps χ k) = embedding χ k := by
  rfl

/-- Two scaled inputs and the compensating output coefficient give an unscaled
product. This is the normalization compensation used by every active hidden pair. -/
theorem scaled_product {s : ℝ} (hs : s ≠ 0) (x y d : ℝ) :
    (s * x) * (s * y) * ((s ^ 2)⁻¹ * d) = x * y * d := by
  field_simp

/-- **The actual SwiGLU branch realizes the powered-division construction.** This
derives every target-dependent coordinate from the ten products and the character
identity. Positive epsilon permits exact compensation for both input normalizers. -/
theorem branch_weights {eps : ℝ} (heps : 0 < eps)
    (χ : MulChar (ZMod p) ℂ) (k : ℕ) (a : ZMod p) {b : ZMod p} (hb : b ≠ 0) :
    branch (weights (K := K) (Z := Z) eps χ k)
      (fun i => branchScale (K := K) eps * raw χ k a b i) = branchVector χ k a b := by
  have hs : branchScale (K := K) eps ≠ 0 := (branchScale_pos heps).ne'
  funext ⟨i, j⟩
  rw [weights, branch_compiledWeights]
  simp only [dense_coordinateProjection]
  simp_rw [scaled_product hs]
  rcases i with i | i
  · fin_cases i <;> fin_cases j <;>
      simp [-map_pow, Fin.sum_univ_succ, gateCoord, valueCoord, outputShape, pairShape, raw, pack,
        branchVector, character_div_pow, zeroFlag_div_pow a hb, RCLike.star_def] <;>
      ring
  · simp [Fin.sum_univ_succ, outputShape, pairShape, pack, branchVector]

/-- The construction uses twenty active hidden units at every exponent. -/
theorem active_hidden_count : Fintype.card (Fin 10 × Fin 2) = 20 := by
  norm_num

omit [Fintype Z] in
/-- Gate selections remain unchanged when the exponent changes: only lookup entries
carry the exponent, which is fixed for a task rather than supplied as another input. -/
theorem gateMatrix_weights_eq (eps : ℝ) (χ : MulChar (ZMod p) ℂ) (k l : ℕ) :
    gateMatrix (weights (K := K) (Z := Z) eps χ k) = gateMatrix (weights eps χ l) := by
  funext i j
  rcases j with j | j <;>
    simp only [gateMatrix, weights, compiledWeights, WithLp.ofLp_toLp]

omit [Fintype Z] in
/-- The value matrix also has no dependence on the task exponent. -/
theorem valueMatrix_weights_eq (eps : ℝ) (χ : MulChar (ZMod p) ℂ) (k l : ℕ) :
    valueMatrix (weights (K := K) (Z := Z) eps χ k) = valueMatrix (weights eps χ l) := by
  funext i j
  rcases j with j | j <;>
    simp only [valueMatrix, weights, compiledWeights, WithLp.ofLp_toLp]

omit [Fintype Z] in
/-- The output matrix also has no dependence on the task exponent. -/
theorem outputMatrix_weights_eq (eps : ℝ) (χ : MulChar (ZMod p) ℂ) (k l : ℕ) :
    outputMatrix (weights (K := K) (Z := Z) eps χ k) = outputMatrix (weights eps χ l) := by
  funext j i
  rcases j with j | j <;>
    simp only [outputMatrix, weights, compiledWeights, WithLp.ofLp_toLp]

example := dense_coordinateProjection (K := Fin 1) gateCoord
  (fun i => if i.2 = 0 then 2 else 3) 0
example := embedding_weights (K := Fin 1) (Z := Fin 23) (p := 3) (1 / 8388608) 1 2
example := scaled_product (s := 2) (by norm_num) 3 4 5
example := branch_weights (K := Fin 1) (Z := Fin 23) (p := 3)
  (eps := 1 / 8388608) (by norm_num) 1 0 0 (b := 2) (by decide)
example := branch_weights (K := Fin 1) (Z := Fin 23) (p := 3)
  (eps := 1 / 8388608) (by norm_num) 1 3 0 (b := 2) (by decide)
example := gateMatrix_weights_eq (K := Fin 1) (Z := Fin 23) (p := 3) (1 / 8388608) 1 0 3
example := valueMatrix_weights_eq (K := Fin 1) (Z := Fin 23) (p := 3) (1 / 8388608) 1 0 3
example := outputMatrix_weights_eq (K := Fin 1) (Z := Fin 23) (p := 3) (1 / 8388608) 1 0 3

end Transformer.Rotate90.Power
