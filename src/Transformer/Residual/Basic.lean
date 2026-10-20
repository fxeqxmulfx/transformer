module

public import Mathlib.Analysis.Normed.Module.Basic
public import Mathlib.Data.Matrix.Mul
public import Mathlib.Data.Matrix.Diagonal
public import Mathlib.Tactic

/-!
# Residual streams and weighted aggregation

Shared algebra for arXiv:2512.24880v2, §1, Eqs. (1)–(4), and
arXiv:2603.15031v1, §§2–3. Features may be vectors in any real normed
space. Stream indices and feature coordinates are distinct types.
-/

@[expose] public section

open scoped BigOperators

namespace Transformer.Residual

variable {ι : Type*} [Fintype ι]
variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- Aggregate features with scalar weights. Sources: arXiv:2512.24880v2,
Eq. (3), and arXiv:2603.15031v1, Eq. (4). -/
noncomputable def weightedSum (w : ι → ℝ) (x : ι → E) : E :=
  ∑ j, w j • x j

/-- Mix the stream axis without changing the feature space.
Source: arXiv:2512.24880v2, Eq. (3), `H_res x`. -/
noncomputable def mix (H : Matrix ι ι ℝ) (x : ι → E) : ι → E :=
  fun i => weightedSum (H i) x

/-- Read one feature vector, transform it, then write it into all streams.
Source: arXiv:2512.24880v2, Eq. (3). Coefficients are supplied at the
current token; input-dependent predictors can be substituted directly. -/
noncomputable def hyperStep (pre post : ι → ℝ) (H : Matrix ι ι ℝ)
    (F : E → E) (x : ι → E) : ι → E :=
  fun i => mix H x i + post i • F (weightedSum pre x)

/-- Standard residual recursion, arXiv:2512.24880v2, Eq. (1).
`F` includes any normalization belonging to the branch. -/
noncomputable def residualRun (F : ℕ → E → E) (x : E) : ℕ → E
  | 0 => x
  | t + 1 => residualRun F x t + F t (residualRun F x t)

omit [NormedSpace ℝ E] in
/-- Unroll standard residuals: every preceding branch has coefficient one.
Source: arXiv:2512.24880v2, Eq. (2), with the initial depth set to zero. -/
theorem residualRun_unroll (F : ℕ → E → E) (x : E) (t : ℕ) :
    residualRun F x t = x + ∑ k ∈ Finset.range t, F k (residualRun F x k) := by
  induction t with
  | zero => simp [residualRun]
  | succ t ih =>
    rw [residualRun, Finset.sum_range_succ, ← add_assoc, ← ih]

/-- Summing equal features scales by total weight; normalization is explicit.
Sources: arXiv:2512.24880v2, §4.1; arXiv:2603.15031v1, §3.1. -/
theorem weightedSum_const (w : ι → ℝ) (v : E) :
    weightedSum w (fun _ => v) = (∑ j, w j) • v := by
  unfold weightedSum
  rw [Finset.sum_smul]

/-- Weighting commutes with feature addition. Auxiliary aggregation algebra
for arXiv:2512.24880v2, Eq. (4). -/
theorem weightedSum_add (w : ι → ℝ) (x y : ι → E) :
    weightedSum w (fun j => x j + y j) = weightedSum w x + weightedSum w y := by
  simp only [weightedSum, smul_add, Finset.sum_add_distrib]

/-- Weighting commutes with feature subtraction. This permits stability
estimates for fixed coefficients, rather than for a learned gate predictor. -/
theorem weightedSum_sub (w : ι → ℝ) (x y : ι → E) :
    weightedSum w (fun j => x j - y j) = weightedSum w x - weightedSum w y := by
  simp only [weightedSum, smul_sub, Finset.sum_sub_distrib]

/-- A positive weighted aggregate is bounded by total weight times the
largest allowed feature norm. This does not bound an arbitrary branch `F`.
Sources: arXiv:2512.24880v2, §4.1; arXiv:2603.15031v1, §3.1. -/
theorem weightedSum_norm_le (w : ι → ℝ) (x : ι → E) (R : ℝ)
    (hw : ∀ j, 0 ≤ w j) (hx : ∀ j, ‖x j‖ ≤ R) :
    ‖weightedSum w x‖ ≤ (∑ j, w j) * R := by
  calc
    ‖weightedSum w x‖ ≤ ∑ j, ‖w j • x j‖ := norm_sum_le _ _
    _ = ∑ j, w j * ‖x j‖ := by
      congr 1
      ext j
      rw [norm_smul, Real.norm_of_nonneg (hw j)]
    _ ≤ ∑ j, w j * R :=
      Finset.sum_le_sum fun j _ => mul_le_mul_of_nonneg_left (hx j) (hw j)
    _ = (∑ j, w j) * R := (Finset.sum_mul _ _ _).symm

/-- The norm-bound hypotheses hold for positive weights and unit features. -/
example : (∀ _ : Fin 4, (0 : ℝ) ≤ 1 / 4) ∧
    (∀ j : Fin 4, ‖(fun _ : Fin 4 => (1 : ℝ)) j‖ ≤ 1) := by
  constructor <;> intro j <;> norm_num

/-- Convex aggregation preserves any common norm bound.
Sources: arXiv:2512.24880v2, §4.1; arXiv:2603.15031v1, §3.1. -/
theorem weightedSum_convex_norm_le (w : ι → ℝ) (x : ι → E) (R : ℝ)
    (hw : ∀ j, 0 ≤ w j) (hs : ∑ j, w j = 1) (hx : ∀ j, ‖x j‖ ≤ R) :
    ‖weightedSum w x‖ ≤ R := by
  calc
    ‖weightedSum w x‖ ≤ (∑ j, w j) * R := weightedSum_norm_le w x R hw hx
    _ = R := by rw [hs, one_mul]

/-- The convex norm-bound hypotheses hold for a nonzero four-stream state. -/
example : (∀ _ : Fin 4, (0 : ℝ) ≤ 1 / 4) ∧
    (∑ _ : Fin 4, (1 / 4 : ℝ)) = 1 ∧
    (∀ j : Fin 4, ‖(fun _ : Fin 4 => (1 : ℝ)) j‖ ≤ 1) := by
  norm_num

/-- A row-normalized mixer preserves the subspace of identical streams.
This is the precise preserved identity mode in arXiv:2512.24880v2, §4.1. -/
theorem mix_const (H : Matrix ι ι ℝ) (v : E)
    (hH : ∀ i, ∑ j, H i j = 1) : mix H (fun _ => v) = fun _ => v := by
  funext i
  change weightedSum (H i) (fun _ => v) = v
  rw [weightedSum_const, hH, one_smul]

/-- The preservation hypothesis holds for uniform mixing of four streams. -/
example : ∀ i : Fin 4, ∑ j : Fin 4, (fun _ _ : Fin 4 => (1 / 4 : ℝ)) i j = 1 := by
  intro i
  norm_num

/-- Matrix identity mixing leaves every stream unchanged. Auxiliary algebra
for the single-stream reduction in arXiv:2512.24880v2, §4.1. -/
theorem mix_one [DecidableEq ι] (x : ι → E) : mix 1 x = x := by
  funext i
  simp [mix, weightedSum, Matrix.one_apply]

/-- Composing stream mixers multiplies their matrices in execution order.
This connects the operator on actual feature vectors with the residual
matrix product in arXiv:2512.24880v2, Eq. (4). -/
theorem mix_mul (A B : Matrix ι ι ℝ) (x : ι → E) :
    mix (A * B) x = mix A (mix B x) := by
  funext i
  simp only [mix, weightedSum, Matrix.mul_apply, Finset.sum_smul,
    Finset.smul_sum, ← mul_smul]
  rw [Finset.sum_comm]

/-- Summing the updated streams retains both the skip contribution and the
write-back contribution. Source: arXiv:2512.24880v2, Eq. (3). -/
theorem sum_hyperStep (pre post : ι → ℝ) (H : Matrix ι ι ℝ)
    (F : E → E) (x : ι → E) :
    ∑ i, hyperStep pre post H F x i =
      (∑ i, mix H x i) + (∑ i, post i) • F (weightedSum pre x) := by
  simp only [hyperStep, Finset.sum_add_distrib, Finset.sum_smul]

/-- A standard residual block is a special case with one stream and all
three coefficients equal to one. This is a specialization of the HC
operator, not a claim that learned sigmoid gates must equal one.
Sources: arXiv:2409.19606v3, §3.1; arXiv:2512.24880v2, Eqs. (1), (3). -/
theorem hyperStep_standard (F : E → E) (x : E) :
    hyperStep (fun _ : Fin 1 => 1) (fun _ => 1) 1 F (fun _ => x) 0 = x + F x := by
  rw [hyperStep, mix_one]
  simp only [weightedSum, Fin.sum_univ_one, one_smul]

end Transformer.Residual
