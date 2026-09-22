/-
# Homogenized Transformers — the entries of a Gaussian head

Assumption (G) of arXiv:2604.01978v1 draws the value matrix `V` and the two
factors of `A = W W'ᵀ` with independent centered Gaussian entries.  This file
reads off what consequence (i) of (G) uses:

* a centered Gaussian of variance `v` is square-integrable, with mean `0` and
  second moment `v` (`memLp_two_of_map_eq_gaussianReal`,
  `integral_eq_zero_of_map_eq_gaussianReal`,
  `integral_sq_of_map_eq_gaussianReal`);
* the entries of `V` are therefore uncorrelated with variance `σ_V²`
  (`integral_value_mul_value`), and their products are integrable
  (`integrable_value_mul_value`);
* `V` is independent of `A = W W'ᵀ` (`indepFun_value_qk`): the entries of `V`
  and those of `(W, W')` are disjoint blocks of one independent family.

Source: arXiv:2604.01978v1, `eq: tformers.at.initialization`.
-/

import Transformer.Homogenized.GaussianInit
import Mathlib.Analysis.Matrix.MeasurableSpace

open scoped BigOperators NNReal
open MeasureTheory ProbabilityTheory

namespace Transformer
namespace Homogenized

variable {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω}

/-! ### A centered Gaussian coordinate -/

/-- A random variable with law `𝒩(0, v)` is square-integrable. -/
theorem memLp_two_of_map_eq_gaussianReal {X : Ω → ℝ} (hX : Measurable X) {v : ℝ≥0}
    (h : P.map X = gaussianReal 0 v) : MemLp X 2 P := by
  have hid := memLp_id_gaussianReal (μ := 0) (v := v) 2
  rw [← h] at hid
  exact (memLp_map_measure_iff hid.aestronglyMeasurable hX.aemeasurable).1 hid

/-- A random variable with law `𝒩(0, v)` has mean `0`. -/
theorem integral_eq_zero_of_map_eq_gaussianReal {X : Ω → ℝ} (hX : Measurable X) {v : ℝ≥0}
    (h : P.map X = gaussianReal 0 v) : ∫ ω, X ω ∂P = 0 := by
  have hmap : ∫ ω, X ω ∂P = ∫ y, y ∂(P.map X) :=
    (integral_map hX.aemeasurable aestronglyMeasurable_id).symm
  rw [hmap, h]
  simp

/-- A random variable with law `𝒩(0, v)` has second moment `v`. -/
theorem integral_sq_of_map_eq_gaussianReal {X : Ω → ℝ} (hX : Measurable X) {v : ℝ≥0}
    (h : P.map X = gaussianReal 0 v) : ∫ ω, X ω ^ 2 ∂P = v := by
  have hmap : ∫ ω, X ω ^ 2 ∂P = ∫ y, y ^ 2 ∂(P.map X) :=
    (integral_map hX.aemeasurable (continuous_pow 2).aestronglyMeasurable).symm
  have hvar := variance_of_integral_eq_zero (μ := gaussianReal 0 v) aemeasurable_id (by simp)
  rw [variance_id_gaussianReal] at hvar
  rw [hmap, h]
  exact hvar.symm

/-- The hypotheses of `memLp_two_of_map_eq_gaussianReal`,
`integral_eq_zero_of_map_eq_gaussianReal` and
`integral_sq_of_map_eq_gaussianReal` are satisfiable: the identity under a
standard Gaussian. -/
example : Measurable (id : ℝ → ℝ) ∧ (gaussianReal 0 1).map id = gaussianReal 0 1 :=
  ⟨measurable_id, Measure.map_id⟩

/-! ### The entries of `(V, W, W')` -/

variable {d : ℕ} {Vr Wr Wr' : Ω → Matrix (Fin d) (Fin d) ℝ}

/-- The entries of `(V, W, W')`, indexed by a block `0, 1, 2` and a position:
the family that assumption (G) declares independent.

Source: arXiv:2604.01978v1, `eq: tformers.at.initialization`. -/
def headEntries (Vr Wr Wr' : Ω → Matrix (Fin d) (Fin d) ℝ) (p : Fin 3 × Fin d × Fin d) :
    Ω → ℝ := fun ω =>
  if p.1 = 0 then Vr ω p.2.1 p.2.2 else if p.1 = 1 then Wr ω p.2.1 p.2.2 else Wr' ω p.2.1 p.2.2

omit [MeasurableSpace Ω] in
/-- Block `0` holds the entries of `V`. -/
theorem headEntries_value (p : Fin d × Fin d) :
    headEntries Vr Wr Wr' (0, p) = fun ω => Vr ω p.1 p.2 := by
  funext ω
  simp [headEntries]

theorem measurable_headEntries (hV : Measurable Vr) (hW : Measurable Wr)
    (hW' : Measurable Wr') (p : Fin 3 × Fin d × Fin d) :
    Measurable (headEntries Vr Wr Wr' p) := by
  unfold headEntries
  split_ifs
  exacts [(hV.eval).eval, (hW.eval).eval, (hW'.eval).eval]

/-- Two distinct entries of `V` are independent. -/
theorem indepFun_value_entries (hind : iIndepFun (headEntries Vr Wr Wr') P)
    {p q : Fin d × Fin d} (hpq : p ≠ q) :
    IndepFun (fun ω => Vr ω p.1 p.2) (fun ω => Vr ω q.1 q.2) P := by
  have h := hind.indepFun (i := ((0 : Fin 3), p)) (j := (0, q)) fun h => hpq (Prod.mk.inj h).2
  rwa [headEntries_value, headEntries_value] at h

/-- **The entries of `V` are uncorrelated with variance `σ_V²`**:
`E[V_p V_q] = σ_V² 1_{p = q}`.

Source: arXiv:2604.01978v1, `eq: tformers.at.initialization`. -/
theorem integral_value_mul_value (hV : Measurable Vr)
    (hind : iIndepFun (headEntries Vr Wr Wr') P) {σV : ℝ≥0}
    (hmV : ∀ i j, P.map (fun ω => Vr ω i j) = gaussianReal 0 (σV ^ 2)) (p q : Fin d × Fin d) :
    ∫ ω, Vr ω p.1 p.2 * Vr ω q.1 q.2 ∂P = if p = q then (σV : ℝ) ^ 2 else 0 := by
  split_ifs with hpq
  · subst hpq
    simp_rw [← sq]
    rw [integral_sq_of_map_eq_gaussianReal (hV.eval.eval) (hmV p.1 p.2)]
    simp
  · rw [(indepFun_value_entries hind hpq).integral_fun_mul_eq_mul_integral
      (hV.eval.eval).aestronglyMeasurable (hV.eval.eval).aestronglyMeasurable,
      integral_eq_zero_of_map_eq_gaussianReal (hV.eval.eval) (hmV p.1 p.2), zero_mul]

/-- The products of two entries of `V` are integrable. -/
theorem integrable_value_mul_value (hV : Measurable Vr) {σV : ℝ≥0}
    (hmV : ∀ i j, P.map (fun ω => Vr ω i j) = gaussianReal 0 (σV ^ 2)) (p q : Fin d × Fin d) :
    Integrable (fun ω => Vr ω p.1 p.2 * Vr ω q.1 q.2) P :=
  (memLp_two_of_map_eq_gaussianReal (hV.eval.eval) (hmV p.1 p.2)).integrable_mul
    (memLp_two_of_map_eq_gaussianReal (hV.eval.eval) (hmV q.1 q.2))

/-- `A = W W'ᵀ` is measurable. -/
theorem measurable_qk (hW : Measurable Wr) (hW' : Measurable Wr') :
    Measurable fun ω => Wr ω * (Wr' ω).transpose := by
  refine Measurable.of_eval fun i => Measurable.of_eval fun j => ?_
  simp only [Matrix.mul_apply, Matrix.transpose_apply]
  exact Finset.measurable_sum _ fun k _ => (hW.eval.eval).mul (hW'.eval.eval)

/-- **`V` is independent of `A = W W'ᵀ`**: `V` is a function of block `0` of the
entries, `A` of blocks `1` and `2`.

Source: arXiv:2604.01978v1, `eq: tformers.at.initialization`. -/
theorem indepFun_value_qk (hV : Measurable Vr) (hW : Measurable Wr)
    (hW' : Measurable Wr') (hind : iIndepFun (headEntries Vr Wr Wr') P) :
    IndepFun Vr (fun ω => Wr ω * (Wr' ω).transpose) P := by
  classical
  set S : Finset (Fin 3 × Fin d × Fin d) := Finset.univ.filter fun p => p.1 = 0
  set T : Finset (Fin 3 × Fin d × Fin d) := Finset.univ.filter fun p => p.1 ≠ 0
  have hST : Disjoint S T := Finset.disjoint_filter.2 fun p _ h h' => h' h
  have h := hind.indepFun_finset S T hST (measurable_headEntries hV hW hW')
  set φ : (S → ℝ) → Matrix (Fin d) (Fin d) ℝ :=
    fun g => Matrix.of fun i j => g ⟨(0, i, j), by simp [S]⟩
  set ψ : (T → ℝ) → Matrix (Fin d) (Fin d) ℝ := fun g =>
    Matrix.of (fun i j => g ⟨(1, i, j), by simp [T]⟩) *
      (Matrix.of fun i j => g ⟨(2, i, j), by simp [T]⟩).transpose
  have hφ : Measurable φ := Continuous.measurable (by fun_prop)
  have hψ : Measurable ψ := Continuous.measurable (by fun_prop)
  convert h.comp hφ hψ using 1
  · funext ω
    ext i j
    simp [φ, headEntries]
  · funext ω
    simp [ψ, headEntries]
    rfl

/-- The hypotheses of `measurable_headEntries`, `indepFun_value_entries`,
`integral_value_mul_value`, `integrable_value_mul_value`, `measurable_qk` and
`indepFun_value_qk` are satisfiable: the zero head on a one-point space, in
dimension `2` so that two distinct entries exist. -/
example :
    Measurable (fun _ : Unit => (0 : Matrix (Fin 2) (Fin 2) ℝ)) ∧
      iIndepFun (headEntries (fun _ : Unit => (0 : Matrix (Fin 2) (Fin 2) ℝ)) 0 0)
        (Measure.dirac ()) ∧
      (∀ i j, (Measure.dirac ()).map (fun _ : Unit => (0 : Matrix (Fin 2) (Fin 2) ℝ) i j) =
        gaussianReal 0 ((0 : ℝ≥0) ^ 2)) ∧
      ((0 : Fin 2), (0 : Fin 2)) ≠ (0, 1) :=
  ⟨measurable_const, iIndepFun_of_unit _, fun i j => by simp [gaussianReal_zero_var], by decide⟩

end Homogenized
end Transformer
