module

public import Transformer.Rotate90.Training.Symmetry
public import Mathlib.Analysis.SpecialFunctions.Log.Deriv
public import Mathlib.Analysis.SpecialFunctions.ExpDeriv

/-!
# Differentiating the actual tied loss at a collapsed embedding

Source: `review/flat_spectrum.md`, *Sampled updates can break aliases*, equations
(S1)--(S3). Every output row initially agrees, but all three embedding occurrences
are differentiated. The hidden-state derivative cancels from cross-entropy:
at this point it adds the same scalar to every logit. It is not frozen or omitted.
The result below permits every finite residual stack and positive RMSNorm epsilon.

The scalar differential is evaluated in the full Euclidean table space. Thus a
direction may perturb only one row and may alter either selected input lookup.
The constant-row condition is imposed only at the evaluation point. It does not
restrict the perturbations to a symmetric subspace, which would erase precisely
the row-separating signal studied in the next module.

The generic hidden-map differentiability premise is discharged for the actual
network by `analyticAt_hidden` in `Sampling.Gradient`; the concrete odd-coordinate
witness below also keeps the positive normalization epsilon at every depth.
-/

@[expose] public section

namespace Transformer.Rotate90.Training.Sampling

open Landscape.Neural
open scoped BigOperators

variable {V C K H : Type*} [Fintype V] [Fintype C] [Fintype K] [Fintype H]

/-- All rows share one vector; this includes nonzero collapsed embeddings. -/
def constantTable (v : C → ℝ) : Table V C := WithLp.toLp 2 fun j => v j.2

/-- One supervised row of the actual tied, normalized finite-depth network. -/
noncomputable def rowLoss (eps : ℝ) (blocks : List (Power.BlockWeights K H))
    (a b target : V) (E : Table V (Channels K)) : ℝ :=
  crossEntropy (Power.stackLogit eps (entry E) blocks a b) target

/-- The singleton objective is analytic with the actual tied lookups retained. -/
theorem analyticAt_rowLoss {eps : ℝ} (heps : 0 < eps)
    (blocks : List (Power.BlockWeights K H)) (a b target : V)
    (E : Table V (Channels K)) : AnalyticAt ℝ (rowLoss eps blocks a b target) E := by
  apply analyticAt_crossEntropy
  intro q
  exact (analyticAt_untiedLogit heps blocks (diagonal E) a b q).comp
    (diagonal.analyticAt E)

/-- At equal logits the derivative of log-sum-exp is the mean logit derivative.
This is a derivative of the full softmax, including every competing class. -/
theorem fderiv_crossEntropy_flat {P : Type*} [NormedAddCommGroup P] [NormedSpace ℝ P]
    {L : P → V → ℝ} {x : P} (target : V)
    (hL : ∀ q, DifferentiableAt ℝ (fun y => L y q) x)
    (hflat : ∀ q, L x q = L x target) (d : P) :
    fderiv ℝ (fun y => crossEntropy (L y) target) x d =
      (∑ q, fderiv ℝ (fun y => L y q) x d) / Fintype.card V -
        fderiv ℝ (fun y => L y target) x d := by
  have hs := HasFDerivAt.fun_sum (u := Finset.univ)
    (fun q _ => (hL q).hasFDerivAt.exp)
  have hf := (hs.log (ne_of_gt (sum_exp_pos (L x) target))).fun_sub
    (hL target).hasFDerivAt
  unfold crossEntropy
  rw [hf.fderiv]
  simp only [sub_apply, smul_apply, sum_apply, smul_eq_mul]
  simp_rw [hflat]
  simp only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, ← Finset.mul_sum]
  have hn : (Fintype.card V : ℝ) ≠ 0 := by
    have : Nonempty V := ⟨target⟩
    exact_mod_cast (ne_of_gt (Fintype.card_pos_iff.mpr this))
  field_simp

/-- The derivative of a tied readout includes the direct output-table term
and the entire indirect hidden-state term from the input-table occurrences. -/
theorem fderiv_tiedReadout {h : Table V C → C → ℝ} {E : Table V C}
    (hh : ∀ i, DifferentiableAt ℝ (fun F => h F i) E) (q : V) (d : Table V C) :
    fderiv ℝ (fun F => tiedReadout (entry F) (h F) q) E d =
      ∑ i, (entry d q i * h E i + entry E q i * fderiv ℝ (fun F => h F i) E d) := by
  have hp (i : C) :=
    (PiLp.proj (𝕜 := ℝ) 2 (fun _ : V × C => ℝ) (q, i)).hasFDerivAt (x := E)
  have hd := HasFDerivAt.fun_sum (u := Finset.univ)
    (fun i _ => (hh i).hasFDerivAt.fun_mul (hp i))
  change fderiv ℝ (fun F => ∑ i, h F i * entry F q i) E d = _
  simp only [PiLp.proj_apply] at hd
  simp only [entry]
  rw [hd.fderiv]
  simp only [sum_apply, add_apply, smul_apply, smul_eq_mul, PiLp.proj_apply]
  apply Finset.sum_congr rfl
  intro i hi
  ring

/-- At repeated rows, the indirect derivative cancels exactly. This is a
calculus identity for a tied readout, not an assumption of fixed hidden features. -/
theorem fderiv_tiedLoss_constant {h : Table V C → C → ℝ} (v : C → ℝ)
    (hh : ∀ i, DifferentiableAt ℝ (fun F => h F i) (constantTable v))
    (target : V) (d : Table V C) :
    fderiv ℝ (fun F => crossEntropy (tiedReadout (entry F) (h F)) target)
      (constantTable v) d =
        (∑ q, ∑ i, entry d q i * h (constantTable v) i) / Fintype.card V -
          ∑ i, entry d target i * h (constantTable v) i := by
  classical
  have hL (q : V) : DifferentiableAt ℝ
      (fun F => tiedReadout (entry F) (h F) q) (constantTable v) := by
    apply DifferentiableAt.fun_sum
    intro i hi
    exact (hh i).mul
      ((PiLp.proj (𝕜 := ℝ) 2 (fun _ : V × C => ℝ) (q, i)).differentiableAt)
  rw [fderiv_crossEntropy_flat target hL (by intro q; rfl)]
  simp_rw [fderiv_tiedReadout hh, Finset.sum_add_distrib]
  simp only [entry, constantTable, WithLp.ofLp_toLp,
    Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  have hn : (Fintype.card V : ℝ) ≠ 0 := by
    have : Nonempty V := ⟨target⟩
    exact_mod_cast (ne_of_gt (Fintype.card_pos_iff.mpr this))
  field_simp
  ring

/-- A nonzero witness at every finite depth and every positive width. Both
outer RMSNorms are retained; the odd coordinate of v + rotate90(v) is two. -/
theorem hidden_ones_pos {eps : ℝ} (heps : 0 < eps) (d : ℕ) (j : K) :
    0 < hidden eps (List.replicate d (Power.zeroBlock (K := K) (H := H)))
      (fun _ => 1) (fun _ => 1) (j, 1) := by
  have hn (x : Channels K → ℝ) (i : Channels K) (hi : 0 < x i) :
      0 < rmsNorm eps x i :=
    div_pos hi (Real.sqrt_pos.mpr (rmsNorm_radicand_pos heps x))
  unfold hidden
  rw [Power.fold_zero_blocks]
  apply hn
  apply hn
  norm_num [rotate90]

example := hidden_ones_pos (K := Fin 8) (H := Fin 43)
  (eps := 1 / 8388608) (by norm_num) 4 0
example := analyticAt_rowLoss (V := Fin 3) (K := Fin 1) (H := Fin 2)
  (eps := 1 / 8388608) (by norm_num) [Power.zeroBlock] 1 1 1
  (constantTable fun _ => 1)
example := fderiv_crossEntropy_flat (L := fun t : ℝ => fun q : Fin 3 => t * q.val)
  (x := 0) 1 (by intro q; fun_prop) (by intro q; simp) 1
example := fderiv_tiedReadout (V := Fin 3) (C := Fin 2)
  (h := fun E i => entry E 0 i ^ 2) (E := constantTable fun _ => 1)
  (by intro i; exact (PiLp.proj (𝕜 := ℝ) 2 (fun _ : Fin 3 × Fin 2 => ℝ)
    (0, i)).differentiableAt.pow 2) 1 (basis 2 0)
example := fderiv_tiedLoss_constant (V := Fin 3) (C := Fin 2)
  (h := fun E i => entry E 0 i ^ 2) (fun _ => 1)
  (by intro i; exact (PiLp.proj (𝕜 := ℝ) 2 (fun _ : Fin 3 × Fin 2 => ℝ)
    (0, i)).differentiableAt.pow 2) 1 (basis 2 0)

end Transformer.Rotate90.Training.Sampling
