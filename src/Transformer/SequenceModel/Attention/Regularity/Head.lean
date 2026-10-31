module

public import Transformer.SequenceModel.Attention.Regularity.Basic
public import Transformer.SequenceModel.Attention.Concentration

/-!
# Local analyticity through normalized rotary attention and XSA

Source: `review/flat_spectrum.md`, *Transformer learning bridges*, R1.
All coordinates vary. The causal softmax denominator is proved positive;
XSA requires only that the selected own value avoid its clipping sphere.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention

open GPTMini Landscape.Neural
open scoped BigOperators

variable {P : Type*} [NormedAddCommGroup P] [NormedSpace ℝ P] {x : P}

/-- R1: the scalar entries of the full matrix representation act by the usual sum. -/
theorem matrixEquiv_apply (m n : ℕ) (A : Matrix (Fin m) (Fin n) ℝ)
    (v : EucSpace n) (i : Fin m) :
    matrixEquiv m n A v i = ∑ j, A i j * v j := rfl

/-- R1: both a dense matrix and its input may vary in every coordinate. -/
theorem analyticAt_matrixApply {m n : ℕ} {A : P → Matrix (Fin m) (Fin n) ℝ}
    {v : P → EucSpace n} (hA : ∀ i j, AnalyticAt ℝ (fun w => A w i j) x)
    (hv : ∀ j, AnalyticAt ℝ (fun w => v w j) x) (i : Fin m) :
    AnalyticAt ℝ (fun w => matrixEquiv m n (A w) (v w) i) x := by
  simp only [matrixEquiv_apply]
  exact Finset.analyticAt_fun_sum _ (fun j _ => (hA i j).fun_mul (hv j))

/-- R1: the actual Euclidean dot product in scalar coordinates. -/
theorem analyticAt_inner {d : ℕ} {u v : P → EucSpace d}
    (hu : ∀ i, AnalyticAt ℝ (fun w => u w i) x)
    (hv : ∀ i, AnalyticAt ℝ (fun w => v w i) x) :
    AnalyticAt ℝ (fun w => inner ℝ (u w) (v w)) x := by
  simp only [EuclideanSpace.inner_eq_star_dotProduct, dotProduct, Pi.star_apply, star_trivial]
  exact Finset.analyticAt_fun_sum _ (fun i _ => (hv i).fun_mul (hu i))

/-- R1: Q/K RMSNorm, arbitrary phases and standard scaling are analytic. -/
theorem analyticAt_headScore {d : ℕ} (hd : 0 < d) {eps : ℝ} (heps : 0 < eps)
    (a b : Fin (d / 2) → ℝ) {q k : P → EucSpace d}
    (hq : ∀ i, AnalyticAt ℝ (fun w => q w i) x)
    (hk : ∀ i, AnalyticAt ℝ (fun w => k w i) x) :
    AnalyticAt ℝ (fun w => headScore d eps a b (q w) (k w)) x := by
  exact (analyticAt_inner
    (analyticAt_rotateHead a (analyticAt_rmsNormEps hd heps hq))
    (analyticAt_rotateHead b (analyticAt_rmsNormEps hd heps hk))).div_const

/-- R1: the causal mask is fixed, and every active denominator is positive. -/
theorem analyticAt_causalWeight {T : ℕ} {s : P → Fin T → Fin T → ℝ}
    (hs : ∀ i j, AnalyticAt ℝ (fun w => s w i j) x) (i j : Fin T) :
    AnalyticAt ℝ (fun w => causalWeight (s w) i j) x := by
  unfold causalWeight
  split_ifs
  · exact (hs i j).rexp'.fun_div
      (Finset.analyticAt_fun_sum _ (fun k _ => (hs i k).rexp'))
      (denominator_pos (s x) i).ne'
  · exact analyticAt_const

/-- R1: local regularity includes the actual value projection and explicit
first-position zero. No global differentiability at the clipping sphere is assumed. -/
theorem analyticAt_headOutput {T d : ℕ} (hd : 0 < d) {eps xeps : ℝ}
    (heps : 0 < eps) (hxeps : 0 < xeps)
    (angle : Fin T → Fin (d / 2) → ℝ) {q k v : P → Fin T → EucSpace d}
    (hq : ∀ j c, AnalyticAt ℝ (fun w => q w j c) x)
    (hk : ∀ j c, AnalyticAt ℝ (fun w => k w j c) x)
    (hv : ∀ j c, AnalyticAt ℝ (fun w => v w j c) x)
    (i : Fin T) (hboundary : ‖v x i‖ ≠ xeps) (c : Fin d) :
    AnalyticAt ℝ (fun w => headOutput eps xeps angle (q w) (k w) (v w) i c) x := by
  let scores := fun w a b => headScore d eps (angle a) (angle b) (q w a) (k w b)
  let y := fun w => ∑ j, causalWeight (scores w) i j • v w j
  let u := fun w => normL2 xeps (v w i)
  have hs := fun a b => analyticAt_headScore hd heps (angle a) (angle b) (hq a) (hk b)
  have hy : ∀ c, AnalyticAt ℝ (fun w => y w c) x := by
    intro c
    simp only [y, WithLp.ofLp_sum, Finset.sum_apply, PiLp.smul_apply, smul_eq_mul]
    exact Finset.analyticAt_fun_sum _ (fun j _ =>
      (analyticAt_causalWeight hs i j).fun_mul (hv j c))
  have hu := analyticAt_normL2 hxeps (hv i) hboundary
  change AnalyticAt ℝ (fun w => (if i.val = 0 then 0 else y w - (inner ℝ (y w) (u w)) • u w) c) x
  split_ifs
  · exact analyticAt_const
  · exact (hy c).fun_sub ((analyticAt_inner hy hu).fun_mul (hu c))

example := analyticAt_matrixApply
  (A := fun t : ℝ => fun i j : Fin 2 => t + i.val + j.val)
  (v := fun t => WithLp.toLp 2 (fun i : Fin 2 => t + i.val)) (x := 0)
  (by intro i j; fun_prop) (by intro i; exact analyticAt_id.fun_add analyticAt_const) 0
example := analyticAt_inner
  (u := fun t : ℝ => WithLp.toLp 2 (fun i : Fin 2 => t + i.val))
  (v := fun t => WithLp.toLp 2 (fun i : Fin 2 => t - i.val)) (x := 0)
  (fun _ => analyticAt_id.fun_add analyticAt_const)
  (fun _ => analyticAt_id.fun_sub analyticAt_const)
example := analyticAt_headScore (d := 2) (by decide) (eps := 1) (by norm_num)
  (fun _ => 1 / 7) (fun _ => 2 / 7)
  (q := fun t : ℝ => WithLp.toLp 2 (fun i => t + i.val))
  (k := fun t => WithLp.toLp 2 (fun i => t - i.val)) (x := 0)
  (fun _ => analyticAt_id.fun_add analyticAt_const)
  (fun _ => analyticAt_id.fun_sub analyticAt_const)
example := analyticAt_causalWeight (s := fun t : ℝ => fun i j : Fin 3 => t * (i.val + j.val))
  (x := 1) (by intro i j; fun_prop) 2 1
example := analyticAt_headOutput (T := 3) (d := 2) (by decide)
  (eps := 1) (xeps := 1) (by norm_num) (by norm_num)
  (fun j _ => (j.val : ℝ) / 7)
  (q := fun t : ℝ => fun _ => WithLp.toLp 2 (fun i => t + i.val))
  (k := fun t => fun _ => WithLp.toLp 2 (fun i => t - i.val))
  (v := fun t => fun _ => WithLp.toLp 2 (fun _ => t)) (x := 0)
  (fun _ _ => analyticAt_id.fun_add analyticAt_const)
  (fun _ _ => analyticAt_id.fun_sub analyticAt_const)
  (fun _ _ => analyticAt_id) 2 (by norm_num [EuclideanSpace.norm_eq]) 0

end Transformer.SequenceModel.Attention
