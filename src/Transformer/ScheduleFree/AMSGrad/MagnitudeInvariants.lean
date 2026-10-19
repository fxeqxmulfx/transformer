module

public import Transformer.ScheduleFree.AMSGrad.MagnitudeDirection
public import Transformer.MagnitudeDirection.SectionA_AMSGradCallbacks
public import Mathlib.Analysis.SpecialFunctions.Sigmoid

/-!
# Query derivatives and projection invariants of Schedule-Free AMSGradMD

These statements check the local composition of arXiv:2405.15682v4, §4.4
with arXiv:2606.25971v2, Appendix A, Algorithm 2. The complete chain rule
is applied at the actual interpolated factor query. Projection controls the
base direction, rather than replacing the averaged or queried directions.
-/

@[expose] public section

noncomputable section

namespace Transformer.ScheduleFree.AMSGrad

open MagnitudeDirection

variable {m n : ℕ}

/-- The smooth raw-gain derivative is the exact-real sigmoid evaluated by PyTorch. -/
theorem softplusDerivative_eq_sigmoid (x : ℝ) : softplusDerivative x = Real.sigmoid x := by
  have hx := Real.exp_pos x
  have hsum : 1 + Real.exp x ≠ 0 := by positivity
  have hinv : 1 + (Real.exp x)⁻¹ ≠ 0 := by positivity
  dsimp [softplusDerivative, Real.sigmoid]
  rw [Real.exp_neg]
  field_simp [hsum, hinv, Real.exp_ne_zero]
  ring

/-- The base proposal is the existing raw AMSGrad callback at beta-one zero,
with the additional §4.4 decay evaluated at the actual interpolated query. -/
theorem proposal_eq_callback {ι : Type*} (p : AdamParameters) (t : ℕ)
    (s : State ι) (g : ι → ℝ) :
    proposal p t s g = fun i =>
      (amsgradVectorCallback p.epsilon 0 p.betaTwo ⟨fun _ => 0, s.second, s.maximum⟩
        s.base g (adamLearningRate p t)).2 i -
      adamLearningRate p t * p.decay * query p s i := by
  funext i
  dsimp [proposal, nextMaximum, nextSecond, amsgradVectorCallback]
  simp only [zero_mul, sub_zero, one_mul, zero_add]
  rw [add_comm p.epsilon]

/-- Unit effective gains preserve the supplied matrix at the first actual query. -/
theorem md_initial_query (p : MDParameters) (weight : Matrix (Fin m) (Fin n) ℝ) :
    mdQueryWeight p (mdInitial weight) = weight := by
  ext i j
  simp [mdQueryWeight, factorWeight, mdQuery, mdInitial, softplus_unit_initialization, fuse]

/-- The initial evaluation weights are the same supplied model weights. -/
theorem md_initial_evaluation (weight : Matrix (Fin m) (Fin n) ℝ) :
    mdEvaluationWeight (mdInitial weight) = weight := by
  ext i j
  simp [mdEvaluationWeight, factorWeight, mdEvaluation, mdInitial, initial,
    softplus_unit_initialization, fuse]

/-- All three actual AMSGrad maxima are nondecreasing in the full MD step. -/
theorem md_maxima_mono (p : MDParameters) (t : ℕ) (s : MDState m n)
    (G : Matrix (Fin m) (Fin n) ℝ) (i : Fin m) (j : Fin n) :
    s.direction.maximum (i, j) ≤ (mdStep p t s G).direction.maximum (i, j) ∧
      s.row.maximum i ≤ (mdStep p t s G).row.maximum i ∧
      s.column.maximum j ≤ (mdStep p t s G).column.maximum j := by
  exact ⟨maximum_mono _ _ _ _ _ _, maximum_mono _ _ _ _ _ _, maximum_mono _ _ _ _ _ _⟩

/-- A positive-radius, nonzero candidate is projected onto the prescribed sphere
before entering the average. A zero effective rate follows `zero_rate_iterates`. -/
theorem md_base_norm (p : MDParameters) (t : ℕ) (s : MDState m n)
    (G : Matrix (Fin m) (Fin n) ℝ) (hc : 0 ≤ s.radius)
    (hη : adamLearningRate (mdDirectionParameters p) t ≠ 0)
    (hD : mdCandidate p t s G ≠ 0) :
    frobeniusNorm (mdBaseDirection (mdStep p t s G)) = s.radius := by
  change frobeniusNorm (fun i j =>
    (if adamLearningRate (mdDirectionParameters p) t = 0 then s.direction.base
      else projectDirection s.radius
        (proposal (mdDirectionParameters p) t s.direction
          (fun ij => (mdGradients (mdQuery p s) G).direction ij.1 ij.2))) (i, j)) = s.radius
  rw [ite_eq_right hη]
  exact matrixProject_norm s.radius (mdCandidate p t s G) hc hD

example : frobeniusNorm (mdBaseDirection (mdStep (m := 1) (n := 1)
    ⟨1, 0, 0, 1/2, 0, 1⟩ 0 (mdInitial (fun _ _ => 2)) 0)) =
    (mdInitial (fun _ _ => (2 : ℝ)) : MDState 1 1).radius := by
  apply md_base_norm (m := 1) (n := 1)
  · exact norm_nonneg _
  · norm_num [mdDirectionParameters, adamLearningRate, warmupFactor]
  · intro h
    have hi := congrFun (congrFun h 0) 0
    norm_num [mdCandidate, proposal, mdGradients, directionGradient, fuse, mdInitial, initial,
      mdDirectionParameters] at hi

/-- The three gradients used by `mdStep` preserve the full objective's actual
directional derivative at the fused Schedule-Free query, including raw-gain
perturbations. The derivative hypothesis is at `mdQueryWeight`, not at a base. -/
theorem md_query_hasDerivAt (L : Matrix (Fin m) (Fin n) ℝ → ℝ)
    (p : MDParameters) (s : MDState m n) (G H : Matrix (Fin m) (Fin n) ℝ)
    (dr : Fin m → ℝ) (dc : Fin n → ℝ)
    (hL : HasFDerivAt L (gradientFunctional G) (mdQueryWeight p s)) :
    let q := mdQuery p s
    HasDerivAt (fun a : ℝ => L (fuse
      (fun i => softplus (q.rawRow i + a * dr i))
      (fun j => softplus (q.rawColumn j + a * dc j)) (q.direction + a • H)))
      (frobeniusPairing (mdGradients q G).direction H +
        (∑ i, (mdGradients q G).row i * dr i) +
        (∑ j, (mdGradients q G).column j * dc j)) 0 := by
  exact softplus_decoupled_loss_hasDerivAt L (mdQuery p s).rawRow dr
    (mdQuery p s).rawColumn dc (mdQuery p s).direction H G hL

/-- Every linear matrix objective supplies an actual differentiable witness to
the query-point chain rule, with arbitrary nonzero or zero perturbations. -/
example (p : MDParameters) (s : MDState m n) (G H : Matrix (Fin m) (Fin n) ℝ)
    (dr : Fin m → ℝ) (dc : Fin n → ℝ) :
    let q := mdQuery p s
    HasDerivAt (fun a : ℝ => gradientFunctional G (fuse
      (fun i => softplus (q.rawRow i + a * dr i))
      (fun j => softplus (q.rawColumn j + a * dc j)) (q.direction + a • H)))
      (frobeniusPairing (mdGradients q G).direction H +
        (∑ i, (mdGradients q G).row i * dr i) +
        (∑ j, (mdGradients q G).column j * dc j)) 0 :=
  md_query_hasDerivAt (gradientFunctional G) p s G H dr dc
    (gradientFunctional G).hasFDerivAt

end Transformer.ScheduleFree.AMSGrad
