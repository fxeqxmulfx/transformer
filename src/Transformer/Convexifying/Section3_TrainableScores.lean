/-
# Trainable softmax scores retain nonconvex prediction geometry

Ergen, Neyshabur, Mehta, arXiv:2211.11052v1, §2–§3.1, equations (2)–(3).
Here query and key weights remain trainable.  Two small training samples
show that the prediction set of one ordinary softmax-attention head is not
convex.  This rules out an exact convex feasible set with affine predictions
for this fixed-width model; it does not rule out other kinds of lifting or
a model with more heads.
-/

import Transformer.Convexifying.Section3_ConvexAttention

open scoped BigOperators

namespace Transformer.Convexifying

/-- The first-row attention weight for two scores `[z, 0]`.
Source: arXiv:2211.11052v1, equation (2), rowwise softmax. -/
private noncomputable def logistic (z : ℝ) : ℝ :=
  Real.exp z / (Real.exp z + 1)

/-- Three selected output coordinates for sample token amplitudes `1` and
`2`, expressed through the trainable products `p = q*k` and `t = v*o`.
Source: arXiv:2211.11052v1, equations (2)–(3). -/
private noncomputable def headTriple (p t : ℝ) : Fin 3 → ℝ :=
  fun j => if j = 0 then t / 2 else
    if j = 1 then t * logistic p else 2 * t * logistic (4 * p)

/-- The midpoint of the attainable predictions with `p = 0` and
`p = log 2`, each with `t = 1`.  Source: equation (3), specialized. -/
private noncomputable def midpointTriple : Fin 3 → ℝ :=
  fun j => if j = 0 then 1 / 2 else if j = 1 then 7 / 12 else 49 / 34

/-- Exponential multiplication used by the second sample's score `4p`.
Source: arXiv:2211.11052v1, equation (3), specialized. -/
private theorem exp_four_mul (x : ℝ) : Real.exp (4 * x) = Real.exp x ^ 4 := by
  have hx : 4 * x = x + x + x + x := by ring
  rw [hx, Real.exp_add, Real.exp_add, Real.exp_add]
  ring

/-- The score `4 log 2` has exponential `16`.
Source: arXiv:2211.11052v1, equation (3), specialized. -/
private theorem exp_four_log_two : Real.exp (4 * Real.log 2) = 16 := by
  rw [exp_four_mul, Real.exp_log (by norm_num : (0 : ℝ) < 2)]
  norm_num

/-- The displayed rational triple is exactly the midpoint of two genuine
trainable-score predictions.  Source: equation (3), specialized. -/
private theorem midpointTriple_eq :
    midpointTriple = fun j =>
      (headTriple 0 1 j + headTriple (Real.log 2) 1 j) / 2 := by
  funext j
  fin_cases j
  · norm_num [midpointTriple, headTriple]
  · norm_num [midpointTriple, headTriple, logistic, Real.exp_zero,
      Real.exp_log (by norm_num : (0 : ℝ) < 2)]
  · norm_num [midpointTriple, headTriple, logistic, Real.exp_zero,
      exp_four_log_two]

/-- No single score product and output amplitude yield the midpoint: its
first coordinates force `t = 1` and `exp p = 7/5`, contradicting the
second sample's score `4p`.  Source: equation (3), specialized. -/
private theorem midpointTriple_unreachable :
    ¬ ∃ p t : ℝ, headTriple p t = midpointTriple := by
  rintro ⟨p, t, h⟩
  have h0 : t / 2 = 1 / 2 := by
    simpa [headTriple, midpointTriple] using
      congrArg (fun f : Fin 3 → ℝ => f 0) h
  have h1 : t * logistic p = 7 / 12 := by
    simpa [headTriple, midpointTriple] using
      congrArg (fun f : Fin 3 → ℝ => f 1) h
  have h2 : 2 * t * logistic (4 * p) = 49 / 34 := by
    simpa [headTriple, midpointTriple] using
      congrArg (fun f : Fin 3 → ℝ => f 2) h
  have ht : t = 1 := by linarith
  have hden : Real.exp p + 1 ≠ 0 := by positivity
  have hp : Real.exp p = 7 / 5 := by
    rw [ht] at h1
    change 1 * (Real.exp p / (Real.exp p + 1)) = 7 / 12 at h1
    simp only [one_mul] at h1
    apply (div_eq_iff hden).mp at h1
    linarith
  have hExp4 : Real.exp (4 * p) = (7 / 5 : ℝ) ^ 4 := by
    rw [exp_four_mul, hp]
  norm_num [logistic, ht, hExp4] at h2

/-- Two samples with token rows `[1, 0]` and `[2, 0]` in one dimension.
Source: arXiv:2211.11052v1, equation (3), specialized. -/
def toyData : Fin 2 → Fin 2 → Vec 1 :=
  fun i r _ => if r = 0 then if i = 0 then 1 else 2 else 0

/-- Genuine query-key scores for the two samples, with trainable scalar
query and key weights.  Source: arXiv:2211.11052v1, equation (3). -/
def toyScores (i : Fin 2) (q k : ℝ) : Mat 2 2 :=
  fun r s => (toyData i r 0 * q) * (toyData i s 0 * k)

/-- Ordinary softmax-attention output, including trainable value and output
weights.  Source: arXiv:2211.11052v1, equations (2)–(3). -/
noncomputable def toyOutput (i r : Fin 2) (q k v o : ℝ) : ℝ :=
  (∑ s, rowSoftmax (toyScores i q k) r s * (toyData i s 0 * v)) * o

/-- Three coordinates of the two-sample, one-head prediction vector.
Source: arXiv:2211.11052v1, equation (3), specialized. -/
noncomputable def toyPrediction (q k v o : ℝ) : Fin 3 → ℝ :=
  fun j => if j = 0 then toyOutput 0 1 q k v o else
    if j = 1 then toyOutput 0 0 q k v o else toyOutput 1 0 q k v o

/-- The reduced products `q*k` and `v*o` describe these genuine softmax
predictions exactly.  Source: arXiv:2211.11052v1, equation (3). -/
theorem toyPrediction_eq_triple (q k v o : ℝ) :
    toyPrediction q k v o = headTriple (q * k) (v * o) := by
  funext j
  fin_cases j
  · simp [toyPrediction, toyOutput, toyScores, toyData, headTriple,
      rowSoftmax, Real.exp_zero]
    ring_nf
  · simp [toyPrediction, toyOutput, toyScores, toyData, headTriple,
      rowSoftmax, logistic, Fin.sum_univ_two, Real.exp_zero]
    ring_nf
  · simp [toyPrediction, toyOutput, toyScores, toyData, headTriple,
      rowSoftmax, logistic, Fin.sum_univ_two, Real.exp_zero]
    ring_nf

/-- The midpoint of two attainable ordinary one-head predictions is not
attainable by any trainable `q`, `k`, `v`, and `o` in one head.
Source: arXiv:2211.11052v1, equations (2)–(3), specialized. -/
theorem toyPrediction_midpoint_unreachable :
    ¬ ∃ q k v o : ℝ,
      toyPrediction q k v o =
        (fun j => (toyPrediction 0 1 1 1 j +
          toyPrediction (Real.log 2) 1 1 1 j) / 2) := by
  rintro ⟨q, k, v, o, h⟩
  have hmid :
      (fun j => (toyPrediction 0 1 1 1 j +
        toyPrediction (Real.log 2) 1 1 1 j) / 2) = midpointTriple := by
    simpa [toyPrediction_eq_triple] using midpointTriple_eq.symm
  have htriple : headTriple (q * k) (v * o) = midpointTriple := by
    calc
      headTriple (q * k) (v * o) = toyPrediction q k v o :=
        (toyPrediction_eq_triple q k v o).symm
      _ = _ := h
      _ = midpointTriple := hmid
  exact midpointTriple_unreachable ⟨q * k, v * o, htriple⟩

/-- The attainable three-coordinate prediction set for one ordinary head.
Source: arXiv:2211.11052v1, equation (3), specialized. -/
def oneHeadPredictionSet : Set (Fin 3 → ℝ) :=
  {u | ∃ q k v o : ℝ, toyPrediction q k v o = u}

/-- **One-head obstruction to an affine convex reparameterization.**  The
attainable predictions are nonconvex even though all four weights remain
trainable.  This concerns exact equality of prediction sets at fixed width;
it does not exclude convex formulations with more heads or nonlinear output
recovery.  Source: arXiv:2211.11052v1, §2–§3.1, equations (2)–(3). -/
theorem oneHeadPredictionSet_not_convex :
    ¬ Convex ℝ oneHeadPredictionSet := by
  intro hc
  have hA : toyPrediction 0 1 1 1 ∈ oneHeadPredictionSet :=
    ⟨0, 1, 1, 1, rfl⟩
  have hB : toyPrediction (Real.log 2) 1 1 1 ∈ oneHeadPredictionSet :=
    ⟨Real.log 2, 1, 1, 1, rfl⟩
  have hmid := (convex_iff_forall_pos.mp hc) hA hB
    (show 0 < (1 / 2 : ℝ) by norm_num)
    (show 0 < (1 / 2 : ℝ) by norm_num)
    (show (1 / 2 : ℝ) + 1 / 2 = 1 by norm_num)
  have hEq :
      (1 / 2 : ℝ) • toyPrediction 0 1 1 1 +
        (1 / 2 : ℝ) • toyPrediction (Real.log 2) 1 1 1 =
      (fun j => (toyPrediction 0 1 1 1 j +
        toyPrediction (Real.log 2) 1 1 1 j) / 2) := by
    funext j
    simp [Pi.add_apply, Pi.smul_apply, smul_eq_mul]
    ring
  rw [hEq] at hmid
  exact toyPrediction_midpoint_unreachable hmid

/-- No finite-dimensional convex parameter set with an affine prediction
map has exactly this one-head softmax prediction set.  This is the precise
obstruction established here; nonlinear recovery or extra heads require
separate analysis.  Source: arXiv:2211.11052v1, §3.1, equation (3). -/
theorem no_finite_affine_convex_lift {m : ℕ}
    (C : Set (Fin m → ℝ)) (f : (Fin m → ℝ) →ᵃ[ℝ] (Fin 3 → ℝ))
    (hC : Convex ℝ C) : f '' C ≠ oneHeadPredictionSet := by
  intro hEq
  have hconv : Convex ℝ oneHeadPredictionSet := by
    rw [← hEq]
    exact Convex.affine_image f hC
  exact oneHeadPredictionSet_not_convex hconv

/-- Convex feasible sets required by the preceding theorem do exist. -/
example : Convex ℝ (Set.univ : Set (Fin 1 → ℝ)) := convex_univ

end Transformer.Convexifying
