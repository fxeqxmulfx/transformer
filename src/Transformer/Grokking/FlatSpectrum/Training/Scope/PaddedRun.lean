module

public import Transformer.Grokking.FlatSpectrum.Training.Scope.RankStart

/-!
# A narrow run inside a wide one

Source: `review/flat_spectrum.md`, *Training reaches the optimal fit* (added), item WD14,
statement 4; the item "Width" of WD11.

Append zero columns to a pair of factors `(u, v)` of width `k₁`: `([u 0], [v 0])` of width
`k₁ + k₂` has the same product and the same weight decay (`fromCols_zero_mul_conjTranspose`,
`frobeniusSq_fromCols_zero`, `decayedLoss_fromCols_zero`). So

* a step of gradient descent with weight decay from the padded pair is the padded step
  (`decayStep_fromCols_zero`), and the run from a padded start is the narrow run with zero
  columns appended, at every step (`decayRun_fromCols_zero`); if the narrow run converges, the
  wide one converges to the padded limit (`tendsto_decayRun_fromCols_zero`);
* the padded pair is a decayed minimum exactly when the narrow pair is
  (`isDecayedMin_fromCols_zero_iff`);
* the stack of a padded pair has rank at most `k₁` (`rank_stack_fromCols_zero_le`).

So the limits of the runs of a width below `min(m, n)` are limits of runs of every larger
width, from the padded starts. For the `2 × 2` identity the runs of width two from a start with
a zero column are the runs of width one, and by `RankStart` none of them ends at a decayed
minimum.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training.Scope

open Matrix Filter Topology

variable {m n k₁ k₂ : Type*} [Fintype m] [Fintype n] [Fintype k₁] [Fintype k₂]
  {G : Matrix m n ℂ → Matrix m n ℂ} {L : Matrix m n ℂ → ℝ} {η lam : ℝ}

omit [Fintype m] [Fintype n] in
/-- Zero columns do not change the product. -/
lemma fromCols_zero_mul_conjTranspose (u : Matrix m k₁ ℂ) (v : Matrix n k₁ ℂ) :
    fromCols u (0 : Matrix m k₂ ℂ) * (fromCols v (0 : Matrix n k₂ ℂ))ᴴ = u * vᴴ := by
  rw [conjTranspose_fromCols_eq_fromRows_conjTranspose, fromCols_mul_fromRows,
    conjTranspose_zero, Matrix.mul_zero, add_zero]

omit [Fintype n] in
/-- Zero columns do not change the norm. -/
lemma frobeniusSq_fromCols_zero (u : Matrix m k₁ ℂ) :
    frobeniusSq (fromCols u (0 : Matrix m k₂ ℂ)) = frobeniusSq u := by
  simp [frobeniusSq, Fintype.sum_sum_type]

/-- **A step from a padded pair is the padded step** (`review/flat_spectrum.md`, item WD14). -/
lemma decayStep_fromCols_zero (G : Matrix m n ℂ → Matrix m n ℂ) (η lam : ℝ)
    (u : Matrix m k₁ ℂ) (v : Matrix n k₁ ℂ) :
    decayStep G η lam (fromCols u (0 : Matrix m k₂ ℂ), fromCols v (0 : Matrix n k₂ ℂ)) =
      (fromCols (decayStep G η lam (u, v)).1 0, fromCols (decayStep G η lam (u, v)).2 0) := by
  simp only [decayStep, fromCols_zero_mul_conjTranspose, Matrix.mul_fromCols, Matrix.mul_zero]
  refine Prod.ext ?_ ?_ <;> ext i (j | j) <;> simp

/-- A run is its previous state after one more step. -/
lemma decayRun_succ {k : Type*} [Fintype k] (G : Matrix m n ℂ → Matrix m n ℂ) (η lam : ℝ)
    (a : Matrix m k ℂ) (b : Matrix n k ℂ) (t : ℕ) :
    decayRun G η lam a b (t + 1) = decayStep G η lam (decayRun G η lam a b t) := by
  rw [decayRun, Function.iterate_succ_apply', ← decayRun]

/-- **The run from a padded start is the narrow run with zero columns appended**
(`review/flat_spectrum.md`, item WD14), at every step, for every gradient field, step and
weight decay. -/
theorem decayRun_fromCols_zero (G : Matrix m n ℂ → Matrix m n ℂ) (η lam : ℝ)
    (u₀ : Matrix m k₁ ℂ) (v₀ : Matrix n k₁ ℂ) (t : ℕ) :
    decayRun G η lam (fromCols u₀ (0 : Matrix m k₂ ℂ)) (fromCols v₀ (0 : Matrix n k₂ ℂ)) t =
      (fromCols (decayRun G η lam u₀ v₀ t).1 0, fromCols (decayRun G η lam u₀ v₀ t).2 0) := by
  induction t with
  | zero => rfl
  | succ t ih => rw [decayRun_succ, decayRun_succ, ih, decayStep_fromCols_zero]

/-- **If the narrow run converges, the run from the padded start converges to the padded
limit** (`review/flat_spectrum.md`, item WD14). -/
theorem tendsto_decayRun_fromCols_zero {u₀ u : Matrix m k₁ ℂ} {v₀ v : Matrix n k₁ ℂ}
    (h : Tendsto (decayRun G η lam u₀ v₀) atTop (𝓝 (u, v))) :
    Tendsto (decayRun G η lam (fromCols u₀ (0 : Matrix m k₂ ℂ))
      (fromCols v₀ (0 : Matrix n k₂ ℂ))) atTop (𝓝 (fromCols u 0, fromCols v 0)) := by
  have hc : Continuous fun w : Matrix m k₁ ℂ × Matrix n k₁ ℂ =>
      ((fromCols w.1 (0 : Matrix m k₂ ℂ), fromCols w.2 (0 : Matrix n k₂ ℂ)) :
        Matrix m (k₁ ⊕ k₂) ℂ × Matrix n (k₁ ⊕ k₂) ℂ) := by
    refine Continuous.prodMk (continuous_matrix fun i j => ?_) (continuous_matrix fun i j => ?_)
    · cases j with
      | inl j => simpa using continuous_fst.matrix_elem i j
      | inr j => simpa using continuous_const
    · cases j with
      | inl j => simpa using continuous_snd.matrix_elem i j
      | inr j => simpa using continuous_const
  exact ((hc.tendsto (u, v)).comp h).congr fun t => (decayRun_fromCols_zero G η lam u₀ v₀ t).symm

/-- Zero columns do not change the weight-decayed loss. -/
lemma decayedLoss_fromCols_zero (L : Matrix m n ℂ → ℝ) (lam : ℝ) (u : Matrix m k₁ ℂ)
    (v : Matrix n k₁ ℂ) :
    decayedLoss L lam (fromCols u (0 : Matrix m k₂ ℂ)) (fromCols v (0 : Matrix n k₂ ℂ)) =
      decayedLoss L lam u v := by
  rw [decayedLoss, decayedLoss, fromCols_zero_mul_conjTranspose, frobeniusSq_fromCols_zero,
    frobeniusSq_fromCols_zero]

/-- **A padded pair is a decayed minimum exactly when the narrow pair is**
(`review/flat_spectrum.md`, item WD14). -/
theorem isDecayedMin_fromCols_zero_iff {u : Matrix m k₁ ℂ} {v : Matrix n k₁ ℂ} :
    IsDecayedMin L lam (fromCols u (0 : Matrix m k₂ ℂ)) (fromCols v (0 : Matrix n k₂ ℂ)) ↔
      IsDecayedMin L lam u v := by
  simp only [IsDecayedMin, decayedLoss_fromCols_zero]

omit [Fintype m] [Fintype n] in
/-- **The stack of a padded pair has rank at most the narrow width**
(`review/flat_spectrum.md`, item WD14). -/
theorem rank_stack_fromCols_zero_le [DecidableEq k₁] (u : Matrix m k₁ ℂ) (v : Matrix n k₁ ℂ) :
    (stack (fromCols u (0 : Matrix m k₂ ℂ)) (fromCols v (0 : Matrix n k₂ ℂ))).rank ≤
      Fintype.card k₁ := by
  have e : stack (fromCols u (0 : Matrix m k₂ ℂ)) (fromCols v (0 : Matrix n k₂ ℂ)) =
      stack u v * fromCols (1 : Matrix k₁ k₁ ℂ) (0 : Matrix k₁ k₂ ℂ) := by
    rw [Matrix.mul_fromCols, Matrix.mul_one, Matrix.mul_zero]
    ext (i | i) (j | j) <;> simp [stack]
  rw [e]
  exact (Matrix.rank_mul_le_left _ _).trans (Matrix.rank_le_card_width _)

/-! The witness: the `2 × 2` identity with `λ = ½`, and the start of width two whose first
columns are `(1, 1)` and whose second columns are zero, in the ball `ρ = 8` with `κ = 2` and
`η = 1/200`. Its run is the run of width one, and it ends at no decayed minimum. -/

example := decayRun_fromCols_zero (k₂ := Fin 1) (fun X : Matrix (Fin 2) (Fin 2) ℂ => X - 1)
  1 1 (of fun _ _ => 1 : Matrix (Fin 2) (Fin 1) ℂ) (of fun _ _ => 1) 3

example (u v : Matrix (Fin 2) (Fin 1) ℂ) :
    ¬ IsDecayedMin (fitLoss Set.univ (1 : Matrix (Fin 2) (Fin 2) ℂ)) (1 / 2)
      (fromCols u (0 : Matrix (Fin 2) (Fin 1) ℂ)) (fromCols v (0 : Matrix (Fin 2) (Fin 1) ℂ)) :=
  fun h => not_isDecayedMin_width_one (by norm_num) (by norm_num) u v
    (isDecayedMin_fromCols_zero_iff.1 h)

example : ∃ U V : Matrix (Fin 2) (Fin 1 ⊕ Fin 1) ℂ,
    Tendsto (decayRun (fun X => mask Set.univ (X - 1)) (1 / 200) (1 / 2)
      (fromCols (of fun _ _ => 1 : Matrix (Fin 2) (Fin 1) ℂ) (0 : Matrix (Fin 2) (Fin 1) ℂ))
      (fromCols (of fun _ _ => 1 : Matrix (Fin 2) (Fin 1) ℂ) (0 : Matrix (Fin 2) (Fin 1) ℂ)))
      atTop (𝓝 (U, V)) ∧
    lift (1 / 2) (mask Set.univ (U * Vᴴ - 1)) * stack U V = 0 ∧
    ¬ IsDecayedMin (fitLoss Set.univ (1 : Matrix (Fin 2) (Fin 2) ℂ)) (1 / 2) U V :=
  exists_tendsto_decayRun_one_not_isDecayedMin (κ := 2) (ρ := 8) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by rw [frobeniusSq_fromCols_zero]
        norm_num [frobeniusSq])
    ((rank_stack_fromCols_zero_le _ _).trans_eq (by simp))

example : ∃ u v : Matrix (Fin 2) (Fin 1) ℂ,
    Tendsto (decayRun (fun X => mask Set.univ (X - 1)) (1 / 200) (1 / 2)
      (fromCols (of fun _ _ => 1 : Matrix (Fin 2) (Fin 1) ℂ) (0 : Matrix (Fin 2) (Fin 1) ℂ))
      (fromCols (of fun _ _ => 1 : Matrix (Fin 2) (Fin 1) ℂ) (0 : Matrix (Fin 2) (Fin 1) ℂ)))
      atTop (𝓝 (fromCols u 0, fromCols v 0)) := by
  obtain ⟨u, v, h, -⟩ := exists_tendsto_decayRun_fitLoss (Ω := (Set.univ : Set (Fin 2 × Fin 2)))
    (K := 1) (η := 1 / 200) (lam := 1 / 2) (κ := 2) (ρ := 8) (by norm_num) (by norm_num)
    (by norm_num) (by rw [frobeniusSq_one_fin_two]; norm_num) (by norm_num) (by norm_num)
    (by norm_num) (U₀ := (of fun _ _ => 1 : Matrix (Fin 2) (Fin 1) ℂ))
    (V₀ := (of fun _ _ => 1 : Matrix (Fin 2) (Fin 1) ℂ)) (by norm_num [frobeniusSq])
  exact ⟨u, v, tendsto_decayRun_fromCols_zero h⟩

end Transformer.Grokking.FlatSpectrum.Training.Scope
