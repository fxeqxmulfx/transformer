module

public import Transformer.Grokking.FlatSpectrum.Training.Convergent
public import Transformer.Grokking.FlatSpectrum.Training.NuclearMin

/-!
# A run does not raise the rank of its stack

Source: `review/flat_spectrum.md`, *Training reaches the optimal fit* (added), item WD14; the
item "Width" of WD11.

A step of gradient descent with weight decay multiplies the stack `W = [U; V]` on the left by
`1 − η M_t` (WD1). So, for every gradient field, step and weight decay:

* a vector `z` with `W_0 z = 0` has `W_t z = 0` at every step
  (`stack_decayRun_mulVec_eq_zero`) and `W z = 0` at a limit of the run
  (`stack_mulVec_eq_zero_of_tendsto`);
* by rank–nullity (`rank_le_of_forall_mulVec_eq_zero`) the rank of the stack does not grow,
  `rank W_t ≤ rank W_0` (`rank_stack_decayRun_le`), and the product of a limit has
  `rank (U Vᴴ) ≤ rank W_0` (`rank_mul_conjTranspose_le_of_tendsto`);
* so a run whose limit is a decayed minimum starts at a stack whose rank is at least that of
  some trained product: if every minimizer of `L + λ ‖·‖_S1` has a rank above `rank W_0`, no
  limit of the run is a decayed minimum (`not_isDecayedMin_of_rank_lt`).

At every width the starts with `rank [U_0; V_0]` below the least rank of a trained product are
therefore in the exceptional set of WD10. `RankStart` has the `2 × 2` identity.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training.Scope

open Matrix Filter Topology

variable {m n k : Type*} [Fintype m] [Fintype n] [Fintype k] [DecidableEq m] [DecidableEq n]
  {G : Matrix m n ℂ → Matrix m n ℂ} {L : Matrix m n ℂ → ℝ} {η lam : ℝ}
  {U₀ U : Matrix m k ℂ} {V₀ V : Matrix n k ℂ}

/-- **A kernel vector of the stack stays one along the run** (`review/flat_spectrum.md`, item
WD14): `[U₀; V₀] z = 0` gives `[U_t; V_t] z = 0` at every step, for every gradient field, step
and weight decay. -/
theorem stack_decayRun_mulVec_eq_zero (G : Matrix m n ℂ → Matrix m n ℂ) (η lam : ℝ) {z : k → ℂ}
    (hz : stack U₀ V₀ *ᵥ z = 0) (t : ℕ) :
    stack (decayRun G η lam U₀ V₀ t).1 (decayRun G η lam U₀ V₀ t).2 *ᵥ z = 0 := by
  induction t with
  | zero => exact hz
  | succ t ih => rw [stack_decayRun_succ, ← Matrix.mulVec_mulVec, ih, Matrix.mulVec_zero]

/-- **A kernel vector of the stack of the start is one of the stack of the limit**
(`review/flat_spectrum.md`, item WD14). -/
theorem stack_mulVec_eq_zero_of_tendsto {z : k → ℂ} (hz : stack U₀ V₀ *ᵥ z = 0)
    (hlim : Tendsto (decayRun G η lam U₀ V₀) atTop (𝓝 (U, V))) : stack U V *ᵥ z = 0 := by
  have hc : Continuous fun W : Matrix m k ℂ × Matrix n k ℂ => stack W.1 W.2 *ᵥ z := by
    simp only [stack, fromRows_mulVec]
    refine continuous_pi fun i => ?_
    cases i with
    | inl i => exact (continuous_apply i).comp (continuous_fst.matrix_mulVec continuous_const)
    | inr i => exact (continuous_apply i).comp (continuous_snd.matrix_mulVec continuous_const)
  exact tendsto_nhds_unique ((hc.tendsto (U, V)).comp hlim)
    (tendsto_const_nhds.congr fun t => (stack_decayRun_mulVec_eq_zero G η lam hz t).symm)

omit [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n] in
/-- Rank–nullity: a matrix that vanishes on the kernel of another has at most its rank. -/
lemma rank_le_of_forall_mulVec_eq_zero {p q : Type*} [Fintype p] [Fintype q]
    {A : Matrix p k ℂ} {B : Matrix q k ℂ} (h : ∀ z, B *ᵥ z = 0 → A *ᵥ z = 0) :
    A.rank ≤ B.rank := by
  have hker : LinearMap.ker B.mulVecLin ≤ LinearMap.ker A.mulVecLin := fun z hz => by
    rw [LinearMap.mem_ker, Matrix.mulVecLin_apply] at hz ⊢
    exact h z hz
  have h1 := LinearMap.finrank_range_add_finrank_ker A.mulVecLin
  have h2 := LinearMap.finrank_range_add_finrank_ker B.mulVecLin
  have h3 := Submodule.finrank_mono hker
  rw [Matrix.rank, Matrix.rank]
  omega

/-- **The rank of the stack does not grow along the run** (`review/flat_spectrum.md`, item
WD14): `rank [U_t; V_t] ≤ rank [U₀; V₀]`. -/
theorem rank_stack_decayRun_le (G : Matrix m n ℂ → Matrix m n ℂ) (η lam : ℝ)
    (U₀ : Matrix m k ℂ) (V₀ : Matrix n k ℂ) (t : ℕ) :
    (stack (decayRun G η lam U₀ V₀ t).1 (decayRun G η lam U₀ V₀ t).2).rank ≤
      (stack U₀ V₀).rank :=
  rank_le_of_forall_mulVec_eq_zero fun _ hz => stack_decayRun_mulVec_eq_zero G η lam hz t

/-- **The product of a limit has at most the rank of the stack of the start**
(`review/flat_spectrum.md`, item WD14): `rank (U Vᴴ) ≤ rank [U₀; V₀]`. -/
theorem rank_mul_conjTranspose_le_of_tendsto
    (hlim : Tendsto (decayRun G η lam U₀ V₀) atTop (𝓝 (U, V))) :
    (U * Vᴴ).rank ≤ (stack U₀ V₀).rank := by
  refine (Matrix.rank_mul_le_left U Vᴴ).trans (rank_le_of_forall_mulVec_eq_zero fun z hz => ?_)
  have h := stack_mulVec_eq_zero_of_tendsto hz hlim
  rw [stack, fromRows_mulVec] at h
  funext i
  exact congrFun h (Sum.inl i)

/-- **A run reaches no trained product of rank above that of its start**
(`review/flat_spectrum.md`, item WD14): for `λ ≥ 0`, if every minimizer of `L + λ ‖·‖_S1` has a
rank above `rank [U₀; V₀]`, no limit of the run is a decayed minimum. -/
theorem not_isDecayedMin_of_rank_lt (hlam : 0 ≤ lam)
    (hrank : ∀ X : Matrix m n ℂ, IsNuclearMin L lam X → (stack U₀ V₀).rank < X.rank)
    (hlim : Tendsto (decayRun G η lam U₀ V₀) atTop (𝓝 (U, V))) : ¬ IsDecayedMin L lam U V :=
  fun h => (hrank _ (h.isNuclearMin hlam)).not_ge (rank_mul_conjTranspose_le_of_tendsto hlim)

/-! The witness of the first statements: the run from the zero factors, which stays there for
every gradient field. The witness of the last one is in `RankStart`. -/

omit [DecidableEq m] [DecidableEq n] in
private lemma decayRun_zero_start (G : Matrix m n ℂ → Matrix m n ℂ) (η lam : ℝ) (t : ℕ) :
    decayRun G η lam (0 : Matrix m k ℂ) (0 : Matrix n k ℂ) t = (0, 0) := by
  induction t with
  | zero => rfl
  | succ t ih =>
    rw [decayRun, Function.iterate_succ_apply', ← decayRun, ih]
    simp [decayStep]

omit [DecidableEq m] [DecidableEq n] in
private lemma tendsto_zero_start (G : Matrix m n ℂ → Matrix m n ℂ) (η lam : ℝ) :
    Tendsto (decayRun G η lam (0 : Matrix m k ℂ) (0 : Matrix n k ℂ)) atTop (𝓝 (0, 0)) :=
  tendsto_const_nhds.congr fun t => (decayRun_zero_start G η lam t).symm

example := stack_decayRun_mulVec_eq_zero (fun X : Matrix (Fin 2) (Fin 2) ℂ => X - 1) 1 1
  (U₀ := (0 : Matrix (Fin 2) (Fin 2) ℂ)) (V₀ := 0) (z := 1)
  (by rw [stack_zero, Matrix.zero_mulVec]) 3

example := stack_mulVec_eq_zero_of_tendsto (z := (1 : Fin 2 → ℂ))
  (by rw [stack_zero, Matrix.zero_mulVec])
  (tendsto_zero_start (fun X : Matrix (Fin 2) (Fin 2) ℂ => X - 1) 1 1)

example := rank_le_of_forall_mulVec_eq_zero (A := (0 : Matrix (Fin 2) (Fin 2) ℂ))
  (B := (1 : Matrix (Fin 2) (Fin 2) ℂ)) fun z _ => Matrix.zero_mulVec z

example := rank_mul_conjTranspose_le_of_tendsto
  (tendsto_zero_start (k := Fin 2) (fun X : Matrix (Fin 2) (Fin 2) ℂ => X - 1) 1 1)

end Transformer.Grokking.FlatSpectrum.Training.Scope
