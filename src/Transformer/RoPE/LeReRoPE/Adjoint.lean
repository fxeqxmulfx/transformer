module

public import Transformer.RoPE.LeRoPE.Section3_LearnedFrequencies
public import Transformer.RoPE.RoFormer.Section3_Rotary
public import Mathlib.Analysis.Calculus.Deriv.Prod
public import Mathlib.Analysis.Complex.RealDeriv
public import Mathlib.Analysis.SpecialFunctions.ExpDeriv

/-!
# LeReRoPE: the frequency gradient from the rotated chunks

In no paper. A kernel does not form the pairs of LeRoPE's Eq. (6) (Karypis et al.,
arXiv:2607.10134v1, §3.2): its backward pass returns the gradient `dx̃_i` of every rotated chunk
`x̃_i = x_i e^{i a_i θ}` of band `m`, and the frequency's gradient is read off them.

* `cgrad`, `apply_eq_sum_inner_cgrad`: a real-linear functional on `ι → ℂ` is
  `v ↦ Σ_i ⟪dx_i, v_i⟫` for the gradients `dx_i ∈ ℂ = ℝ²` of its coordinates;
* `torque`: `Im(x̃_i^* dx̃_i)`, `x̃_re dx̃_im − x̃_im dx̃_re` in `ℝ²`, the turn the gradient gives
  the chunk;
* `hasDerivAt_rotate_adjoint`: for every loss differentiable in the rotated chunks,
  `∂L/∂θ = Σ_i a_i Im(x̃_i^* dx̃_i)`, a sum over positions, not pairs;
* `adjoint_eq_pairwise`: when the gradients come from the weights `G_ij` of the products of the
  rotated queries `x̃_i` and keys `ỹ_j`, `dx̃_i = Σ_j G_ij ỹ_j` and `dỹ_j = Σ_i G_ij x̃_i`, the sum
  is Eq. (6)'s pairwise `Σ_i Σ_j G_ij (a_i − b_j) Im(x̃_i^* ỹ_j)`;
* `hasDerivAt_rerope_log_frequency`: ReRoPE (Su, kexue.fm/archives/9708) rotates the near queries
  to `s`, the near keys to `t`, the far queries to the window `w`, and not the far keys
  (`Transformer.RoPE.ReRoPE.rerope_attend_eq_onlineMerge`). In LeRoPE's parameter,
  `θ̂ = e^α θ` (Eq. (5)), `∂L/∂α = θ̂ (Σ_s s τ_s + Σ_t t τ'_t + w Σ_s τ''_s)` for the torques of the
  near queries, near keys and far queries: the far keys add nothing.

Leaky ReRoPE rotates the far queries to `w + (s − w)/k` and the far keys to `t/k`;
`hasDerivAt_rotate_adjoint` applies to those positions as well.
-/

@[expose] public section

open scoped ComplexConjugate InnerProductSpace
open Finset

namespace Transformer.RoPE.LeReRoPE

open Transformer.RoPE.LeRoPE Transformer.RoPE.RoFormer

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- The rotation's derivative: `d/dθ (x e^{iaθ}) = i a x e^{iaθ}`. -/
theorem hasDerivAt_rotate (a : ℝ) (x : ℂ) (θ : ℝ) :
    HasDerivAt (fun θ => rotate θ a x) (a * Complex.I * rotate θ a x) θ := by
  refine (((((hasDerivAt_id' (x := θ)).const_mul a).ofReal_comp).mul_const
    Complex.I).cexp.const_mul x).congr_deriv ?_
  simp only [rotate, mul_one]
  ring

/-- The gradient `dx_i ∈ ℂ = ℝ²` of a real-linear functional `L'` on `ι → ℂ` in the
coordinate `i`: its values on `1` and on `i` there. -/
noncomputable def cgrad (L' : (ι → ℂ) →L[ℝ] ℝ) (i : ι) : ℂ :=
  ⟨L' (Pi.single i (1 : ℂ) : ι → ℂ), L' (Pi.single i Complex.I : ι → ℂ)⟩

omit [Fintype ι] in
/-- On one coordinate a real-linear functional is the inner product with its gradient there. -/
theorem apply_single_eq_inner_cgrad (L' : (ι → ℂ) →L[ℝ] ℝ) (i : ι) (z : ℂ) :
    L' (Pi.single i z) = ⟪cgrad L' i, z⟫_ℝ := by
  have h : Pi.single i z =
      z.re • (Pi.single i (1 : ℂ) : ι → ℂ) + z.im • (Pi.single i Complex.I : ι → ℂ) := by
    ext j
    by_cases hj : j = i
    · subst hj
      simp only [Pi.add_apply, Pi.smul_apply, Pi.single_eq_same, Complex.real_smul, mul_one]
      exact (Complex.re_add_im z).symm
    · simp [hj]
  rw [h, map_add, map_smul, map_smul, Complex.inner, cgrad]
  simp only [smul_eq_mul, Complex.mul_re, Complex.conj_re, Complex.conj_im]
  ring

/-- **A real-linear functional is the sum of its gradients' inner products**:
`L' v = Σ_i ⟪dx_i, v_i⟫`. -/
theorem apply_eq_sum_inner_cgrad (L' : (ι → ℂ) →L[ℝ] ℝ) (v : ι → ℂ) :
    L' v = ∑ i, ⟪cgrad L' i, v i⟫_ℝ := by
  conv_lhs => rw [← Finset.univ_sum_single v]
  rw [map_sum]
  exact sum_congr rfl fun i _ => apply_single_eq_inner_cgrad L' i (v i)

/-- The torque `Im(x̃_i^* dx̃_i)` of the gradient `dx̃_i` on the chunk `x̃_i`. -/
noncomputable def torque (L' : (ι → ℂ) →L[ℝ] ℝ) (x : ι → ℂ) (i : ι) : ℝ :=
  (conj (x i) * cgrad L' i).im

/-- **The frequency gradient from the rotated chunks**: for a loss `L` of the chunks `x_i` rotated
to the positions `a_i`, differentiable at `x̃(θ)`, `x̃_i = x_i e^{i a_i θ}`, with the derivative
`L'`, `∂L/∂θ = Σ_i a_i Im(x̃_i^* dx̃_i)`. -/
theorem hasDerivAt_rotate_adjoint (a : ι → ℝ) (x : ι → ℂ) {L : (ι → ℂ) → ℝ}
    {L' : (ι → ℂ) →L[ℝ] ℝ} {θ : ℝ} (hL : HasFDerivAt L L' fun i => rotate θ (a i) (x i)) :
    HasDerivAt (fun θ => L fun i => rotate θ (a i) (x i))
      (∑ i, a i * torque L' (fun i => rotate θ (a i) (x i)) i) θ := by
  have hx : HasDerivAt (fun θ i => rotate θ (a i) (x i))
      (fun i => (a i : ℂ) * Complex.I * rotate θ (a i) (x i)) θ :=
    hasDerivAt_pi.2 fun i => hasDerivAt_rotate (a i) (x i) θ
  refine (hL.comp_hasDerivAt θ hx).congr_deriv ?_
  rw [apply_eq_sum_inner_cgrad]
  refine sum_congr rfl fun i _ => ?_
  simp only [torque, Complex.inner, Complex.mul_re, Complex.mul_im, Complex.conj_re,
    Complex.conj_im, Complex.I_re, Complex.I_im, Complex.ofReal_re, Complex.ofReal_im]
  ring

/-- The hypothesis of `hasDerivAt_rotate_adjoint` holds for the loss that reads the real part of
one rotated chunk. -/
example (a : Fin 2 → ℝ) (x : Fin 2 → ℂ) (θ : ℝ) :=
  hasDerivAt_rotate_adjoint a x (θ := θ)
    (Complex.reCLM.comp (ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : Fin 2 => ℂ) 0)).hasFDerivAt

omit [DecidableEq ι] in
/-- **The torque sum is Eq. (6)'s pairwise sum**: when the gradients of the queries `x̃_i` at the
positions `a_i` and of the keys `ỹ_j` at `b_j` come from the weights `G_ij` of their products,
`Σ_i a_i Im(x̃_i^* Σ_j G_ij ỹ_j) + Σ_j b_j Im(ỹ_j^* Σ_i G_ij x̃_i)
  = Σ_i Σ_j G_ij (a_i − b_j) Im(x̃_i^* ỹ_j)`. -/
theorem adjoint_eq_pairwise {κ : Type*} [Fintype κ] (a : ι → ℝ) (b : κ → ℝ) (x : ι → ℂ)
    (y : κ → ℂ) (G : ι → κ → ℝ) :
    ∑ i, a i * (conj (x i) * ∑ j, (G i j : ℂ) * y j).im +
        ∑ j, b j * (conj (y j) * ∑ i, (G i j : ℂ) * x i).im =
      ∑ i, ∑ j, G i j * (a i - b j) * (conj (x i) * y j).im := by
  have h : ∑ j, b j * (conj (y j) * ∑ i, (G i j : ℂ) * x i).im =
      ∑ i, ∑ j, b j * (conj (y j) * ((G i j : ℂ) * x i)).im := by
    simp only [mul_sum, Complex.im_sum]
    exact sum_comm
  rw [h]
  simp only [mul_sum, Complex.im_sum]
  rw [← sum_add_distrib]
  refine sum_congr rfl fun i _ => ?_
  rw [← sum_add_distrib]
  refine sum_congr rfl fun j _ => ?_
  simp only [Complex.mul_im, Complex.mul_re, Complex.conj_re, Complex.conj_im, Complex.ofReal_re,
    Complex.ofReal_im]
  ring

variable {n : ℕ}

/-- ReRoPE's rotated chunks of band `m` in three families: `(0, s)` the near query at `s`, `(1, t)`
the near key at `t`, `(2, s)` the far query at the window `w`. The far keys are not rotated. -/
noncomputable def reropePosition (w : ℝ) (p : Fin 3 × Fin n) : ℝ :=
  ![((p.2 : ℕ) : ℝ), ((p.2 : ℕ) : ℝ), w] p.1

/-- **LeReRoPE's frequency gradient, as a kernel computes it**: for a loss of band `m`'s chunks
rotated to ReRoPE's positions at `θ̂ = e^α θ`, differentiable there,
`∂L/∂α = θ̂ (Σ_s s τ(0, s) + Σ_t t τ(1, t) + w Σ_s τ(2, s))` with the torques `τ` of the near
queries, the near keys and the far queries. -/
theorem hasDerivAt_rerope_log_frequency (w : ℝ) (x : Fin 3 × Fin n → ℂ)
    {L : (Fin 3 × Fin n → ℂ) → ℝ} {L' : (Fin 3 × Fin n → ℂ) →L[ℝ] ℝ} {θ α : ℝ}
    (hL : HasFDerivAt L L' fun p => rotate (learnedFrequency θ α) (reropePosition w p) (x p)) :
    HasDerivAt (fun α => L fun p => rotate (learnedFrequency θ α) (reropePosition w p) (x p))
      (learnedFrequency θ α *
        (∑ s : Fin n, ((s : ℕ) : ℝ) * torque L'
            (fun p => rotate (learnedFrequency θ α) (reropePosition w p) (x p)) (0, s) +
          ∑ t : Fin n, ((t : ℕ) : ℝ) * torque L'
            (fun p => rotate (learnedFrequency θ α) (reropePosition w p) (x p)) (1, t) +
          w * ∑ s : Fin n, torque L'
            (fun p => rotate (learnedFrequency θ α) (reropePosition w p) (x p)) (2, s))) α := by
  refine ((hasDerivAt_rotate_adjoint (reropePosition w) x hL).comp α
    (learned_frequency_hasDerivAt θ α)).congr_deriv ?_
  rw [Fintype.sum_prod_type, Fin.sum_univ_three, mul_sum]
  simp only [reropePosition, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two,
    Matrix.head_cons, Matrix.tail_cons]
  ring

/-- The hypothesis of `hasDerivAt_rerope_log_frequency` holds for the window `1024` and the loss
that reads the real part of the near query at `0`. -/
example (x : Fin 3 × Fin 2 → ℂ) (θ α : ℝ) :=
  hasDerivAt_rerope_log_frequency 1024 x (θ := θ) (α := α)
    (Complex.reCLM.comp (ContinuousLinearMap.proj (R := ℝ)
      (φ := fun _ : Fin 3 × Fin 2 => ℂ) (0, 0))).hasFDerivAt

end Transformer.RoPE.LeReRoPE
