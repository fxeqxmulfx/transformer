module

public import Transformer.RoPE.LeReRoPE.Adjoint
public import Transformer.RoPE.LeReRoPE.Window
public import Mathlib.Analysis.Calculus.Deriv.Comp

/-!
# LeReRoPE: a learned window and slope in the kernel

In no paper. One pass computes a head's learned window `w` and slope `s`
(`Transformer.RoPE.LeReRoPE.route_eq_windowPosition`): it rotates the near query to `t`, the near
key to `j`, the far query to `w + s(t − w)` and the far key to `s j`. The kernel's backward pass
returns the gradients `dx̃_i` of the rotated chunks, and the window's and the slope's gradients are
read off the torques `Im(x̃_i^* dx̃_i)`, as the frequency's is (`Transformer.RoPE.LeReRoPE.Adjoint`).

* `hasDerivAt_position_adjoint`: for chunks rotated at the frequencies `θ_i` to the positions
  `P_i(u)`, `∂L/∂u = Σ_i θ_i P_i'(u) Im(x̃_i^* dx̃_i)`;
* `windowChunkPosition`, `windowChunkPosition_sub`: the pass's four families of chunks; the query
  at `t` and the key at `j` meet through the near chunks inside the window and through the far
  chunks beyond it, `p(t − j)` apart either way;
* `hasDerivAt_window_chunk_slope`: over the head's bands `m`,
  `∂L/∂s = Σ_m θ_m (Σ_t (t − w) τ(m, 2, t) + Σ_j j τ(m, 3, j))`, the far queries' torques weighed
  by their excess over the window and the far keys' by their positions;
* `hasDerivAt_window_chunk_window`: `∂L/∂w = (1 − s) Σ_m θ_m Σ_t τ(m, 2, t)`, the far queries'
  torques unweighed, and no far key's. ReRoPE's frequency gradient already sums them
  (`Transformer.RoPE.LeReRoPE.hasDerivAt_rerope_log_frequency`).

`Transformer.RoPE.LeReRoPE.hasDerivAt_rotate_adjoint` gives the frequency's gradient at these
positions too.
-/

@[expose] public section

open scoped ComplexConjugate InnerProductSpace
open Finset

namespace Transformer.RoPE.LeReRoPE

open Transformer.RoPE.RoFormer

/-- The rotation's derivative in the position: `d/dm (x e^{imθ}) = i θ x e^{imθ}`. -/
theorem hasDerivAt_rotate_position (θ : ℝ) (x : ℂ) (m : ℝ) :
    HasDerivAt (fun m => rotate θ m x) (θ * Complex.I * rotate θ m x) m := by
  refine (((((hasDerivAt_id' (x := m)).mul_const θ).ofReal_comp).mul_const
    Complex.I).cexp.const_mul x).congr_deriv ?_
  simp only [rotate, one_mul]
  ring

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- **A position's gradient from the rotated chunks**: for a loss `L` of the chunks `x_i` rotated
at the frequencies `θ_i` to the positions `P_i(u)`, with the derivatives `P_i'` in a parameter
`u`, differentiable at the rotated chunks with the derivative `L'`,
`∂L/∂u = Σ_i θ_i P_i' Im(x̃_i^* dx̃_i)`. -/
theorem hasDerivAt_position_adjoint (θ : ι → ℝ) (x : ι → ℂ) {P : ℝ → ι → ℝ} {P' : ι → ℝ}
    {u : ℝ} (hP : ∀ i, HasDerivAt (fun u => P u i) (P' i) u) {L : (ι → ℂ) → ℝ}
    {L' : (ι → ℂ) →L[ℝ] ℝ} (hL : HasFDerivAt L L' fun i => rotate (θ i) (P u i) (x i)) :
    HasDerivAt (fun u => L fun i => rotate (θ i) (P u i) (x i))
      (∑ i, θ i * P' i * torque L' (fun i => rotate (θ i) (P u i) (x i)) i) u := by
  have hx : HasDerivAt (fun u i => rotate (θ i) (P u i) (x i))
      (fun i => P' i • ((θ i : ℂ) * Complex.I * rotate (θ i) (P u i) (x i))) u := by
    refine hasDerivAt_pi.2 fun i => ?_
    have h := (hasDerivAt_rotate_position (θ i) (x i) (P u i)).scomp u (hP i)
    exact h
  refine (hL.comp_hasDerivAt u hx).congr_deriv ?_
  rw [apply_eq_sum_inner_cgrad]
  refine sum_congr rfl fun i _ => ?_
  simp only [torque, Complex.real_smul, Complex.inner, Complex.mul_re, Complex.mul_im,
    Complex.conj_re, Complex.conj_im, Complex.I_re, Complex.I_im, Complex.ofReal_re,
    Complex.ofReal_im]
  ring

/-- The hypotheses of `hasDerivAt_position_adjoint` hold for two chunks at the position `u` and
the loss that reads the real part of the first. -/
example (θ : Fin 2 → ℝ) (x : Fin 2 → ℂ) (u : ℝ) :=
  hasDerivAt_position_adjoint θ x (P := fun u _ => u) (fun _ => hasDerivAt_id' (x := u))
    (Complex.reCLM.comp (ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : Fin 2 => ℂ) 0)).hasFDerivAt

variable {β : Type*} [Fintype β] [DecidableEq β] {n : ℕ}

/-- A pass's rotated chunks of one band in four families: `(0, t)` the near query at `t`, `(1, j)`
the near key at `j`, `(2, t)` the far query at `w + s(t − w)`, `(3, j)` the far key at `s j`. -/
noncomputable def windowChunkPosition (w s : ℝ) (p : Fin 4 × Fin n) : ℝ :=
  ![((p.2 : ℕ) : ℝ), ((p.2 : ℕ) : ℝ), w + s * (((p.2 : ℕ) : ℝ) - w), s * ((p.2 : ℕ) : ℝ)] p.1

/-- **The pass computes the map**: the query at `t` and the key at `j` meet through the near
chunks when `t − j < w` and through the far chunks otherwise, `p(t − j)` apart either way. -/
theorem windowChunkPosition_sub (w s : ℝ) (t j : Fin n) :
    (if ((t : ℕ) : ℝ) - ((j : ℕ) : ℝ) < w then
        windowChunkPosition w s (0, t) - windowChunkPosition w s (1, j)
      else windowChunkPosition w s (2, t) - windowChunkPosition w s (3, j)) =
      windowPosition w s (((t : ℕ) : ℝ) - ((j : ℕ) : ℝ)) := by
  simp only [windowChunkPosition, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two,
    Matrix.cons_val_three, Matrix.head_cons, Matrix.tail_cons]
  exact route_eq_windowPosition w s _ _

/-- **The slope's gradient, as a kernel computes it**: for a loss of a head's chunks, band `m`
rotated at `θ_m` to the pass's positions, differentiable there,
`∂L/∂s = Σ_m θ_m (Σ_t (t − w) τ(m, 2, t) + Σ_j j τ(m, 3, j))` with the torques `τ` of the far
queries and the far keys. -/
theorem hasDerivAt_window_chunk_slope (θ : β → ℝ) (w : ℝ) (x : β × Fin 4 × Fin n → ℂ)
    {L : (β × Fin 4 × Fin n → ℂ) → ℝ} {L' : (β × Fin 4 × Fin n → ℂ) →L[ℝ] ℝ} {s : ℝ}
    (hL : HasFDerivAt L L' fun p => rotate (θ p.1) (windowChunkPosition w s p.2) (x p)) :
    HasDerivAt (fun σ => L fun p => rotate (θ p.1) (windowChunkPosition w σ p.2) (x p))
      (∑ m, θ m *
        (∑ t : Fin n, (((t : ℕ) : ℝ) - w) *
            torque L' (fun p => rotate (θ p.1) (windowChunkPosition w s p.2) (x p)) (m, 2, t) +
          ∑ j : Fin n, ((j : ℕ) : ℝ) *
            torque L' (fun p => rotate (θ p.1) (windowChunkPosition w s p.2) (x p)) (m, 3, j)))
      s := by
  have hP : ∀ p : β × Fin 4 × Fin n, HasDerivAt (fun σ => windowChunkPosition w σ p.2)
      (![0, 0, ((p.2.2 : ℕ) : ℝ) - w, ((p.2.2 : ℕ) : ℝ)] p.2.1) s := by
    rintro ⟨m, k, i⟩
    dsimp only
    have hf : (fun σ => windowChunkPosition w σ (k, i)) = fun σ =>
        ![((i : ℕ) : ℝ), ((i : ℕ) : ℝ), w, 0] k +
          σ * ![0, 0, ((i : ℕ) : ℝ) - w, ((i : ℕ) : ℝ)] k := by
      funext σ
      fin_cases k <;> simp [windowChunkPosition]
    have h := ((hasDerivAt_id' (x := s)).mul_const
      (![0, 0, ((i : ℕ) : ℝ) - w, ((i : ℕ) : ℝ)] k)).const_add
        (![((i : ℕ) : ℝ), ((i : ℕ) : ℝ), w, 0] k)
    rw [one_mul] at h
    rw [hf]
    exact h
  refine (hasDerivAt_position_adjoint (fun p => θ p.1) x hP hL).congr_deriv ?_
  rw [Fintype.sum_prod_type]
  refine sum_congr rfl fun m _ => ?_
  rw [Fintype.sum_prod_type, Fin.sum_univ_four]
  simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two,
    Matrix.cons_val_three, Matrix.head_cons, Matrix.tail_cons, mul_zero, zero_mul,
    sum_const_zero, zero_add, mul_add, mul_sum, mul_assoc]

/-- The hypothesis of `hasDerivAt_window_chunk_slope` holds for the window `16` and the loss that
reads the real part of the far query at `0`. -/
example (θ : Unit → ℝ) (x : Unit × Fin 4 × Fin 2 → ℂ) (s : ℝ) :=
  hasDerivAt_window_chunk_slope θ 16 x (s := s)
    (Complex.reCLM.comp (ContinuousLinearMap.proj (R := ℝ)
      (φ := fun _ : Unit × Fin 4 × Fin 2 => ℂ) ((), 2, 0))).hasFDerivAt

/-- **The window's gradient, as a kernel computes it**: for the same loss,
`∂L/∂w = (1 − s) Σ_m θ_m Σ_t τ(m, 2, t)`, the far queries' torques unweighed by their positions,
and none of the far keys'. -/
theorem hasDerivAt_window_chunk_window (θ : β → ℝ) (s : ℝ) (x : β × Fin 4 × Fin n → ℂ)
    {L : (β × Fin 4 × Fin n → ℂ) → ℝ} {L' : (β × Fin 4 × Fin n → ℂ) →L[ℝ] ℝ} {w : ℝ}
    (hL : HasFDerivAt L L' fun p => rotate (θ p.1) (windowChunkPosition w s p.2) (x p)) :
    HasDerivAt (fun ω => L fun p => rotate (θ p.1) (windowChunkPosition ω s p.2) (x p))
      ((1 - s) * ∑ m, θ m * ∑ t : Fin n,
        torque L' (fun p => rotate (θ p.1) (windowChunkPosition w s p.2) (x p)) (m, 2, t)) w := by
  have hP : ∀ p : β × Fin 4 × Fin n, HasDerivAt (fun ω => windowChunkPosition ω s p.2)
      (![0, 0, 1 - s, 0] p.2.1) w := by
    rintro ⟨m, k, i⟩
    dsimp only
    have hf : (fun ω => windowChunkPosition ω s (k, i)) = fun ω =>
        ![((i : ℕ) : ℝ), ((i : ℕ) : ℝ), s * ((i : ℕ) : ℝ), s * ((i : ℕ) : ℝ)] k +
          ω * ![0, 0, 1 - s, 0] k := by
      funext ω
      fin_cases k <;> simp [windowChunkPosition]
      ring
    have h := ((hasDerivAt_id' (x := w)).mul_const (![0, 0, 1 - s, 0] k)).const_add
      (![((i : ℕ) : ℝ), ((i : ℕ) : ℝ), s * ((i : ℕ) : ℝ), s * ((i : ℕ) : ℝ)] k)
    rw [one_mul] at h
    rw [hf]
    exact h
  refine (hasDerivAt_position_adjoint (fun p => θ p.1) x hP hL).congr_deriv ?_
  rw [mul_sum, Fintype.sum_prod_type]
  refine sum_congr rfl fun m _ => ?_
  rw [Fintype.sum_prod_type, Fin.sum_univ_four]
  simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two,
    Matrix.cons_val_three, Matrix.head_cons, Matrix.tail_cons, mul_zero, zero_mul,
    sum_const_zero, zero_add, add_zero, mul_sum]
  refine sum_congr rfl fun t _ => ?_
  ring

/-- The hypothesis of `hasDerivAt_window_chunk_window` holds for the slope `1/16` and the same
loss. -/
example (θ : Unit → ℝ) (x : Unit × Fin 4 × Fin 2 → ℂ) (w : ℝ) :=
  hasDerivAt_window_chunk_window θ (1 / 16) x (w := w)
    (Complex.reCLM.comp (ContinuousLinearMap.proj (R := ℝ)
      (φ := fun _ : Unit × Fin 4 × Fin 2 => ℂ) ((), 2, 0))).hasFDerivAt

end Transformer.RoPE.LeReRoPE
