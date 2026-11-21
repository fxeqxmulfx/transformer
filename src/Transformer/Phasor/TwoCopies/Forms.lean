module

public import Transformer.Phasor.TwoCopies.States

/-!
# Two copies: complex forms on the states

Source: `review/flat_spectrum.md`, *Functions of n variables*, NV8 (added), task T3c.

The gate and the value of a unit of main's FFN are real-linear forms into `ℂ`. On a state of the
construction a form with weights on the codes reads one code per position, `∑_j W(j, t_j) c_j`,
because the position `j` holds the code of its own token only (`slotForm_codeWeight`). A form that
reads a phase plane as `x₀ + i x₁` gets `P_c ψ(w_c)` (`slotForm_planeWeight`), with
`ψ(v) = cos θ_v + i sin θ_v` (`phase`), and `ψ(v + w) = ψ(v) ψ(w)` (`phase_add`). A state with
nothing on the bias and the codes is its phase planes and its output plane (`vec_stateC_planes`).
-/

@[expose] public section

open Finset Real

namespace Transformer.Phasor.TwoCopies

variable {D p : ℕ}

/-- `ψ(v) = e^{2πi v/p}` as a complex number, `cos θ_v + i sin θ_v`. -/
noncomputable def phase (p : ℕ) (v : Fin p) : ℂ := ⟨cos (angle p v), sin (angle p v)⟩

/-- **`ψ(v + w) = ψ(v) ψ(w)`.** -/
theorem phase_add (v w : Fin p) : phase p (v + w) = phase p v * phase p w := by
  apply Complex.ext
  · simp only [phase, Complex.mul_re, cos_angle_add, cos_add]
  · simp only [phase, Complex.mul_im, sin_angle_add, sin_add]
    ring

/-- The coefficient of the code of the position `j` in a state: `k` for the query's own token,
`a` for the others. -/
def codeCoef (k a : ℝ) (j : Fin 4) : ℝ := if j = 3 then k else a

/-- A complex form with the weights `W` on the codes and none elsewhere. -/
noncomputable def codeWeight (W : Fin 4 × Fin p → ℂ) : Slot p → ℂ
  | .inl _ => 0
  | .inr (.inl x) => W x
  | .inr (.inr _) => 0

variable {e : Slot p → EucSpace D}

/-- **A form on the codes reads one code per position**: `∑_j W(j, t_j) c_j`. -/
theorem slotForm_codeWeight (he : Orthonormal ℝ e) (W : Fin 4 × Fin p → ℂ) (b k a : ℝ)
    (t : Fin 4 → Fin p) (P : Fin 2 → ℝ) (w : Fin 2 → Fin p) (M : ℝ) (u : Fin p) :
    slotForm e (codeWeight W) (vec e (stateC b k a t P w M u)) =
      ∑ j, W (j, t j) * codeCoef k a j := by
  rw [slotForm_vec he, Fintype.sum_sum_type, Fintype.sum_sum_type]
  simp only [codeWeight, zero_mul, sum_const_zero, zero_add, add_zero]
  rw [Fintype.sum_prod_type]
  refine sum_congr rfl fun j _ => ?_
  rw [sum_eq_single (t j) (fun y _ hy => by simp [stateC, hy]) (by simp)]
  simp [stateC, codeCoef]

/-- A complex form that reads the phase plane `c` as `x₀ + i x₁`. -/
noncomputable def planeWeight (c : Fin 2) : Slot p → ℂ
  | .inl _ => 0
  | .inr (.inl _) => 0
  | .inr (.inr (.inl x)) => if x.1 = c then (if x.2 = 0 then 1 else Complex.I) else 0
  | .inr (.inr (.inr _)) => 0

/-- **The plane `c` of a state reads as `P_c ψ(w_c)`.** -/
theorem slotForm_planeWeight (he : Orthonormal ℝ e) (c : Fin 2) (b k a : ℝ) (t : Fin 4 → Fin p)
    (P : Fin 2 → ℝ) (w : Fin 2 → Fin p) (M : ℝ) (u : Fin p) :
    slotForm e (planeWeight c) (vec e (stateC b k a t P w M u)) = P c * phase p (w c) := by
  rw [slotForm_vec he, Fintype.sum_sum_type, Fintype.sum_sum_type, Fintype.sum_sum_type]
  simp only [planeWeight, zero_mul, sum_const_zero, zero_add, add_zero]
  rw [Fintype.sum_prod_type, sum_eq_single c (fun c' _ hc' => by simp [hc']) (by simp)]
  apply Complex.ext <;>
    simp [Fin.sum_univ_two, stateC, circ, phase, Complex.cos_ofReal_re, Complex.sin_ofReal_re]

/-- **A state with nothing on the bias and the codes** is its phase planes and its output
plane. -/
theorem vec_stateC_planes (e : Slot p → EucSpace D) (b : ℝ) (t : Fin 4 → Fin p) (P : Fin 2 → ℝ)
    (w : Fin 2 → Fin p) (M : ℝ) (u : Fin p) :
    vec e (stateC b 0 0 t P w M u) =
      ∑ c, ∑ r, (P c * circ p (w c) r) • e (.inr (.inr (.inl (c, r)))) +
        ∑ r, (M * circ p u r) • e (.inr (.inr (.inr r))) := by
  rw [vec, Fintype.sum_sum_type, Fintype.sum_sum_type, Fintype.sum_sum_type,
    Fintype.sum_prod_type (fun x => stateC b 0 0 t P w M u (.inr (.inr (.inl x))) • _)]
  simp [stateC]

/-- The output plane alone has the squared length `M²`. -/
theorem sum_stateC_out_sq (b : ℝ) (t : Fin 4 → Fin p) (w : Fin 2 → Fin p) (M : ℝ) (u : Fin p) :
    ∑ s, stateC b 0 0 t 0 w M u s ^ 2 = M ^ 2 := by
  have h := sum_circ_sq u
  rw [Fin.sum_univ_two] at h
  rw [Fintype.sum_sum_type, Fintype.sum_sum_type, Fintype.sum_sum_type]
  simp [stateC, Fin.sum_univ_two]
  linear_combination M ^ 2 * h

/-- At `p = 29` on the coordinate slots of `D = 128`: a form on the codes. -/
example (W : Fin 4 × Fin 29 → ℂ) (b k a : ℝ) (t : Fin 4 → Fin 29) (P : Fin 2 → ℝ)
    (w : Fin 2 → Fin 29) (M : ℝ) (u : Fin 29) :=
  slotForm_codeWeight (orthonormal_coordSlots (by norm_num : 4 * 29 + 7 ≤ 128)) W b k a t P w M u

/-- The same with the plane `0`. -/
example (b k a : ℝ) (t : Fin 4 → Fin 29) (P : Fin 2 → ℝ) (w : Fin 2 → Fin 29) (M : ℝ)
    (u : Fin 29) :=
  slotForm_planeWeight (orthonormal_coordSlots (by norm_num : 4 * 29 + 7 ≤ 128)) 0 b k a t P w M u

end Transformer.Phasor.TwoCopies
