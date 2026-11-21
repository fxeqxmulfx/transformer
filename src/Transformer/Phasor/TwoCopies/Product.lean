module

public import Transformer.Phasor.TwoCopies.Forms

/-!
# Two copies: the feed-forward branches multiply phases

Source: `review/flat_spectrum.md`, *Functions of n variables*, NV8 (added), task T3c.

Main's FFN is a sum of products `(g_u z)(v_u z)/√2` of two complex forms, read out by its real and
imaginary parts. A branch with fewer units than `F` puts them at some places and zeros elsewhere
(`ffnOf`, `plainFFN_ffnOf`).

The first branch computes the phases of the two copies, any `q_c : ℤ/p × ℤ/p → ℤ/p`. Its unit
`(c, a)` multiplies the indicator of `x_c = a`, a form on the code of `x_c`, by `ψ(q_c(a, y_c))`,
a form on the code of `y_c`; only the unit `a = x_c` is on, so the plane `c` receives
`P_c ψ(q_c(x_c, y_c))` (`plainFFN_ffn0`). That is `p` units per copy, the point-mass route of NV7:
one product per value of `x_c`, since `ψ(q_c)` is not a product of a function of `x_c` and one of
`y_c`. The second branch has one unit, the product of the two planes read as complex numbers:
`P₀ ψ(w₀) · P₁ ψ(w₁) = P₀ P₁ ψ(w₀ + w₁)`, written into the output plane (`plainFFN_ffn1`).
-/

@[expose] public section

open Finset Real
open Transformer.ComplexFFN

namespace Transformer.Phasor.TwoCopies

variable {D F p : ℕ}

/-- A feed-forward branch with the units `U` at the places `ι` among its `F` units; the other units
are zero. -/
noncomputable def ffnOf {U : Type*} (ι : U ↪ Fin F) (g v : U → EucSpace D →ₗ[ℝ] ℂ)
    (oRe oIm : U → EucSpace D) : FFN D F where
  gate := Function.extend ι g 0
  value := Function.extend ι v 0
  outRe := Function.extend ι oRe 0
  outIm := Function.extend ι oIm 0

/-- **The zero units add nothing**: the branch is the plain FFN of its units `U`. -/
theorem plainFFN_ffnOf {U : Type*} [Fintype U] (ι : U ↪ Fin F) (g v : U → EucSpace D →ₗ[ℝ] ℂ)
    (oRe oIm : U → EucSpace D) (z : EucSpace D) :
    plainFFN (ffnOf ι g v oRe oIm).gate (ffnOf ι g v oRe oIm).value (ffnOf ι g v oRe oIm).outRe
      (ffnOf ι g v oRe oIm).outIm z = plainFFN g v oRe oIm z := by
  simp only [plainFFN, complexReadout, plainHidden, ffnOf]
  symm
  refine Fintype.sum_of_injective ι ι.injective _ _ (fun k hk => ?_) (fun u => ?_)
  · have h : ¬∃ u, ι u = k := by simpa using hk
    simp [Function.extend_apply' _ _ _ h]
  · simp [ι.injective.extend_apply]

/-- The operands of the copy `c`: `x_c` at the position `2c`, `y_c` at `2c + 1`. -/
def xPos : Fin 2 → Fin 4 := ![0, 2]

/-- The position `2c + 1` of `y_c`. -/
def yPos : Fin 2 → Fin 4 := ![1, 3]

/-- The direction of the slot `(c, r)` of a phase plane. -/
noncomputable def planeDir (e : Slot p → EucSpace D) (c r : Fin 2) : EucSpace D :=
  e (.inr (.inr (.inl (c, r))))

/-- **The first branch**: the unit `(c, a)` multiplies the indicator of `x_c = a` by
`ψ(q_c(a, y_c))` and writes the product into the phase plane `c`. -/
noncomputable def ffn0 (e : Slot p → EucSpace D) (ι : Fin 2 × Fin p ↪ Fin F)
    (q : Fin 2 → Fin p → Fin p → Fin p) : FFN D F :=
  ffnOf ι (fun u => slotForm e (codeWeight fun x => if x = (xPos u.1, u.2) then 1 else 0))
    (fun u => slotForm e (codeWeight fun x =>
      if x.1 = yPos u.1 then phase p (q u.1 u.2 x.2) else 0))
    (fun u => planeDir e u.1 0) (fun u => planeDir e u.1 1)

variable {e : Slot p → EucSpace D}

/-- The phase plane `c` of a state with nothing else: `P_c (cos θ e_{c,0} + sin θ e_{c,1})`. -/
theorem vec_stateC_planes_zero (b : ℝ) (t : Fin 4 → Fin p) (P : Fin 2 → ℝ) (w : Fin 2 → Fin p)
    (u : Fin p) : vec e (stateC b 0 0 t P w 0 u) =
      ∑ c, ((P c * cos (angle p (w c))) • planeDir e c 0 +
        (P c * sin (angle p (w c))) • planeDir e c 1) := by
  simp [vec_stateC_planes, Fin.sum_univ_two, circ, planeDir]

/-- **The first branch computes both phases**: on a state with the code coefficients `k`, `a`, it
writes `P_c ψ(q_c(t_{x_c}, t_{y_c}))` into the plane `c`, with `P_c` the product of the two code
coefficients over `√2`. -/
theorem plainFFN_ffn0 (he : Orthonormal ℝ e) (ι : Fin 2 × Fin p ↪ Fin F)
    (q : Fin 2 → Fin p → Fin p → Fin p) (b k a : ℝ) (t : Fin 4 → Fin p) (w : Fin 2 → Fin p)
    (u : Fin p) :
    plainFFN (ffn0 e ι q).gate (ffn0 e ι q).value (ffn0 e ι q).outRe (ffn0 e ι q).outIm
        (vec e (stateC b k a t 0 w 0 u)) =
      vec e (stateC b 0 0 t
        (fun c => codeCoef k a (xPos c) * codeCoef k a (yPos c) / Real.sqrt 2)
        (fun c => q c (t (xPos c)) (t (yPos c))) 0 u) := by
  have hg (c : Fin 2) (a' : Fin p) :
      (∑ j, (if (j, t j) = (xPos c, a') then (1 : ℂ) else 0) * (codeCoef k a j : ℂ)) =
        if t (xPos c) = a' then (codeCoef k a (xPos c) : ℂ) else 0 := by
    simp only [Prod.mk.injEq, ite_and, ite_mul, one_mul, zero_mul, sum_ite_eq', mem_univ, ite_true]
  have hv (c : Fin 2) (a' : Fin p) :
      (∑ j, (if j = yPos c then phase p (q c a' (t j)) else 0) * (codeCoef k a j : ℂ)) =
        phase p (q c a' (t (yPos c))) * codeCoef k a (yPos c) := by
    simp only [ite_mul, zero_mul, sum_ite_eq', mem_univ, ite_true]
  have hc (c : Fin 2) : ((codeCoef k a (xPos c) : ℂ) * (phase p (q c (t (xPos c)) (t (yPos c))) *
      codeCoef k a (yPos c)) / (Real.sqrt 2 : ℂ)) =
      ((codeCoef k a (xPos c) * codeCoef k a (yPos c) / Real.sqrt 2 : ℝ) : ℂ) *
        phase p (q c (t (xPos c)) (t (yPos c))) := by
    push_cast; ring
  rw [ffn0, plainFFN_ffnOf, vec_stateC_planes_zero]
  simp only [plainFFN, complexReadout, plainHidden, slotForm_codeWeight he, hg, hv]
  rw [Fintype.sum_prod_type]
  refine sum_congr rfl fun c _ => ?_
  rw [sum_eq_single (t (xPos c)) (fun a' _ ha' => by simp [Ne.symm ha']) (by simp), ite_eq_left rfl,
    hc c, Complex.re_ofReal_mul, Complex.im_ofReal_mul]
  rfl

/-- **The second branch**: one unit, the product of the two phase planes read as complex numbers,
written into the output plane with the gain `g`. -/
noncomputable def ffn1 (e : Slot p → EucSpace D) (ι : Unit ↪ Fin F) (g : ℝ) : FFN D F :=
  ffnOf ι (fun _ => slotForm e (planeWeight 0)) (fun _ => slotForm e (planeWeight 1))
    (fun _ => g • e (.inr (.inr (.inr 0)))) (fun _ => g • e (.inr (.inr (.inr 1))))

/-- **The second branch multiplies the phases**: on a state with the planes `P_c ψ(w_c)` it writes
`g P₀ P₁/√2 · ψ(w₀ + w₁)` into the output plane. -/
theorem plainFFN_ffn1 (he : Orthonormal ℝ e) (ι : Unit ↪ Fin F) (g b k a : ℝ) (t : Fin 4 → Fin p)
    (P : Fin 2 → ℝ) (w : Fin 2 → Fin p) (u : Fin p) :
    plainFFN (ffn1 e ι g).gate (ffn1 e ι g).value (ffn1 e ι g).outRe (ffn1 e ι g).outIm
        (vec e (stateC b k a t P w 0 u)) =
      vec e (stateC b 0 0 t 0 w (g * (P 0 * P 1 / Real.sqrt 2)) (w 0 + w 1)) := by
  have hc : (P 0 : ℂ) * phase p (w 0) * (P 1 * phase p (w 1)) / (Real.sqrt 2 : ℂ) =
      ((P 0 * P 1 / Real.sqrt 2 : ℝ) : ℂ) * phase p (w 0 + w 1) := by
    rw [phase_add]; push_cast; ring
  rw [ffn1, plainFFN_ffnOf, vec_stateC_planes]
  simp only [plainFFN, complexReadout, plainHidden, slotForm_planeWeight he, hc,
    Complex.re_ofReal_mul, Complex.im_ofReal_mul]
  simp [Fin.sum_univ_two, circ, phase, smul_smul]
  congr 1 <;> congr 1 <;> ring

/-- At `p = 29` on the real coordinate slots of `D = 128`, with main's `F = 171` units: the first
branch. -/
example (ι : Fin 2 × Fin 29 ↪ Fin 171) (q : Fin 2 → Fin 29 → Fin 29 → Fin 29) (b k a : ℝ)
    (t : Fin 4 → Fin 29) (w : Fin 2 → Fin 29) (u : Fin 29) :=
  plainFFN_ffn0 (orthonormal_coordSlots (by norm_num : 4 * 29 + 7 ≤ 128)) ι q b k a t w u

/-- The second branch. -/
example (ι : Unit ↪ Fin 171) (g b k a : ℝ) (t : Fin 4 → Fin 29) (P : Fin 2 → ℝ)
    (w : Fin 2 → Fin 29) (u : Fin 29) :=
  plainFFN_ffn1 (orthonormal_coordSlots (by norm_num : 4 * 29 + 7 ≤ 128)) ι g b k a t P w u

end Transformer.Phasor.TwoCopies
