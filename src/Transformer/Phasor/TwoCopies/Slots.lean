module

public import Transformer.Phasor.OneStream

/-!
# Two copies in the one-stream model: slots in the stream

Source: `review/flat_spectrum.md`, *Functions of n variables*, NV8 (added), task T3c.

The construction of NV8 keeps every quantity in directions of its own: a bias, four codes of a
token (one per operand), two phase planes (one per copy) and the output plane, `4p + 7` orthonormal
directions `e_s` (`card_slot`; the coordinate directions `coordSlots` are one choice,
`orthonormal_coordSlots`). A vector with the coefficients `c` on orthonormal directions,
`vec e c = ∑_s c_s e_s`, has the inner products and the length of its coefficients
(`inner_vec_vec`, `norm_sq_vec`), so the RMS of the model rescales the coefficients by
`√D / √(∑_s c_s² + D ε)` (`rmsNormEps_vec`). The projections of the construction read slots: a real
map `z ↦ ⟨e_s, z⟩ u` (`slotMap`) or a complex form `z ↦ ∑_s w_s ⟨e_s, z⟩` (`slotForm`), which give
`c_s u` and `∑_s w_s c_s` on `vec e c`.
-/

@[expose] public section

open Finset
open scoped RealInnerProductSpace

namespace Transformer.Phasor.TwoCopies

/-- The slots: the bias, the codes `(j, a)` of the token `a` of the position `j`, the phase planes
`(c, r)` of the copies `c` (`r = 0` the real part) and the output plane `r`. -/
abbrev Slot (p : ℕ) : Type := Unit ⊕ (Fin 4 × Fin p) ⊕ (Fin 2 × Fin 2) ⊕ Fin 2

theorem card_slot (p : ℕ) : Fintype.card (Slot p) = 4 * p + 7 := by
  simp only [Fintype.card_sum, Fintype.card_unit, Fintype.card_prod, Fintype.card_fin]
  ring

variable {D d p : ℕ}

/-- The coordinate directions of `EucSpace D`. -/
noncomputable def coords (D : ℕ) : Fin D → EucSpace D := fun i => EuclideanSpace.single i 1

theorem orthonormal_coords (D : ℕ) : Orthonormal ℝ (coords D) :=
  EuclideanSpace.orthonormal_single

/-- The slots as coordinate directions of a stream of width `D ≥ 4p + 7`. -/
noncomputable def coordSlots (h : 4 * p + 7 ≤ D) : Slot p → EucSpace D :=
  fun s => coords D (Fin.castLE ((card_slot p).trans_le h) (Fintype.equivFin (Slot p) s))

/-- **`4p + 7` orthonormal directions fit into a stream of width `D ≥ 4p + 7`.** -/
theorem orthonormal_coordSlots (h : 4 * p + 7 ≤ D) : Orthonormal ℝ (coordSlots h) :=
  (orthonormal_coords D).comp _ ((Fin.castLE_injective _).comp (Fintype.equivFin _).injective)

/-- Main's width `D = 128` holds the slots of `p = 29` as real directions. -/
example : Orthonormal ℝ (coordSlots (p := 29) (D := 128) (by norm_num)) :=
  orthonormal_coordSlots _

/-- `p` codes as coordinate directions of a head of width `d ≥ p`. -/
noncomputable def coordCodes (h : p ≤ d) : Fin p → EucSpace d :=
  fun a => coords d (Fin.castLE h a)

theorem orthonormal_coordCodes (h : p ≤ d) : Orthonormal ℝ (coordCodes h) :=
  (orthonormal_coords d).comp _ (Fin.castLE_injective h)

/-- Main's head width `d = 32` holds the codes of `p = 29`. -/
example : Orthonormal ℝ (coordCodes (p := 29) (d := 32) (by norm_num)) := orthonormal_coordCodes _

variable {ι : Type*} [Fintype ι] {e : ι → EucSpace D}

/-- The vector with the coefficients `c` on the directions `e`. -/
noncomputable def vec (e : ι → EucSpace D) (c : ι → ℝ) : EucSpace D := ∑ s, c s • e s

theorem inner_vec_vec (he : Orthonormal ℝ e) (c c' : ι → ℝ) :
    ⟪vec e c, vec e c'⟫ = ∑ s, c s * c' s := by
  simpa [vec] using he.inner_sum c c' univ

theorem norm_sq_vec (he : Orthonormal ℝ e) (c : ι → ℝ) : ‖vec e c‖ ^ 2 = ∑ s, c s ^ 2 := by
  rw [← real_inner_self_eq_norm_sq, inner_vec_vec he]
  simp only [sq]

/-- The slot `s` of `vec e c` is `c s`. -/
theorem inner_vec (he : Orthonormal ℝ e) (c : ι → ℝ) (s : ι) : ⟪e s, vec e c⟫ = c s :=
  he.inner_right_fintype c s

theorem vec_add (e : ι → EucSpace D) (c c' : ι → ℝ) : vec e (c + c') = vec e c + vec e c' := by
  simp [vec, add_smul, sum_add_distrib]

theorem vec_smul (e : ι → EucSpace D) (a : ℝ) (c : ι → ℝ) : vec e (a • c) = a • vec e c := by
  simp [vec, smul_sum, smul_smul]

/-- The coordinate directions of `ℝ³`: `⟨vec c, vec c'⟩ = ∑ c_s c'_s` and `‖vec c‖² = ∑ c_s²`. -/
example (c c' : Fin 3 → ℝ) : ⟪vec (coords 3) c, vec (coords 3) c'⟫ = ∑ s, c s * c' s ∧
    ‖vec (coords 3) c‖ ^ 2 = ∑ s, c s ^ 2 ∧ ⟪coords 3 0, vec (coords 3) c⟫ = c 0 :=
  ⟨inner_vec_vec (orthonormal_coords 3) c c', norm_sq_vec (orthonormal_coords 3) c,
    inner_vec (orthonormal_coords 3) c 0⟩

/-- The factor of the RMS on a vector of squared length `r`: `√D / √(r + D ε)`. -/
noncomputable def rmsScale (D : ℕ) (eps r : ℝ) : ℝ := Real.sqrt D / Real.sqrt (r + D * eps)

theorem rmsNormEps_eq_smul (eps : ℝ) (x : EucSpace D) :
    GPTMini.rmsNormEps eps x = rmsScale D eps (‖x‖ ^ 2) • x :=
  rfl

/-- **The RMS rescales the coefficients** by `rmsScale` of their squared length. -/
theorem rmsNormEps_vec (he : Orthonormal ℝ e) (eps : ℝ) (c : ι → ℝ) :
    GPTMini.rmsNormEps eps (vec e c) = vec e (rmsScale D eps (∑ s, c s ^ 2) • c) := by
  rw [rmsNormEps_eq_smul, norm_sq_vec he, vec_smul]

/-- The RMS of `vec e (a • c)`: the factor `rmsScale D ε (a² ∑ c_s²) · a` on `c`. -/
theorem rmsNormEps_vec_smul (he : Orthonormal ℝ e) (eps a : ℝ) (c : ι → ℝ) :
    GPTMini.rmsNormEps eps (vec e (a • c)) =
      vec e ((rmsScale D eps (a ^ 2 * ∑ s, c s ^ 2) * a) • c) := by
  rw [rmsNormEps_vec he, smul_smul]
  simp only [Pi.smul_apply, smul_eq_mul, mul_pow, ← mul_sum]

/-- The coordinate directions of `ℝ³` at main's `ε = 2⁻²³`. -/
example (a : ℝ) (c : Fin 3 → ℝ) :
    GPTMini.rmsNormEps (1 / 8388608) (vec (coords 3) c) =
        vec (coords 3) (rmsScale 3 (1 / 8388608) (∑ s, c s ^ 2) • c) ∧
      GPTMini.rmsNormEps (1 / 8388608) (vec (coords 3) (a • c)) =
        vec (coords 3) ((rmsScale 3 (1 / 8388608) (a ^ 2 * ∑ s, c s ^ 2) * a) • c) :=
  ⟨rmsNormEps_vec (orthonormal_coords 3) _ c, rmsNormEps_vec_smul (orthonormal_coords 3) _ a c⟩

theorem rmsScale_pos {eps r : ℝ} (hD : 0 < D) (heps : 0 ≤ eps) (hr : 0 < r) :
    0 < rmsScale D eps r := by
  have : (0 : ℝ) < D := Nat.cast_pos.2 hD
  unfold rmsScale
  positivity

example : 0 < rmsScale 128 (1 / 8388608) 3 := rmsScale_pos (by norm_num) (by norm_num) (by norm_num)

/-- `rmsScale D ε r · √r ≤ √D`: the RMS never lengthens beyond `√D`. -/
theorem rmsScale_mul_sqrt_le {eps r : ℝ} (heps : 0 ≤ eps) :
    rmsScale D eps r * Real.sqrt r ≤ Real.sqrt D := by
  unfold rmsScale
  rcases eq_or_lt_of_le (Real.sqrt_nonneg (r + D * eps)) with h | h
  · rw [← h, div_zero, zero_mul]; exact Real.sqrt_nonneg _
  · rw [div_mul_eq_mul_div, div_le_iff₀ h]
    exact mul_le_mul_of_nonneg_left
      (Real.sqrt_le_sqrt (le_add_of_nonneg_right (by positivity))) (Real.sqrt_nonneg _)

example : rmsScale 4 1 5 * Real.sqrt 5 ≤ Real.sqrt 4 :=
  rmsScale_mul_sqrt_le (by norm_num)

/-- The real map `z ↦ ⟨e_s, z⟩ u`: it reads the slot `s` into the direction `u`. -/
noncomputable def slotMap (e : ι → EucSpace D) (s : ι) (u : EucSpace d) :
    EucSpace D →ₗ[ℝ] EucSpace d :=
  (innerₛₗ ℝ (e s)).smulRight u

theorem slotMap_vec (he : Orthonormal ℝ e) (s : ι) (u : EucSpace d) (c : ι → ℝ) :
    slotMap e s u (vec e c) = c s • u := by
  rw [slotMap, LinearMap.smulRight_apply, innerₛₗ_apply_apply, inner_vec he]

example (u : EucSpace 2) (c : Fin 3 → ℝ) : slotMap (coords 3) 0 u (vec (coords 3) c) = c 0 • u :=
  slotMap_vec (orthonormal_coords 3) 0 u c

/-- The complex form `z ↦ ∑_s w_s ⟨e_s, z⟩`, with complex weights `w`. -/
noncomputable def slotForm (e : ι → EucSpace D) (w : ι → ℂ) : EucSpace D →ₗ[ℝ] ℂ where
  toFun z := ∑ s, w s * (⟪e s, z⟫ : ℂ)
  map_add' z z' := by
    simp only [inner_add_right, Complex.ofReal_add, mul_add, sum_add_distrib]
  map_smul' a z := by
    simp only [real_inner_smul_right, Complex.ofReal_mul, RingHom.id_apply, Complex.real_smul,
      mul_sum]
    exact sum_congr rfl fun s _ => by ring

theorem slotForm_vec (he : Orthonormal ℝ e) (w : ι → ℂ) (c : ι → ℝ) :
    slotForm e w (vec e c) = ∑ s, w s * c s := by
  simp only [slotForm, LinearMap.coe_mk, AddHom.coe_mk, inner_vec he]

/-- The form with the weight `i` on every coordinate of `ℝ³`. -/
example (c : Fin 3 → ℝ) :
    slotForm (coords 3) (fun _ => Complex.I) (vec (coords 3) c) = ∑ s, Complex.I * c s :=
  slotForm_vec (orthonormal_coords 3) _ c

end Transformer.Phasor.TwoCopies
