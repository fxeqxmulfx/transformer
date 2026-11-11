module

public import Mathlib.Analysis.Calculus.Deriv.Mul
public import Mathlib.Analysis.Calculus.Deriv.Slope
public import Transformer.Grokking.FlatSpectrum.Training.Continuity

/-!
# The update is the gradient step of the weight-decayed loss

Source: `review/flat_spectrum.md`, *Training reaches the optimal fit* (added), item WD3.
The loss `L` enters the other items only through a field `G` with the first-order inequality
`L X + ⟨G X, Y − X⟩ ≤ L Y`. If `G` is continuous, this makes `G` the gradient of `L`: along a
curve `X + t H(t)` the inequality at `X` and at `X + t H(t)` gives

    t ⟨G X, H(t)⟩ ≤ L (X + t H(t)) − L X ≤ t ⟨G (X + t H(t)), H(t)⟩,

and both sides have slope `⟨G X, H(0)⟩` at `t = 0` (`hasDerivAt_comp_add_smul`). The squared
Frobenius norm is such a function with `G X = 2 X` (`frobeniusSq_first_order`). So the
weight-decayed loss has the directional derivative

    d/dt [L ((U + tA)(V + tB)ᴴ) + λ ½ (‖U + tA‖² + ‖V + tB‖²)] = ⟨G V + λ U, A⟩ + ⟨Gᴴ U + λ V, B⟩

at `t = 0`, with `G = G (U Vᴴ)` (`hasDerivAt_decayedLoss`): its gradient is the pair
`(G V + λ U, Gᴴ U + λ V)` that `decayStep` subtracts, times `η`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix Filter Topology

variable {m n k : Type*} [Fintype m] [Fintype n] [Fintype k]

/-- From `t lo ≤ y ≤ t hi` with `t ≠ 0`, the slope `t⁻¹ y` lies between `lo` and `hi`. -/
lemma min_le_inv_mul_and_le_max {t lo hi y : ℝ} (ht : t ≠ 0) (h1 : t * lo ≤ y)
    (h2 : y ≤ t * hi) : min lo hi ≤ t⁻¹ * y ∧ t⁻¹ * y ≤ max lo hi := by
  rcases lt_or_gt_of_ne ht with hneg | hpos
  · have hinv : t⁻¹ ≤ 0 := (inv_lt_zero.mpr hneg).le
    have a1 := mul_le_mul_of_nonpos_left h2 hinv
    have a2 := mul_le_mul_of_nonpos_left h1 hinv
    rw [inv_mul_cancel_left₀ ht] at a1 a2
    exact ⟨(min_le_right _ _).trans a1, a2.trans (le_max_left _ _)⟩
  · have hinv : 0 ≤ t⁻¹ := (inv_pos.mpr hpos).le
    have a1 := mul_le_mul_of_nonneg_left h1 hinv
    have a2 := mul_le_mul_of_nonneg_left h2 hinv
    rw [inv_mul_cancel_left₀ ht] at a1 a2
    exact ⟨(min_le_left _ _).trans a1, a2.trans (le_max_right _ _)⟩

example := min_le_inv_mul_and_le_max (t := 2) (lo := 1) (hi := 3) (y := 4) two_ne_zero
  (by norm_num) (by norm_num)

/-- The first-order inequality at both ends of a segment:
`t ⟨G X, H⟩ ≤ L (X + t H) − L X ≤ t ⟨G (X + t H), H⟩`. -/
lemma mul_pairing_le_sub_and_le {L : Matrix m n ℂ → ℝ} {G : Matrix m n ℂ → Matrix m n ℂ}
    (hconv : ∀ X Y, L X + pairing (G X) (Y - X) ≤ L Y) (X H : Matrix m n ℂ) (t : ℝ) :
    t * pairing (G X) H ≤ L (X + t • H) - L X ∧
      L (X + t • H) - L X ≤ t * pairing (G (X + t • H)) H := by
  have h1 := hconv X (X + t • H)
  have h2 := hconv (X + t • H) X
  have e : X - (X + t • H) = (-t) • H := by module
  rw [add_sub_cancel_left, pairing_smul_right] at h1
  rw [e, pairing_smul_right] at h2
  constructor <;> linarith

/-- The squared Frobenius norm satisfies the first-order inequality with gradient `2 X`. -/
lemma frobeniusSq_first_order (X Y : Matrix m n ℂ) :
    frobeniusSq X + pairing ((2 : ℝ) • X) (Y - X) ≤ frobeniusSq Y := by
  have h := frobeniusSq_add X (Y - X)
  rw [add_sub_cancel] at h
  rw [pairing_smul_left]
  linarith [frobeniusSq_nonneg (Y - X)]

example := mul_pairing_le_sub_and_le (L := frobeniusSq) (G := fun X => (2 : ℝ) • X)
  frobeniusSq_first_order (1 : Matrix (Fin 1) (Fin 1) ℂ) 1 1

/-- **A field with the first-order inequality is the gradient** (`review/flat_spectrum.md`,
item WD3): if `L X + ⟨G X, Y − X⟩ ≤ L Y` for all `X`, `Y` and `G` is continuous, then along a
continuous curve `X + t H(t)` the function `L` has derivative `⟨G X, H(0)⟩` at `t = 0`. -/
theorem hasDerivAt_comp_add_smul {L : Matrix m n ℂ → ℝ} {G : Matrix m n ℂ → Matrix m n ℂ}
    (hconv : ∀ X Y, L X + pairing (G X) (Y - X) ≤ L Y) (hG : Continuous G) (X : Matrix m n ℂ)
    {H : ℝ → Matrix m n ℂ} (hH : Continuous H) :
    HasDerivAt (fun t : ℝ => L (X + t • H t)) (pairing (G X) (H 0)) 0 := by
  rw [hasDerivAt_iff_tendsto_slope_zero]
  have hlo : Tendsto (fun t : ℝ => pairing (G X) (H t)) (𝓝[≠] 0) (𝓝 (pairing (G X) (H 0))) :=
    ((continuous_pairing continuous_const hH).tendsto 0).mono_left nhdsWithin_le_nhds
  have hhi : Tendsto (fun t : ℝ => pairing (G (X + t • H t)) (H t)) (𝓝[≠] 0)
      (𝓝 (pairing (G X) (H 0))) := by
    have hc : Continuous fun t : ℝ => X + t • H t :=
      continuous_const.fun_add (continuous_id.fun_smul hH)
    have h := ((continuous_pairing (hG.comp hc) hH).tendsto 0).mono_left
      (nhdsWithin_le_nhds (s := {0}ᶜ))
    simpa using h
  have hmin := hlo.min hhi
  have hmax := hlo.max hhi
  rw [min_self] at hmin
  rw [max_self] at hmax
  refine tendsto_of_tendsto_of_tendsto_of_le_of_le' hmin hmax ?_ ?_
  · filter_upwards [self_mem_nhdsWithin] with t ht
    have h := mul_pairing_le_sub_and_le hconv X (H t) t
    simpa using (min_le_inv_mul_and_le_max ht h.1 h.2).1
  · filter_upwards [self_mem_nhdsWithin] with t ht
    have h := mul_pairing_le_sub_and_le hconv X (H t) t
    simpa using (min_le_inv_mul_and_le_max ht h.1 h.2).2

example := hasDerivAt_comp_add_smul (L := frobeniusSq) (G := fun X => (2 : ℝ) • X)
  frobeniusSq_first_order (continuous_id.fun_const_smul 2) (1 : Matrix (Fin 1) (Fin 1) ℂ)
  (H := fun _ => 1) continuous_const

omit [Fintype m] [Fintype n] in
/-- `(U + tA)(V + tB)ᴴ = U Vᴴ + t (A Vᴴ + U Bᴴ + t A Bᴴ)`. -/
lemma add_smul_mul_conjTranspose (U A : Matrix m k ℂ) (V B : Matrix n k ℂ) (t : ℝ) :
    (U + t • A) * (V + t • B)ᴴ = U * Vᴴ + t • (A * Vᴴ + U * Bᴴ + t • (A * Bᴴ)) := by
  rw [conjTranspose_add, conjTranspose_smul, star_trivial]
  simp only [Matrix.add_mul, Matrix.mul_add, Matrix.smul_mul, Matrix.mul_smul]
  module

/-- The derivative of the weight-decayed loss along `(A, B)`, from the derivative of the loss
along the curve of the products: a field `G₀` with the latter gives the pair
`(G₀ V + λ U, G₀ᴴ U + λ V)` (`review/flat_spectrum.md`, item WD3). No convexity is used. -/
theorem hasDerivAt_decayedLoss_of_hasDerivAt {L : Matrix m n ℂ → ℝ} {G₀ : Matrix m n ℂ}
    (lam : ℝ) (U A : Matrix m k ℂ) (V B : Matrix n k ℂ)
    (hloss : HasDerivAt (fun t : ℝ => L (U * Vᴴ + t • (A * Vᴴ + U * Bᴴ + t • (A * Bᴴ))))
      (pairing G₀ (A * Vᴴ + U * Bᴴ + (0 : ℝ) • (A * Bᴴ))) 0) :
    HasDerivAt (fun t : ℝ => decayedLoss L lam (U + t • A) (V + t • B))
      (pairing (G₀ * V + lam • U) A + pairing (G₀ᴴ * U + lam • V) B) 0 := by
  have h2 : Continuous fun X : Matrix m k ℂ => (2 : ℝ) • X := continuous_id.fun_const_smul 2
  have h2' : Continuous fun X : Matrix n k ℂ => (2 : ℝ) • X := continuous_id.fun_const_smul 2
  have hU := hasDerivAt_comp_add_smul frobeniusSq_first_order h2 U (H := fun _ => A)
    continuous_const
  have hV := hasDerivAt_comp_add_smul frobeniusSq_first_order h2' V (H := fun _ => B)
    continuous_const
  have hfun : (fun t : ℝ => decayedLoss L lam (U + t • A) (V + t • B)) = fun t : ℝ =>
      L (U * Vᴴ + t • (A * Vᴴ + U * Bᴴ + t • (A * Bᴴ))) +
        lam * ((frobeniusSq (U + t • A) + frobeniusSq (V + t • B)) / 2) := by
    funext t
    rw [decayedLoss, add_smul_mul_conjTranspose]
  have e1 : pairing (G₀ * V) A = pairing G₀ (A * Vᴴ) := pairing_mul_right _ _ _
  have e2 : pairing (G₀ᴴ * U) B = pairing G₀ (U * Bᴴ) := by
    rw [pairing_mul_left, conjTranspose_conjTranspose, pairing_comm G₀, pairing_mul_right,
      conjTranspose_conjTranspose]
  rw [hfun]
  convert hloss.fun_add (((hU.fun_add hV).div_const 2).const_mul lam) using 1
  simp only [zero_smul, add_zero, pairing_add_right, pairing_add_left, pairing_smul_left, e1, e2]
  ring

/-- **The update is the gradient step** (`review/flat_spectrum.md`, item WD3): for `L` with
the first-order inequality and a continuous `G`, the weight-decayed loss of the factors moved
along `(A, B)` has derivative `⟨G V + λ U, A⟩ + ⟨Gᴴ U + λ V, B⟩` at the start, `G = G (U Vᴴ)`.
Its gradient is the pair that `decayStep` subtracts, times the step `η`. -/
theorem hasDerivAt_decayedLoss {L : Matrix m n ℂ → ℝ} {G : Matrix m n ℂ → Matrix m n ℂ}
    (hconv : ∀ X Y, L X + pairing (G X) (Y - X) ≤ L Y) (hG : Continuous G) (lam : ℝ)
    (U A : Matrix m k ℂ) (V B : Matrix n k ℂ) :
    HasDerivAt (fun t : ℝ => decayedLoss L lam (U + t • A) (V + t • B))
      (pairing (G (U * Vᴴ) * V + lam • U) A + pairing ((G (U * Vᴴ))ᴴ * U + lam • V) B) 0 :=
  hasDerivAt_decayedLoss_of_hasDerivAt lam U A V B (hasDerivAt_comp_add_smul hconv hG (U * Vᴴ)
    (continuous_const.fun_add (continuous_id.fun_smul continuous_const)))

/-- The linear loss `−Re X` of a `1 × 1` matrix: its weight-decayed loss is differentiable
along every direction, with the gradient of `decayStep`. -/
example := hasDerivAt_decayedLoss (L := fun X : Matrix (Fin 1) (Fin 1) ℂ => pairing (-1) X)
  (G := fun _ => -1) (fun X Y => by rw [pairing_sub_right]; exact (add_sub_cancel _ _).le)
  continuous_const 1 (1 : Matrix (Fin 1) (Fin 1) ℂ) 1 (1 : Matrix (Fin 1) (Fin 1) ℂ) 1

/-- The same from the derivative of the loss along the curve of the products. -/
example := hasDerivAt_decayedLoss_of_hasDerivAt
  (L := fun X : Matrix (Fin 1) (Fin 1) ℂ => pairing (-1) X) (G₀ := -1) 1
  (1 : Matrix (Fin 1) (Fin 1) ℂ) 1 (1 : Matrix (Fin 1) (Fin 1) ℂ) 1
  (hasDerivAt_comp_add_smul (G := fun _ => -1)
    (fun X Y => by rw [pairing_sub_right]; exact (add_sub_cancel _ _).le) continuous_const _
    (continuous_const.fun_add (continuous_id.fun_smul continuous_const)))

end Transformer.Grokking.FlatSpectrum.Training
