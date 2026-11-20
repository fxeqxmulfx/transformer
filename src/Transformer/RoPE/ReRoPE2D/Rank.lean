module

public import Transformer.RoPE.ReRoPE.Section3_Rank
public import Transformer.RoPE.ReRoPE2D.Basic

/-!
# 2D ReRoPE: no single pass of a fixed dimension on large grids

2D ReRoPE is in no paper; see `Transformer.RoPE.ReRoPE2D.Defs`.
`Transformer.RoPE.ReRoPE2D.SinglePass` rules out a single pass in the head's dimension. A pass of
any dimension `|ι|` maps the query `q` at the patch `X` and the key `k` at the patch `Y` to vectors
`φ(q, X)`, `ψ(k, Y)` of `ℝ^ι` and scores their dot product. On a line of patches along one axis,
2D ReRoPE scores the unit query and key of a plane of that axis as one-dimensional ReRoPE does,
so the rank bound of one dimension, `Transformer.RoPE.ReRoPE.le_mul_of_dotProduct_eq`, applies:

* `single_dotProduct_axialRotary`: axial RoPE at `x` scores the unit query and key of the plane
  `l` `cos(x_{ax l} θ_l)`;
* `le_mul_of_features`: if the plane `l` has `cos(w θ_l) ≠ 1` at the window `w` of its axis, a
  pass of dimension `|ι|` gives 2D ReRoPE's scores on a grid with `n` patches along that axis only
  if `n ≤ (|ι| + 1) w`;
* `not_single_pass_features`: on grids with more patches along that axis no pass of dimension
  `|ι|` gives them. RoPE's frequency `θ_0 = 1` has `cos(w θ_0) ≠ 1` at every window `w ≥ 1`
  (`Transformer.RoPE.ReRoPE.cos_natCast_ne_one`), and both encoders turn a plane of each axis at
  the frequency `1`.
-/

@[expose] public section

open Matrix
open Transformer.RoPE.RoFormer
open Transformer.RoPE.ReRoPE (rectifiedPosition le_mul_of_dotProduct_eq cos_natCast_ne_one)

namespace Transformer.RoPE.ReRoPE2D

variable {p : ℕ}

/-- Axial RoPE at `x` scores the unit query and key of the plane `l` `cos(x_{ax l} θ_l)`. -/
theorem single_dotProduct_axialRotary (θ : Fin p → ℝ) (ax : Fin p → Fin 2) (x : Fin 2 → ℝ)
    (l : Fin p) :
    Pi.single (0, l) 1 ⬝ᵥ (axialRotary θ ax x *ᵥ Pi.single (0, l) 1) =
      Real.cos (x (ax l) * θ l) := by
  simp [axialRotary, blockDiagonal_apply, rotation]

/-- **A pass of a fixed dimension serves a bounded grid**: if the plane `l` has `cos(w θ_l) ≠ 1`
at the window `w` of its axis, and maps `φ`, `ψ` from a query or a key and its patch to `ℝ^ι`
give 2D ReRoPE's scores `φ(q, X) ⬝ᵥ ψ(k, Y)` for all patches `X`, `Y` of a grid and all `q`, `k`,
then the grid has at most `(|ι| + 1) w` patches along that axis. -/
theorem le_mul_of_features (θ : Fin p → ℝ) (ax : Fin p → Fin 2) (l : Fin p) {W S : Fin 2 → ℕ}
    (hθ : Real.cos (W (ax l) * θ l) ≠ 1) (hS0 : ∀ a, 0 < S a) {ι : Type*} [Fintype ι]
    (φ ψ : (Fin 2 × Fin p → ℝ) → (Fin 2 → ℕ) → ι → ℝ)
    (hφψ : ∀ X Y, (∀ a, X a < S a ∧ Y a < S a) → ∀ q k, φ q X ⬝ᵥ ψ k Y =
      score θ ax (fun a => (W a : ℝ)) (fun a => (X a : ℝ)) (fun a => (Y a : ℝ)) q k) :
    S (ax l) ≤ (Fintype.card ι + 1) * W (ax l) := by
  have hgrid : ∀ c, c < S (ax l) → ∀ a, (Pi.single (ax l) c : Fin 2 → ℕ) a < S a :=
    fun _ hc => single_lt hS0 hc
  have hc : (fun _ : Unit => Real.cos (W (ax l) * θ l)) ⬝ᵥ (fun _ => 1) =
      Real.cos (W (ax l) * θ l) := by
    simp [dotProduct]
  refine (le_mul_of_dotProduct_eq (κ := Unit) (n := S (ax l)) (w := W (ax l))
    (fun i => φ (Pi.single (0, l) 1) (Pi.single (ax l) i))
    (fun j => ψ (Pi.single (0, l) 1) (Pi.single (ax l) j))
    (fun _ _ => Real.cos (W (ax l) * θ l)) (fun _ _ => 1)
    (fun d => Real.cos (rectifiedPosition (W (ax l)) d * θ l))
    (fun i j hji hin => ?_) (fun i j hij _ => ?_) (fun _ _ => ?_)).trans_eq
    (by rw [Fintype.card_unit])
  · have hd : -(W (ax l) : ℝ) ≤ (i : ℝ) - j := by
      have hj : (j : ℝ) ≤ i := by exact_mod_cast hji
      linarith [Nat.cast_nonneg (α := ℝ) (W (ax l))]
    rw [hφψ _ _ fun a => ⟨hgrid i hin a, hgrid j (by omega) a⟩, score,
      single_dotProduct_axialRotary]
    simp only [Pi.neg_apply, clampOffset, Pi.single_eq_same]
    rw [neg_mul, Real.cos_neg, Nat.cast_sub hji,
      clampPosition_eq_rectifiedPosition (Nat.cast_nonneg _) hd]
  · have hd : (W (ax l) : ℝ) ≤ ((i - j : ℕ) : ℝ) := by
      exact_mod_cast (show W (ax l) ≤ i - j by omega)
    rw [hc, rectifiedPosition, min_eq_right hd]
  · rw [hc, Nat.cast_zero, rectifiedPosition, min_eq_left (Nat.cast_nonneg (α := ℝ) _), zero_mul,
      Real.cos_zero]
    exact fun h => hθ h.symm

/-- The hypotheses of `le_mul_of_features` hold for axial RoPE itself, a pass of the head's
dimension, on MoonViT's table of `512 × 512` positions with the windows `512`, inside which 2D
ReRoPE is axial RoPE. Its plane `1` turns with the row at the frequency `1`. -/
example : 512 ≤ (Fintype.card (Fin 2 × Fin 32) + 1) * 512 :=
  le_mul_of_features (kimiFreq 10000) kimiAxis 1 (W := fun _ => 512) (S := fun _ => 512)
    (by simpa [kimiFreq, frequency] using cos_natCast_ne_one (m := 512) (by norm_num))
    (fun _ => by norm_num)
    (fun q X => axialRotary (kimiFreq 10000) kimiAxis (fun a => (X a : ℝ)) *ᵥ q)
    (fun k Y => axialRotary (kimiFreq 10000) kimiAxis (fun a => (Y a : ℝ)) *ᵥ k)
    (fun X Y hXY q k => (score_of_abs_le _ _ (fun a => by
      have hX : (X a : ℝ) < 512 := by exact_mod_cast (hXY a).1
      have hY : (Y a : ℝ) < 512 := by exact_mod_cast (hXY a).2
      have hX0 : (0 : ℝ) ≤ X a := Nat.cast_nonneg _
      have hY0 : (0 : ℝ) ≤ Y a := Nat.cast_nonneg _
      rw [abs_le]
      constructor <;> push_cast <;> linarith) q k).symm)

/-- **No single pass of a fixed dimension on large grids**: if the plane `l` has `cos(w θ_l) ≠ 1`
at the window `w` of its axis, no maps `φ`, `ψ` to `ℝ^ι` give 2D ReRoPE's scores on a grid with
more than `(|ι| + 1) w` patches along that axis. -/
theorem not_single_pass_features (θ : Fin p → ℝ) (ax : Fin p → Fin 2) (l : Fin p)
    {W S : Fin 2 → ℕ} (hθ : Real.cos (W (ax l) * θ l) ≠ 1) (hS0 : ∀ a, 0 < S a) {ι : Type*}
    [Fintype ι] (hS : (Fintype.card ι + 1) * W (ax l) < S (ax l)) :
    ¬ ∃ φ ψ : (Fin 2 × Fin p → ℝ) → (Fin 2 → ℕ) → ι → ℝ, ∀ X Y,
      (∀ a, X a < S a ∧ Y a < S a) → ∀ q k, φ q X ⬝ᵥ ψ k Y =
        score θ ax (fun a => (W a : ℝ)) (fun a => (X a : ℝ)) (fun a => (Y a : ℝ)) q k :=
  fun ⟨φ, ψ, h⟩ => absurd (le_mul_of_features θ ax l hθ hS0 φ ψ h) (not_le.mpr hS)

/-- In GLM-5.3-Flash's layout, whose plane `0` turns with the first axis at the frequency `1`, no
pass of dimension `256` gives 2D ReRoPE with the windows `16` on a grid of `4113` patches along
that axis. -/
example : ¬ ∃ φ ψ : (Fin 2 × Fin 32 → ℝ) → (Fin 2 → ℕ) → Fin 256 → ℝ, ∀ X Y,
    (∀ a, X a < 4113 ∧ Y a < 4113) → ∀ q k, φ q X ⬝ᵥ ψ k Y =
      score (glmFreq 10000) glmAxis (fun _ => ((16 : ℕ) : ℝ)) (fun a => (X a : ℝ))
        (fun a => (Y a : ℝ)) q k :=
  not_single_pass_features (glmFreq 10000) glmAxis 0 (W := fun _ => 16) (S := fun _ => 4113)
    (by simpa [glmFreq, frequency] using cos_natCast_ne_one (m := 16) (by norm_num))
    (fun _ => by norm_num) (by norm_num)

end Transformer.RoPE.ReRoPE2D
