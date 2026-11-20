module

public import Transformer.RoPE.ReRoPE.Section3_SinglePass
public import Transformer.RoPE.ReRoPE2D.Regions

/-!
# 2D ReRoPE: no single pass in the head's dimension

2D ReRoPE is in no paper; see `Transformer.RoPE.ReRoPE2D.Defs`. Su notes that ReRoPE needs two
attention matrices, since one RoPE pass gives only linear relative positions (kexue.fm/archives/9708,
计算 (Computation); `Transformer.RoPE.ReRoPE.not_single_pass_rectified`, for RoPE passes at any
positions). For 2D ReRoPE the statement here is stronger. No matrices `A(X)` and `B(Y)` of the head's
dimension, depending on the query's patch `X` and on the key's patch `Y`, give its scores
`(A(X) q)ᵀ(B(Y) k)` for all `q` and `k`, already on a grid of `w + 2` patches along one axis. This
excludes one axial RoPE pass at any positions, one rotation per plane by any angles, and any
linear map of the head's dimension.

* `transpose_mul_eq_of_dotProduct`: scores `(A q)ᵀ(B k) = qᵀ M k` for all `q`, `k` fix `AᵀB = M`;
* `mul_transpose_mul_eq`: if `A₀ᵀB` is orthogonal, the scores of a rectangle of positions are tied:
  `(AᵀB)(A₀ᵀB)ᵀ(A₀ᵀB₀) = AᵀB₀`. This needs square matrices: from `A₀ᵀB(BᵀA₀) = 1` follows
  `B(BᵀA₀)A₀ᵀ = 1`;
* `exists_int_of_axialRotary_eq`: two axial rotations agree only if on every plane `l` their
  coordinates on its axis differ by whole turns at `θ_l`;
* `not_single_pass`: on a line along the axis of a plane with `0 < θ_l < 2π`, 2D ReRoPE's clamped
  offsets break the tie of the rectangle `X ∈ {w + 1, w}`, `Y ∈ {0, 1}` by `θ_l`.

The passes of `Transformer.RoPE.ReRoPE2D.Regions` compute each score once, but rotate the query
by a map that depends on the key's region, which a single pass does not allow. A feature map of a
larger dimension escapes `mul_transpose_mul_eq`; in one dimension
`Transformer.RoPE.ReRoPE.le_mul_of_features_rectified` bounds the length of the sequences that a
pass of a fixed dimension serves.
-/

@[expose] public section

open Matrix
open scoped Real
open Transformer.RoPE.ReRoPE (ne_two_pi_mul_int)

namespace Transformer.RoPE.ReRoPE2D

variable {p : ℕ}

/-- Scores `(A q)ᵀ(B k) = qᵀ M k` for all `q` and `k` fix the matrix `AᵀB = M`. -/
theorem transpose_mul_eq_of_dotProduct {ι : Type*} [Fintype ι] [DecidableEq ι]
    {A B M : Matrix ι ι ℝ} (h : ∀ q k, (A *ᵥ q) ⬝ᵥ (B *ᵥ k) = q ⬝ᵥ (M *ᵥ k)) : Aᵀ * B = M := by
  ext i j
  have hij := h (Pi.single i 1) (Pi.single j 1)
  rw [RoFormer.mulVec_dotProduct, mulVec_mulVec] at hij
  simpa using hij

/-- The hypothesis of `transpose_mul_eq_of_dotProduct` holds for RoFormer's rotary matrices
(arXiv:2104.09864v5, Eq. (16)). -/
example (θ : Fin 64 → ℝ) (m n : ℝ) :
    (RoFormer.rotary θ m)ᵀ * RoFormer.rotary θ n = RoFormer.rotary θ (n - m) :=
  transpose_mul_eq_of_dotProduct fun q k => RoFormer.rotary_dotProduct θ q k m n

/-- **The rectangle tie** of one pass: if `A₀ᵀB` is orthogonal, the scores of the queries `A`,
`A₀` and the keys `B`, `B₀` satisfy `(AᵀB)(A₀ᵀB)ᵀ(A₀ᵀB₀) = AᵀB₀`. -/
theorem mul_transpose_mul_eq {ι : Type*} [Fintype ι] [DecidableEq ι] (A A₀ B B₀ : Matrix ι ι ℝ)
    (h : A₀ᵀ * B * (A₀ᵀ * B)ᵀ = 1) :
    Aᵀ * B * (A₀ᵀ * B)ᵀ * (A₀ᵀ * B₀) = Aᵀ * B₀ := by
  rw [transpose_mul, transpose_transpose] at h ⊢
  have h1 : A₀ᵀ * (B * (Bᵀ * A₀)) = 1 := by simpa only [Matrix.mul_assoc] using h
  have h2 := mul_eq_one_comm.mp h1
  calc Aᵀ * B * (Bᵀ * A₀) * (A₀ᵀ * B₀) = Aᵀ * (B * (Bᵀ * A₀) * A₀ᵀ) * B₀ := by
        simp only [Matrix.mul_assoc]
    _ = Aᵀ * B₀ := by rw [h2, Matrix.mul_one]

/-- Axial RoPE is orthogonal: `R_x R_xᵀ = 1`. -/
theorem axialRotary_mul_transpose (θ : Fin p → ℝ) (ax : Fin p → Fin 2) (x : Fin 2 → ℝ) :
    axialRotary θ ax x * (axialRotary θ ax x)ᵀ = 1 := by
  rw [axialRotary_transpose, ← axialRotary_add, add_neg_cancel, axialRotary_zero]

/-- The hypothesis of `mul_transpose_mul_eq` holds for queries and keys of axial RoPE. -/
example (θ : Fin 32 → ℝ) (x₀ y : Fin 2 → ℝ) (A B₀ : Matrix (Fin 2 × Fin 32) (Fin 2 × Fin 32) ℝ) :
    Aᵀ * axialRotary θ kimiAxis y * ((axialRotary θ kimiAxis x₀)ᵀ * axialRotary θ kimiAxis y)ᵀ *
        ((axialRotary θ kimiAxis x₀)ᵀ * B₀) = Aᵀ * B₀ :=
  mul_transpose_mul_eq A _ _ B₀ (by
    rw [axialRotary_transpose, ← axialRotary_add]
    exact axialRotary_mul_transpose θ kimiAxis _)

/-- Two axial rotations agree only if on every plane `l` the coordinates on its axis differ by a
whole number of turns at `θ_l`. -/
theorem exists_int_of_axialRotary_eq (θ : Fin p → ℝ) (ax : Fin p → Fin 2) {u v : Fin 2 → ℝ}
    (h : axialRotary θ ax u = axialRotary θ ax v) (l : Fin p) :
    ∃ m : ℤ, (u (ax l) - v (ax l)) * θ l = 2 * π * m := by
  have hc := congrArg (fun M : Matrix (Fin 2 × Fin p) (Fin 2 × Fin p) ℝ => M (0, l) (0, l)) h
  have hs := congrArg (fun M : Matrix (Fin 2 × Fin p) (Fin 2 × Fin p) ℝ => M (1, l) (0, l)) h
  simp only [axialRotary, blockDiagonal_apply, RoFormer.rotation] at hc hs
  simp at hc hs
  rw [sub_mul]
  exact Real.Angle.angle_eq_iff_two_pi_dvd_sub.mp (Real.Angle.cos_sin_inj hc hs)

/-- The hypothesis of `exists_int_of_axialRotary_eq` holds for two positions a whole turn apart on
a plane turning at `1`. -/
example (θ : Fin 32 → ℝ) (ax : Fin 32 → Fin 2) (u : Fin 2 → ℝ) :
    ∃ m : ℤ, (u (ax 0) - u (ax 0)) * θ 0 = 2 * π * m :=
  exists_int_of_axialRotary_eq θ ax rfl 0

/-- **No single pass in the head's dimension**: if some plane `l` turns at `0 < θ_l < 2π` and its
axis has the window `w ≥ 1` and at least `w + 2` patches, no matrices `A(X)` and `B(Y)` of the
head's dimension give 2D ReRoPE's scores `(A(X) q)ᵀ(B(Y) k)` for all patches `X`, `Y` of the grid
and all `q`, `k`. -/
theorem not_single_pass (θ : Fin p → ℝ) (ax : Fin p → Fin 2) (l : Fin p) (hθ0 : 0 < θ l)
    (hθ : θ l < 2 * π) {W S : Fin 2 → ℕ} (hW : 1 ≤ W (ax l)) (hS : W (ax l) + 2 ≤ S (ax l))
    (hS0 : ∀ a, 0 < S a) :
    ¬ ∃ A B : (Fin 2 → ℕ) → Matrix (Fin 2 × Fin p) (Fin 2 × Fin p) ℝ,
      ∀ X Y, (∀ a, X a < S a ∧ Y a < S a) → ∀ q k,
        (A X *ᵥ q) ⬝ᵥ (B Y *ᵥ k) =
          score θ ax (fun a => (W a : ℝ)) (fun a => (X a : ℝ)) (fun a => (Y a : ℝ)) q k := by
  rintro ⟨A, B, h⟩
  have hgrid : ∀ c, c ≤ W (ax l) + 1 → ∀ a, (Pi.single (ax l) c : Fin 2 → ℕ) a < S a := by
    intro c hc a
    by_cases ha : a = ax l
    · subst ha
      rw [Pi.single_eq_same]
      omega
    · rw [Pi.single_eq_of_ne ha]
      exact hS0 a
  have hM : ∀ c d, c ≤ W (ax l) + 1 → d ≤ W (ax l) + 1 →
      (A (Pi.single (ax l) c))ᵀ * B (Pi.single (ax l) d) =
        axialRotary θ ax (-clampOffset (fun a => (W a : ℝ))
          (fun a => ((Pi.single (ax l) c : Fin 2 → ℕ) a : ℝ))
          (fun a => ((Pi.single (ax l) d : Fin 2 → ℕ) a : ℝ))) :=
    fun c d hc hd => transpose_mul_eq_of_dotProduct
      (h _ _ fun a => ⟨hgrid c hc a, hgrid d hd a⟩)
  have key := mul_transpose_mul_eq (A (Pi.single (ax l) (W (ax l) + 1)))
    (A (Pi.single (ax l) (W (ax l)))) (B (Pi.single (ax l) 0)) (B (Pi.single (ax l) 1))
    (by rw [hM _ _ (by omega) (by omega)]; exact axialRotary_mul_transpose θ ax _)
  rw [hM _ _ le_rfl (by omega), hM _ _ (by omega) (by omega), hM _ _ (by omega) (by omega),
    hM _ _ le_rfl (by omega), axialRotary_transpose, ← axialRotary_add,
    ← axialRotary_add] at key
  obtain ⟨m, hm⟩ := exists_int_of_axialRotary_eq θ ax key l
  have hW0 : (0 : ℝ) ≤ W (ax l) := Nat.cast_nonneg _
  have hW1 : (1 : ℝ) ≤ W (ax l) := by exact_mod_cast hW
  have e1 : clampPosition (W (ax l)) ((W (ax l) : ℝ) + 1) = W (ax l) :=
    clampPosition_of_le hW0 (by linarith)
  have e2 : clampPosition (W (ax l)) (W (ax l) : ℝ) = W (ax l) := clampPosition_of_le hW0 le_rfl
  have e3 : clampPosition (W (ax l)) ((W (ax l) : ℝ) - 1) = W (ax l) - 1 :=
    clampPosition_of_mem_Icc ⟨by linarith, by linarith⟩
  simp only [Pi.add_apply, Pi.neg_apply, clampOffset, Pi.single_eq_same, Nat.cast_add,
    Nat.cast_one, Nat.cast_zero, sub_zero, add_sub_cancel_right, e1, e2, e3] at hm
  exact ne_two_pi_mul_int hθ0 hθ m (by linear_combination hm)

/-- In MoonViT's layout the plane `1` turns with the row at the frequency `1`: with the windows
`256` no single pass of the head's dimension gives 2D ReRoPE on a grid of `258 × 258` patches. -/
example : ¬ ∃ A B : (Fin 2 → ℕ) → Matrix (Fin 2 × Fin 32) (Fin 2 × Fin 32) ℝ,
    ∀ X Y, (∀ a, X a < 258 ∧ Y a < 258) → ∀ q k,
      (A X *ᵥ q) ⬝ᵥ (B Y *ᵥ k) = score (kimiFreq 10000) kimiAxis (fun _ => ((256 : ℕ) : ℝ))
        (fun a => (X a : ℝ)) (fun a => (Y a : ℝ)) q k :=
  not_single_pass (kimiFreq 10000) kimiAxis 1 (by simp [kimiFreq, frequency])
    (by simp [kimiFreq, frequency]; linarith [Real.pi_gt_three]) (W := fun _ => 256)
    (S := fun _ => 258) (by norm_num) (by norm_num) (fun _ => by norm_num)

/-- In particular no single axial RoPE pass, the query at `F X` and the key at `G Y` on each
axis, gives 2D ReRoPE in GLM-5.3-Flash's layout, whose plane `0` turns with the first axis at
the frequency `1`. -/
example (F G : (Fin 2 → ℕ) → Fin 2 → ℝ) : ¬ ∀ X Y, (∀ a, X a < 258 ∧ Y a < 258) → ∀ q k,
    (axialRotary (glmFreq 10000) glmAxis (F X) *ᵥ q) ⬝ᵥ
        (axialRotary (glmFreq 10000) glmAxis (G Y) *ᵥ k) =
      score (glmFreq 10000) glmAxis (fun _ => ((256 : ℕ) : ℝ)) (fun a => (X a : ℝ))
        (fun a => (Y a : ℝ)) (q : Fin 2 × Fin 32 → ℝ) k := fun h =>
  not_single_pass (glmFreq 10000) glmAxis 0 (by simp [glmFreq, frequency])
    (by simp [glmFreq, frequency]; linarith [Real.pi_gt_three]) (W := fun _ => 256)
    (S := fun _ => 258) (by norm_num) (by norm_num) (fun _ => by norm_num)
    ⟨fun X => axialRotary (glmFreq 10000) glmAxis (F X),
      fun Y => axialRotary (glmFreq 10000) glmAxis (G Y), h⟩

end Transformer.RoPE.ReRoPE2D
