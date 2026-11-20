module

public import Mathlib.Analysis.Real.Pi.Irrational
public import Mathlib.Data.Matrix.ColumnRowPartitioned
public import Mathlib.LinearAlgebra.Matrix.Block
public import Mathlib.LinearAlgebra.Matrix.Rank
public import Transformer.RoPE.ReRoPE.Section3_SinglePass

/-!
# ReRoPE, 计算 (Computation): no single pass of a fixed dimension on long sequences

Su, *Transformer升级之路：12、无限外推的ReRoPE？*, kexue.fm/archives/9708, section 计算
(Computation): "the matrices (leaky-rerope), (rerope) are nonlinear, piecewise linear; realizing
them takes computing the attention matrix twice and combining the two". `Section3_SinglePass`
shows that no single pass of RoPE gives either. Here a pass is arbitrary: it maps the query `q`
at `i` and the key `k` at `j` to any vectors `φ(q, i)`, `ψ(k, j)` of a fixed finite dimension `D`
and scores their dot product.

* `le_mul_of_dotProduct_eq`: let `f i ⬝ᵥ g j = h (i − j)` for all `j ≤ i < n`, in dimension `D`,
  and let `h (i − j)` be a dot product `u i ⬝ᵥ v j` in dimension `D'` from the distance `w` on,
  which misses `h 0` on the diagonal. Then `n ≤ (D + D') w`. Otherwise the positions
  `0, w, …, (D + D') w` give a square matrix of `f ⬝ᵥ g − u ⬝ᵥ v` that vanishes below the
  diagonal and not on it, so has rank `D + D' + 1`, yet is a product through `D + D'` dimensions;
* `le_mul_of_features_rectified`: in a plane of frequency `θ_l` ReRoPE scores the unit query and
  key `cos(min(d, w) θ_l)`, constant from the window on (`D' = 1`). If `cos(w θ_l) ≠ 1`, a pass
  of dimension `D` gives ReRoPE's scores for all queries and keys only while `n ≤ (D + 1) w`;
* `not_single_pass_features_rectified`: on longer sequences no pass of dimension `D` gives them.
  RoPE's frequency `θ_0 = 1` has `cos(w θ_0) ≠ 1` at every window `w ≥ 1`
  (`cos_natCast_ne_one`).

Leaky ReRoPE is `Section3_RankLeaky`. `Section3_Merge` computes the same attention exactly with
two passes over disjoint sets of keys.
-/

@[expose] public section

open Matrix

namespace Transformer.RoPE.ReRoPE

open RoFormer

/-- If the dot products of vectors in `ℝ^ι` are `f i ⬝ᵥ g j = h (i − j)` for all `j ≤ i < n`, and
from the distance `w` on `h (i − j)` is the dot product `u i ⬝ᵥ v j` of vectors in `ℝ^κ`, which
misses `h 0` on the diagonal, then `n ≤ (|ι| + |κ|) w`. -/
theorem le_mul_of_dotProduct_eq {ι κ : Type*} [Fintype ι] [Fintype κ] {n w : ℕ}
    (f g : ℕ → ι → ℝ) (u v : ℕ → κ → ℝ) (h : ℕ → ℝ)
    (hfg : ∀ i j, j ≤ i → i < n → f i ⬝ᵥ g j = h (i - j))
    (hfar : ∀ i j, j + w ≤ i → i < n → h (i - j) = u i ⬝ᵥ v j)
    (h0 : ∀ i, i < n → h 0 ≠ u i ⬝ᵥ v i) : n ≤ (Fintype.card ι + Fintype.card κ) * w := by
  by_contra hn
  push Not at hn
  have hpt : ∀ r : Fin (Fintype.card ι + Fintype.card κ + 1), (r : ℕ) * w < n := fun r =>
    lt_of_le_of_lt (Nat.mul_le_mul_right w (by have := r.2; omega)) hn
  let A : Matrix (Fin (Fintype.card ι + Fintype.card κ + 1)) (ι ⊕ κ) ℝ :=
    fromCols (of fun r l => f (r * w) l) (of fun r l => -u (r * w) l)
  let B : Matrix (ι ⊕ κ) (Fin (Fintype.card ι + Fintype.card κ + 1)) ℝ :=
    fromRows (of fun l s => g (s * w) l) (of fun l s => v (s * w) l)
  have hentry : ∀ r s, (A * B) r s = f (r * w) ⬝ᵥ g (s * w) - u (r * w) ⬝ᵥ v (s * w) := by
    intro r s
    simp [A, B, Matrix.mul_apply, dotProduct, sub_eq_add_neg]
  have htri : (A * B).IsUpperTriangular := by
    intro r s hsr
    have hsw : (s : ℕ) * w + w ≤ r * w := by
      simpa [Nat.add_mul] using Nat.mul_le_mul_right w (show (s : ℕ) + 1 ≤ r from hsr)
    rw [hentry, hfg _ _ (le_trans (Nat.le_add_right _ _) hsw) (hpt r), hfar _ _ hsw (hpt r),
      sub_self]
  have hdet : (A * B).det ≠ 0 := by
    rw [det_of_isUpperTriangular htri]
    refine Finset.prod_ne_zero_iff.mpr fun r _ => ?_
    rw [hentry, hfg _ _ le_rfl (hpt r), Nat.sub_self]
    exact sub_ne_zero.mpr (h0 _ (hpt r))
  have hrank : (A * B).rank = Fintype.card ι + Fintype.card κ + 1 := by
    rw [rank_of_isUnit _ ((isUnit_iff_isUnit_det _).mpr (isUnit_iff_ne_zero.mpr hdet)),
      Fintype.card_fin]
  have hle : (A * B).rank ≤ Fintype.card ι + Fintype.card κ :=
    (rank_mul_le_left A B).trans ((rank_le_card_width A).trans_eq (by simp))
  omega

/-- One-hot features of dimension `n` give any scores `h (i − j)` on `n` positions. -/
example : 4 ≤ (Fintype.card (Fin 4) + Fintype.card Unit) * 2 :=
  le_mul_of_dotProduct_eq (n := 4) (w := 2)
    (fun i l => if (l : ℕ) = i then 1 else 0) (fun j l => if (l : ℕ) - j < 2 then 1 else 0)
    (fun _ _ => 0) (fun _ _ => 0) (fun d => if d < 2 then 1 else 0)
    (fun i j _ hi => by
      simp only [dotProduct, Fin.sum_univ_four]
      interval_cases i <;> simp)
    (fun i j hij _ => by rw [ite_eq_right (show ¬i - j < 2 by omega)]; simp [dotProduct])
    (fun _ _ => by simp [dotProduct])

/-- `cos m ≠ 1` at every natural `m ≠ 0`: `m = 2πk` would make `π` rational. -/
theorem cos_natCast_ne_one {m : ℕ} (hm : m ≠ 0) : Real.cos m ≠ 1 := by
  intro h
  obtain ⟨k, hk⟩ := (Real.cos_eq_one_iff _).mp h
  have hk0 : k ≠ 0 := by
    rintro rfl
    have hm0 : (m : ℝ) = 0 := by
      rw [← hk]
      simp
    exact hm (by exact_mod_cast hm0)
  have hk' : ((2 * k : ℤ) : ℝ) * Real.pi = m := by
    rw [← hk]
    push_cast
    ring
  exact (irrational_pi.intCast_mul (by omega)).ne_nat m hk'

example : Real.cos (512 : ℕ) ≠ 1 :=
  cos_natCast_ne_one (by norm_num)

/-- Computation: if some frequency has `cos(w θ_l) ≠ 1` and maps `φ`, `ψ` from a query or a key
and its position to `ℝ^ι` give ReRoPE's scores `φ(q, i) ⬝ᵥ ψ(k, j) = q_iᵀ R^{−min(i−j, w)} k_j`
for all `j ≤ i < n` and all `q`, `k`, then `n ≤ (|ι| + 1) w`. -/
theorem le_mul_of_features_rectified {p : ℕ} (θ : Fin p → ℝ) (l : Fin p) {w : ℕ}
    (hθ : Real.cos (w * θ l) ≠ 1) {ι : Type*} [Fintype ι] {n : ℕ}
    (φ ψ : (Fin 2 × Fin p → ℝ) → ℕ → ι → ℝ)
    (hφψ : ∀ i j, j ≤ i → i < n → ∀ q key,
      φ q i ⬝ᵥ ψ key j = q ⬝ᵥ (rotary θ (-rectifiedPosition w ((i - j : ℕ) : ℝ)) *ᵥ key)) :
    n ≤ (Fintype.card ι + 1) * w := by
  have hc : (fun _ : Unit => Real.cos (w * θ l)) ⬝ᵥ (fun _ => 1) = Real.cos (w * θ l) := by
    simp [dotProduct]
  refine (le_mul_of_dotProduct_eq (κ := Unit) (φ (Pi.single (0, l) 1)) (ψ (Pi.single (0, l) 1))
    (fun _ _ => Real.cos (w * θ l)) (fun _ _ => 1) (fun d => Real.cos (rectifiedPosition w d * θ l))
    (fun i j hji hin => ?_) (fun i j hij _ => ?_) (fun _ _ => ?_)).trans_eq
    (by rw [Fintype.card_unit])
  · rw [hφψ i j hji hin, (single_dotProduct_rotary θ _ l).1, neg_mul, Real.cos_neg]
  · have hd : (w : ℝ) ≤ ((i - j : ℕ) : ℝ) := by exact_mod_cast (show w ≤ i - j by omega)
    rw [hc, rectifiedPosition, min_eq_right hd]
  · rw [hc, Nat.cast_zero, rectifiedPosition, min_eq_left (Nat.cast_nonneg (α := ℝ) w), zero_mul,
      Real.cos_zero]
    exact fun h => hθ h.symm

example : 512 ≤ (Fintype.card (Fin 2 × Fin 64) + 1) * 512 :=
  le_mul_of_features_rectified (fun l : Fin 64 => frequency 10000 128 l) 0 (w := 512)
    (by simpa [frequency] using cos_natCast_ne_one (m := 512) (by norm_num))
    (fun q i => rotary (fun l : Fin 64 => frequency 10000 128 l) i *ᵥ q)
    (fun key j => rotary (fun l : Fin 64 => frequency 10000 128 l) j *ᵥ key)
    (fun i j hji hi q key => by
      have hd : ((i - j : ℕ) : ℝ) ≤ (512 : ℕ) := by exact_mod_cast (show i - j ≤ 512 by omega)
      rw [rotary_dotProduct, rectifiedPosition, min_eq_left hd, Nat.cast_sub hji, neg_sub])

/-- Computation: on sequences longer than `(|ι| + 1) w` no maps `φ`, `ψ` to `ℝ^ι` give ReRoPE's
scores for all queries and keys, if some frequency has `cos(w θ_l) ≠ 1`. -/
theorem not_single_pass_features_rectified {p : ℕ} (θ : Fin p → ℝ) (l : Fin p) {w : ℕ}
    (hθ : Real.cos (w * θ l) ≠ 1) {ι : Type*} [Fintype ι] {n : ℕ}
    (hn : (Fintype.card ι + 1) * w < n) :
    ¬ ∃ φ ψ : (Fin 2 × Fin p → ℝ) → ℕ → ι → ℝ, ∀ i j, j ≤ i → i < n → ∀ q key,
      φ q i ⬝ᵥ ψ key j = q ⬝ᵥ (rotary θ (-rectifiedPosition w ((i - j : ℕ) : ℝ)) *ᵥ key) :=
  fun ⟨φ, ψ, hφψ⟩ => absurd (le_mul_of_features_rectified θ l hθ φ ψ hφψ) (not_le.mpr hn)

example : ¬ ∃ φ ψ : (Fin 2 × Fin 64 → ℝ) → ℕ → Fin 256 → ℝ, ∀ i j, j ≤ i → i < 262144 →
    ∀ q key, φ q i ⬝ᵥ ψ key j = q ⬝ᵥ (rotary (fun l : Fin 64 => frequency 10000 128 l)
      (-rectifiedPosition (512 : ℕ) ((i - j : ℕ) : ℝ)) *ᵥ key) :=
  not_single_pass_features_rectified _ 0
    (by simpa [frequency] using cos_natCast_ne_one (m := 512) (by norm_num)) (by norm_num)

end Transformer.RoPE.ReRoPE
