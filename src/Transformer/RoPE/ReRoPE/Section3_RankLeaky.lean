module

public import Transformer.RoPE.ReRoPE.Section3_Rank

/-!
# Leaky ReRoPE, 计算 (Computation): no single pass of a fixed dimension on long sequences

Su, *Transformer升级之路：12、无限外推的ReRoPE？*, kexue.fm/archives/9708, section 计算
(Computation): "whether ReRoPE or Leaky ReRoPE, computing the attention matrix twice is
unavoidable". `Section3_Rank` shows that no single pass of a fixed dimension gives ReRoPE's
scores on long sequences; they are constant beyond the window. Leaky ReRoPE's are not, but have
rank two there:

* `le_mul_of_features_leaky`: in a plane of frequency `θ_l` Leaky ReRoPE scores the unit query
  and key `cos((w + (d − w)/k) θ_l)` beyond the window, which at `d = i − j` is
  `cos(α + βi) cos(βj) + sin(α + βi) sin(βj)` with `α = (w − w/k) θ_l` and `β = θ_l/k`
  (`D' = 2` in `le_mul_of_dotProduct_eq`). If `cos α ≠ 1`, a pass of dimension `D` gives its
  scores for all queries and keys only while `n ≤ (D + 2) w`;
* `not_single_pass_features_leaky`: on longer sequences no pass of dimension `D` gives them. At
  RoPE's frequency `θ_0 = 1`, `w = 512` and `k = 16`, `α = 480` and `cos α ≠ 1`.

The source's `k` interpolates, `k > 1`, which it leaves implicit and `not_single_pass_leaky`
assumes; here `k` is any real. `k = 1` is RoPE and makes `α = 0`; Lean's `x / 0 = 0` makes
`k = 0` ReRoPE.
-/

@[expose] public section

open Matrix

namespace Transformer.RoPE.ReRoPE

open RoFormer

/-- Computation: if some frequency has `cos((w − w/k) θ_l) ≠ 1` and maps `φ`, `ψ` to `ℝ^ι` give
Leaky ReRoPE's scores `φ(q, i) ⬝ᵥ ψ(k, j) = q_iᵀ R^{−leaky(i−j)} k_j` for all `j ≤ i < n` and all
`q`, `k`, then `n ≤ (|ι| + 2) w`. -/
theorem le_mul_of_features_leaky {p : ℕ} (θ : Fin p → ℝ) (l : Fin p) {w : ℕ} (k : ℝ)
    (hθ : Real.cos ((w - w / k) * θ l) ≠ 1) {ι : Type*} [Fintype ι] {n : ℕ}
    (φ ψ : (Fin 2 × Fin p → ℝ) → ℕ → ι → ℝ)
    (hφψ : ∀ i j, j ≤ i → i < n → ∀ q key,
      φ q i ⬝ᵥ ψ key j = q ⬝ᵥ (rotary θ (-leakyPosition w k ((i - j : ℕ) : ℝ)) *ᵥ key)) :
    n ≤ (Fintype.card ι + 2) * w := by
  have huv (x y : ℝ) :
      ![Real.cos x, Real.sin x] ⬝ᵥ ![Real.cos y, Real.sin y] = Real.cos (x - y) := by
    simp only [dotProduct, Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one]
    rw [Real.cos_sub]
  refine (le_mul_of_dotProduct_eq (κ := Fin 2) (φ (Pi.single (0, l) 1)) (ψ (Pi.single (0, l) 1))
    (fun i => ![Real.cos ((w - w / k) * θ l + θ l / k * i),
      Real.sin ((w - w / k) * θ l + θ l / k * i)])
    (fun j => ![Real.cos (θ l / k * j), Real.sin (θ l / k * j)])
    (fun d => Real.cos (leakyPosition w k d * θ l))
    (fun i j hji hin => ?_) (fun i j hij _ => ?_) (fun i _ => ?_)).trans_eq
    (by rw [Fintype.card_fin])
  · rw [hφψ i j hji hin, (single_dotProduct_rotary θ _ l).1, neg_mul, Real.cos_neg]
  · have hd : ¬((i - j : ℕ) : ℝ) < w :=
      not_lt.mpr (by exact_mod_cast (show w ≤ i - j by omega))
    rw [huv, leakyPosition, ite_eq_right hd, Nat.cast_sub (by omega : j ≤ i)]
    congr 1
    ring
  · have hl0 : leakyPosition (w : ℝ) k 0 = 0 := (Nat.cast_nonneg (α := ℝ) w).lt_or_eq.elim
      (leakyPosition_of_lt k) fun hw => by rw [leakyPosition, ← hw]; norm_num
    rw [huv, Nat.cast_zero, hl0, zero_mul, Real.cos_zero, add_sub_cancel_right]
    exact fun h => hθ h.symm

example : 512 ≤ (Fintype.card (Fin 2 × Fin 64) + 2) * 512 :=
  le_mul_of_features_leaky (fun l : Fin 64 => frequency 10000 128 l) 0 (w := 512) 16
    (by norm_num [frequency]; simpa using cos_natCast_ne_one (m := 480) (by norm_num))
    (fun q i => rotary (fun l : Fin 64 => frequency 10000 128 l) i *ᵥ q)
    (fun key j => rotary (fun l : Fin 64 => frequency 10000 128 l) j *ᵥ key)
    (fun i j hji hi q key => by
      have hd : ((i - j : ℕ) : ℝ) < (512 : ℕ) := by exact_mod_cast (show i - j < 512 by omega)
      rw [rotary_dotProduct, leakyPosition, ite_eq_left hd, Nat.cast_sub hji, neg_sub])

/-- Computation: on sequences longer than `(|ι| + 2) w` no maps `φ`, `ψ` to `ℝ^ι` give Leaky
ReRoPE's scores for all queries and keys, if some frequency has `cos((w − w/k) θ_l) ≠ 1`. -/
theorem not_single_pass_features_leaky {p : ℕ} (θ : Fin p → ℝ) (l : Fin p) {w : ℕ} (k : ℝ)
    (hθ : Real.cos ((w - w / k) * θ l) ≠ 1) {ι : Type*} [Fintype ι] {n : ℕ}
    (hn : (Fintype.card ι + 2) * w < n) :
    ¬ ∃ φ ψ : (Fin 2 × Fin p → ℝ) → ℕ → ι → ℝ, ∀ i j, j ≤ i → i < n → ∀ q key,
      φ q i ⬝ᵥ ψ key j = q ⬝ᵥ (rotary θ (-leakyPosition w k ((i - j : ℕ) : ℝ)) *ᵥ key) :=
  fun ⟨φ, ψ, hφψ⟩ => absurd (le_mul_of_features_leaky θ l k hθ φ ψ hφψ) (not_le.mpr hn)

example : ¬ ∃ φ ψ : (Fin 2 × Fin 64 → ℝ) → ℕ → Fin 256 → ℝ, ∀ i j, j ≤ i → i < 262144 →
    ∀ q key, φ q i ⬝ᵥ ψ key j = q ⬝ᵥ (rotary (fun l : Fin 64 => frequency 10000 128 l)
      (-leakyPosition (512 : ℕ) 16 ((i - j : ℕ) : ℝ)) *ᵥ key) :=
  not_single_pass_features_leaky _ 0 16
    (by norm_num [frequency]; simpa using cos_natCast_ne_one (m := 480) (by norm_num))
    (by norm_num)

end Transformer.RoPE.ReRoPE
