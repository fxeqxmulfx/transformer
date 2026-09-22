/-
# The emergence of clusters in self-attention dynamics — `l:boundedother`, the mean

The function `g` of the proof of `l:boundedother` (§7 of arXiv:2305.05465v6)
is the softmax mean `θ ↦ Σ_k a_k e^{θ a_k} / Σ_k e^{θ a_k}` of the exponential
rates `a_k = γ_k`, with `γ_{i₀} = 0` for the bounded token (the `1` of the
source's denominator).  Here, for any finite family `a`:

* `softmaxMean_sub_ge` — the mean increases, quantitatively, from the strict
  convexity of the log-sum-exp (`sq_le_softmax_monotone`);
* `tendsto_softmaxMean`, `continuous_softmaxMean` — continuous in `a` and in `θ`;
* `exists_softmaxMean_root` — with one positive and one negative rate, it
  vanishes at exactly one `θ₀`, with sign changing across it.

**What the source says and what is carried here.**  The source says `g`
"takes value `-∞` at `-∞`, and `+∞` at `+∞`".  It does not: `g` is an average
of the `γ_k` and stays in `[γ_1, γ_n]`.  Its limits are `γ_1 < 0` and
`γ_n > 0`, which is all that is used, and the root `θ₀` exists as claimed.

Source: arXiv:2305.05465v6, proof of `l:boundedother`, the function `g`.
-/

import Transformer.Clusters.Section7_OnlyOneSpread
import Mathlib.Topology.Order.IntermediateValue

open scoped BigOperators
open Real Filter Topology

namespace Transformer
namespace Clusters

variable {m : ℕ}

/-- The softmax mean of `a` at inverse temperature `θ`,
`Σ_k a_k e^{θ a_k} / Σ_k e^{θ a_k}`: the function `g` of the proof of
`l:boundedother`.

Source: arXiv:2305.05465v6, proof of `l:boundedother`. -/
noncomputable def softmaxMean (a : Idx (m + 1) → ℝ) (θ : ℝ) : ℝ :=
  ∑ k, Perspective.softmaxWeight (fun l => θ * a l) k * a k

/-- A softmax weight along converging scores converges. -/
theorem tendsto_softmaxWeight {F : Filter ℝ} {u : ℝ → Idx (m + 1) → ℝ} {v : Idx (m + 1) → ℝ}
    (hu : ∀ k, Tendsto (fun t => u t k) F (𝓝 (v k))) (j : Idx (m + 1)) :
    Tendsto (fun t => Perspective.softmaxWeight (u t) j) F
      (𝓝 (Perspective.softmaxWeight v j)) := by
  unfold Perspective.softmaxWeight
  exact ((hu j).rexp).div
    (tendsto_finsetSum _ fun k _ => (hu k).rexp)
    (Perspective.softmaxPartition_pos (Nat.succ_pos m) v).ne'

/-- The softmax mean along converging families converges. -/
theorem tendsto_softmaxMean {a : ℝ → Idx (m + 1) → ℝ} {γ : Idx (m + 1) → ℝ}
    (ha : ∀ k, Tendsto (fun t => a t k) atTop (𝓝 (γ k))) (θ : ℝ) :
    Tendsto (fun t => softmaxMean (a t) θ) atTop (𝓝 (softmaxMean γ θ)) := by
  unfold softmaxMean
  exact tendsto_finsetSum _ fun k _ =>
    (tendsto_softmaxWeight (fun l => (ha l).const_mul θ) k).mul (ha k)

/-- The softmax mean is continuous in the temperature. -/
theorem continuous_softmaxMean (a : Idx (m + 1) → ℝ) : Continuous (softmaxMean a) := by
  refine continuous_iff_continuousAt.2 fun θ => ?_
  unfold softmaxMean
  exact tendsto_finsetSum _ fun k _ =>
    (tendsto_softmaxWeight (F := 𝓝 θ) (fun l => (continuous_id.tendsto θ).mul_const (a l)) k).mul
      tendsto_const_nhds

/-- Reversing the family reverses the mean: `g_{-a}(θ) = -g_a(-θ)`. -/
theorem softmaxMean_neg (a : Idx (m + 1) → ℝ) (θ : ℝ) :
    softmaxMean (fun k => -a k) θ = -softmaxMean a (-θ) := by
  simp only [softmaxMean, mul_neg, neg_mul, Finset.sum_neg_distrib]

/-- **The mean increases, quantitatively.**  For `θ₁ < θ₂` and `p ≠ q`,
`g(θ₂) - g(θ₁) ≥ c (θ₂ - θ₁) (a_p - a_q)²` for some `c > 0`: the derivative
of `g` is a variance.

Source: arXiv:2305.05465v6, proof of `l:boundedother`, "`g` has a positive
derivative". -/
theorem softmaxMean_sub_ge (a : Idx (m + 1) → ℝ) {θ₁ θ₂ : ℝ} (h : θ₁ < θ₂)
    {p q : Idx (m + 1)} (hpq : p ≠ q) :
    ∃ c : ℝ, 0 < c ∧ c * (θ₂ - θ₁) * (a p - a q) ^ 2 ≤ softmaxMean a θ₂ - softmaxMean a θ₁ := by
  set S := ∑ k, |a k|
  set K := (|θ₁| + |θ₂|) * S
  have hS : ∀ j, |a j| ≤ S := fun j =>
    Finset.single_le_sum (f := fun k => |a k|) (fun k _ => abs_nonneg _) (Finset.mem_univ j)
  have hK : ∀ θ, |θ| ≤ |θ₁| + |θ₂| → ∀ j, |θ * a j| ≤ K := fun θ hθ j => by
    rw [abs_mul]
    exact mul_le_mul hθ (hS j) (abs_nonneg _) (by positivity)
  have := sq_le_softmax_monotone (fun l => θ₂ * a l) (fun l => θ₁ * a l)
    (hK θ₂ (le_add_of_nonneg_left (abs_nonneg _))) (hK θ₁ (le_add_of_nonneg_right (abs_nonneg _)))
    hpq
  set c := Real.exp (-(2 * K)) / ((m : ℝ) + 1) / 2
  have hc : 0 < c := by positivity
  have hR : ∑ j, (Perspective.softmaxWeight (fun l => θ₂ * a l) j -
      Perspective.softmaxWeight (fun l => θ₁ * a l) j) * (θ₂ * a j - θ₁ * a j) =
      (θ₂ - θ₁) * (softmaxMean a θ₂ - softmaxMean a θ₁) := by
    rw [softmaxMean, softmaxMean, ← Finset.sum_sub_distrib, Finset.mul_sum]
    exact Finset.sum_congr rfl fun j _ => by ring
  rw [hR] at this
  refine ⟨c, hc, le_of_mul_le_mul_left ?_ (sub_pos.2 h)⟩
  have hL : c * ((θ₂ * a p - θ₁ * a p) - (θ₂ * a q - θ₁ * a q)) ^ 2 =
      (θ₂ - θ₁) * (c * (θ₂ - θ₁) * (a p - a q) ^ 2) := by ring
  linarith

/-- With two tokens, the mean is nondecreasing. -/
theorem softmaxMean_mono (a : Idx (m + 1) → ℝ) (hm : 0 < m) : Monotone (softmaxMean a) := by
  intro θ₁ θ₂ h
  rcases h.lt_or_eq with h | h
  · have hne : (0 : Idx (m + 1)) ≠ Fin.last m := fun e => by
      have := congrArg Fin.val e; simp only [Fin.val_zero, Fin.val_last] at this; omega
    obtain ⟨c, hc, hle⟩ := softmaxMean_sub_ge a h hne
    nlinarith [mul_nonneg (mul_nonneg hc.le (sub_pos.2 h).le) (sq_nonneg (a 0 - a (Fin.last m)))]
  · rw [h]

/-- With two distinct rates, the mean is strictly increasing. -/
theorem softmaxMean_strictMono (a : Idx (m + 1) → ℝ) {p q : Idx (m + 1)} (hpq : a p ≠ a q) :
    StrictMono (softmaxMean a) := by
  intro θ₁ θ₂ h
  obtain ⟨c, hc, hle⟩ := softmaxMean_sub_ge a h (p := p) (q := q) fun e => hpq (by rw [e])
  have : 0 < c * (θ₂ - θ₁) * (a p - a q) ^ 2 :=
    mul_pos (mul_pos hc (sub_pos.2 h)) (by positivity)
  linarith

/-- A positive rate makes the mean positive at a large temperature. -/
theorem exists_softmaxMean_pos (a : Idx (m + 1) → ℝ) {p : Idx (m + 1)} (hp : 0 < a p) :
    ∃ θ, 0 < softmaxMean a θ := by
  set S := ∑ k, |a k|
  set θ := (S + 1) / a p ^ 2
  have hθ : 0 ≤ θ := by positivity
  have hterm : ∀ k, -|a k| ≤ Real.exp (θ * a k) * a k := fun k => by
    rcases le_or_gt 0 (a k) with hk | hk
    · linarith [abs_nonneg (a k), mul_nonneg (Real.exp_pos (θ * a k)).le hk]
    · have h1 : Real.exp (θ * a k) ≤ 1 :=
        Real.exp_le_one_iff.2 (mul_nonpos_of_nonneg_of_nonpos hθ hk.le)
      rw [abs_of_neg hk]
      nlinarith
  have hsum := Finset.add_sum_erase Finset.univ (fun k => Real.exp (θ * a k) * a k)
    (Finset.mem_univ p)
  have hrest : -S ≤ ∑ k ∈ Finset.univ.erase p, Real.exp (θ * a k) * a k := by
    have h1 := Finset.sum_le_sum fun k (_ : k ∈ Finset.univ.erase p) => hterm k
    have h2 : ∑ k ∈ Finset.univ.erase p, |a k| ≤ S :=
      Finset.sum_le_sum_of_subset_of_nonneg (Finset.erase_subset _ _) fun k _ _ => abs_nonneg _
    rw [Finset.sum_neg_distrib] at h1
    linarith
  have hexp : S + 1 + a p ≤ Real.exp (θ * a p) * a p := by
    have h1 := Real.add_one_le_exp (θ * a p)
    have h2 : θ * a p * a p = S + 1 := by simp only [θ]; field_simp
    nlinarith
  refine ⟨θ, ?_⟩
  have hnum : 0 < ∑ k, Real.exp (θ * a k) * a k := by linarith
  have hZ := Perspective.softmaxPartition_pos (Nat.succ_pos m) (fun l => θ * a l)
  have : softmaxMean a θ = (∑ k, Real.exp (θ * a k) * a k) / ∑ k, Real.exp (θ * a k) := by
    simp only [softmaxMean, Perspective.softmaxWeight, Finset.sum_div, div_mul_eq_mul_div]
  rw [this]
  exact div_pos hnum hZ

/-- **The root `θ₀` of `g`.**  With a positive and a negative rate, the mean
changes sign exactly once: it is negative left of some `θ₀` and positive right
of it.

Source: arXiv:2305.05465v6, proof of `l:boundedother`, the definition of `θ₀`. -/
theorem exists_softmaxMean_root (a : Idx (m + 1) → ℝ) {p q : Idx (m + 1)} (hp : 0 < a p)
    (hq : a q < 0) :
    ∃ θ₀, ∀ θ, (θ < θ₀ → softmaxMean a θ < 0) ∧ (θ₀ < θ → 0 < softmaxMean a θ) := by
  have hmono := softmaxMean_strictMono a (p := p) (q := q) (by linarith)
  obtain ⟨θp, hθp⟩ := exists_softmaxMean_pos a hp
  obtain ⟨θq, hθq⟩ := exists_softmaxMean_pos (fun k => -a k) (p := q) (by linarith)
  rw [softmaxMean_neg] at hθq
  have hlt : -θq < θp := hmono.lt_iff_lt.1 (by linarith)
  have hcont := continuous_softmaxMean a
  obtain ⟨θ₀, -, h0⟩ := intermediate_value_Icc hlt.le hcont.continuousOn
    (⟨by linarith, hθp.le⟩ : (0 : ℝ) ∈ Set.Icc (softmaxMean a (-θq)) (softmaxMean a θp))
  exact ⟨θ₀, fun θ => ⟨fun h => h0 ▸ hmono h, fun h => h0 ▸ hmono h⟩⟩

/-- The hypotheses of `exists_softmaxMean_root` are satisfiable: the rates
`(-1, 0, 1)`, whose root is `θ₀ = 0`. -/
example : ∃ a : Idx 3 → ℝ, 0 < a 2 ∧ a 0 < 0 :=
  ⟨fun k => (k : ℝ) - 1, by norm_num, by norm_num⟩

end Clusters
end Transformer
