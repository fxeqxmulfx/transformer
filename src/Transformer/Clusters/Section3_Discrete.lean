/-
# The emergence of clusters in self-attention dynamics — the discrete rescaled
  tokens

`r:discreterescaling` of arXiv:2305.05465v6, §3: the discrete analogue of
`e:Rres`, for the forward Euler scheme `e:discreteequation`.

**What the source says and what is carried here.**

* `r:discreterescaling` sets `R = I_d + ΔtV` and *assumes* `R` invertible.
  The inverse is carried as an explicit two-sided inverse `Rinv`, so that the
  assumption is a hypothesis rather than an instance to be discharged.

Source: arXiv:2305.05465v6, `r:discreterescaling`.
-/

import Transformer.Clusters.Section3_Rescaled

open scoped BigOperators
open Real

namespace Transformer
namespace Clusters

variable {d n : ℕ}

/-- **Remark (r:discreterescaling).**  The discrete analogue of `e:Rres`: with
`R = I_d + ΔtV` invertible and `z_i^{[k]} = R^{-k}x_i(kΔt)`,

  `z_i^{[k+1]} = z_i^{[k]} + Δt Σ_j P_ij(R^k z^{[k]}) R^{-1}V(z_j^{[k]} - z_i^{[k]})`.

Source: arXiv:2305.05465v6, `r:discreterescaling`. -/
def DiscreteRescaled (Δt : ℝ) (Q K V R Rinv : ParamMatrix d)
    (Z : ℕ → Idx n → EucSpace d) : Prop :=
  ∀ (k : ℕ) (i : Idx n),
    Z (k + 1) i = Z k i + Δt • ∑ j : Idx n,
      attentionMatrix Q K (fun l => (R ^ k) (Z k l)) i j • Rinv (V (Z k j - Z k i))

/-- With `V = 0` the constant sequence solves the discrete rescaled
dynamics. -/
theorem discreteRescaled_const (Δt : ℝ) (Q K R Rinv : ParamMatrix d)
    (Z : Idx n → EucSpace d) : DiscreteRescaled Δt Q K 0 R Rinv (fun _ => Z) := by
  intro k i
  simp

/-- **Remark (r:discreterescaling), the change of variables.**  Under
`x_i(kΔt) = R^k z_i^{[k]}` with `R = I_d + ΔtV` invertible, the discrete
transformer `e:discreteequation` and the discrete rescaled dynamics are the
same equation.

As in the source: `V` commutes with `R`, so `R^{-(k+1)}VR^k = R^{-1}V`, and
`R^{-1} = I_d - ΔtR^{-1}V`; the rows of `eq:P` summing to one turn
`Σ_j P_ij z_j - z_i` into `Σ_j P_ij (z_j - z_i)`.  Applied to both sides,
`R^{k+1}` is injective, which gives the two directions at once.

Source: arXiv:2305.05465v6, `r:discreterescaling`. -/
theorem discreteTransformer_iff_rescaled (Δt : ℝ) (Q K V R Rinv : ParamMatrix d)
    (hR : R = 1 + Δt • V) (hRinv : R * Rinv = 1 ∧ Rinv * R = 1)
    (X Z : ℕ → Idx n → EucSpace d) (hXZ : ∀ (k : ℕ) (i : Idx n), X k i = (R ^ k) (Z k i)) :
    DiscreteTransformer Δt Q K V X ↔ DiscreteRescaled Δt Q K V R Rinv Z := by
  have hX : ∀ k, X k = fun l => (R ^ k) (Z k l) := fun k => funext (hXZ k)
  have hVR : ∀ k, V * R ^ k = R ^ k * V := fun k => by
    refine Commute.pow_right ?_ k
    show V * R = R * V
    ext x; simp [hR]
  have hinj : ∀ k, Function.Injective ⇑(R ^ k) := fun k => by
    have hk : Rinv ^ k * R ^ k = 1 := by
      have hc : Commute Rinv R := hRinv.2.trans hRinv.1.symm
      rw [← hc.mul_pow, hRinv.2, one_pow]
    intro a b h
    calc a = (Rinv ^ k * R ^ k) a := by rw [hk]; rfl
      _ = (Rinv ^ k) ((R ^ k) a) := rfl
      _ = (Rinv ^ k * R ^ k) b := by rw [h]; rfl
      _ = b := by rw [hk]; rfl
  have key : ∀ k i, (R ^ (k + 1)) (Z k i + Δt • ∑ j : Idx n,
      attentionMatrix Q K (fun l => (R ^ k) (Z k l)) i j • Rinv (V (Z k j - Z k i)))
      = X k i + Δt • ∑ j : Idx n, attentionMatrix Q K (X k) i j • V (X k j) := by
    intro k i
    have hn : 0 < n := Fin.pos_iff_nonempty.mpr ⟨i⟩
    set P := attentionMatrix Q K (fun l => (R ^ k) (Z k l)) i
    have hP : ∑ j, P j = 1 := sum_attentionMatrix hn Q K _ i
    have hRk : ∀ v, (R ^ (k + 1)) (Rinv v) = (R ^ k) v := fun v => by
      rw [pow_succ]
      show (R ^ k) ((R * Rinv) v) = _
      rw [hRinv.1]; rfl
    rw [hX, map_add, map_smul, map_sum]
    simp only [map_smul, hRk, map_sub]
    have hVc : ∀ y, (R ^ k) (V y) = V ((R ^ k) y) := fun y =>
      (congrArg (fun M : ParamMatrix d => M y) (hVR k)).symm
    have hR1 : (R ^ (k + 1)) (Z k i) = (R ^ k) (Z k i) + Δt • V ((R ^ k) (Z k i)) := by
      rw [pow_succ]
      show (R ^ k) (R (Z k i)) = _
      rw [show R (Z k i) = Z k i + Δt • V (Z k i) by rw [hR]; simp, map_add, map_smul, hVc]
    simp only [smul_sub, Finset.sum_sub_distrib, ← Finset.sum_smul, hP, one_smul, hVc, hR1]
    simp only [P]
    abel
  constructor
  · intro h k i
    apply hinj (k + 1)
    rw [key, ← h k i, hXZ]
  · intro h k i
    rw [hXZ, h k i, key]

/-- The hypotheses of `discreteTransformer_iff_rescaled` are satisfiable: at
`V = 0` one has `R = I_d`, which is its own inverse, and every sequence is its
own rescaling. -/
example (Δt : ℝ) (Z : ℕ → Idx n → EucSpace d) :
    (1 : ParamMatrix d) = 1 + Δt • (0 : ParamMatrix d) ∧
      ((1 : ParamMatrix d) * 1 = 1 ∧ (1 : ParamMatrix d) * 1 = 1) ∧
      ∀ (k : ℕ) (i : Idx n), Z k i = ((1 : ParamMatrix d) ^ k) (Z k i) := by
  have h : Δt • (0 : ParamMatrix d) = 0 := by ext x; simp
  refine ⟨by rw [h, add_zero], ⟨one_mul 1, one_mul 1⟩, fun k i => ?_⟩
  rw [one_pow]
  rfl

end Clusters
end Transformer
