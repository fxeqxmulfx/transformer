module

public import Transformer.Perspective.SphereCritical

/-!
# Strict saddles of `𝖤_β` on `(𝕊^{d-1})^n` for `d ≥ 3`

Criscitiello, Rebjock, McRae, Boumal — arXiv:2405.18273, `thm:spheres` and
`cor:spheres` through Appendix C, for the `𝖤_β` of arXiv:2312.10794v5.

The curve `s ↦ (x_i + s v_i)/‖x_i + s v_i‖` of a tangent `v` is smooth
(`contDiff_normalizedLine`) with velocity `v` (`hasDerivAt_normalizedLine`),
so at a critical point the second derivative of `𝖤_β` along it is
`2 · hessQuad`. Along `v_i = Proj_{x_i} e_k` for the `e_k` of
`exists_hessQuad_proj_pos` it is positive: for `d ≥ 3` and `β > 0`, every
critical point with two distinct tokens is a strict saddle
(`exists_secondDerivEBetaAt_pos_of_nonTrivial`). The synchronized
configurations are global maxima (`selfEnergy_le_of_not_nonTrivial`), and
with them this is `thm:spheres` for `𝖤_β`
(`synchronized_of_secondDerivEBetaAt_nonpos`).
-/

@[expose] public section

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

variable (d n : ℕ)

/-- A point `x` of the sphere moved along a tangent `v` never reaches the
origin: `⟨x, x + s v⟩ = 1`. -/
theorem add_smul_ne_zero {x v : EucSpace d} (hx : ‖x‖ = 1) (hv : inner (𝕜 := ℝ) x v = 0)
    (s : ℝ) : x + s • v ≠ 0 := by
  intro h
  have h' := congrArg (inner (𝕜 := ℝ) x) h
  rw [inner_add_right, real_inner_smul_right, hv, inner_self_eq_one_of_norm_eq_one hx,
    inner_zero_right] at h'
  norm_num at h'

/-- **The normalized line `s ↦ (x + s v)/‖x + s v‖` is smooth**, for `x` on
the sphere and `v` tangent at `x`. -/
theorem contDiff_normalizedLine {x v : EucSpace d} (hx : ‖x‖ = 1)
    (hv : inner (𝕜 := ℝ) x v = 0) :
    ContDiff ℝ 2 fun s : ℝ => ‖x + s • v‖⁻¹ • (x + s • v) := by
  have hw : ContDiff ℝ 2 fun s : ℝ => x + s • v :=
    contDiff_const.add (contDiff_id.smul contDiff_const)
  exact ((hw.norm ℝ (add_smul_ne_zero d hx hv)).inv
    fun s => norm_ne_zero_iff.2 (add_smul_ne_zero d hx hv s)).smul hw

/-- **The normalized line has velocity `v` at `s = 0`**: the norm `‖x + s v‖`
has derivative `⟨x, v⟩ = 0` there. -/
theorem hasDerivAt_normalizedLine {x v : EucSpace d} (hx : ‖x‖ = 1)
    (hv : inner (𝕜 := ℝ) x v = 0) :
    HasDerivAt (fun s : ℝ => ‖x + s • v‖⁻¹ • (x + s • v)) v 0 := by
  have hw : HasDerivAt (fun s : ℝ => x + s • v) v 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).smul_const v).const_add x
  have hsq : HasDerivAt (fun s : ℝ => ‖x + s • v‖ ^ 2) 0 0 := by
    simpa [hv] using hw.norm_sq
  have hn : HasDerivAt (fun s : ℝ => ‖x + s • v‖) 0 0 := by
    simpa [Real.sqrt_sq_eq_abs] using hsq.sqrt (by simp [hx])
  convert (hn.inv (by simp [hx])).smul hw using 1
  simp [hx]

/-- **Strict saddles of `𝖤_β` for `d ≥ 3`.**  For `β > 0`, every critical
point of `𝖤_β` on `(𝕊^{d-1})^n` with two distinct tokens is a strict saddle:
the normalized line `s ↦ (x_i + s v_i)/‖x_i + s v_i‖` of some tangent `v` sees
a positive second derivative of `𝖤_β`, so the ascent leaves it.

The tangent is `v_i = Proj_{x_i} e_k` for a basis vector `e_k`, which moves
every token toward the same `e_k`. The second derivative along the line is
`2 · hessQuad(X, v)` (`secondDerivEBetaAt_eq_two_mul_hessQuad`), and
`exists_hessQuad_proj_pos` gives a `k` that makes it positive.

**What the source says and what is changed here.**  The source proves the
contrapositive, `thm:spheres` (`synchronized_of_secondDerivEBetaAt_nonpos`);
here the direction is exhibited, as `IsStrictSaddleTorusEnergy` asks on the
circle. Its Gaussian `γ` is replaced by the `d` basis vectors, which have the
same second moments. The term `⟨S, 2XᵀX⟩` that it removes with the
first-order condition `XS = 0` vanishes identically, so criticality is used
only to identify the second derivative with `2 · hessQuad`.

Source: arXiv:2405.18273, Appendix C; arXiv:2312.10794v5, §6, `thm: boumal`,
whose landscape half it is. -/
theorem exists_secondDerivEBetaAt_pos_of_nonTrivial (hd : 3 ≤ d) (β : ℝ) (hβ : 0 < β)
    (X : SphereTuple d n) (hcrit : IsCriticalEBeta d n β X) (hnt : NonTrivialTuple d n X) :
    ∃ (v : Idx n → EucSpace d) (Y : ℝ → SphereTuple d n) (c : ℝ),
      (∀ i, inner (𝕜 := ℝ) ((X i : EucSpace d)) (v i) = 0) ∧
      (∀ s i, (Y s i : EucSpace d)
        = ‖(X i : EucSpace d) + s • v i‖⁻¹ • ((X i : EucSpace d) + s • v i)) ∧
      SecondDerivEBetaAt d n β Y c ∧ 0 < c := by
  obtain ⟨k, hk⟩ := exists_hessQuad_proj_pos d n hd β hβ X hnt
  set v : Idx n → EucSpace d := fun i => proj d (X i) (EuclideanSpace.basisFun (Fin d) ℝ k)
  have hX : ∀ i, ‖(X i : EucSpace d)‖ = 1 := fun i => mem_sphere_zero_iff_norm.mp (X i).2
  have hv : ∀ i, inner (𝕜 := ℝ) ((X i : EucSpace d)) (v i) = 0 := fun i =>
    inner_proj_eq_zero (hX i) _
  let Y : ℝ → SphereTuple d n := fun s i =>
    ⟨‖(X i : EucSpace d) + s • v i‖⁻¹ • ((X i : EucSpace d) + s • v i), by
      rw [mem_sphere_zero_iff_norm, norm_smul, norm_inv, norm_norm,
        inv_mul_cancel₀ (norm_ne_zero_iff.2 (add_smul_ne_zero d (hX i) (hv i) s))]⟩
  have hY0 : Y 0 = X := funext fun i => Subtype.ext (by simp [Y, hX i])
  have hf : ContDiff ℝ 2 fun s => selfEnergy d n β (Y s) := by
    simp only [selfEnergy, particleEnergy, ContinuousLinearMap.id_apply]
    exact contDiff_const.mul (ContDiff.sum fun i _ => ContDiff.sum fun j _ =>
      (contDiff_const.mul ((contDiff_normalizedLine d (hX i) (hv i)).inner ℝ
        (contDiff_normalizedLine d (hX j) (hv j)))).exp)
  have hc : SecondDerivEBetaAt d n β Y (deriv (deriv fun s => selfEnergy d n β (Y s)) 0) :=
    ⟨_, fun t => ((hf.differentiable (by norm_num)) t).hasDerivAt,
      (hf.differentiable_deriv_two 0).hasDerivAt⟩
  refine ⟨v, Y, _, hv, fun s i => rfl, hc, ?_⟩
  rw [secondDerivEBetaAt_eq_two_mul_hessQuad d n β X hcrit Y v hY0
    (fun i => hasDerivAt_normalizedLine d (hX i) (hv i)) _ hc]
  linarith

/-- **Synchronized configurations are global maxima of `𝖤_β`** for `β > 0`:
when all tokens coincide, every `⟨x_i, x_j⟩` is `1`, its largest value. -/
theorem selfEnergy_le_of_not_nonTrivial (β : ℝ) (hβ : 0 < β) (X : SphereTuple d n)
    (hX : ¬ NonTrivialTuple d n X) (Z : SphereTuple d n) :
    selfEnergy d n β Z ≤ selfEnergy d n β X := by
  simp only [NonTrivialTuple, not_exists, not_not] at hX
  simp only [selfEnergy, particleEnergy, ContinuousLinearMap.id_apply]
  refine mul_le_mul_of_nonneg_left (Finset.sum_le_sum fun i _ => Finset.sum_le_sum fun j _ => ?_)
    (inv_nonneg.2 (by linarith))
  rw [hX i j, inner_self_eq_one_of_norm_eq_one (mem_sphere_zero_iff_norm.mp (X j).2)]
  refine exp_le_exp.2 (mul_le_mul_of_nonneg_left ?_ hβ.le)
  have := real_inner_le_norm ((Z i : EucSpace d)) ((Z j : EucSpace d))
  rwa [mem_sphere_zero_iff_norm.mp (Z i).2, mem_sphere_zero_iff_norm.mp (Z j).2,
    one_mul] at this

/-- **Theorem (thm:spheres of arXiv:2405.18273), for `𝖤_β`.**  *For `d ≥ 3`
and `β > 0`, a critical point of `𝖤_β` on `(𝕊^{d-1})^n` where the Hessian is
negative semidefinite is synchronized, `x_1 = ⋯ = x_n`, and a global maximum.*

**What the source says and what is changed here.**  The source states it for
`n ≥ 1`, a connected graph and `φ' > 0`, `φ'' ≥ 0` on `[-1, 1]`; `cor:spheres`
applies it to `φ_β(t) = e^{βt}/β`, `β > 0`; `𝖤_β` is the complete graph with
unit weights, and `n` is free. The Hessian is negative semidefinite when
every curve through `X` that is differentiable at `0` has a non-positive
second derivative of `𝖤_β`; at a critical point that is `hessQuad ≤ 0` on the
tangent space (`secondDerivEBetaAt_eq_two_mul_hessQuad`).

Source: arXiv:2405.18273, Theorem `thm:spheres`, Corollary `cor:spheres`,
Appendix C. -/
theorem synchronized_of_secondDerivEBetaAt_nonpos (hd : 3 ≤ d) (β : ℝ) (hβ : 0 < β)
    (X : SphereTuple d n) (hcrit : IsCriticalEBeta d n β X)
    (hneg : ∀ (Y : ℝ → SphereTuple d n) (v : Idx n → EucSpace d) (c : ℝ), Y 0 = X →
      (∀ i, HasDerivAt (fun s => (Y s i : EucSpace d)) (v i) 0) →
        SecondDerivEBetaAt d n β Y c → c ≤ 0) :
    (∀ i j, X i = X j) ∧ ∀ Z, selfEnergy d n β Z ≤ selfEnergy d n β X := by
  have hX : ∀ i, ‖(X i : EucSpace d)‖ = 1 := fun i => mem_sphere_zero_iff_norm.mp (X i).2
  have hnt : ¬ NonTrivialTuple d n X := by
    intro hnt
    obtain ⟨v, Y, c, hv, hY, hc, hcpos⟩ :=
      exists_secondDerivEBetaAt_pos_of_nonTrivial d n hd β hβ X hcrit hnt
    have hY0 : Y 0 = X := funext fun i => Subtype.ext (by rw [hY]; simp [hX i])
    have hYd : ∀ i, HasDerivAt (fun s => (Y s i : EucSpace d)) (v i) 0 := fun i => by
      simpa only [hY] using hasDerivAt_normalizedLine d (hX i) (hv i)
    exact absurd (hneg Y v c hY0 hYd hc) (not_le.2 hcpos)
  refine ⟨fun i j => Subtype.ext ?_, selfEnergy_le_of_not_nonTrivial d n β hβ X hnt⟩
  simp only [NonTrivialTuple, not_exists, not_not] at hnt
  exact hnt i j

/-! ### The hypotheses are satisfiable -/

/-- The hypotheses of `add_smul_ne_zero`, `contDiff_normalizedLine` and
`hasDerivAt_normalizedLine` are satisfiable: `e_0` is a unit vector of `ℝ²`
and `e_1` is tangent at it. -/
example : ‖(EuclideanSpace.single (0 : Fin 2) (1 : ℝ) : EucSpace 2)‖ = 1 ∧
    inner (𝕜 := ℝ) (EuclideanSpace.single (0 : Fin 2) (1 : ℝ) : EucSpace 2)
      (EuclideanSpace.single 1 1) = 0 := by
  simp [EuclideanSpace.inner_single_left]

/-- The hypotheses of `exists_secondDerivEBetaAt_pos_of_nonTrivial` hold for
the antipodal pair of `𝕊²`, a critical point of `𝖤_1` with distinct tokens. -/
example : 3 ≤ 3 ∧ (0 : ℝ) < 1 ∧ IsCriticalEBeta 3 2 1 (antipodalPair 3 (basePoint 2)) ∧
    NonTrivialTuple 3 2 (antipodalPair 3 (basePoint 2)) :=
  ⟨le_rfl, one_pos, antipodalPair_isCriticalEBeta 3 1 (basePoint 2),
    (antipodalPair_critical_nonTrivial 3 (basePoint 2)).2⟩

/-- The hypotheses of `selfEnergy_le_of_not_nonTrivial` and
`synchronized_of_secondDerivEBetaAt_nonpos` are satisfiable: a single token of
`𝕊²` at `β = 1` is critical, and `𝖤_1` is constant on `(𝕊²)^1`
(`selfEnergy_one`), so every second derivative along a curve is `0`. -/
example : 3 ≤ 3 ∧ (0 : ℝ) < 1 ∧ ¬ NonTrivialTuple 3 1 (fun _ => basePoint 2) ∧
    IsCriticalEBeta 3 1 1 (fun _ => basePoint 2) ∧
    ∀ (Y : ℝ → SphereTuple 3 1) (v : Idx 1 → EucSpace 3) (c : ℝ), Y 0 = (fun _ => basePoint 2) →
      (∀ i, HasDerivAt (fun s => (Y s i : EucSpace 3)) (v i) 0) →
        SecondDerivEBetaAt 3 1 1 Y c → c ≤ 0 := by
  have hx : ‖((basePoint 2 : SSphere 3) : EucSpace 3)‖ = 1 :=
    mem_sphere_zero_iff_norm.mp (basePoint 2).2
  refine ⟨le_rfl, one_pos, fun ⟨i, j, h⟩ => h (by rw [Subsingleton.elim i j]),
    fun i => ?_, fun Y v c _ _ ⟨f', hf', hc⟩ => ?_⟩
  · rw [Fin.sum_univ_one]
    exact proj_smul_self hx _
  · have h0 : f' = fun _ => 0 := funext fun t => (hf' t).unique (by
      simp only [selfEnergy_one]
      exact hasDerivAt_const t _)
    rw [h0] at hc
    exact (hc.unique (hasDerivAt_const 0 0)).le

end Perspective
end Transformer
