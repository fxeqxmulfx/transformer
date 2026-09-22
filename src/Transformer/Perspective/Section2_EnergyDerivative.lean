/-
# §3.2 — Differentiating the interaction energy along a continuity equation

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*, §3.2–§3.3.

Along a solution of `eq: CE` with a continuous bounded velocity `v`,

  `d/dt 𝖤_β[μ(t)] = ∫ ⟨∫ e^{β⟨x,y⟩} y dμ(t,y), v(t,x)⟩ dμ(t,x)`

(`hasDerivAt_interactionEnergy`).  This is the computation behind both
`eq: dissipation.softmax` and `lem: dissipation`; the two differ only in `v`.

The proof expands `e^{β⟨x,y⟩} = Σ_k β^k ⟨x,y⟩^k / k!`.  The `k`-th term of the
energy is `Σ_I (∫ m_I dμ)²` (`integral_prod_inner_pow`), whose derivative the
continuity equation gives term by term; summed back over `I` it is
`2 ∫∫ k ⟨x,y⟩^{k-1} ⟨y, v(x)⟩` (`sum_mono_mul_fderiv_mono`), bounded by `2kM`,
so the series may be differentiated term by term (`hasDerivAt_tsum`).
-/

import Transformer.Perspective.Section2_EnergySeries
import Mathlib.MeasureTheory.Function.L2Space
import Mathlib.Analysis.Calculus.SmoothSeries

open scoped BigOperators Nat
open Real MeasureTheory

namespace Transformer
namespace Perspective

variable {d : ℕ}

/-- The kernel of the `k`-th derivative term is at most `k M`. -/
theorem abs_kernel_le (k : ℕ) (w : SSphere d → EucSpace d) (M : ℝ) (hM : ∀ x, ‖w x‖ ≤ M)
    (p : SSphere d × SSphere d) :
    |k * inner (𝕜 := ℝ) (p.1 : EucSpace d) (p.2 : EucSpace d) ^ (k - 1) *
        inner (𝕜 := ℝ) (p.2 : EucSpace d) (w p.1)| ≤ k * M := by
  have hy : ‖(p.2 : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp p.2.2
  have h1 : |inner (𝕜 := ℝ) (p.1 : EucSpace d) (p.2 : EucSpace d)| ^ (k - 1) ≤ 1 :=
    pow_le_one₀ (abs_nonneg _) (abs_inner_sphere_le_one p.1 p.2)
  have h2 : |inner (𝕜 := ℝ) (p.2 : EucSpace d) (w p.1)| ≤ M := by
    calc _ ≤ ‖(p.2 : EucSpace d)‖ * ‖w p.1‖ := abs_real_inner_le_norm _ _
      _ ≤ M := by rw [hy, one_mul]; exact hM p.1
  rw [abs_mul, abs_mul, abs_pow, Nat.abs_cast]
  calc (k : ℝ) * |inner (𝕜 := ℝ) (p.1 : EucSpace d) (p.2 : EucSpace d)| ^ (k - 1) *
        |inner (𝕜 := ℝ) (p.2 : EucSpace d) (w p.1)|
      ≤ k * 1 * M := by gcongr
    _ = k * M := by ring

/-- **The derivative of the interaction energy along `eq: CE`.**  For a
velocity `v` continuous on the sphere and bounded by `M`,

  `d/dt 𝖤_β[μ(t)] = ∫ ⟨∫ e^{β⟨x,y⟩} y dμ(t,y), v(t,x)⟩ dμ(t,x)`.

Source: arXiv:2312.10794v5, §3.2, the computation behind
`eq: dissipation.softmax` (and §3.3, `lem: dissipation`). -/
theorem hasDerivAt_interactionEnergy (β : ℝ) (hβ : β ≠ 0) (μ : ℝ → ProbSphere d)
    (v : ℝ → EucSpace d → EucSpace d) (hCE : auxCE d μ v)
    (hv : ∀ s, Continuous fun x : SSphere d => v s x) (M : ℝ)
    (hM : ∀ s (x : SSphere d), ‖v s x‖ ≤ M) (t : ℝ) :
    HasDerivAt (fun s => interactionEnergy d β (μ s))
      (∫ x, inner (𝕜 := ℝ)
          (∫ y, exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) • (y : EucSpace d)
            ∂(μ t : Measure (SSphere d))) (v t x) ∂(μ t : Measure (SSphere d))) t := by
  set ρ : ℝ → Measure (SSphere d) := fun s => (μ s : Measure (SSphere d)) with hρ
  set J : ℕ → ℝ → ℝ := fun k s => ∫ p : SSphere d × SSphere d,
    k * inner (𝕜 := ℝ) (p.1 : EucSpace d) (p.2 : EucSpace d) ^ (k - 1) *
      inner (𝕜 := ℝ) (p.2 : EucSpace d) (v s p.1) ∂(ρ s).prod (ρ s) with hJ
  set g : ℕ → ℝ → ℝ := fun k s => (2 * β)⁻¹ * (β ^ k / k ! *
    ∑ I : Fin k → Fin d, (∫ x : SSphere d, mono I (x : EucSpace d) ∂ρ s) ^ 2) with hg
  set g' : ℕ → ℝ → ℝ := fun k s => (2 * β)⁻¹ * (β ^ k / k ! * (2 * J k s)) with hg'
  have hJle : ∀ k s, |J k s| ≤ k * M := fun k s => by
    have hle : ∀ p : SSphere d × SSphere d,
        ‖(k : ℝ) * inner (𝕜 := ℝ) (p.1 : EucSpace d) (p.2 : EucSpace d) ^ (k - 1) *
          inner (𝕜 := ℝ) (p.2 : EucSpace d) (v s p.1)‖ ≤ k * M := fun p => by
      rw [Real.norm_eq_abs]; exact abs_kernel_le k (fun x => v s x) M (hM s) p
    have := norm_integral_le_of_norm_le_const (μ := (ρ s).prod (ρ s)) (ae_of_all _ hle)
    simpa [J, Real.norm_eq_abs] using this
  have hderiv : ∀ k s, HasDerivAt (g k) (g' k s) s := fun k s => by
    have ha : ∀ I : Fin k → Fin d, HasDerivAt (fun s => ∫ x : SSphere d, mono I (x : EucSpace d)
        ∂ρ s) (∫ x : SSphere d, inner (𝕜 := ℝ) (gradient (mono I) (x : EucSpace d)) (v s x)
          ∂ρ s) s := fun I => hCE _ (contDiff_mono I 1) s
    have := ((HasDerivAt.fun_sum (u := Finset.univ) fun I _ => (ha I).fun_pow 2).const_mul
      (β ^ k / k !)).const_mul (2 * β)⁻¹
    convert this using 1
    show (2 * β)⁻¹ * (β ^ k / k ! * (2 * J k s)) = _
    rw [hJ]; dsimp only
    rw [← sum_integral_mono_mul_integral_gradient k (ρ s) _ (hv s), Finset.mul_sum]
    congr 2
    refine Finset.sum_congr rfl fun I _ => ?_
    simp only [Nat.cast_ofNat, pow_one, Nat.add_one_sub_one]
    ring
  set u : ℕ → ℝ := fun k => |(2 * β)⁻¹| * (|β| ^ k / k ! * (k * 1 ^ (k - 1))) * (2 * M)
    with hu
  have hsum1 : Summable fun k : ℕ => |β| ^ k / k ! * (k * (1 : ℝ) ^ (k - 1)) :=
    (hasSum_deriv_exp_mul |β| 1).summable
  have hbound : ∀ k s, ‖g' k s‖ ≤ u k := fun k s => by
    have h := hJle k s
    calc ‖g' k s‖ = |(2 * β)⁻¹| * (|β| ^ k / k ! * (2 * |J k s|)) := by
          rw [Real.norm_eq_abs, hg']; dsimp only
          rw [abs_mul, abs_mul, abs_mul, abs_div, abs_pow, Nat.abs_cast, abs_two]
      _ ≤ |(2 * β)⁻¹| * (|β| ^ k / k ! * (2 * (k * M))) := by gcongr
      _ = u k := by rw [hu]; dsimp only; rw [one_pow, mul_one]; ring
  have hE : (fun s => interactionEnergy d β (μ s)) = fun s => ∑' k, g k s :=
    funext fun s => (hasSum_interactionEnergy β (μ s)).tsum_eq.symm
  have hmain := hasDerivAt_tsum ((hsum1.mul_left _).mul_right _) hderiv hbound
    (hasSum_interactionEnergy β (μ t)).summable t
  rw [hE]
  convert hmain using 1
  refine (HasSum.tsum_eq ?_).symm
  have hK := continuous_inner_sphere (d := d)
  have hw : Continuous fun p : SSphere d × SSphere d =>
      inner (𝕜 := ℝ) (p.2 : EucSpace d) (v t p.1) :=
    (continuous_subtype_val.comp continuous_snd).inner ((hv t).comp continuous_fst)
  have hS := hasSum_integral_of_abs_le ((ρ t).prod (ρ t))
    (F := fun k p => β ^ k / k ! * (k * inner (𝕜 := ℝ) (p.1 : EucSpace d) (p.2 : EucSpace d)
      ^ (k - 1) * inner (𝕜 := ℝ) (p.2 : EucSpace d) (v t p.1)))
    (G := fun p => β * exp (β * inner (𝕜 := ℝ) (p.1 : EucSpace d) (p.2 : EucSpace d)) *
      inner (𝕜 := ℝ) (p.2 : EucSpace d) (v t p.1))
    (u := fun k => |β| ^ k / k ! * (k * (1 : ℝ) ^ (k - 1)) * M)
    (fun k => continuous_const.mul ((continuous_const.mul (hK.pow _)).mul hw))
    (hsum1.mul_right M) (fun k p => ?_) (fun p => ?_)
  rotate_left
  · calc _ = |β| ^ k / k ! * |k * inner (𝕜 := ℝ) (p.1 : EucSpace d) (p.2 : EucSpace d)
          ^ (k - 1) * inner (𝕜 := ℝ) (p.2 : EucSpace d) (v t p.1)| := by
          rw [abs_mul, abs_div, abs_pow, Nat.abs_cast]
      _ ≤ |β| ^ k / k ! * (k * M) := by
          gcongr; exact abs_kernel_le k (fun x => v t x) M (hM t) p
      _ = _ := by rw [one_pow, mul_one]; ring
  · convert (hasSum_deriv_exp_mul β _).mul_right
      (inner (𝕜 := ℝ) (p.2 : EucSpace d) (v t p.1)) using 1
    funext k; ring
  have hint : ∀ x : SSphere d, Integrable (fun y : SSphere d =>
      exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) • (y : EucSpace d)) (ρ t) :=
    fun x => integrable_of_continuous_compact
      ((continuous_expInner_right d β x).smul continuous_subtype_val) _
  have hG : ∫ p, β * exp (β * inner (𝕜 := ℝ) (p.1 : EucSpace d) (p.2 : EucSpace d)) *
      inner (𝕜 := ℝ) (p.2 : EucSpace d) (v t p.1) ∂(ρ t).prod (ρ t) =
      β * ∫ x, inner (𝕜 := ℝ) (∫ y, exp (β * inner (𝕜 := ℝ) (x : EucSpace d)
        (y : EucSpace d)) • (y : EucSpace d) ∂ρ t) (v t x) ∂ρ t := by
    simp_rw [mul_assoc]
    have hi : Integrable (fun p : SSphere d × SSphere d =>
        exp (β * inner (𝕜 := ℝ) (p.1 : EucSpace d) (p.2 : EucSpace d)) *
          inner (𝕜 := ℝ) (p.2 : EucSpace d) (v t p.1)) ((ρ t).prod (ρ t)) :=
      integrable_of_continuous_compact ((continuous_const.mul hK).rexp.mul hw) _
    rw [integral_const_mul, integral_prod _ hi]
    congr 1
    refine integral_congr_ae (ae_of_all _ fun x => ?_)
    dsimp only
    rw [real_inner_comm (v t x), ← integral_inner (hint x)]
    refine integral_congr_ae (ae_of_all _ fun y => ?_)
    dsimp only
    rw [inner_smul_right, real_inner_comm (v t x) (y : EucSpace d)]
  convert hS.mul_left ((2 * β)⁻¹ * 2) using 1
  · funext k
    rw [hg']; dsimp only
    rw [integral_const_mul, hJ]; dsimp only
    ring
  · rw [hG]; field_simp; rfl

end Perspective
end Transformer
