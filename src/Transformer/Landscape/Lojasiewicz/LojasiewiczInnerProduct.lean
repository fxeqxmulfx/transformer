module

public import Transformer.Landscape.Lojasiewicz.Modulated

/-!
# Łojasiewicz convergence in any finite-dimensional inner product space

`lojasiewicz_modulated` is stated in `ℝ^N`. A linear isometry carries gradients to
gradients (`gradient_comp_linearIsometryEquiv_symm`), and every finite-dimensional
real inner product space is isometric to `ℝ^N` (`stdOrthonormalBasis`), so the
theorem holds verbatim in all of them (`lojasiewicz_modulated_of_finiteDimensional`):
for instance on tuples of vectors with the `ℓ²` norm, where `Perspective` writes the
sphere flows.

Source: arXiv:2510.22026v2, Appendix D.1, `lem: loj`.
-/

@[expose] public section

namespace Transformer
namespace Landscape.Lojasiewicz

variable {E F : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  [NormedAddCommGroup F] [InnerProductSpace ℝ F]

/-- **Linear isometries carry gradients:** `∇(f ∘ e⁻¹)(y) = e ∇f(e⁻¹ y)`, at every
point, differentiable or not. -/
theorem gradient_comp_linearIsometryEquiv_symm [CompleteSpace E] [CompleteSpace F]
    (e : E ≃ₗᵢ[ℝ] F) (f : E → ℝ) (y : F) :
    gradient (f ∘ e.symm) y = e (gradient f (e.symm y)) := by
  apply (InnerProductSpace.toDual ℝ F).injective
  rw [toDual_gradient]
  ext v
  rw [show f ∘ e.symm = f ∘ e.symm.toContinuousLinearEquiv from rfl,
    ContinuousLinearEquiv.comp_right_fderiv, InnerProductSpace.toDual_apply_apply,
    e.inner_map_eq_flip, ← toDual_gradient, ContinuousLinearMap.comp_apply,
    InnerProductSpace.toDual_apply_apply]
  rfl

/-- The hypotheses of `gradient_comp_linearIsometryEquiv_symm` hold for the identity
of `ℝ²`. -/
example : gradient ((fun _ : EucSpace 2 => (0 : ℝ)) ∘ (LinearIsometryEquiv.refl ℝ
    (EucSpace 2)).symm) 0 = LinearIsometryEquiv.refl ℝ (EucSpace 2)
      (gradient (fun _ : EucSpace 2 => (0 : ℝ)) 0) :=
  gradient_comp_linearIsometryEquiv_symm _ _ _

/-- **Lemma (lem: loj) in a finite-dimensional inner product space `E`.**  Let `M(t)`
be symmetric with `C λ(t) I ≻ M(t) ≻ λ(t) I`, `λ(t) > 0` and `∫_0^∞ λ = ∞`, and let `f`
be analytic on an open `U ⊆ E`. Then any trajectory of `ẋ = -M(t) ∇f(x)` that stays
in a compact subset of `U` converges to a critical point of `f`.

The hypotheses are those of `lojasiewicz_modulated`, with `E` for `ℝ^N`; the
theorem is that one, read through an isometry `E ≃ ℝ^N`.

Source: arXiv:2510.22026v2, Appendix D.1, `lem: loj`. -/
theorem lojasiewicz_modulated_of_finiteDimensional [FiniteDimensional ℝ E]
    (U : Set E) (hU : IsOpen U) (f : E → ℝ) (hf : AnalyticOnNhd ℝ f U)
    (C : ℝ) (lam : ℝ → ℝ) (M : ℝ → E →L[ℝ] E)
    (hlam : ∀ t : ℝ, 0 ≤ t → 0 < lam t)
    (hMsymm : ∀ t : ℝ, 0 ≤ t → ∀ v w : E,
      inner (𝕜 := ℝ) (M t v) w = inner (𝕜 := ℝ) v (M t w))
    (hMlower : ∀ t : ℝ, 0 ≤ t → ∀ v : E, v ≠ 0 →
      lam t * ‖v‖ ^ 2 < inner (𝕜 := ℝ) (M t v) v)
    (hMupper : ∀ t : ℝ, 0 ≤ t → ∀ v : E, v ≠ 0 →
      inner (𝕜 := ℝ) (M t v) v < C * lam t * ‖v‖ ^ 2)
    (hint : Filter.Tendsto (fun T : ℝ => ∫ s in (0 : ℝ)..T, lam s)
      Filter.atTop Filter.atTop)
    (x : ℝ → E) (K : Set E) (hK : IsCompact K) (hKU : K ⊆ U)
    (hxK : ∀ t : ℝ, 0 ≤ t → x t ∈ K)
    (hx : ∀ t : ℝ, 0 ≤ t → HasDerivAt x (-(M t (gradient f (x t)))) t) :
    ∃ xstar ∈ U, Filter.Tendsto x Filter.atTop (nhds xstar) ∧
      gradient f xstar = 0 := by
  set N := Module.finrank ℝ E
  set e : E ≃ₗᵢ[ℝ] EucSpace N := (stdOrthonormalBasis ℝ E).repr
  have hg : ∀ y, gradient (f ∘ e.symm) y = e (gradient f (e.symm y)) :=
    gradient_comp_linearIsometryEquiv_symm e f
  set M' : ℝ → ParamMatrix N := fun t =>
    (e.toContinuousLinearEquiv : E →L[ℝ] EucSpace N).comp
      ((M t).comp (e.symm.toContinuousLinearEquiv : EucSpace N →L[ℝ] E)) with hM'
  have hM'v : ∀ t v, M' t v = e (M t (e.symm v)) := fun _ _ => rfl
  have hnorm : ∀ v : EucSpace N, ‖e.symm v‖ = ‖v‖ := e.symm.norm_map
  have hne : ∀ v : EucSpace N, v ≠ 0 → e.symm v ≠ 0 := fun v hv h =>
    hv (e.symm.map_eq_zero_iff.1 h)
  obtain ⟨y, hyU, hy, hgy⟩ := lojasiewicz_modulated N (e.symm ⁻¹' U)
    (hU.preimage e.symm.continuous) (f ∘ e.symm)
    (fun z hz => (hf _ hz).comp (e.symm.analyticAt z)) C lam M' hlam
    (fun t ht v w => by
      rw [hM'v, hM'v, e.inner_map_eq_flip, hMsymm t ht, e.symm.inner_map_eq_flip,
        LinearIsometryEquiv.symm_symm])
    (fun t ht v hv => by
      rw [hM'v, e.inner_map_eq_flip, ← hnorm v]
      exact hMlower t ht _ (hne v hv))
    (fun t ht v hv => by
      rw [hM'v, e.inner_map_eq_flip, ← hnorm v]
      exact hMupper t ht _ (hne v hv))
    hint (fun t => e (x t)) (e '' K) (hK.image e.continuous)
    (fun _ ⟨z, hz, hze⟩ => by rw [Set.mem_preimage, ← hze, e.symm_apply_apply]; exact hKU hz)
    (fun t ht => Set.mem_image_of_mem e (hxK t ht))
    (fun t ht => by
      rw [hg, e.symm_apply_apply, hM'v, e.symm_apply_apply]
      exact (e.toContinuousLinearEquiv.hasFDerivAt.comp_hasDerivAt t (hx t ht)).congr_deriv
        (by simp))
  refine ⟨e.symm y, hyU, ?_, ?_⟩
  · refine ((e.symm.continuous.tendsto y).comp hy).congr fun t => ?_
    simp
  · rw [hg] at hgy
    exact e.map_eq_zero_iff.1 hgy

/-- All inputs of `lojasiewicz_modulated_of_finiteDimensional` are satisfiable on
`ℝ × ℝ` with the `ℓ²` norm, for a constant energy and trajectory, `M = I`,
`λ = 1/2` and `C = 4`. -/
example : ∃ z ∈ (Set.univ : Set (WithLp 2 (ℝ × ℝ))),
    Filter.Tendsto (fun _ : ℝ => (0 : WithLp 2 (ℝ × ℝ))) Filter.atTop (nhds z) ∧
      gradient (fun _ : WithLp 2 (ℝ × ℝ) => (0 : ℝ)) z = 0 := by
  refine lojasiewicz_modulated_of_finiteDimensional Set.univ isOpen_univ
    (fun _ => 0) analyticOnNhd_const 4 (fun _ => 1 / 2)
    (fun _ => ContinuousLinearMap.id ℝ _) (fun _ _ => by norm_num) (fun _ _ _ _ => rfl)
    ?_ ?_ ?_ (fun _ => 0) {0} isCompact_singleton (Set.subset_univ _)
    (fun _ _ => Set.mem_singleton 0) fun t _ => by simpa using hasDerivAt_const t _
  · intro t ht w hw
    rw [ContinuousLinearMap.id_apply, real_inner_self_eq_norm_sq]
    nlinarith [sq_pos_of_pos (norm_pos_iff.mpr hw)]
  · intro t ht w hw
    rw [ContinuousLinearMap.id_apply, real_inner_self_eq_norm_sq]
    nlinarith [sq_pos_of_pos (norm_pos_iff.mpr hw)]
  · have heq : (fun T : ℝ => ∫ _s in (0 : ℝ)..T, (1 / 2 : ℝ)) = fun T : ℝ => T / 2 := by
      ext T
      simp
      ring
    rw [heq]
    exact Filter.tendsto_id.atTop_div_const (by norm_num)

end Landscape.Lojasiewicz
end Transformer
