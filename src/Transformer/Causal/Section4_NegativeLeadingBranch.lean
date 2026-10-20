module

public import Transformer.Causal.Section4_TransverseQuotient
public import Transformer.Causal.Section4_SubspaceFollowerLimits
public import Transformer.Causal.Section4_LeadingDirection

/-!
# The exceptional branch for an arbitrary original follower

ArXiv:2411.04990v2, §4, `thm2`, under its original spectral hypotheses.
Once all original predecessors converge to a leading unit vector `u`,
the actual follower either converges to `u`, or its longitudinal overlap
stays uniformly negative and all its transverse leading components tend
to zero. Its nonzero leading projection then has normalized limit `-u`.
The results have no token-count or hemisphere restriction.

The second alternative really occurs, for stationary antipodal pairs.
The assertion that its basin has measure zero for arbitrary original
prefixes is still unproved; this module does not assume or assert it.
-/

@[expose] public section

open Real Filter
open scoped Topology

namespace Transformer.Causal

variable {d n : ℕ}

/-- Every actual last token with a converging original leading prefix
either converges to the prefix limit, or remains uniformly negative
and loses all leading components orthogonal to that limit.
Source: arXiv:2411.04990v2, §4, a count-unrestricted reduction of `thm2`.
The exceptional basin's measure is not part of this conclusion. -/
theorem csaLast_tendsto_or_negative_leading (β : ℝ) (Q K V : ParamMatrix d)
    (L : Submodule ℝ (EucSpace d)) (lam : ℝ) (hlam : 0 < lam)
    (heig : ∀ v ∈ L, V v = lam • v)
    (hinv : ∀ z ∈ Lᗮ, V z ∈ Lᗮ)
    (hdecay : ∀ z ∈ Lᗮ, z ≠ 0 → inner ℝ (V z) z < lam * ‖z‖ ^ 2)
    (u : SSphere d) (hu : (u : EucSpace d) ∈ L)
    (X : ℝ → SphereTuple d (n + 1)) (hX : CSA d (n + 1) β Q K V X) (hn : 1 ≤ n)
    (hp : ∀ j : Idx n, Tendsto (fun t => (X t j.castSucc : EucSpace d)) atTop
      (nhds (u : EucSpace d))) :
    Tendsto (fun t => (X t (Fin.last n) : EucSpace d)) atTop (nhds (u : EucSpace d)) ∨
      ∃ δ : ℝ, 0 < δ ∧
        (∀ᶠ t in atTop, inner ℝ (u : EucSpace d) (X t (Fin.last n) : EucSpace d) ≤ -δ) ∧
        ∀ v ∈ L, inner ℝ v u = 0 →
          Tendsto (fun t => inner ℝ v (X t (Fin.last n) : EucSpace d)) atTop (nhds 0) := by
  let q : ℝ → ℝ := fun t => inner ℝ (u : EucSpace d) (X t (Fin.last n) : EucSpace d)
  obtain ⟨c, hc, hf⟩ := eventually_csaLastLeadingForcing_pos β Q K lam hlam X hn u hp
  obtain ⟨T, hT⟩ := eventually_atTop.mp hf
  obtain ⟨δ, hδ, hpos | hneg⟩ := eventually_nonneg_or_uniformly_negative
    (csaLastLeadingCoefficient β Q K V lam X) (csaLastLeadingForcing β Q K lam X u) q
    (hasDerivAt_csaLast_leading_forcing β Q K V L lam heig hinv _ hu X hX)
    (fun t => csaSphereOverlap_bounds u (X t (Fin.last n)))
    (|lam| + ‖V‖ + 1) c T (by positivity) hc
    (fun t _ => (abs_csaLastLeadingCoefficient_le β Q K V lam X t).trans (by linarith)) hT
  · exact Or.inl (csaLast_tendsto_of_eventually_nonneg_leading β Q K V L lam hlam
      heig hinv hdecay u hu X hX hn hp hpos)
  · right
    have hneg' := hneg.mono (fun _ h => h.le)
    exact ⟨δ, hδ, hneg', fun v hv hvu =>
      csaLast_transverse_tendsto_of_uniformly_negative β Q K V L lam hlam
        heig hinv u hu v hv hvu X hX hn hp δ hδ hneg'⟩

/-- Failure of actual follower convergence has the stated uniformly
negative, transverse-vanishing form. Source: arXiv:2411.04990v2, §4,
`thm2`. This necessary condition does not replace the missing basin-nullity
step with a stronger initialization assumption. -/
theorem csaLast_negative_leading_of_not_tendsto (β : ℝ) (Q K V : ParamMatrix d)
    (L : Submodule ℝ (EucSpace d)) (lam : ℝ) (hlam : 0 < lam)
    (heig : ∀ v ∈ L, V v = lam • v)
    (hinv : ∀ z ∈ Lᗮ, V z ∈ Lᗮ)
    (hdecay : ∀ z ∈ Lᗮ, z ≠ 0 → inner ℝ (V z) z < lam * ‖z‖ ^ 2)
    (u : SSphere d) (hu : (u : EucSpace d) ∈ L)
    (X : ℝ → SphereTuple d (n + 1)) (hX : CSA d (n + 1) β Q K V X) (hn : 1 ≤ n)
    (hp : ∀ j : Idx n, Tendsto (fun t => (X t j.castSucc : EucSpace d)) atTop
      (nhds (u : EucSpace d)))
    (hbad : ¬Tendsto (fun t => (X t (Fin.last n) : EucSpace d)) atTop
      (nhds (u : EucSpace d))) :
    ∃ δ : ℝ, 0 < δ ∧
      (∀ᶠ t in atTop, inner ℝ (u : EucSpace d) (X t (Fin.last n) : EucSpace d) ≤ -δ) ∧
      ∀ v ∈ L, inner ℝ v u = 0 →
        Tendsto (fun t => inner ℝ v (X t (Fin.last n) : EucSpace d)) atTop (nhds 0) :=
  (csaLast_tendsto_or_negative_leading β Q K V L lam hlam heig hinv hdecay
    u hu X hX hn hp).resolve_left hbad

/-- For any number of original predecessors converging to `u`, the last
token either converges to `u` or its leading projection stays away from
zero and its normalized direction converges to `-u`. The full token
need not have a limit in the exceptional alternative.
Source: arXiv:2411.04990v2, §4, a count-unrestricted angular dichotomy
for `thm2`; it leaves the exceptional basin's nullity open. -/
theorem csaLast_tendsto_or_negative_direction (β : ℝ) (Q K V : ParamMatrix d)
    (L : Submodule ℝ (EucSpace d)) (lam : ℝ) (hlam : 0 < lam)
    (heig : ∀ v ∈ L, V v = lam • v)
    (hinv : ∀ z ∈ Lᗮ, V z ∈ Lᗮ)
    (hdecay : ∀ z ∈ Lᗮ, z ≠ 0 → inner ℝ (V z) z < lam * ‖z‖ ^ 2)
    (u : SSphere d) (hu : (u : EucSpace d) ∈ L)
    (X : ℝ → SphereTuple d (n + 1)) (hX : CSA d (n + 1) β Q K V X) (hn : 1 ≤ n)
    (hp : ∀ j : Idx n, Tendsto (fun t => (X t j.castSucc : EucSpace d)) atTop
      (nhds (u : EucSpace d))) :
    Tendsto (fun t => (X t (Fin.last n) : EucSpace d)) atTop (nhds (u : EucSpace d)) ∨
      ∃ δ : ℝ, 0 < δ ∧
        (∀ᶠ t in atTop, inner ℝ (u : EucSpace d) (X t (Fin.last n) : EucSpace d) ≤ -δ ∧
          δ ≤ ‖L.starProjection (X t (Fin.last n) : EucSpace d)‖) ∧
        Tendsto (fun t => ‖L.starProjection (X t (Fin.last n) : EucSpace d)‖⁻¹ •
          L.starProjection (X t (Fin.last n) : EucSpace d)) atTop (nhds (-(u : EucSpace d))) := by
  rcases csaLast_tendsto_or_negative_leading β Q K V L lam hlam heig hinv hdecay
    u hu X hX hn hp with hgood | ⟨δ, hδ, hneg, htransverse⟩
  · exact Or.inl hgood
  · right
    refine ⟨δ, hδ, hneg.mono (fun t ht => ⟨ht,
      norm_starProjection_ge_of_negative L u hu _ δ ht⟩), ?_⟩
    exact tendsto_normalized_projection_neg L u (fun t => (X t (Fin.last n) : EucSpace d))
      (tendsto_leadingOffAxis L u hu _ htransverse) δ hδ hneg

/-- The actual stationary antipodal pair satisfies all original spectral
and prefix hypotheses and fails to converge to its leader. It realizes
the exceptional alternative, so it cannot be deleted pointwise.
Source: arXiv:2411.04990v2, §4, `thm2`, an exceptional initialization. -/
example : ∃ (X : ℝ → SphereTuple 2 2) (u : SSphere 2),
    (0 : ℝ) < 1 ∧ (1 : ℕ) ≤ 1 ∧
    (∀ v ∈ (⊤ : Submodule ℝ (EucSpace 2)),
      (ContinuousLinearMap.id ℝ (EucSpace 2)) v = (1 : ℝ) • v) ∧
    (∀ z ∈ (⊤ : Submodule ℝ (EucSpace 2))ᗮ,
      (ContinuousLinearMap.id ℝ (EucSpace 2)) z ∈ (⊤ : Submodule ℝ (EucSpace 2))ᗮ) ∧
    (∀ z ∈ (⊤ : Submodule ℝ (EucSpace 2))ᗮ, z ≠ 0 →
      inner ℝ ((ContinuousLinearMap.id ℝ (EucSpace 2)) z) z < (1 : ℝ) * ‖z‖ ^ 2) ∧
    (u : EucSpace 2) ∈ (⊤ : Submodule ℝ (EucSpace 2)) ∧
    CSA 2 2 1 (ContinuousLinearMap.id ℝ _) (ContinuousLinearMap.id ℝ _)
      (ContinuousLinearMap.id ℝ _) X ∧
    (∀ j : Idx 1, Tendsto (fun t => (X t j.castSucc : EucSpace 2)) atTop
      (nhds (u : EucSpace 2))) ∧
    ¬Tendsto (fun t => (X t (Fin.last 1) : EucSpace 2)) atTop (nhds (u : EucSpace 2)) := by
  let u : SSphere 2 := basePoint 1
  refine ⟨fun _ => antipodalCircleTuple u, u, one_pos, le_rfl,
    fun v _ => by simp, fun z hz => hz, ?_, Submodule.mem_top,
    csa_const_antipodal 1 u, ?_, ?_⟩
  · intro z hz hz0
    exact (hz0 (by
      simpa only [Submodule.top_orthogonal_eq_bot, Submodule.mem_bot] using hz)).elim
  · intro j
    have hj : j = 0 := by ext; omega
    simpa only [hj, Fin.castSucc_zero, antipodalCircleTuple, Matrix.cons_val_zero] using
      (tendsto_const_nhds : Tendsto (fun _ : ℝ => (u : EucSpace 2)) atTop (nhds (u : EucSpace 2)))
  · intro h
    have h' : Tendsto (fun _ : ℝ => -(u : EucSpace 2)) atTop (nhds (u : EucSpace 2)) := by
      simpa only [antipodalCircleTuple, show Fin.last 1 = (1 : Fin 2) from rfl, Matrix.cons_val_one,
        Matrix.cons_val_zero] using h
    have he : -(u : EucSpace 2) = u := tendsto_nhds_unique tendsto_const_nhds h'
    have hi := congrArg (fun x : EucSpace 2 => inner ℝ (u : EucSpace 2) x) he
    norm_num [u, basePoint] at hi

end Transformer.Causal
