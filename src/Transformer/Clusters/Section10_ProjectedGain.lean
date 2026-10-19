module

public import Transformer.Clusters.Section10_CorrectedEnergy
public import Transformer.Clusters.Section8_GainLimit

/-!
# The maximal leading attention gain vanishes

ArXiv:2305.05465v6, §10, Step 2' and the adaptation of Steps 3--4
from §8.1 in the proof of `t:multiplicity`.
The leading projected positions are bounded by their initial hull, and
their actual consensus velocities are bounded by `2 λ R`. This makes
the projected tuple Lipschitz on future times and its maximal leading
attention gain uniformly continuous there.

The convergent corrected energy dominates `2 λ` times this nonnegative
gain. The proved derivative-domination principle forces the gain to
vanish, without assuming the source's quantitative candidate-distance
claim or the false projected-root remainder estimate.
-/

@[expose] public section

open scoped BigOperators
open Real Filter Topology

namespace Transformer.Clusters

variable {d n : ℕ}

/-- A positive-speed attention consensus drift in a radius-`R` ball
has norm at most `2 λ R`. Source: arXiv:2305.05465v6, §10, the actual
projected ODE and the bounded-speed observation used in Steps 3--4. -/
theorem norm_attentionConsensusDrift_le (Q K : ParamMatrix d)
    (X Y : Idx n → EucSpace d) (lam R : ℝ) (hlam : 0 ≤ lam)
    (hY : ∀ j, ‖Y j‖ ≤ R) (i : Idx n) :
    ‖∑ j : Idx n, attentionMatrix Q K X i j • (lam • (Y j - Y i))‖ ≤ 2 * lam * R := by
  apply norm_sum_attentionMatrix_smul_le Q K X i
  intro j
  rw [norm_smul, Real.norm_of_nonneg hlam]
  have hsub : ‖Y j - Y i‖ ≤ 2 * R :=
    (norm_sub_le (Y j) (Y i)).trans (by linarith [hY i, hY j])
  have h := mul_le_mul_of_nonneg_left hsub hlam
  nlinarith

/-- The scalar and complete position-bound hypotheses hold for two
origin tokens at unit speed and radius.
Source: arXiv:2305.05465v6, §10, the projected ODE. -/
example : (0 : ℝ) ≤ 1 ∧ ∀ j : Idx 2,
    ‖(fun _ : Idx 2 => (0 : EucSpace 1)) j‖ ≤ (1 : ℝ) :=
  ⟨zero_le_one, fun _ => by simp only [norm_zero, zero_le_one]⟩

/-- The actual projected ODE gives the displacement estimate on every
future interval. Source: arXiv:2305.05465v6, §10, Steps 3--4, using the
projected consensus drift rather than a bound on the full configuration. -/
theorem RescaledDynamics.norm_sub_projected_le (Q K V proj : ParamMatrix d)
    (F G : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (hV : IsParanormalOn V F G lam) (hproj : IsProjOnto proj F G)
    (Z : ℝ → Idx n → EucSpace d) (hZ : RescaledDynamics Q K V Z) (R : ℝ)
    (hR : ∀ (i : Idx n) (t : ℝ), 0 ≤ t → ‖proj (Z t i)‖ ≤ R) (i : Idx n)
    {t s : ℝ} (ht : 0 ≤ t) (hts : t ≤ s) :
    ‖proj (Z s i) - proj (Z t i)‖ ≤ (2 * lam * R) * (s - t) :=
  norm_image_sub_le_of_norm_deriv_le_segment' (f := fun u => proj (Z u i))
    (fun u _ => (hZ.hasDerivAt_projected Q K V proj F G lam hV hproj Z u i).hasDerivWithinAt)
    (fun u hu => norm_attentionConsensusDrift_le Q K (fun j => expTime V u (Z u j))
      (fun j => proj (Z u j)) lam R hV.2.2.2.1.le
      (fun j => hR j u (ht.trans hu.1)) i) s ⟨hts, le_rfl⟩

/-- All ODE, splitting, projection, radius, and interval hypotheses hold
for an actual stationary origin configuration on `[0,1]`.
Source: arXiv:2305.05465v6, §3, `e:Rres`, and §10. -/
example : IsParanormalOn (1 : ParamMatrix 1) ⊤ ⊥ 1 ∧
    IsProjOnto (1 : ParamMatrix 1) ⊤ ⊥ ∧
    RescaledDynamics (1 : ParamMatrix 1) 1 1 (fun _ (_ : Idx 2) => 0) ∧
    (∀ (i : Idx 2) (t : ℝ), 0 ≤ t →
      ‖(1 : ParamMatrix 1) ((fun _ (_ : Idx 2) => (0 : EucSpace 1)) t i)‖ ≤ 1) ∧
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 :=
  ⟨(isGoodTripleMulti_one 1).2, isProjOnto_one 1, rescaledDynamics_one_const 1 1 _,
    fun _ _ _ => by simp only [map_zero, norm_zero, zero_le_one], le_rfl, zero_le_one⟩

/-- The complete leading tuple is Lipschitz on future times in its
product norm. Source: arXiv:2305.05465v6, §10, Steps 3--4. -/
theorem RescaledDynamics.lipschitzOn_projected_tuple (Q K V proj : ParamMatrix d)
    (F G : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (hV : IsParanormalOn V F G lam) (hproj : IsProjOnto proj F G)
    (Z : ℝ → Idx n → EucSpace d) (hZ : RescaledDynamics Q K V Z)
    (R : ℝ) (hR0 : 0 ≤ R)
    (hR : ∀ (i : Idx n) (t : ℝ), 0 ≤ t → ‖proj (Z t i)‖ ≤ R) :
    LipschitzOnWith ⟨2 * lam * R,
      mul_nonneg (mul_nonneg (by norm_num) hV.2.2.2.1.le) hR0⟩
      (fun t i => proj (Z t i)) (Set.Ici 0) := by
  have hlam : 0 < lam := hV.2.2.2.1
  apply lipschitzOnWith_iff_norm_sub_le.mpr
  intro t ht s hs
  change ‖(fun i => proj (Z t i)) - (fun i => proj (Z s i))‖ ≤ (2 * lam * R) * ‖t - s‖
  apply (pi_norm_le_iff_of_nonneg (mul_nonneg (by positivity) (norm_nonneg _))).mpr
  intro i
  change ‖proj (Z t i) - proj (Z s i)‖ ≤ (2 * lam * R) * ‖t - s‖
  rcases le_or_gt s t with hst | hst
  · rw [Real.norm_of_nonneg (sub_nonneg.mpr hst)]
    exact hZ.norm_sub_projected_le Q K V proj F G lam hV hproj Z R hR i hs hst
  · rw [norm_sub_rev, Real.norm_eq_abs, abs_of_nonpos (sub_nonpos.mpr hst.le)]
    have h := hZ.norm_sub_projected_le Q K V proj F G lam hV hproj Z R hR i ht hst.le
    linarith

/-- The full Lipschitz inputs coexist for stationary origin tokens in
a unit ball. Source: arXiv:2305.05465v6, §10, Steps 3--4. -/
example : IsParanormalOn (1 : ParamMatrix 1) ⊤ ⊥ 1 ∧
    IsProjOnto (1 : ParamMatrix 1) ⊤ ⊥ ∧
    RescaledDynamics (1 : ParamMatrix 1) 1 1 (fun _ (_ : Idx 2) => 0) ∧
    (0 : ℝ) ≤ 1 ∧ ∀ (i : Idx 2) (t : ℝ), 0 ≤ t →
      ‖(1 : ParamMatrix 1) ((fun _ (_ : Idx 2) => (0 : EucSpace 1)) t i)‖ ≤ 1 :=
  ⟨(isGoodTripleMulti_one 1).2, isProjOnto_one 1, rescaledDynamics_one_const 1 1 _,
    zero_le_one, fun _ _ _ => by simp only [map_zero, norm_zero, zero_le_one]⟩

/-- The maximal leading gain is uniformly continuous along the actual
future projected dynamics. Source: arXiv:2305.05465v6, §10, Steps 2'--4,
using compact confinement of the projected tuple only. -/
theorem RescaledDynamics.uniformContinuousOn_projected_attentionGain
    (Q K V proj A : ParamMatrix d) (F G : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (hV : IsParanormalOn V F G lam) (hproj : IsProjOnto proj F G)
    (Z : ℝ → Idx n → EucSpace d) (hZ : RescaledDynamics Q K V Z) (i : Idx n) :
    UniformContinuousOn (fun t => attentionGain A (fun j => proj (Z t j)) i) (Set.Ici 0) := by
  obtain ⟨R, hR, hbound⟩ := hZ.exists_bound_projected Q K V proj F G lam hV hproj Z
  have hcompact : IsCompact {Y : Idx n → EucSpace d |
      ∀ j, Y j ∈ tokenHull (fun k => proj (Z 0 k))} :=
    isCompact_pi_infinite fun _ => isCompact_tokenHull _
  exact (hcompact.uniformContinuousOn_of_continuous (continuous_attentionGain A i).continuousOn).comp
    (hZ.lipschitzOn_projected_tuple Q K V proj F G lam hV hproj Z R hR.le hbound).uniformContinuousOn
    (fun t ht j => hZ.mem_projected_tokenHull_zero Q K V proj F G lam hV hproj Z j ht)

/-- The uniform-continuity inputs hold for an actual nonzero stationary
rescaled configuration. Source: arXiv:2305.05465v6, §3, `e:Rres`, and §10. -/
example : IsParanormalOn (1 : ParamMatrix 1) ⊤ ⊥ 1 ∧
    IsProjOnto (1 : ParamMatrix 1) ⊤ ⊥ ∧
    RescaledDynamics (1 : ParamMatrix 1) 1 1
      (fun _ (_ : Idx 2) => (basePoint 0 : EucSpace 1)) :=
  ⟨(isGoodTripleMulti_one 1).2, isProjOnto_one 1, rescaledDynamics_one_const 1 1 _⟩

/-- **The actual maximal leading attention gain tends to zero.**
Source: arXiv:2305.05465v6, §10, Step 2' and the subsequent energy argument.
Its nonnegativity and uniform continuity, together with a convergent
energy dominating it, rule out late persistent positive gaps. -/
theorem RescaledDynamics.tendsto_projected_attentionGain_zero
    (Q K V proj A : ParamMatrix d) (F G : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (hV : IsParanormalOn V F G lam) (hproj : IsProjOnto proj F G)
    (hA : IsAttentionRoot Q K A) (Z : ℝ → Idx n → EucSpace d)
    (hZ : RescaledDynamics Q K V Z) (i : Idx n) :
    Tendsto (fun t => attentionGain A (fun j => proj (Z t j)) i) atTop (nhds 0) := by
  obtain ⟨δ, D, _, _, hdiff, hlim, hbound⟩ :=
    hZ.exists_convergent_corrected_projected_energy Q K V proj A F G lam hV hproj hA Z
  obtain ⟨L, hL⟩ := hlim i
  apply tendsto_zero_of_deriv_domination
    (fun t => correctedProjectedEnergy A proj Z lam δ D i t / (2 * lam))
    (fun t => deriv (correctedProjectedEnergy A proj Z lam δ D i) t / (2 * lam))
    (fun t => attentionGain A (fun j => proj (Z t j)) i) (L / (2 * lam))
    (fun t => ((hdiff i t).hasDerivAt).div_const (2 * lam))
  · intro t ht
    unfold attentionGain
    apply (Finset.sup'_le_iff _ _).mpr
    intro j _
    apply (le_div_iff₀ (mul_pos (show (0 : ℝ) < 2 by norm_num) hV.2.2.2.1)).mpr
    simpa only [mul_comm] using hbound t ht i j
  · exact fun t _ => attentionGain_nonneg A (fun j => proj (Z t j)) i
  · exact hZ.uniformContinuousOn_projected_attentionGain Q K V proj A F G lam hV hproj Z i
  · exact hL.div_const (2 * lam)

/-- The root, complete splitting, projection, and coupled ODE coexist
for actual stationary tokens. Source: arXiv:2305.05465v6, §10, Step 2'. -/
example : IsParanormalOn (1 : ParamMatrix 1) ⊤ ⊥ 1 ∧
    IsProjOnto (1 : ParamMatrix 1) ⊤ ⊥ ∧ IsAttentionRoot (1 : ParamMatrix 1) 1 1 ∧
    RescaledDynamics (1 : ParamMatrix 1) 1 1
      (fun _ (_ : Idx 2) => (basePoint 0 : EucSpace 1)) :=
  ⟨(isGoodTripleMulti_one 1).2, isProjOnto_one 1, isAttentionRoot_id 1,
    rescaledDynamics_one_const 1 1 _⟩

end Transformer.Clusters
