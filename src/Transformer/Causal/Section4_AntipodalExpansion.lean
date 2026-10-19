module

public import Transformer.Causal.Section4_AntipodalField
public import Transformer.Causal.Section4_FollowerLimits
public import Transformer.Causal.Section5_StrongCenterWitness

/-!
# A fixed original prefix has at most one antipodal follower trajectory

ArXiv:2411.04990v2, §4, proof of `thm1`, antipodal instability.
The proved original field estimate is strictly expanding in a fixed
explicit neighborhood of the antipodal configuration. Its radius
uses only the original positive weight floor and the proved sum of
original weight sensitivities. Every original converging prefix and
two followers tending to its antipode eventually enter this same
neighborhood.

The actual squared follower distance then has a persistent positive
relative derivative while remaining bounded by four. The scalar
expansion lemma forces that distance to vanish. The original shared
prefix gives equality of the entire original state at a finite time;
uniqueness of the original smooth causal flow, after a time shift,
gives equality of the full original trajectories.
-/

@[expose] public section

open scoped BigOperators
open Real Filter Topology

namespace Transformer.Causal

variable {d n : ℕ}

/-- An explicit actual neighborhood radius for the original conditional antipodal expansion.
Source: arXiv:2411.04990v2, §4, the positive antipodal spherical linearization in `thm1`. -/
noncomputable def csaAntipodalRadius (β : ℝ) (Q K : ParamMatrix d) (n : ℕ) : ℝ :=
  min (1 / 4) (csaWeightFloor β Q K (n + 1) / (8 * (csaPredecessorLipschitz β Q K n + 1)))

/-- The actual original expansion neighborhood has strictly positive radius.
Source: arXiv:2411.04990v2, §4, the local antipodal instability in `thm1`. -/
theorem csaAntipodalRadius_pos (β : ℝ) (Q K : ParamMatrix d) :
    0 < csaAntipodalRadius β Q K n := by
  have hf := csaWeightFloor_pos β Q K (Nat.zero_lt_succ n)
  have hL := csaPredecessorLipschitz_nonneg (n := n) β Q K
  apply lt_min_iff.mpr
  constructor <;> positivity

/-- The actual expansion radius stays within the geometric quarter-neighborhood.
Source: arXiv:2411.04990v2, §4, the neighborhood of the antipodal equilibrium in `thm1`. -/
theorem csaAntipodalRadius_le_quarter (β : ℝ) (Q K : ParamMatrix d) :
    csaAntipodalRadius β Q K n ≤ 1 / 4 := min_le_left _ _

/-- At the actual radius, all original weight-sensitivity errors consume at most one quarter of the original floor.
Source: arXiv:2411.04990v2, §4, continuity of the original antipodal expansion in `thm1`. -/
theorem csaAntipodalRadius_error (β : ℝ) (Q K : ParamMatrix d) :
    2 * csaAntipodalRadius β Q K n * csaPredecessorLipschitz β Q K n ≤ csaWeightFloor β Q K (n + 1) / 4 := by
  have hL := csaPredecessorLipschitz_nonneg (n := n) β Q K
  have hr := (csaAntipodalRadius_pos (n := n) β Q K).le
  have hden : 0 < 8 * (csaPredecessorLipschitz β Q K n + 1) := by positivity
  have h := (le_div_iff₀ hden).mp (min_le_right (1 / 4)
    (csaWeightFloor β Q K (n + 1) / (8 * (csaPredecessorLipschitz β Q K n + 1))))
  change csaAntipodalRadius β Q K n * (8 * (csaPredecessorLipschitz β Q K n + 1)) ≤ _ at h
  nlinarith

/-- Inside the actual original neighborhood, the actual conditional field uniformly expands squared follower distance.
Source: arXiv:2411.04990v2, §4, the positive spherical antipodal Jacobian in `thm1`. -/
theorem csaLastField_expands (β : ℝ) (Q K : ParamMatrix d)
    (Y Z : SphereTuple d (n + 1)) (a : SSphere d) (hn : 1 ≤ n)
    (hYZ : ∀ j : Idx n, Y j.castSucc = Z j.castSucc)
    (hp : ∀ j : Idx n, ‖(Y j.castSucc : EucSpace d) - (a : EucSpace d)‖ ≤ csaAntipodalRadius β Q K n)
    (hY : ‖(Y (Fin.last n) : EucSpace d) + (a : EucSpace d)‖ ≤ csaAntipodalRadius β Q K n)
    (hZ : ‖(Z (Fin.last n) : EucSpace d) + (a : EucSpace d)‖ ≤ csaAntipodalRadius β Q K n) :
    csaWeightFloor β Q K (n + 1) / 4 * ‖(Y (Fin.last n) : EucSpace d) - (Z (Fin.last n) : EucSpace d)‖ ^ 2 ≤
      inner ℝ ((Y (Fin.last n) : EucSpace d) - (Z (Fin.last n) : EucSpace d))
        (csaAmbientField β Q K (ContinuousLinearMap.id ℝ _) (fun j => (Y j : EucSpace d)) (Fin.last n) -
          csaAmbientField β Q K (ContinuousLinearMap.id ℝ _) (fun j => (Z j : EucSpace d)) (Fin.last n)) := by
  have h := csaLastField_separation_lower β Q K Y Z a hn (csaAntipodalRadius β Q K n)
    (csaAntipodalRadius_le_quarter β Q K) hYZ hp hY hZ
  refine le_trans (mul_le_mul_of_nonneg_right ?_ (sq_nonneg _)) h
  linarith [csaAntipodalRadius_error (n := n) β Q K]

/-- An actual follower tending to the original antipode has vanishing original antipodal distance.
Source: arXiv:2411.04990v2, §4, entering the antipodal neighborhood in `thm1`. -/
theorem csaNorm_add_tendsto_of_antipode (x : ℝ → SSphere d) (a : SSphere d)
    (h : Tendsto x atTop (𝓝 (Perspective.antipode d a))) :
    Tendsto (fun t => ‖(x t : EucSpace d) + (a : EucSpace d)‖) atTop (𝓝 0) := by
  have hc : Tendsto (fun t => (x t : EucSpace d)) atTop (𝓝 (-(a : EucSpace d))) := by
    simpa only [Function.comp_def, Perspective.antipode] using
      (continuous_subtype_val.tendsto (Perspective.antipode d a)).comp h
  simpa only [neg_add_cancel, norm_zero] using (hc.add_const (a : EucSpace d)).norm

/-- Two actual original trajectories with the same converging original prefix cannot both approach its antipode unless equal.
Source: arXiv:2411.04990v2, §4, the conditional antipodal instability step in `thm1`.
The original nonlinear field supplies the expansion; no center-stable manifold result is assumed. -/
theorem CSA_eq_of_shared_prefix_antipodal_limits (β : ℝ) (Q K : ParamMatrix d)
    (X Y : ℝ → SphereTuple d (n + 1))
    (hX : CSA d (n + 1) β Q K (ContinuousLinearMap.id ℝ _) X)
    (hY : CSA d (n + 1) β Q K (ContinuousLinearMap.id ℝ _) Y) (a : SSphere d) (hn : 1 ≤ n)
    (hp : ∀ t, ∀ j : Idx n, X t j.castSucc = Y t j.castSucc)
    (hlim : ∀ j : Idx n, Tendsto (fun t => X t j.castSucc) atTop (𝓝 a))
    (hx : Tendsto (fun t => X t (Fin.last n)) atTop (𝓝 (Perspective.antipode d a)))
    (hy : Tendsto (fun t => Y t (Fin.last n)) atTop (𝓝 (Perspective.antipode d a))) : X = Y := by
  have hrad := csaAntipodalRadius_pos (n := n) β Q K
  obtain ⟨T, hT⟩ := eventually_atTop.mp
    (((csaNorm_add_tendsto_of_antipode _ a hx).eventually_lt_const hrad).and
      (((csaNorm_add_tendsto_of_antipode _ a hy).eventually_lt_const hrad).and
        ((tendsto_csaPrefixError X a hlim).eventually_lt_const hrad)))
  let r (t : ℝ) := ‖(X t (Fin.last n) : EucSpace d) - (Y t (Fin.last n) : EucSpace d)‖ ^ 2
  let r' (t : ℝ) := 2 * inner ℝ ((X t (Fin.last n) : EucSpace d) - (Y t (Fin.last n) : EucSpace d))
    (csaAmbientField β Q K (ContinuousLinearMap.id ℝ _) (fun j => (X t j : EucSpace d)) (Fin.last n) -
      csaAmbientField β Q K (ContinuousLinearMap.id ℝ _) (fun j => (Y t j : EucSpace d)) (Fin.last n))
  have hd (t : ℝ) : HasDerivAt r (r' t) t := by
    have h := ((CSA_iff_csaAmbientField β Q K _ X).mp hX t (Fin.last n)).sub
      ((CSA_iff_csaAmbientField β Q K _ Y).mp hY t (Fin.last n))
    simpa! only [r, r', Pi.sub_apply] using h.norm_sq
  have hr (t : ℝ) : 0 ≤ r t ∧ r t ≤ 4 := by
    have ht : ‖(X t (Fin.last n) : EucSpace d) - (Y t (Fin.last n) : EucSpace d)‖ ≤ 2 := by
      simpa only [mem_sphere_zero_iff_norm.mp (X t (Fin.last n)).2,
        mem_sphere_zero_iff_norm.mp (Y t (Fin.last n)).2, one_add_one_eq_two] using
          norm_sub_le (X t (Fin.last n) : EucSpace d) (Y t (Fin.last n) : EucSpace d)
    refine ⟨sq_nonneg _, ?_⟩
    dsimp [r]
    nlinarith [norm_nonneg ((X t (Fin.last n) : EucSpace d) - (Y t (Fin.last n) : EucSpace d))]
  have he : ∀ t, T ≤ t → (csaWeightFloor β Q K (n + 1) / 2) * r t ≤ r' t := by
    intro t ht
    have hp' (j : Idx n) : ‖(X t j.castSucc : EucSpace d) - (a : EucSpace d)‖ ≤ csaAntipodalRadius β Q K n :=
      (Finset.single_le_sum (s := Finset.univ) (fun l hl => norm_nonneg
        ((X t l.castSucc : EucSpace d) - (a : EucSpace d))) (Finset.mem_univ j)).trans (hT t ht).2.2.le
    have h := csaLastField_expands β Q K (X t) (Y t) a hn (hp t) hp' (hT t ht).1.le (hT t ht).2.1.le
    dsimp [r, r']
    linarith
  have hzero := csaScalar_eq_zero_of_expansion r r' hd 4 (csaWeightFloor β Q K (n + 1) / 2) T hr
    (div_pos (csaWeightFloor_pos β Q K (Nat.zero_lt_succ n)) (by norm_num)) he T le_rfl
  have hlast : X T (Fin.last n) = Y T (Fin.last n) :=
    Subtype.ext (sub_eq_zero.mp (norm_eq_zero.mp (sq_eq_zero_iff.mp hzero)))
  have htime : X T = Y T := funext (fun i => Fin.lastCases hlast (fun j => hp T j) i)
  have hs := CSA_unique β Q K (ContinuousLinearMap.id ℝ _) (fun t => X (t + T)) (fun t => Y (t + T))
    (CSA_time_shift β Q K _ X hX T) (CSA_time_shift β Q K _ Y hY T) (by simpa only [zero_add] using htime)
  funext t
  simpa only [sub_add_cancel] using congrFun hs (t - T)

/-- A genuine original antipodal equilibrium jointly realizes both actual solutions and all convergence premises.
Source: arXiv:2411.04990v2, §4, the original antipodal equilibrium in `thm1`. -/
example (a : SSphere 2) :
    CSA 2 2 1 (ContinuousLinearMap.id ℝ _) (ContinuousLinearMap.id ℝ _)
      (ContinuousLinearMap.id ℝ _) (fun _ => antipodalCircleTuple a) ∧ (1 : ℕ) ≤ 1 ∧
    (∀ t : ℝ, ∀ j : Idx 1, (fun _ : ℝ => antipodalCircleTuple a) t j.castSucc =
      (fun _ : ℝ => antipodalCircleTuple a) t j.castSucc) ∧
    (∀ j : Idx 1, Tendsto (fun _ : ℝ => antipodalCircleTuple a j.castSucc) atTop (𝓝 a)) ∧
    Tendsto (fun _ : ℝ => antipodalCircleTuple a (Fin.last 1)) atTop (𝓝 (Perspective.antipode 2 a)) := by
  refine ⟨csa_const_antipodal 1 a, le_rfl, fun _ _ => rfl, ?_, ?_⟩
  · intro j
    have hj : j = 0 := Subsingleton.elim j 0
    subst j
    exact tendsto_const_nhds
  · exact tendsto_const_nhds

/-- The actual original antipodal configuration also lies in every proved original expansion neighborhood.
Source: arXiv:2411.04990v2, §4, the antipodal equilibrium in `thm1`. -/
example (β : ℝ) (Q K : ParamMatrix 2) (a : SSphere 2) :
    let Y : SphereTuple 2 2 := Fin.snoc (fun _ : Idx 1 => a) (Perspective.antipode 2 a)
    (∀ j : Idx 1, ‖(Y j.castSucc : EucSpace 2) - (a : EucSpace 2)‖ ≤ csaAntipodalRadius β Q K 1) ∧
      ‖(Y (Fin.last 1) : EucSpace 2) + (a : EucSpace 2)‖ ≤ csaAntipodalRadius β Q K 1 ∧
        (∀ j : Idx 1, Y j.castSucc = Y j.castSucc) := by
  dsimp only
  refine ⟨?_, ?_, fun _ => rfl⟩
  · intro j
    simpa only [Fin.snoc_castSucc, sub_self, norm_zero] using (csaAntipodalRadius_pos (n := 1) β Q K).le
  · simpa only [Fin.snoc_last, Perspective.antipode, neg_add_cancel, norm_zero] using
      (csaAntipodalRadius_pos (n := 1) β Q K).le

end Transformer.Causal
