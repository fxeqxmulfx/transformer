module

public import Transformer.Landscape.GlobalFlow.TimeDependent

/-!
# The flow of a bounded time-dependent Lipschitz field

For a field `w : ℝ → E → E` continuous in `(t, x)`, `L`-Lipschitz in `x` and
bounded by `M` (`IsFlowField`), `flow w s t x` is the position at time `t` of
the trajectory through `x` at time `s`, which exists and is unique by
`exists_timeDependent_through`. This file proves

* the flow property `Φ_{s',t} ∘ Φ_{s,s'} = Φ_{s,t}` (`flow_flow`);
* the speed bound `‖Φ_{s,t} x - x‖ ≤ M |t - s|` (`norm_flow_sub_le`);
* the Lipschitz bound `e^{L |t - s|}` in `x`, forwards by Grönwall and
  backwards through the reversed field `-w(-t, x)` (`dist_flow_le`);
* continuity in the initial time and point together (`continuous_flow_left`);
* compact supports survive composition with `Φ_{s,t}`
  (`hasCompactSupport_comp_flow`).

These are the characteristics of the linear continuity equation, used in the
uniqueness proof of `c:wellposedtransformers` of arXiv:2305.05465v6, §6.
-/

@[expose] public section

open Set Metric Filter Topology
open scoped NNReal

namespace Transformer
namespace Landscape.GlobalFlow

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- `w` is continuous in `(t, x)`, `L`-Lipschitz in `x` at each time, and
bounded by `M`: the hypotheses under which its flow exists. -/
structure IsFlowField (w : ℝ → E → E) (L : ℝ≥0) (M : ℝ) : Prop where
  continuous : Continuous (Function.uncurry w)
  lipschitz : ∀ t, LipschitzWith L (w t)
  bound : ∀ t x, ‖w t x‖ ≤ M

/-- The flow of `w`: `flow w s t x` is the value at time `t` of the trajectory
of `w` through `x` at time `s`. -/
noncomputable def flow (w : ℝ → E → E) (s t : ℝ) (x : E) : E :=
  Classical.epsilon (fun γ : ℝ → E => γ s = x ∧ ∀ t, HasDerivAt γ (w t (γ t)) t) t

variable {w : ℝ → E → E} {L : ℝ≥0} {M : ℝ}

omit [NormedSpace ℝ E] in
theorem IsFlowField.nonneg (hw : IsFlowField w L M) : 0 ≤ M :=
  (norm_nonneg _).trans (hw.bound 0 0)

omit [NormedSpace ℝ E] in
theorem IsFlowField.continuous_time (hw : IsFlowField w L M) (x : E) :
    Continuous fun t => w t x :=
  hw.continuous.comp (continuous_id.prodMk continuous_const)

omit [NormedSpace ℝ E] in
/-- The reversed field `-w(-t, x)`, which runs the trajectories of `w`
backwards, satisfies the same hypotheses. -/
theorem IsFlowField.reverse (hw : IsFlowField w L M) : IsFlowField (fun t x => -w (-t) x) L M where
  continuous := (hw.continuous.comp (continuous_fst.neg.prodMk continuous_snd)).neg
  lipschitz t := (hw.lipschitz (-t)).neg
  bound t x := by rw [norm_neg]; exact hw.bound _ _

/-- **Uniqueness**: two trajectories of `w` meeting at one time coincide. -/
theorem IsFlowField.eq_of_hasDerivAt (hw : IsFlowField w L M) {γ δ : ℝ → E} {s : ℝ}
    (hγ : ∀ t, HasDerivAt γ (w t (γ t)) t) (hδ : ∀ t, HasDerivAt δ (w t (δ t)) t)
    (hs : γ s = δ s) : γ = δ := by
  funext t
  exact timeDependent_eqOn_Ioo w L hw.lipschitz γ δ (s - |t - s| - 1) (s + |t - s| + 1) s
    ⟨by linarith [abs_nonneg (t - s)], by linarith [abs_nonneg (t - s)]⟩
    (fun t _ => hγ t) (fun t _ => hδ t) hs
    ⟨by linarith [neg_abs_le (t - s)], by linarith [le_abs_self (t - s)]⟩

variable [CompleteSpace E]

theorem IsFlowField.flow_spec (hw : IsFlowField w L M) (s : ℝ) (x : E) :
    flow w s s x = x ∧ ∀ t, HasDerivAt (fun t => flow w s t x) (w t (flow w s t x)) t :=
  Classical.epsilon_spec (p := fun γ : ℝ → E => γ s = x ∧ ∀ t, HasDerivAt γ (w t (γ t)) t)
    (exists_timeDependent_through w L M hw.nonneg hw.lipschitz hw.continuous_time hw.bound s x)

theorem IsFlowField.flow_self (hw : IsFlowField w L M) (s : ℝ) (x : E) : flow w s s x = x :=
  (hw.flow_spec s x).1

theorem IsFlowField.hasDerivAt_flow (hw : IsFlowField w L M) (s t : ℝ) (x : E) :
    HasDerivAt (fun t => flow w s t x) (w t (flow w s t x)) t :=
  (hw.flow_spec s x).2 t

/-- Every trajectory of `w` is a trajectory of the flow. -/
theorem IsFlowField.flow_eq (hw : IsFlowField w L M) {γ : ℝ → E} (hγ : ∀ t, HasDerivAt γ (w t (γ t)) t)
    (s t : ℝ) : flow w s t (γ s) = γ t :=
  congrFun (hw.eq_of_hasDerivAt (hw.hasDerivAt_flow s · (γ s)) hγ (hw.flow_self s _)) t

/-- **The flow property** `Φ_{s',t} ∘ Φ_{s,s'} = Φ_{s,t}`. -/
theorem IsFlowField.flow_flow (hw : IsFlowField w L M) (s s' t : ℝ) (x : E) :
    flow w s' t (flow w s s' x) = flow w s t x :=
  hw.flow_eq (γ := fun t => flow w s t x) (hw.hasDerivAt_flow s · x) s' t

theorem IsFlowField.continuous_flow_time (hw : IsFlowField w L M) (s : ℝ) (x : E) :
    Continuous fun t => flow w s t x :=
  continuous_iff_continuousAt.2 fun t => (hw.hasDerivAt_flow s t x).continuousAt

/-- **Trajectories move at speed at most `M`.** -/
theorem IsFlowField.norm_flow_sub_le (hw : IsFlowField w L M) (s t : ℝ) (x : E) :
    ‖flow w s t x - x‖ ≤ M * |t - s| := by
  have h := (convex_univ : Convex ℝ (univ : Set ℝ)).norm_image_sub_le_of_norm_deriv_le
    (f := fun t => flow w s t x) (C := M) (fun t _ => (hw.hasDerivAt_flow s t x).differentiableAt)
    (fun t _ => by rw [(hw.hasDerivAt_flow s t x).deriv]; exact hw.bound _ _)
    (mem_univ s) (mem_univ t)
  simpa [hw.flow_self, Real.norm_eq_abs] using h

/-- Trajectories from `B̄(0, R)` stay in `B̄(0, R + M |t - s|)`. -/
theorem IsFlowField.norm_flow_le (hw : IsFlowField w L M) {R : ℝ} {x : E} (hx : ‖x‖ ≤ R)
    (s t : ℝ) : ‖flow w s t x‖ ≤ R + M * |t - s| := by
  linarith [norm_le_norm_add_norm_sub' (flow w s t x) x, hw.norm_flow_sub_le s t x]

theorem IsFlowField.flow_reverse (hw : IsFlowField w L M) (s t : ℝ) (x : E) :
    flow (fun t x => -w (-t) x) (-s) (-t) x = flow w s t x := by
  have hγ : ∀ τ, HasDerivAt (fun τ => flow w s (-τ) x) (-w (-τ) (flow w s (-τ) x)) τ := by
    intro τ
    simpa [Function.comp_def] using (hw.hasDerivAt_flow s (-τ) x).scomp τ (hasDerivAt_neg τ)
  have := hw.reverse.flow_eq hγ (-s) (-t)
  simpa [hw.flow_self] using this

/-- **Lipschitz dependence on the initial point**, forwards in time (Grönwall). -/
theorem IsFlowField.dist_flow_le_of_le (hw : IsFlowField w L M) {s t : ℝ} (hst : s ≤ t)
    (x y : E) : dist (flow w s t x) (flow w s t y) ≤ dist x y * Real.exp (L * (t - s)) := by
  have hc : ∀ z, ContinuousOn (fun t => flow w s t z) (Icc s t) := fun z =>
    (hw.continuous_flow_time s z).continuousOn
  have hd : ∀ z, ∀ τ ∈ Ico s t,
      HasDerivWithinAt (fun t => flow w s t z) (w τ (flow w s τ z)) (Ici τ) τ := fun z τ _ =>
    (hw.hasDerivAt_flow s τ z).hasDerivWithinAt
  exact dist_le_of_trajectories_ODE hw.lipschitz (hc x) (hd x) (hc y) (hd y)
    (le_of_eq (by rw [hw.flow_self, hw.flow_self])) t ⟨hst, le_rfl⟩

/-- **Lipschitz dependence on the initial point**: `Φ_{s,t}` is
`e^{L |t - s|}`-Lipschitz, in both directions of time. -/
theorem IsFlowField.dist_flow_le (hw : IsFlowField w L M) (s t : ℝ) (x y : E) :
    dist (flow w s t x) (flow w s t y) ≤ dist x y * Real.exp (L * |t - s|) := by
  rcases le_total s t with h | h
  · rw [abs_of_nonneg (sub_nonneg.2 h)]
    exact hw.dist_flow_le_of_le h x y
  · have := hw.reverse.dist_flow_le_of_le (neg_le_neg h) x y
    rwa [hw.flow_reverse, hw.flow_reverse, show -t - -s = |t - s| by
      rw [abs_of_nonpos (sub_nonpos.2 h)]; ring] at this

theorem IsFlowField.continuous_flow (hw : IsFlowField w L M) (s t : ℝ) :
    Continuous (flow w s t) :=
  (LipschitzWith.of_dist_le_mul (K := (Real.exp (L * |t - s|)).toNNReal) fun x y => by
    rw [Real.coe_toNNReal _ (Real.exp_pos _).le, mul_comm]
    exact hw.dist_flow_le s t x y).continuous

/-- **Joint continuity** in the initial time and point. -/
theorem IsFlowField.continuous_flow_left (hw : IsFlowField w L M) (t : ℝ) :
    Continuous fun p : ℝ × E => flow w p.1 t p.2 := by
  refine continuous_iff_continuousAt.2 fun p₀ => ?_
  have h1 : Tendsto (fun p : ℝ × E => flow w p.1 p₀.1 p.2) (𝓝 p₀) (𝓝 p₀.2) := by
    refine tendsto_iff_dist_tendsto_zero.2 (squeeze_zero (fun _ => dist_nonneg) (fun p => ?_)
      (g := fun p : ℝ × E => M * |p₀.1 - p.1| + dist p.2 p₀.2) ?_)
    · calc dist (flow w p.1 p₀.1 p.2) p₀.2
          ≤ dist (flow w p.1 p₀.1 p.2) p.2 + dist p.2 p₀.2 := dist_triangle _ _ _
        _ ≤ _ := by rw [dist_eq_norm]; gcongr; exact hw.norm_flow_sub_le _ _ _
    · have : Continuous fun p : ℝ × E => M * |p₀.1 - p.1| + dist p.2 p₀.2 := by fun_prop
      simpa using this.tendsto p₀
  have h2 := ((hw.continuous_flow p₀.1 t).tendsto p₀.2).comp h1
  refine h2.congr fun p => ?_
  exact hw.flow_flow p.1 p₀.1 t p.2

/-- Composing with a flow map keeps the support compact: `g ∘ Φ_{s,t}`
vanishes off `Φ_{t,s}(tsupport g)`. -/
theorem IsFlowField.hasCompactSupport_comp_flow (hw : IsFlowField w L M)
    {F : Type*} [NormedAddCommGroup F] {g : E → F} (hg : HasCompactSupport g) (s t : ℝ) :
    HasCompactSupport (g ∘ flow w s t) := by
  refine HasCompactSupport.intro (hg.isCompact.image (hw.continuous_flow t s)) fun x hx => ?_
  by_contra h
  exact hx ⟨flow w s t x, subset_tsupport _ (Function.mem_support.2 h),
    by rw [hw.flow_flow, hw.flow_self]⟩

/-- With `δ = 0`, the Grönwall bound is linear in the error. -/
theorem gronwallBound_zero_mul (K a b x : ℝ) :
    gronwallBound 0 K (a * b) x = a * gronwallBound 0 K b x := by
  by_cases hK : K = 0
  · simp only [hK, gronwallBound_K0]; ring
  · simp only [gronwallBound_of_K_ne_0 hK]; ring

/-- The hypotheses of the results above are satisfiable: the field `sin x` is
`1`-Lipschitz and bounded by `1`, the constant `0` solves `ẋ = sin x` and lies
in `B̄(0, 0)`, `0 ≤ 1`, and the zero function has compact support. -/
example : IsFlowField (fun _ x : ℝ => Real.sin x) 1 1 ∧
    (∀ t, HasDerivAt (fun _ : ℝ => (0 : ℝ)) (Real.sin 0) t) ∧ ‖(0 : ℝ)‖ ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧
    HasCompactSupport (0 : ℝ → ℝ) :=
  ⟨⟨Real.continuous_sin.comp continuous_snd, fun _ => Real.lipschitzWith_sin,
      fun _ x => by simpa using Real.abs_sin_le_one x⟩,
    fun t => by simpa using hasDerivAt_const t (0 : ℝ), by simp, zero_le_one,
    HasCompactSupport.zero⟩

end Landscape.GlobalFlow
end Transformer
