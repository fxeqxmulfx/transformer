/-
# The emergence of clusters in self-attention dynamics — well-posedness of the
  ODEs

§6 of arXiv:2305.05465v6: `p:wellposedparticles` and
`p:wellposednessrescaledparticles`, global existence and uniqueness for
`eq:trans_dyn` and for its rescaled form `e:Rres`.

**What the source says and what is carried here.**

* Both propositions assert a *unique Lipschitz continuous* `t ↦ x(t)` defined
  on all of `ℝ`.  "Lipschitz continuous" cannot be read globally on `ℝ`: for
  `n = 1` the only attention weight is `1`, the dynamics is the linear
  `ẋ = Vx`, and its solution `x(t) = e^{tV}x⁰` grows exponentially, so it is
  not Lipschitz on `ℝ` for any `V` with a positive eigenvalue.  It is read
  here as Lipschitz on every compact interval, which is what the proof of
  `c:wellposedtransformers` delivers and what the uniqueness argument uses.
  This is the only deviation from the source.

* Uniqueness is asserted among curves with that regularity, so it is carried
  inside the `∃!`.

* The source postpones the proof of `p:wellposedparticles` to the
  well-posedness of the continuity equation.  It is proved here directly:
  the field of `eq:trans_dyn` is smooth and grows linearly
  (`Section6_Field.lean`), so `GlobalFlow.exists_global` and
  `GlobalFlow.eq_of_hasDerivAt` apply.

* `p:wellposednessrescaledparticles` is deduced, as in the source, through
  the change of variables `x_i = e^{tV}z_i`
  (`transformerDynamics_iff_rescaled`).

Source: arXiv:2305.05465v6, `p:wellposedparticles`,
`p:wellposednessrescaledparticles`.
-/

import Transformer.Clusters.Section6_Field

open scoped BigOperators NNReal
open Real

namespace Transformer
namespace Clusters

variable {d n : ℕ}

/-- **Lipschitz on every compact interval**, the regularity
`p:wellposedparticles` asks of its solution. -/
def IsLocLipschitzCurve (X : ℝ → Idx n → EucSpace d) : Prop :=
  ∀ (i : Idx n) (a b : ℝ), ∃ L : ℝ≥0, LipschitzOnWith L (fun t => X t i) (Set.Icc a b)

/-- A constant curve is Lipschitz on every interval, with constant `0`. -/
theorem isLocLipschitzCurve_const (X : Idx n → EucSpace d) :
    IsLocLipschitzCurve (fun _ => X) :=
  fun i _ _ => ⟨0, LipschitzOnWith.of_dist_le_mul fun _ _ _ _ => by simp⟩

/-- A curve whose tokens have continuous derivatives is Lipschitz on every
compact interval. -/
theorem isLocLipschitzCurve_of_hasDerivAt {X X' : ℝ → Idx n → EucSpace d}
    (hX : ∀ t i, HasDerivAt (fun s => X s i) (X' t i) t) (hX' : ∀ i, Continuous fun t => X' t i) :
    IsLocLipschitzCurve X :=
  fun i a b => lipschitzOnWith_Icc_of_hasDerivAt (fun t => hX t i) (hX' i) a b

/-- **Proposition (p:wellposedparticles).**  For any initial sequence of `n`
tokens there is exactly one curve, Lipschitz on compact intervals, solving
`eq:trans_dyn` from it, and it is defined for all time.

Source: arXiv:2305.05465v6, `p:wellposedparticles`. -/
theorem wellposed_particles (Q K V : ParamMatrix d) (X₀ : Idx n → EucSpace d) :
    ∃! X : ℝ → Idx n → EucSpace d,
      TransformerDynamics Q K V X ∧ X 0 = X₀ ∧ IsLocLipschitzCurve X := by
  have hF := (contDiff_transformerField (n := n) Q K V).locallyLipschitz
  obtain ⟨γ, hγ0, hγ⟩ := GlobalFlow.exists_global hF (norm_nonneg V)
    (norm_transformerField_le Q K V) X₀
  have hγc : Continuous γ := continuous_iff_continuousAt.2 fun t => (hγ t).continuousAt
  refine ⟨γ, ⟨(transformerDynamics_iff_hasDerivAt Q K V γ).2 hγ, hγ0,
    isLocLipschitzCurve_of_hasDerivAt (fun t i => (hasDerivAt_pi.1 (hγ t)) i) fun i =>
      (continuous_apply i).comp ((contDiff_transformerField Q K V).continuous.comp hγc)⟩, ?_⟩
  rintro Y ⟨hY, hY0, -⟩
  exact GlobalFlow.eq_of_hasDerivAt hF ((transformerDynamics_iff_hasDerivAt Q K V Y).1 hY) hγ
    (hY0.trans hγ0.symm)

/-- **Proposition (p:wellposednessrescaledparticles).**  The same for the
rescaled dynamics `e:Rres`, through `x_i(t) = e^{tV}z_i(t)`.

Source: arXiv:2305.05465v6, `p:wellposednessrescaledparticles`. -/
theorem wellposed_rescaled_particles (Q K V : ParamMatrix d) (Z₀ : Idx n → EucSpace d) :
    ∃! Z : ℝ → Idx n → EucSpace d,
      RescaledDynamics Q K V Z ∧ Z 0 = Z₀ ∧ IsLocLipschitzCurve Z := by
  obtain ⟨X, ⟨hX, hX0, -⟩, -⟩ := wellposed_particles Q K V Z₀
  have hinv : ∀ t y, expTime V (-t) (expTime V t y) = y := fun t y => by
    simpa using congrArg (fun M : ParamMatrix d => M y) (expTime_neg_mul V t)
  have hinv' : ∀ t y, expTime V t (expTime V (-t) y) = y := fun t y => by
    simpa using hinv (-t) y
  set Z : ℝ → Idx n → EucSpace d := fun t i => expTime V (-t) (X t i)
  have hXZ : ∀ t i, X t i = expTime V t (Z t i) := fun t i => (hinv' t _).symm
  have hZ := (transformerDynamics_iff_rescaled Q K V X Z hXZ).1 hX
  have hXc : Continuous X :=
    continuous_pi fun k => continuous_iff_continuousAt.2 fun t => (hX t k).continuousAt
  have hZc : ∀ j, Continuous fun t => Z t j := fun j =>
    continuous_iff_continuousAt.2 fun t => (hZ t j).continuousAt
  refine ⟨Z, ⟨hZ, by simp [Z, hX0, expTime_zero], isLocLipschitzCurve_of_hasDerivAt hZ fun i => ?_⟩, ?_⟩
  · have hX' : ∀ t, (fun l => expTime V t (Z t l)) = X t := fun t => funext fun l => (hXZ t l).symm
    simp only [hX']
    exact continuous_finsetSum _ fun j _ =>
      ((contDiff_attentionMatrix Q K i j).continuous.comp hXc).smul
        (V.continuous.comp ((hZc j).sub (hZc i)))
  rintro Y ⟨hY, hY0, -⟩
  set X' : ℝ → Idx n → EucSpace d := fun t i => expTime V t (Y t i)
  have hX' := (transformerDynamics_iff_rescaled Q K V X' Y fun _ _ => rfl).2 hY
  have hXX : X' = X := GlobalFlow.eq_of_hasDerivAt
    (contDiff_transformerField (n := n) Q K V).locallyLipschitz
    ((transformerDynamics_iff_hasDerivAt Q K V X').1 hX')
    ((transformerDynamics_iff_hasDerivAt Q K V X).1 hX) (by simp [X', hY0, hX0, expTime_zero])
  funext t i
  rw [show Z t i = expTime V (-t) (X' t i) by rw [hXX]]
  exact (hinv t _).symm

/-- The stationary curve of `transformerDynamics_const` has all three
properties at `V = 0`, so the conjunction inside the two `∃!` above is not
empty of witnesses. -/
example (Q K : ParamMatrix d) (X₀ : Idx n → EucSpace d) :
    TransformerDynamics Q K 0 (fun _ => X₀) ∧ (fun _ => X₀) 0 = X₀ ∧
      IsLocLipschitzCurve (fun _ : ℝ => X₀) :=
  ⟨transformerDynamics_const Q K X₀, rfl, isLocLipschitzCurve_const X₀⟩

/-- The same for the rescaled dynamics. -/
example (Q K : ParamMatrix d) (Z₀ : Idx n → EucSpace d) :
    RescaledDynamics Q K 0 (fun _ => Z₀) ∧ (fun _ => Z₀) 0 = Z₀ ∧
      IsLocLipschitzCurve (fun _ : ℝ => Z₀) :=
  ⟨rescaledDynamics_const Q K Z₀, rfl, isLocLipschitzCurve_const Z₀⟩

end Clusters
end Transformer
