module

public import Transformer.Causal.Section4_AntipodalFibers
public import Transformer.Causal.Section4_ConditionalMeasure

/-!
# Almost-everywhere consensus for the actual original causal system

ArXiv:2411.04990v2, §4, Theorem `thm1` and its proof.
The induction is on the actual original number of tokens. The
one-token leader is fixed. For a good original prefix, the actual
measurable antipodal last-token fiber has at most one initial point.
The original invariant spherical marginal has zero singleton masses
in dimension at least two, so Fubini makes this exceptional event
null under the original independent initialization.

The scalar dichotomy then makes the actual last token converge to
the original first initial token. Prefix consistency supplies the
same limit for every original preceding token. The induction proves
actual real-time convergence of the entire original canonical flow.
Original flow uniqueness gives the original quantifier over every
actual CSA solution, with arbitrary original query and key matrices.

The estimates require no sign condition on the temperature. This
stronger all-real-temperature helper implies the manuscript's
nonnegative-temperature statement without altering its dynamics,
initial law, or conclusion. No unstable-manifold result is assumed.
-/

@[expose] public section

open MeasureTheory Real Filter Topology

namespace Transformer.Causal

variable {d n : ℕ}

/-- Almost-sure original prefix consensus extends to the actual next original follower under the original atomless product law.
Source: arXiv:2411.04990v2, §4, the induction step of `thm1`. -/
theorem ae_csaFlow_consensus_step (β : ℝ) (Q K : ParamMatrix d)
    (σ : Measure (SSphere d)) [IsProbabilityMeasure σ] (hσ : ∀ x, σ {x} = 0) (hn : 1 ≤ n)
    (hgood : ∀ᵐ Y ∂Measure.pi (fun _ : Idx n => σ), ∀ j : Idx n,
      Tendsto (fun t => csaFlow β Q K (ContinuousLinearMap.id ℝ _) Y t j) atTop (𝓝 (Y ⟨0, hn⟩))) :
    ∀ᵐ X₀ ∂Measure.pi (fun _ : Idx (n + 1) => σ), ∀ k,
      Tendsto (fun t => csaFlow β Q K (ContinuousLinearMap.id ℝ _) X₀ t k) atTop (𝓝 (X₀ 0)) := by
  let B : Set (SphereTuple d (n + 1)) := {X₀ | Tendsto
    (fun m : ℕ => csaFlow β Q K (ContinuousLinearMap.id ℝ _) X₀ (m : ℝ) (Fin.last n))
      atTop (𝓝 (Perspective.antipode d (X₀ 0)))}
  have hB : MeasurableSet B := measurableSet_csaAntipodalInitial β Q K
  have hf : ∀ᵐ Y ∂Measure.pi (fun _ : Idx n => σ), {z : SSphere d | Fin.snoc Y z ∈ B}.Subsingleton := by
    filter_upwards [hgood] with Y hY
    exact subsingleton_csaSnoc_antipodal_nat_fiber β Q K Y hn hY
  have hbad := csa_ae_not_mem_of_subsingleton_snoc_fibers σ hσ B hB hf
  have hprefix := csa_ae_prefix_of_ae σ _ hgood
  filter_upwards [hbad, hprefix] with X₀ hnot hpre
  have hp : ∀ j : Idx n, Tendsto
      (fun t => csaFlow β Q K (ContinuousLinearMap.id ℝ _) X₀ t j.castSucc) atTop (𝓝 (X₀ 0)) := by
    intro j
    rw [show (fun t => csaFlow β Q K (ContinuousLinearMap.id ℝ _) X₀ t j.castSucc) =
      (fun t => csaFlow β Q K (ContinuousLinearMap.id ℝ _) (fun l => X₀ l.castSucc) t j) from
        funext (fun t => csaFlow_castSucc β Q K _ X₀ t j)]
    exact hpre j
  intro k
  refine Fin.lastCases ?_ (fun j => hp j) k
  exact csaFlow_last_tendsto_of_not_antipodal_nat β Q K X₀ hn hp hnot

/-- Every actual original nonempty finite causal flow converges almost surely under its original atomless product initialization.
Source: arXiv:2411.04990v2, §4, the full induction in the proof of `thm1`. -/
theorem ae_csaFlow_consensus_succ (β : ℝ) (Q K : ParamMatrix d)
    (σ : Measure (SSphere d)) [IsProbabilityMeasure σ] (hσ : ∀ x, σ {x} = 0) (n : ℕ) :
    ∀ᵐ X₀ ∂Measure.pi (fun _ : Idx (n + 1) => σ), ∀ k,
      Tendsto (fun t => csaFlow β Q K (ContinuousLinearMap.id ℝ _) X₀ t k) atTop (𝓝 (X₀ 0)) := by
  induction n with
  | zero =>
    apply ae_of_all
    intro X₀ k
    have hk : k = 0 := by apply Fin.ext; omega
    subst k
    simpa only [csaFlow_singleton_id] using
      (tendsto_const_nhds : Tendsto (fun _ : ℝ => X₀ 0) atTop (𝓝 (X₀ 0)))
  | succ n ih =>
    exact ae_csaFlow_consensus_step β Q K σ hσ (Nat.succ_le_succ (Nat.zero_le n)) ih

/-- The full original almost-sure causal consensus statement holds for every real temperature and every original solution.
Source: arXiv:2411.04990v2, §4, `thm1`.
This strengthens only the source's temperature range; all original matrices, dynamics, laws, and limits remain. -/
theorem single_cluster_all_real (hd : 2 ≤ d) (β : ℝ) (Q K : ParamMatrix d) (hn : 1 ≤ n)
    (P : Measure (SphereTuple d n)) (hP : Perspective.UniformTuple d n P) :
    ∀ᵐ X₀ ∂P, ∀ X : ℝ → SphereTuple d n, X 0 = X₀ →
      CSA d n β Q K (ContinuousLinearMap.id ℝ _) X → ∀ k,
        Tendsto (fun t => X t k) atTop (𝓝 (X₀ ⟨0, hn⟩)) := by
  obtain ⟨σ, hσprob, hσinv, rfl⟩ := hP
  have := hσprob
  have hσ : ∀ x : SSphere d, σ {x} = 0 :=
    Perspective.measure_singleton_eq_zero_of_invariant d hd σ hσinv
  cases n with
  | zero => omega
  | succ m =>
    have h := ae_csaFlow_consensus_succ β Q K σ hσ m
    filter_upwards [h] with X₀ hlim
    intro X h0 hX k
    have he := csaFlow_eq_of_CSA β Q K (ContinuousLinearMap.id ℝ _) X hX
    rw [h0] at he
    have hi : (⟨0, hn⟩ : Idx (m + 1)) = 0 := by apply Fin.ext; simp only [Fin.val_zero]
    rw [hi]
    simpa only [he] using hlim k

/-- The actual whole original tuple converges almost surely to the constant tuple at its original first token.
Source: arXiv:2411.04990v2, §4, the single-cluster conclusion of `thm1`. -/
theorem ae_csaFlow_tendsto_consensus (hd : 2 ≤ d) (β : ℝ) (Q K : ParamMatrix d) (hn : 1 ≤ n)
    (P : Measure (SphereTuple d n)) (hP : Perspective.UniformTuple d n P) :
    ∀ᵐ X₀ ∂P, Tendsto (csaFlow β Q K (ContinuousLinearMap.id ℝ _) X₀) atTop
      (𝓝 (fun _ : Idx n => X₀ ⟨0, hn⟩)) := by
  filter_upwards [single_cluster_all_real hd β Q K hn P hP] with X₀ h
  exact tendsto_pi_nhds.mpr (h _ (csaFlow_zero β Q K _ X₀) (csaFlow_spec β Q K _ X₀).2)

/-- The original measurable integer-time consensus event has probability one under the original uniform initialization.
Source: arXiv:2411.04990v2, §4, the probability-one convergence assertion in `thm1`.
The preceding theorem proves the stronger real-time convergence; integer times give this explicit Borel event. -/
theorem measure_csaFlow_consensus_nat_eq_one (hd : 2 ≤ d) (β : ℝ) (Q K : ParamMatrix d)
    (P : Measure (SphereTuple d (n + 1))) (hP : Perspective.UniformTuple d (n + 1) P) :
    P {X₀ | ∀ k, Tendsto (fun m : ℕ => csaFlow β Q K (ContinuousLinearMap.id ℝ _) X₀ (m : ℝ) k)
      atTop (𝓝 (X₀ 0))} = 1 := by
  have hprob : IsProbabilityMeasure P := by
    obtain ⟨σ, hσ, _, rfl⟩ := hP
    have := hσ
    infer_instance
  have ha := single_cluster_all_real hd β Q K (Nat.succ_le_succ (Nat.zero_le n)) P hP
  have hn : ∀ᵐ X₀ ∂P, ∀ k, Tendsto
      (fun m : ℕ => csaFlow β Q K (ContinuousLinearMap.id ℝ _) X₀ (m : ℝ) k) atTop (𝓝 (X₀ 0)) := by
    filter_upwards [ha] with X₀ h k
    exact (h _ (csaFlow_zero β Q K _ X₀) (csaFlow_spec β Q K _ X₀).2 k).comp tendsto_natCast_atTop_atTop
  exact (mem_ae_iff_prob_eq_one
    (measurableSet_csaFlow_consensus_nat β Q K (ContinuousLinearMap.id ℝ _) 0)).mp hn

/-- The actual invariant uniform circle law jointly realizes the dimension, size and original law hypotheses.
Source: arXiv:2411.04990v2, §4, the original independent uniform initialization in `thm1`. -/
example : 2 ≤ 2 ∧ (1 : ℕ) ≤ 2 ∧ Perspective.UniformTuple 2 2 anisotropicUniformLaw :=
  ⟨le_rfl, by norm_num, uniformTuple_anisotropicUniformLaw⟩

/-- The actual uniform marginal and the original fixed one-token flow jointly realize every induction-step premise.
Source: arXiv:2411.04990v2, §4, the original base case of `thm1`. -/
example (β : ℝ) (Q K : ParamMatrix 2) :
    IsProbabilityMeasure (MeanField.uniformLaw 2) ∧
      (∀ x : SSphere 2, MeanField.uniformLaw 2 {x} = 0) ∧ (1 : ℕ) ≤ 1 ∧
      (∀ᵐ Y ∂Measure.pi (fun _ : Idx 1 => MeanField.uniformLaw 2), ∀ j : Idx 1,
        Tendsto (fun t => csaFlow β Q K (ContinuousLinearMap.id ℝ _) Y t j) atTop (𝓝 (Y 0))) := by
  have h := Metastability.isUniformOn_uniformLaw (d := 2) (by norm_num)
  refine ⟨h.1, fun x => ?_, le_rfl, ?_⟩
  · have := h.1
    exact Perspective.measure_singleton_eq_zero_of_invariant 2 le_rfl _ h.2 x
  · apply ae_of_all
    intro Y j
    have hj : j = 0 := Subsingleton.elim j 0
    subst j
    simpa only [csaFlow_singleton_id] using
      (tendsto_const_nhds : Tendsto (fun _ : ℝ => Y 0) atTop (𝓝 (Y 0)))

end Transformer.Causal
