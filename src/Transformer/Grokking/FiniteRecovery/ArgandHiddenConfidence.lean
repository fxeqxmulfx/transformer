module

public import Transformer.Grokking.FiniteRecovery.ObservationAnchor
public import Transformer.Grokking.FiniteRecovery.ArgandEnergyRecovery
public import Transformer.Grokking.FiniteRecovery.ArgandRecoveryObstruction
public import Transformer.Grokking.FiniteRecovery.RecoveryStability
public import Transformer.Grokking.Protocol.AbsentBatches
public import Transformer.Optimization.PAGE.AnchoredDecay

/-! A single PAGE run on the revealed cross-entropies of full Argand.
The class and observed graph determine the center of the quadratic penalty.
One confidence event, step size and penalty work for the whole finite class.
The proof gives no efficient method for finding the reference weights. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Transformer.Optimization.PAGE Filter Metric MeasureTheory Set
open scoped Topology ENNReal Classical

variable {A : Type*} [Fintype A] [Nonempty A] {H d L T : ℕ}

/-- A component reads precisely one revealed label at its final position. -/
noncomputable def graphComponent (r : ℕ) (part : Finset (Word A (T+1) × A))
    (i : part) : WeightSpace A H d L → ℝ :=
  fullGradientObjective r [⟨T,i.val.1,Fin.last T,i.val.2⟩] 0

/-- PAGE with constant step η and coupled penalty ‖w-c‖²/(2η).
The data and the reference are both functions of the same revealed graph. -/
noncomputable def observedAnchoredPAGE (r : ℕ) (family : Set (Word A (T+1) → A))
    (part : Finset (Word A (T+1) × A)) (η : ℝ) (start : Weights A H d L)
    (path : ℕ → Option part) (t : ℕ) : Weights A H d L :=
  let c := WithLp.toLp 2 (observationAnchor (H := H) (d := d) (L := L) r family part)
  WithLp.ofLp (run (anchoredLoss (graphComponent r part) c η⁻¹) η (WithLp.toLp 2 start) path t).weights

/-- All winners are strict after each update and at the limit of the
entire weight trajectory. This includes all hidden inputs simultaneously. -/
def StrictRecoveryAfterUpdate (r : ℕ) (f : Word A (T+1) → A) (w : ℕ → Weights A H d L) : Prop :=
  (∀ t, (Model.fromWeights (w (t+1))).realizes r f) ∧
    ∃ star, Tendsto w atTop (𝓝 star) ∧ (Model.fromWeights star).realizes r f

private theorem bounded_anchored_recovery (r : ℕ) (family : Set (Word A (T+1) → A))
    (S : Finset (Word A (T+1))) (hS : S.Nonempty) (hid : InjOn (observe S) family)
    (f : Word A (T+1) → A) (hf : f ∈ family)
    (hr : ∃ w : Weights A H d L, (Model.fromWeights w).realizes r f) (D : ℝ) :
    ∃ a > 0, ∀ η, 0 < η → η ≤ a → ∀ w : Weights A H d L,
      ‖(WithLp.toLp 2 w : WeightSpace A H d L)‖ ≤ D → ∀ path,
        StrictRecoveryAfterUpdate r f (observedAnchoredPAGE r family (observe S f) η w path) := by
  obtain ⟨x,hx⟩ := hS
  let _ : Nonempty (observe S f) := ⟨⟨(x,f x),mem_observe.mpr ⟨hx,rfl⟩⟩⟩
  let C := PiLp.continuousLinearEquiv 2 ℝ (fun _ : Parameter A H d L => ℝ)
  let c := C.symm (observationAnchor r family (observe S f))
  let G := {w : WeightSpace A H d L | (Model.fromWeights (C w)).realizes r f}
  have hG : IsOpen G := (isOpen_realizingWeights r f).preimage C.continuous
  have hc : c ∈ G := observationAnchor_realizes r family S hid hf hr
  obtain ⟨a,ha,fit⟩ := anchored_page_converges (graphComponent r (observe S f))
    (fun _ => (contDiff_fullGradientObjective _ _ _).differentiable one_ne_zero)
    (fun _ => locallyLipschitz_fullGradient _ _ _) hG hc (D+‖c‖)
  refine ⟨a,ha,fun η hη hsmall w hw path => ?_⟩
  have hy : C.symm w ∈ closedBall c (D+‖c‖) := by
    rw [mem_closedBall,dist_eq_norm]
    exact (norm_sub_le (C.symm w) c).trans (add_le_add hw le_rfl)
  obtain ⟨hw',star,hstar,ht,_⟩ := fit η hη hsmall _ hy path
  exact ⟨hw',C star,(C.continuous.tendsto star).comp ht,hstar⟩

/-- One event and step bound must serve every representable member of the
supplied class on the prescribed mask, even for different sampling paths. -/
def AnchoredRecoveryAt (mu : Measure (ComplexParameter A H d L → ℂ)) (q : ℝ≥0∞) (r : ℕ)
    (family : Set (Word A (T+1) → A)) (S : Finset (Word A (T+1))) : Prop :=
  ∃ U : Set (ComplexParameter A H d L → ℂ), MeasurableSet U ∧ q < mu U ∧
    ∃ a > 0, ∀ η, 0 < η → η ≤ a → ∀ f ∈ family,
      (∃ w : Weights A H d L, (Model.fromWeights w).realizes r f) → ∀ draw ∈ U, ∀ path,
        StrictRecoveryAfterUpdate r f
          (observedAnchoredPAGE r family (observe S f) η (sgdInitialWeights true draw) path)

/-- With a bounded initialization, a single fixed step and penalty give a
deterministic guarantee. The bound on the initial norm is independent of the
target and imposes no condition on the initial predictions. -/
theorem anchoredPAGE_bounded_initialization (r : ℕ) (family : Set (Word A (T+1) → A))
    (S : Finset (Word A (T+1))) (hS : S.Nonempty) (hid : InjOn (observe S) family) (D : ℝ) :
    ∃ a > 0, ∀ η, 0 < η → η ≤ a → ∀ f ∈ family,
      (∃ w : Weights A H d L, (Model.fromWeights w).realizes r f) → ∀ w : Weights A H d L,
        ‖(WithLp.toLp 2 w : WeightSpace A H d L)‖ ≤ D → ∀ path,
          StrictRecoveryAfterUpdate r f (observedAnchoredPAGE r family (observe S f) η w path) := by
  let V := {f ∈ family | ∃ w : Weights A H d L, (Model.fromWeights w).realizes r f}
  choose δ hδ fit using fun f : V => bounded_anchored_recovery r family S hS hid f f.2.1 f.2.2 D
  obtain ⟨a,ha,hle⟩ := exists_pos_le_all δ hδ
  exact ⟨a,ha,fun η hη hs f hf hr => fit ⟨f,hf,hr⟩ η hη (hs.trans (hle ⟨f,hf,hr⟩))⟩

/-- On one measurable event of native initializations of probability above
q, the same constant step and decay yield strict recovery for every strictly
representable target in the class and every sampler. The reference search
uses the entire supplied class, including its nonrepresentable members. -/
theorem anchoredPAGE_hidden_confidence (mu : Measure (ComplexParameter A H d L → ℂ))
    [IsProbabilityMeasure mu] {q : ℝ≥0∞} (hq : q < 1) (r : ℕ)
    (family : Set (Word A (T+1) → A)) (S : Finset (Word A (T+1))) (hS : S.Nonempty)
    (hid : InjOn (observe S) family) : AnchoredRecoveryAt mu q r family S := by
  obtain ⟨n,U,hU,hp,hb⟩ := exists_initial_norm_bound mu true hq
  obtain ⟨a,ha,fit⟩ := anchoredPAGE_bounded_initialization (H := H) (d := d) (L := L) r family S hS hid n
  exact ⟨U,hU,hp,a,ha,fun η hη hs f hf hr draw hd =>
    fit η hη hs f hf hr _ (hb draw hd)⟩

omit [Nonempty A] in
/-- Ordinary zero-centered L2 can erase every strict prediction. With
λ = 1/η large enough, the complete Argand PAGE weights converge to zero
above any prescribed confidence when full refreshes occur infinitely often.
This uses the actual cross-entropies of any nonempty revealed graph. -/
theorem zero_centeredPAGE_collapses (mu : Measure (ComplexParameter A H d L → ℂ))
    [IsProbabilityMeasure mu] {q : ℝ≥0∞} (hq : q < 1) (r : ℕ)
    (part : Finset (Word A (T+1) × A)) (hp : part.Nonempty) :
    ∃ U : Set (ComplexParameter A H d L → ℂ), MeasurableSet U ∧ q < mu U ∧
      ∃ a > 0, ∀ η, 0 < η → η ≤ a → ∀ draw ∈ U, ∀ path : ℕ → Option part,
        (∃ᶠ t in atTop, path t = none) →
        Tendsto (fun t => (run (anchoredLoss (graphComponent r part) 0 η⁻¹) η
          (WithLp.toLp 2 (sgdInitialWeights true draw)) path t).weights) atTop (𝓝 0) := by
  obtain ⟨p,hp⟩ := hp
  let _ : Nonempty part := ⟨⟨p,hp⟩⟩
  let loss := graphComponent (H := H) (d := d) (L := L) r part
  have hz (i : part) : gradient (loss i) 0 = 0 := by
    change gradient (fullGradientObjective r _ 0) (0 : WeightSpace A H d L) = 0
    rw [fullGradientObjective_gradient, WithLp.ofLp_zero, batchGradient_eq_zero_of_zeroEmbeddings]
    · simp
    · intro a j b; rfl
  have hzero : full loss 0 = 0 := by simp only [full,hz,mean_const]
  obtain ⟨n,U,hU,hprob,hbound⟩ := exists_initial_norm_bound mu true hq
  obtain ⟨a,ha,fit⟩ := anchored_page_converges loss
    (fun _ => (contDiff_fullGradientObjective _ _ _).differentiable one_ne_zero)
    (fun _ => locallyLipschitz_fullGradient _ _ _) isOpen_univ (mem_univ (0 : WeightSpace A H d L)) n
  refine ⟨U,hU,hprob,a,ha,fun η hη hb draw hd path hfreq => ?_⟩
  obtain ⟨_,star,_,ht,hfixed⟩ := fit η hη hb _ (mem_closedBall_zero_iff.mpr (hbound draw hd)) path
  simpa only [(hfixed hfreq).2 hzero] using ht

omit [Nonempty A] in
/-- A fixed positive probability of an independent full refresh is enough
for the zero limit for almost every sampling path on the whole start event. -/
theorem zero_centeredPAGE_iid (mu : Measure (ComplexParameter A H d L → ℂ))
    [IsProbabilityMeasure mu] {q : ℝ≥0∞} (hq : q < 1) (r : ℕ)
    (part : Finset (Word A (T+1) × A)) (hp : part.Nonempty)
    [MeasurableSpace (Option part)] [MeasurableSingletonClass (Option part)]
    (sample : Measure (Option part)) [IsProbabilityMeasure sample] (href : sample {none} ≠ 0) :
    ∃ U : Set (ComplexParameter A H d L → ℂ), MeasurableSet U ∧ q < mu U ∧
      ∃ a > 0, ∀ η, 0 < η → η ≤ a → ∀ draw ∈ U,
        ∀ᵐ path ∂Measure.infinitePi (fun _ : ℕ => sample),
          Tendsto (fun t => (run (anchoredLoss (graphComponent r part) 0 η⁻¹) η
            (WithLp.toLp 2 (sgdInitialWeights true draw)) path t).weights) atTop (𝓝 0) := by
  obtain ⟨U,hU,hprob,a,ha,fit⟩ := zero_centeredPAGE_collapses mu hq r part hp
  refine ⟨U,hU,hprob,a,ha,fun η hη hb draw hd => ?_⟩
  filter_upwards [Protocol.ae_frequently_mem_infinitePi sample (measurableSet_singleton none) href]
    with path hpath using fit η hη hb draw hd path hpath

end Transformer.Grokking.FiniteRecovery.Argand
