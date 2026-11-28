module

public import Transformer.Grokking.FiniteRecovery.ProjectionRows
public import Transformer.Grokking.FiniteRecovery.ArgandSoftsign

/-! Softsign Lion followed by Euclidean projection of the decoder's complex
output rows. The stock input draw is used directly. Clipping begins only
after the first update and leaves the momentum exactly as computed. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.ProjectionRows

open SoftsignMomentum
open Transformer.AMSGradW.Kernite (clip)

variable {ι G : Type*} [Fintype ι]

noncomputable def run (tags : ι → Option G) (R decay : ℝ) (rate : ℕ → ℝ)
    (grad : ℕ → (ι → ℝ) → ι → ℝ) (initial : ι → ℝ) : ℕ → State ι
  | 0 => ⟨initial,0⟩
  | t+1 =>
    let s := run tags R decay rate grad initial t
    let next := step decay (rate t) (grad t s.weights) s
    ⟨project tags R next.weights,next.moment⟩

variable (tags : ι → Option G) (R decay : ℝ) (rate : ℕ → ℝ)
  (grad : ℕ → (ι → ℝ) → ι → ℝ) (initial : ι → ℝ)

/-- Projection can only decrease coordinate magnitudes, so positive decay
still bounds the complete parameter vector for all minibatch sequences. -/
theorem weight_bound (hR : 0 ≤ R) (hdc : 0 < decay) (hr : ∀ t, 0 ≤ rate t ∧ rate t*decay ≤ 1)
    (i : ι) (t : ℕ) : |(run tags R decay rate grad initial t).weights i| ≤ max ‖initial‖ (1/decay) := by
  let s := run tags R decay rate grad initial
  let d t i := direction (9/10*(s t).moment i + 1/10*clip (grad t (s t).weights) i)
  exact DecoupledDecay.coordinate_bound_of_abs_le (fun t => (s t).weights) d rate hdc hr
    (fun t j => abs_project_le tags hR (step decay (rate t) (grad t (s t).weights) (s t)).weights j)
    (fun _ _ => direction_bound _) i t

/-- At a feasible representative the same directional alignment gives the same
finite ordinary-trajectory bound after projection. The gradients and momenta
are evaluated on the projected trajectory itself. -/
theorem aligned_error_bound [Fintype G] (hR : 0 ≤ R) (u : ι → ℝ) (hu : ∀ b, normOn (row tags (some b)) u ≤ R)
    (B : ℕ) {η μ M : ℝ} (hη : 0 < η) (hμ : 0 < μ) (hM : 0 ≤ M) (hc : 2*η*μ ≤ 1)
    (hr : ∀ j, rate (B+j) = η)
    (ha : ∀ j, let s := run tags R decay rate grad initial (B+j)
      μ*squaredError u s.weights ≤ ∑ i, (s.weights i-u i)*force decay s (grad (B+j) s.weights) i)
    (hm : ∀ j, let s := run tags R decay rate grad initial (B+j)
      ∑ i, (force decay s (grad (B+j) s.weights) i)^2 ≤ M) (k : ℕ) :
    squaredError u (run tags R decay rate grad initial (B+k)).weights ≤
      (1-2*η*μ)^k*squaredError u (run tags R decay rate grad initial B).weights+η*M/(2*μ) := by
  let s := run tags R decay rate grad initial
  let e j := squaredError u (s (B+j)).weights
  have hs (j : ℕ) : e (j+1) ≤ (1-2*η*μ)*e j+η^2*M := by
    let s₀ := s (B+j)
    let g := grad (B+j) s₀.weights
    change squaredError u (project tags R (step decay (rate (B+j)) g s₀).weights) ≤ _
    apply (projected_error_le tags hR u _ hu).trans
    rw [step_error,hr j]
    have hx := mul_le_mul_of_nonneg_left (ha j) (by positivity : 0 ≤ 2*η)
    have hy := mul_le_mul_of_nonneg_left (hm j) (sq_nonneg η)
    exact (add_le_add (sub_le_sub_left hx _) hy).trans_eq (by dsimp [e,s₀,s]; ring)
  simpa only [s,e,Nat.add_zero] using geometric_error_bound e hη hμ hM hc hs k

end Transformer.Grokking.FiniteRecovery.ProjectionRows

namespace Transformer.Grokking.FiniteRecovery.Argand

open Set Metric FlatSpectrum
open scoped NNReal

variable {A X Z : Type*} [Fintype A] {H d L T : ℕ}

/-- An output coordinate of one of the seven attention and FFN projections.
The real and imaginary parts of all inputs share one row radius. -/
inductive ProjectionRow (H d L : ℕ) where
  | attentionIn : Fin L → Fin 3 → Fin H → Fin d → ProjectionRow H d L
  | attentionOut : Fin L → Fin (H*d) → ProjectionRow H d L
  | feedForwardIn : Fin L → Fin 2 → Fin (4*(H*d)) → ProjectionRow H d L
  | feedForwardOut : Fin L → Fin (H*d) → ProjectionRow H d L
  deriving Fintype, DecidableEq

def projectionRow : Parameter A H d L → Option (ProjectionRow H d L)
  | .inl (.attentionIn l k _ h j, _) => some (.attentionIn l k h j)
  | .inl (.attentionOut l _ _ o, _) => some (.attentionOut l o)
  | .inl (.feedForwardIn l k _ o, _) => some (.feedForwardIn l k o)
  | .inl (.feedForwardOut l _ o, _) => some (.feedForwardOut l o)
  | _ => none

noncomputable def projectedSoftsignRun (r : ℕ) (R decay : ℝ) (batch : ℕ → List (LabeledPosition A))
    (rate : ℕ → ℝ) (draw : ComplexParameter A H d L → ℂ) :=
  ProjectionRows.run projectionRow R decay rate (fun t => batchGradient r (batch t)) (sgdInitialWeights true draw)

/-- Independent perturbed copies are scored without projection, exactly as in
the runner. All ordinary checkpoints come from the row-projected optimizer. -/
theorem projectedSoftsign_checkpoint_recovers [Finite Z] (r : ℕ) (R decay : ℝ) (hR : 0 ≤ R) (hdc : 0 < decay)
    (schedule : ℕ → ℕ) (rate : ℕ → ℝ) (hr : ∀ t, 0 ≤ rate t ∧ rate t*decay ≤ 1)
    (batch : ℕ → List (LabeledPosition A)) (draw : ComplexParameter A H d L → ℂ)
    (prompt : Z → Word A (T+1)) (hidden : Z → X) (S : Set X) (family : Set (X → A))
    (hid : InjOn (fun g : X → A => fun x : S => g x) family) (f : X → A) (hf : f ∈ family)
    (hrep : ∃ w : Weights A H d L, StrictOn
      (fun x => (Model.fromWeights w).logits r (prompt x) (Fin.last T)) (f ∘ hidden))
    (v : ℝ≥0) (hv : 0 < v) :
    PathCheckpointRecovery r (fun j => (projectedSoftsignRun r R decay batch rate draw (schedule j)).weights)
      prompt (f ∘ hidden) (hiddenCertificate S family (fun x => f x) hidden) v := by
  let w := sgdInitialWeights (H := H) (d := d) (layers := L) true draw
  apply bounded_path_checkpoint_recovers r _ (R := max ‖w‖ (1/decay)) _ prompt hidden S family hid f hf hrep v hv
  intro j
  rw [mem_closedBall,dist_zero_right]
  apply (pi_norm_le_iff_of_nonneg ((norm_nonneg w).trans (le_max_left _ _))).2
  exact fun i => ProjectionRows.weight_bound projectionRow R decay rate (fun t => batchGradient r (batch t))
    w hR hdc hr i (schedule j)

open Division.Factors

/-- The projected pilot on any identifying training mask for modular division. -/
theorem factor_projectedSoftsign_checkpoint_recovers [Finite Z] (hidden : Z → DivisionInput) (S : Set DivisionInput)
    (hc : (patternGraph (unitMask S)).Preconnected) (hz : ∃ j, (none,j) ∈ S)
    (f : DivisionInput → Fin 239) (hf : f ∈ factorClass unitToken)
    (batch : ℕ → List (LabeledPosition (Fin 239))) (draw : ComplexParameter (Fin 239) 4 16 6 → ℂ) :
    PathCheckpointRecovery 1 (fun j => (projectedSoftsignRun 1 (7/4) (1/10) batch (fun t =>
      3/10*wsdRate 20000 t) draw (256*(32+j))).weights) (divisionPrompt ∘ hidden) (f ∘ hidden)
      (hiddenCertificate S (factorClass unitToken) (fun x => f x) hidden) ((1/1000000 : ℝ≥0)^2) := by
  obtain ⟨w,hw⟩ := class_native_realization f hf
  apply projectedSoftsign_checkpoint_recovers 1 (7/4) (1/10) (by norm_num) (by norm_num) _ _ _ batch draw _ hidden S _
    (factorClass_identified _ _ hc hz) f hf ⟨sgdInitialWeights true w,fun z => hw (hidden z)⟩ _ (by norm_num)
  intro t
  obtain ⟨h₀,h₁⟩ := wsdRate_bounds 20000 t
  exact ⟨by positivity, by linarith⟩

example := factor_projectedSoftsign_checkpoint_recovers id (factorMask (1 : DivisionUnit))
  (by rw [unitMask_factorMask]; exact (diagRow_connected _).preconnected) ⟨1,mem_insert _ _⟩
  divisionAnswer divisionAnswer_mem

end Transformer.Grokking.FiniteRecovery.Argand
