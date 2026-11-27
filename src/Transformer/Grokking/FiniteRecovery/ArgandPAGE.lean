module

public import Transformer.Optimization.PAGE.Convergence
public import Transformer.Grokking.FiniteRecovery.ArgandModularSum
public import Transformer.Grokking.FiniteRecovery.ArgandRecoveryObstruction

/-! PAGE applied to the actual observed binary cross-entropy of full Argand.
The optimizer is independent of this example. At zero embedding rows every
sample gradient vanishes. A strictly realizable target has lower loss at a
witness, ruling out global PL for this very objective on any nonempty mask. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Transformer.Optimization Transformer.Optimization.PAGE Finset

variable {I : Type*} [Fintype I] [Nonempty I] {H d layers T : ℕ}

/-- Each component reads a single observed last-position label, with no weight
penalty. The generic mean gives the empirical cross-entropy. -/
noncomputable def observedComponent (repeats : ℕ) (words : I → Word (ZMod 2) (T+1))
    (target : Word (ZMod 2) (T+1) → ZMod 2) (i : I) : WeightSpace (ZMod 2) H d layers → ℝ :=
  fullGradientObjective repeats [⟨T, words i, Fin.last T, target (words i)⟩] 0

omit [Fintype I] [Nonempty I] in
private theorem component_value (r : ℕ) (u : I → Word (ZMod 2) (T+1))
    (f : Word (ZMod 2) (T+1) → ZMod 2) (i : I) (w : WeightSpace (ZMod 2) H d layers) :
    observedComponent r u f i w =
      (Model.fromWeights (WithLp.ofLp w)).labelLoss r (u i) (f (u i)) := by
  simp [observedComponent, fullGradientObjective, batchLoss, Model.labelLoss]

omit [Fintype I] [Nonempty I] in
private theorem component_gradient_zero (r : ℕ) (u : I → Word (ZMod 2) (T+1))
    (f : Word (ZMod 2) (T+1) → ZMod 2) (w : Weights (ZMod 2) H d layers)
    (hw : ZeroEmbeddings w) (i : I) :
    gradient (observedComponent r u f i) (WithLp.toLp 2 w) = 0 := by
  rw [observedComponent, fullGradientObjective_gradient, WithLp.ofLp_toLp,
    batchGradient_eq_zero_of_zeroEmbeddings _ _ _ hw]
  simp

omit [Nonempty I] in
/-- This includes native draws with zero embedding rows and an initialized
nonzero score. It holds for every refresh and sampling schedule. -/
theorem page_zero_embeddings_fixed (r : ℕ) (u : I → Word (ZMod 2) (T+1))
    (f : Word (ZMod 2) (T+1) → ZMod 2) (w : Weights (ZMod 2) H d layers)
    (hw : ZeroEmbeddings w) (η : ℝ) (draw : ℕ → Option I) (t : ℕ) :
    (run (observedComponent r u f) η (WithLp.toLp 2 w) draw t).weights = WithLp.toLp 2 w ∧
    ¬ (Model.fromWeights (WithLp.ofLp (run (observedComponent r u f) η (WithLp.toLp 2 w) draw t).weights)).realizes r f := by
  rw [run_fixed _ _ _ (component_gradient_zero r u f w hw)]
  exact ⟨rfl, not_realizes_of_zeroEmbeddings _ hw _ _⟩

private theorem strict_loss_lt_uniform (model : Model (ZMod 2) H d layers) (r : ℕ)
    (f : Word (ZMod 2) (T+1) → ZMod 2) (hr : model.realizes r f) (x : Word (ZMod 2) (T+1)) :
    model.labelLoss r x (f x) < -Real.log (1/2) := by
  let z := model.logits r x (Fin.last T)
  have hz (a : ZMod 2) (ha : a ≠ f x) : Real.exp (z a) < Real.exp (z (f x)) :=
    Real.exp_lt_exp.mpr (hr x a ha)
  have hsum : (∑ j, Real.exp (z j)) = Real.exp (z 0) + Real.exp (z 1) := Fin.sum_univ_two _
  apply neg_lt_neg
  apply Real.log_lt_log (by norm_num : (0:ℝ) < 1/2)
  change 1/2 < Real.exp (z (f x)) / ∑ j, Real.exp (z j)
  rw [hsum, lt_div_iff₀ (add_pos (Real.exp_pos _) (Real.exp_pos _))]
  have hb : ∀ a : ZMod 2, a = 0 ∨ a = 1 := by decide
  rcases hb (f x) with h | h
  · have hh := hz 1 (by rw [h]; decide)
    rw [h] at hh ⊢
    linarith
  · have hh := hz 0 (by rw [h]; decide)
    rw [h] at hh ⊢
    linarith

/-- Even on a sample admitting strict full-native weights, plain empirical
cross-entropy violates global PL for every positive constant at every lower bound. -/
theorem observedComponent_not_global_PL (r : ℕ) (u : I → Word (ZMod 2) (T+1))
    (f : Word (ZMod 2) (T+1) → ZMod 2) (w : Weights (ZMod 2) H d layers)
    (hr : (Model.fromWeights w).realizes r f) {μ floor : ℝ} (hμ : 0 < μ)
    (hb : ∀ x : WeightSpace (ZMod 2) H d layers, floor ≤ objective (observedComponent r u f) x) :
    ¬ Dominated (objective (observedComponent (H := H) (d := d) (layers := layers) r u f)) μ floor := by
  let loss : I → WeightSpace (ZMod 2) H d layers → ℝ := observedComponent r u f
  let z : Weights (ZMod 2) H d layers := 0
  have hz : ZeroEmbeddings z := by intro a j b; rfl
  have hg : gradient (objective loss) (WithLp.toLp 2 z) = 0 := by
    rw [gradient_objective loss (fun i => (contDiff_fullGradientObjective _ _ _).differentiable one_ne_zero)]
    change mean (fun i => gradient (observedComponent r u f i) (WithLp.toLp 2 z)) = 0
    simp [component_gradient_zero r u f z hz, mean]
  have zeroLoss (i : I) : loss i (WithLp.toLp 2 z) = -Real.log (1/2) := by
    change observedComponent r u f i _ = _
    rw [component_value]
    have he (a : ZMod 2) : (Model.fromWeights z).embedding a = 0 := by ext j; rfl
    simp [Model.labelLoss, Model.probability, Model.logits, Model.readout, he, ZMod.card]
  have hzero : objective loss (WithLp.toLp 2 z) = -Real.log (1/2) := by
    simp only [objective, zeroLoss, mean_const]
  have each (i : I) : loss i (WithLp.toLp 2 w) < -Real.log (1/2) := by
    change observedComponent r u f i _ < _
    rw [component_value]
    exact strict_loss_lt_uniform _ _ _ hr _
  have hfit : objective loss (WithLp.toLp 2 w) < -Real.log (1/2) := by
    rw [← mean_const (I := I) (-Real.log (1/2))]
    exact mul_lt_mul_of_pos_left (sum_lt_sum_of_nonempty univ_nonempty (fun i _ => each i)) (by positivity)
  intro h
  have hp := h (WithLp.toLp 2 z)
  change 2*μ*(objective loss (WithLp.toLp 2 z) - floor) ≤ ‖gradient (objective loss) (WithLp.toLp 2 z)‖^2 at hp
  rw [hg, norm_zero, zero_pow (by decide : 2 ≠ 0), hzero] at hp
  exact (not_lt_of_ge hp) (mul_pos (by positivity) (sub_pos.mpr ((hb _).trans_lt hfit)))

example {k s : ℕ} (e : Fin (s+1) ≃ Word (ZMod 2) (k+1)) (hk : (k+1)+(k+1) ≤ 32)
    (u : I → Word (ZMod 2) ((k+1)+(k+1))) {μ : ℝ} (hμ : 0 < μ) :
    ¬ Dominated (objective (observedComponent (H := 4) (d := 16) (layers := 6)
      (T := (k+1)+k) 1 u (modularSum e))) μ 0 := by
  apply observedComponent_not_global_PL 1 u (modularSum e) _ (modularSum_native_realizes e hk) hμ
  intro x
  rw [← mean_const (I := I) (0 : ℝ)]
  apply mean_mono
  intro i
  rw [component_value]
  exact Model.labelLoss_nonneg _ _ _ _

end Transformer.Grokking.FiniteRecovery.Argand
