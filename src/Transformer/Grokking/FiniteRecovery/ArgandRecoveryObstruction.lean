module

public import Transformer.Grokking.FiniteRecovery.ZeroEmbeddings
public import Transformer.Grokking.FiniteRecovery.ArgandFullGradientConvergence
public import Transformer.Grokking.FiniteRecovery.Examples

/-! Exact prediction from every admissible draw is false for the full
Argand momentum run. The native initialization accepts zero sampled
embedding entries, with the usual nonzero score multiplier. -/

@[expose] public section

open Filter
open scoped Topology

namespace Transformer.Grokking.FiniteRecovery.Argand

open Transformer.AMSGradW Transformer.Optimization Transformer.Grokking.FlatSpectrum

variable {A : Type*} [Fintype A] {H d layers T : ℕ}

/-- The full gradient, global clipping, momentum and quadratic decay
preserve zero initial embeddings and zero embedding velocity. -/
theorem fullGradientMomentumRun_zeroEmbeddings (realScoreInit : Bool) (repeats : ℕ)
    (batch : List (LabeledPosition A)) (lam : ℝ) (rate : ℕ → ℝ)
    (draw : ComplexParameter A H d layers → ℂ)
    (hdraw : ∀ a j, draw (.embedding a j) = 0) (t : ℕ) :
    let run := fullGradientMomentumRun realScoreInit repeats batch lam rate draw
    ZeroEmbeddings (run t).weights ∧ ZeroEmbeddings (run t).velocity := by
  induction t with
  | zero =>
    refine ⟨?_, fun _ _ _ => rfl⟩
    intro a j b
    cases b <;> simp [fullGradientMomentumRun, ClippedMomentum.run, sgdInitialWeights,
      initialWeights, hdraw]
  | succ t ih =>
    let run := fullGradientMomentumRun realScoreInit repeats batch lam rate draw
    let w := (run t).weights
    let v := (run t).velocity
    let f : WeightSpace A H d layers → ℝ := fullGradientObjective repeats batch lam
    let g := WithLp.ofLp (gradient f (WithLp.toLp 2 w))
    obtain ⟨hw, hv⟩ := ih
    have hg : g = lam • w := by
      dsimp only [g, f]
      rw [fullGradientObjective_gradient, WithLp.ofLp_toLp,
        batchGradient_eq_zero_of_zeroEmbeddings repeats batch w hw]
      simp
    constructor <;> intro a j b
    · let i : Parameter A H d layers := .inl (.embedding a j, b)
      change w i - rate t * ((9 / 10 : ℝ) * v i + Kernite.clip g i) = 0
      have hw' : w i = 0 := hw a j b
      have hv' : v i = 0 := hv a j b
      simp [Kernite.clip, hg, hw', hv']
    · let i : Parameter A H d layers := .inl (.embedding a j, b)
      change (9 / 10 : ℝ) * v i + Kernite.clip g i = 0
      have hw' : w i = 0 := hw a j b
      have hv' : v i = 0 := hv a j b
      simp [Kernite.clip, hg, hw', hv']

omit [Fintype A] in
theorem not_realizes_of_zeroEmbeddings [Nontrivial A] (w : Weights A H d layers)
    (hw : ZeroEmbeddings w) (repeats : ℕ) (target : Word A (T + 1) → A) :
    ¬ (Model.fromWeights w).realizes repeats target := by
  have he (a : A) : (Model.fromWeights w).embedding a = 0 := by
    ext j
    exact Complex.ext (hw a j false) (hw a j true)
  let z : Word A (T + 1) := fun _ => Classical.choice (inferInstance : Nonempty A)
  obtain ⟨b, hb⟩ := exists_ne (target z)
  intro h
  have hlt := h z b hb
  simp only [Model.logits, Model.readout, he, inner_zero_left, Complex.zero_re, zero_div, lt_self_iff_false] at hlt

/-- Every finite step of this native run has tied logits for any target.
The conclusion holds on the full input domain, including hidden words. -/
theorem fullGradientMomentumRun_zeroEmbeddings_not_realizes [Nontrivial A]
    (realScoreInit : Bool) (repeats : ℕ) (batch : List (LabeledPosition A))
    (lam : ℝ) (rate : ℕ → ℝ) (draw : ComplexParameter A H d layers → ℂ)
    (hdraw : ∀ a j, draw (.embedding a j) = 0)
    (target : Word A (T + 1) → A) (t : ℕ) :
    let run := fullGradientMomentumRun realScoreInit repeats batch lam rate draw
    ¬ (Model.fromWeights (run t).weights).realizes repeats target :=
  not_realizes_of_zeroEmbeddings _
    (fullGradientMomentumRun_zeroEmbeddings realScoreInit repeats batch lam rate draw hdraw t).1
    repeats target

/-- A finite weight limit of the same run still has tied logits. -/
theorem fullGradientMomentumRun_zeroEmbeddings_no_correct_limit [Nontrivial A]
    (realScoreInit : Bool) (repeats : ℕ) (batch : List (LabeledPosition A))
    (lam : ℝ) (rate : ℕ → ℝ) (draw : ComplexParameter A H d layers → ℂ)
    (hdraw : ∀ a j, draw (.embedding a j) = 0) (target : Word A (T + 1) → A)
    (star : Weights A H d layers)
    (h : Tendsto (fun t => (fullGradientMomentumRun realScoreInit repeats batch lam rate draw t).weights)
      atTop (𝓝 star)) :
    ¬ (Model.fromWeights star).realizes repeats target := by
  apply not_realizes_of_zeroEmbeddings star _ repeats target
  intro a j b
  have hc := (continuous_apply (Sum.inl (ComplexParameter.embedding a j, b))).tendsto star |>.comp h
  have hz : Tendsto (fun t =>
      (fullGradientMomentumRun realScoreInit repeats batch lam rate draw t).weights
        (.inl (.embedding a j, b))) atTop (𝓝 0) :=
    tendsto_const_nhds.congr fun t =>
      ((fullGradientMomentumRun_zeroEmbeddings realScoreInit repeats batch lam rate draw hdraw t).1 a j b).symm
  exact tendsto_nhds_unique hc hz

/-- The binary modular sum has a unique nuclear completion from three
cells. Full gradient momentum can still converge to tied logits on this
same training mask. -/
example (prefixRate : ℕ → ℝ) (N : ℕ) :
    let e := wordEnumeration 2 1
    let cut := enumeratedCut e
    let target := modularSum e
    let inputs := inputsAtCells cut (staircase 1)
    let batch := observedLastPositions (T := 1) target inputs
    NuclearRecovers (observedCells cut inputs) (targetKernel cut ZMod.stdAddChar target) ∧
      ∃ η > 0, ∃ star : Weights (ZMod 2) 4 16 1,
        Tendsto (fun t => (fullGradientMomentumRun false 1 batch (1 / 10)
          (fun k => if k < N then prefixRate k else η) (fun _ => 0) t).weights) atTop (𝓝 star) ∧
        ¬ (Model.fromWeights star).realizes 1 target := by
  dsimp only
  constructor
  · rw [observedCells_inputsAtCells]
    exact modularSum_nuclear_recovers (wordEnumeration 2 1)
  let e := wordEnumeration 2 1
  let cut := enumeratedCut e
  let target := modularSum e
  let batch := observedLastPositions (T := 1) target (inputsAtCells cut (staircase 1))
  let draw : ComplexParameter (ZMod 2) 4 16 1 → ℂ := fun _ => 0
  obtain ⟨η, hη, hrun⟩ := fullGradientMomentumRun_converges
    false 1 batch (show (0 : ℝ) < 1 / 10 by norm_num) draw prefixRate N
  obtain ⟨star, hlim, _, _, _, _⟩ := hrun η hη le_rfl
  let coords := PiLp.continuousLinearEquiv 2 ℝ (fun _ : Parameter (ZMod 2) 4 16 1 => ℝ)
  have hraw := (coords.continuous.tendsto star).comp hlim
  refine ⟨η, hη, WithLp.ofLp star, hraw, ?_⟩
  exact fullGradientMomentumRun_zeroEmbeddings_no_correct_limit false 1 batch _
    _ draw (fun _ _ => rfl) target _ hraw

end Transformer.Grokking.FiniteRecovery.Argand
