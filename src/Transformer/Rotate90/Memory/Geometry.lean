module

public import Transformer.Rotate90.Memory.Symmetry
public import Transformer.Rotate90.Memory.Basic

/-!
# Correct powered-division parameters need not average correctly

Source: `review/flat_spectrum.md`, *Optimizer memory*, M3--M5. The existing
constructive solution supplies two antipodal solutions of every prime-powered
task. Their midpoint is the zero model. This is an obstruction in the actual
finite parameter space, not in a surrogate loss or an infinite-width limit.
The constructed optimizer state below is admissible, but no claim is made
that random initialization reaches this particular state.
-/

@[expose] public section

noncomputable section

namespace Transformer.Rotate90.Memory

open Landscape.Neural Learning ScheduleFree

variable {V K H : Type*} [Fintype K] [Fintype H]

/-- Reindex the already constructed one-block model into the full-stack space. -/
def liftBlock (w : GatedWeights V K H) : Weights V K H (Fin 1) :=
  WithLp.toLp 2 (fun i => match i with
    | .inl j => w.ofLp (.inl j)
    | .inr (_, j) => w.ofLp (.inr j))

/-- This reindexing preserves the entire actual forward pass. -/
theorem score_liftBlock (eps : ℝ) (w : GatedWeights V K H) (a b q : V) :
    score eps [0] (liftBlock w) a b q = logit eps w a b q := by
  change Power.stackLogit eps (gatedEmbedding w) [Power.blockOfWeights w] a b q = _
  exact Power.stackLogit_singleton eps w a b q

variable {p : ℕ} [Fact p.Prime] {Z : Type*} [Fintype Z]

/-- M4 has genuine perfect-classifier witnesses for every prime and exponent.
Both endpoints solve all valid inputs; their average has zero logits everywhere. -/
theorem antipodal_solvers {eps : ℝ} (heps : 0 < eps) (k : ℕ) :
    ∃ w : Weights (ZMod p) (Power.PairIndex K) ((Fin 10 × Fin 2) ⊕ Z) (Fin 1),
      (∀ a b : ZMod p, b ≠ 0 → ∀ q : ZMod p, q ≠ (a / b) ^ k →
        score eps [0] w a b q < score eps [0] w a b ((a / b) ^ k)) ∧
      (∀ a b : ZMod p, b ≠ 0 → ∀ q : ZMod p, q ≠ (a / b) ^ k →
        score eps [0] (-w) a b q < score eps [0] (-w) a b ((a / b) ^ k)) ∧
      (∀ a b q : ZMod p,
        score eps [0] ((1/2 : ℝ) • w + (1/2 : ℝ) • (-w)) a b q = 0) := by
  obtain ⟨w, hw⟩ := Power.exists_weights (p := p) (K := K) (Z := Z) heps k
  refine ⟨liftBlock w, ?_, ?_, ?_⟩
  · intro a b hb q hq
    simpa only [score_liftBlock] using hw a b hb q hq
  · intro a b hb q hq
    simpa only [score_neg, score_liftBlock] using hw a b hb q hq
  · intro a b q
    simp only [smul_neg, add_neg_cancel, score_zero]

/-- The whole-table strict-solver region is nonconvex for every powered task.
This forbids transferring a convex prediction guarantee to parameter averages. -/
theorem solver_region_not_convex {eps : ℝ} (heps : 0 < eps) (k : ℕ) :
    ¬ Convex ℝ {w : Weights (ZMod p) (Power.PairIndex K) ((Fin 10 × Fin 2) ⊕ Z) (Fin 1) |
      ∀ a b : ZMod p, b ≠ 0 → ∀ q : ZMod p, q ≠ (a / b) ^ k →
        score eps [0] w a b q < score eps [0] w a b ((a / b) ^ k)} := by
  obtain ⟨w, hw, hn, hm⟩ := antipodal_solvers (p := p) (K := K) (Z := Z) heps k
  intro hc
  have hmid := hc hw hn (show (0 : ℝ) ≤ 1/2 by norm_num)
    (show (0 : ℝ) ≤ 1/2 by norm_num) (show (1/2 : ℝ) + 1/2 = 1 by norm_num)
  have hbad := hmid 1 1 one_ne_zero 0 (by simp)
  simp only [hm] at hbad
  exact (lt_irrefl 0) hbad

variable {I : Type*}

/-- A legal saved two-buffer state with evaluation w, base -w and W=M²=1.
The query is determined by beta1 exactly as in the source's pack(). -/
def antipodalState (p : AdamParameters) (w : I → ℝ) : AdamW.TrainingState I :=
  AdamW.pack (halt p) ⟨w, fun i => -w i, ⟨fun _ => 1, 1, 1⟩⟩

/-- M5: one zero-rate update can hit the zero midpoint of two equivalent
classifiers. Real gradients may be arbitrary; their moment update is retained. -/
theorem antipodal_step (p : AdamParameters) (hβ : p.betaOne ≠ 0)
    (w g : I → ℝ) (t : ℕ) :
    AdamW.evaluation (halt p)
      (AdamW.trainingStep (halt p) t (antipodalState p w) g) = 0 := by
  have hx : AdamW.evaluation (halt p) (antipodalState p w) = w :=
    AdamW.evaluation_pack (halt p) _ hβ
  funext i
  rw [halt_evaluation p t _ g (by norm_num [antipodalState, AdamW.pack]), hx]
  simp only [antipodalState, AdamW.pack]
  norm_num

/-- The preceding source step collapses every score of the actual model. -/
theorem antipodal_step_score {D : Type*} (eps : ℝ) (order : List D)
    (p : AdamParameters) (hβ : p.betaOne ≠ 0) (w : Weights V K H D)
    (g : Index V K H D → ℝ) (t : ℕ) (a b q : V) :
    score eps order
      (evaluated (halt p) (AdamW.trainingStep (halt p) t (antipodalState p w.ofLp) g))
      a b q = 0 := by
  unfold evaluated
  rw [antipodal_step p hβ]
  exact score_zero eps order a b q

example := antipodal_solvers (p := 3) (K := Fin 1) (Z := Fin 23)
  (eps := 1/8388608) (by norm_num) 5
example := solver_region_not_convex (p := 3) (K := Fin 1) (Z := Fin 23)
  (eps := 1/8388608) (by norm_num) 0
example := antipodal_step (I := Fin 2)
  ⟨3/1000, 1, 10, 9/10, 98/100, 1/100000000⟩ (by norm_num)
  (fun i => i.val + 1) (fun _ => 2) 750
example := antipodal_step_score (V := Fin 3) (K := Fin 8) (H := Fin 43)
  (1/8388608) (List.finRange 3)
  ⟨3/1000, 1, 10, 9/10, 98/100, 1/100000000⟩ (by norm_num)
  (WithLp.toLp 2 (fun _ => 1)) (fun _ => 2) 750 0 1 2

end Transformer.Rotate90.Memory
