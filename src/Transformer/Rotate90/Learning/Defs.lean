module

public import Transformer.Rotate90.Training.Basic
public import Transformer.ScheduleFree.AdamW.Basic

/-!
# All trainable coordinates of the finite Rotate90 stack

Source: `review/flat_spectrum.md`, *Learning transfer*, equations T1--T3;
`python/src/lab/infrastructure/nn/peri_rotary.py`, `forward`.
The shared embedding and every gate, value and output entry vary together.
The list of layer indices fixes their order; `List.finRange d` instantiates
the actual depth-d model. Nothing here freezes the branch or the readout.
-/

@[expose] public section

noncomputable section

namespace Transformer.Rotate90.Learning

open Landscape.Neural

/-- One shared embedding and the three independent matrices of each layer. -/
abbrev Index (V K H D : Type*) := (V × Channels K) ⊕
  (D × ((Channels K × H) ⊕ ((Channels K × H) ⊕ (H × Channels K))))

/-- The ordinary Euclidean metric on all actual scalar parameters. -/
abbrev Weights (V K H D : Type*) := EuclideanSpace ℝ (Index V K H D)

variable {V K H D : Type*} [Fintype V] [Fintype K] [Fintype H] [Fintype D]

/-- The same coordinates serve both lookups and the output projection. -/
def embedding (w : Weights V K H D) (q : V) (i : Channels K) : ℝ :=
  w.ofLp (.inl (q, i))

/-- The trainable matrices of layer d, in source gate/value/output order. -/
def matrices (w : Weights V K H D) (d : D) : Power.BlockWeights K H :=
  (fun i j => w.ofLp (.inr (d, .inl (i, j))),
   fun i j => w.ofLp (.inr (d, .inr (.inl (i, j)))),
   fun i j => w.ofLp (.inr (d, .inr (.inr (i, j)))))

/-- The existing actual finite-stack forward pass, with every weight variable. -/
def score (eps : ℝ) (order : List D) (w : Weights V K H D) (a b q : V) : ℝ :=
  Power.stackLogit eps (embedding w) (order.map (matrices w)) a b q

/-- Full softmax cross-entropy on a fixed set of operand pairs and labels. -/
def loss (eps : ℝ) (order : List D) (rows : Finset (V × V))
    (target : V × V → V) (w : Weights V K H D) : ℝ :=
  meanCrossEntropy rows (fun r q => score eps order w r.1 r.2 q) target

/-- T1's actual scalar parameter count, including every independently varying
layer. This matches the lab's combined gate/value storage exactly. -/
theorem finrank_weights : Module.finrank ℝ (Weights V K H D) =
    Fintype.card V * (2 * Fintype.card K) +
      Fintype.card D * (3 * (2 * Fintype.card K) * Fintype.card H) := by
  change Module.finrank ℝ (EuclideanSpace ℝ (Index V K H D)) = _
  rw [finrank_euclideanSpace]
  simp only [Index, Channels, Fintype.card_sum, Fintype.card_prod, Fintype.card_fin]
  ring

example : Module.finrank ℝ (Weights (Fin 97) (Fin 8) (Fin 43) (Fin 1)) = 3616 := by
  rw [finrank_weights]
  norm_num
example : Module.finrank ℝ (Weights (Fin 193) (Fin 32) (Fin 171) (Fin 1)) = 45184 := by
  rw [finrank_weights]
  norm_num

/-- Analyticity of one actual block jointly in its weights and incoming state. -/
theorem analyticAt_layer {eps : ℝ} (heps : 0 < eps) (d : D)
    {v : Weights V K H D → Channels K → ℝ} {x : Weights V K H D}
    (hv : ∀ i, AnalyticAt ℝ (fun w => v w i) x) (i : Channels K) :
    AnalyticAt ℝ (fun w => Power.block eps (matrices w d) (v w) i) x := by
  have hr := fun j => analyticAt_rmsNorm heps hv j
  have hg := fun j => analyticAt_dense
    (fun a b => analyticAt_parameter (.inr (d, .inl (a, b))) x) hr j
  have hu := fun j => analyticAt_dense
    (fun a b => analyticAt_parameter (.inr (d, .inr (.inl (a, b)))) x) hr j
  have hout := fun j => analyticAt_dense
    (fun a b => analyticAt_parameter (.inr (d, .inr (.inr (a, b)))) x)
    (fun h => analyticAt_swiglu_comp (hg h) (hu h)) j
  exact (hv i).fun_add (analyticAt_rmsNorm heps hout i)

/-- Induction through all varying layers, with arbitrary analytic input. -/
theorem analyticAt_stack {eps : ℝ} (heps : 0 < eps) (order : List D)
    {v : Weights V K H D → Channels K → ℝ} {x : Weights V K H D}
    (hv : ∀ i, AnalyticAt ℝ (fun w => v w i) x) (i : Channels K) :
    AnalyticAt ℝ (fun w => (order.map (matrices w)).foldl
      (fun z B => Power.block eps B z) (v w) i) x := by
  induction order generalizing v with
  | nil => exact hv i
  | cons d order ih =>
    exact ih (fun j => analyticAt_layer heps d hv j)

/-- Joint analyticity includes both outer normalizers and all tied occurrences. -/
theorem analyticAt_score {eps : ℝ} (heps : 0 < eps) (order : List D)
    (x : Weights V K H D) (a b q : V) :
    AnalyticAt ℝ (fun w => score eps order w a b q) x := by
  have hE : ∀ r i, AnalyticAt ℝ (fun w : Weights V K H D => embedding w r i) x :=
    fun r i => by simpa only [embedding] using
      (analyticAt_parameter (I := Index V K H D) (.inl (r, i)) x)
  have hinput := fun i => analyticAt_rmsNorm heps (analyticAt_tiedInput hE a b) i
  exact analyticAt_tiedReadout hE
    (fun i => analyticAt_rmsNorm heps (fun j => analyticAt_stack heps order hinput j) i) q

/-- The actual finite-model, arbitrary-split loss is analytic in every parameter. -/
theorem analyticOnNhd_loss {eps : ℝ} (heps : 0 < eps) (order : List D)
    (rows : Finset (V × V)) (target : V × V → V) :
    AnalyticOnNhd ℝ (loss (K := K) (H := H) eps order rows target) Set.univ := by
  intro x hx
  exact analyticAt_meanCrossEntropy rows
    (fun r _ q => analyticAt_score heps order x r.1 r.2 q) target

/-- The gradient and segment-curvature bounds below concern an actual C² loss. -/
theorem contDiff_loss {eps : ℝ} (heps : 0 < eps) (order : List D)
    (rows : Finset (V × V)) (target : V × V → V) :
    ContDiff ℝ 2 (loss (K := K) (H := H) eps order rows target) :=
  (analyticOnNhd_loss heps order rows target).contDiff

example := analyticAt_layer (V := Fin 3) (K := Fin 2) (H := Fin 4)
  (eps := 1 / 8388608) (by norm_num) (0 : Fin 3)
  (v := fun w => embedding w 1) (x := 0)
  (by intro i; simpa only [embedding] using
    (analyticAt_parameter (I := Index (Fin 3) (Fin 2) (Fin 4) (Fin 3)) (.inl (1, i)) 0)) (0, 0)
example := analyticAt_stack (V := Fin 3) (K := Fin 2) (H := Fin 4)
  (eps := 1 / 8388608) (by norm_num) (List.finRange 3)
  (v := fun w => embedding w 1) (x := 0)
  (by intro i; simpa only [embedding] using
    (analyticAt_parameter (I := Index (Fin 3) (Fin 2) (Fin 4) (Fin 3)) (.inl (1, i)) 0)) (0, 0)
example := analyticAt_score (V := Fin 3) (K := Fin 2) (H := Fin 4)
  (eps := 1 / 8388608) (by norm_num) (List.finRange 3) 0 0 1 2
example := analyticOnNhd_loss (K := Fin 8) (H := Fin 43)
  (eps := 1 / 8388608) (by norm_num) (List.finRange 3)
  (Finset.univ : Finset (ZMod 3 × ZMod 3)) (fun r => (r.1 / r.2) ^ 2)
example := contDiff_loss (K := Fin 8) (H := Fin 43)
  (eps := 1 / 8388608) (by norm_num) (List.finRange 3)
  (Finset.univ : Finset (ZMod 3 × ZMod 3)) (fun r => (r.1 / r.2) ^ 2)


open ScheduleFree

variable {I : Type*}

/-- The evaluation point recovered from the optimizer's actual two buffers. -/
def evaluated (p : AdamW.Parameters) (s : AdamW.TrainingState I) : EuclideanSpace ℝ I :=
  WithLp.toLp 2 (AdamW.evaluation p s)

/-- The corrected, coordinatewise normalized gradient before weight decay. -/
def normalized (p : AdamW.Parameters) (t : ℕ) (s : AdamW.TrainingState I)
    (g : I → ℝ) : EuclideanSpace ℝ I :=
  WithLp.toLp 2 (fun i => g i / AdamW.denominator p t s.history g i)


end Transformer.Rotate90.Learning
