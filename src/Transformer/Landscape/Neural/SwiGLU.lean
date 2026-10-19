module

public import Transformer.Landscape.Neural.SiLU

/-!
# The tied Rotate90 classifier with a full SwiGLU branch

Source: `landscape_grokking_plan.md`, Activation comparisons, and the browser
model recorded under Model and protocol. The original one-matrix ReLU branch
is replaced by `(SiLU(r G) * (r U)) D`. E remains shared by both operands and
the readout, and the residual and two positive-epsilon RMSNorms remain.

The gate, value and output matrices have separate free coordinates. The
intermediate index H controls their parameter count; SwiGLU is not merely
SiLU applied to the original dense output.
-/

@[expose] public section

namespace Transformer.Landscape.Neural

/-- Distinct entries of E, the gate, value and output projections. -/
abbrev GatedIndex (V K H : Type*) :=
  ((V × Channels K) ⊕ ((Channels K × H) ⊕ ((Channels K × H) ⊕ (H × Channels K))))

/-- One shared E and independent gate, value and output matrices, all in
the Euclidean metric of their actual scalar parameters. -/
abbrev GatedWeights (V K H : Type*) := EuclideanSpace ℝ (GatedIndex V K H)

variable {V K H : Type*} [Fintype V] [Fintype K] [Fintype H]

/-- The same embedding will be reused by both operands and the readout. -/
def gatedEmbedding (weights : GatedWeights V K H) (q : V) (i : Channels K) : ℝ :=
  weights.ofLp (.inl (q, i))

/-- SwiGLU's gate projection G, distinct from its value projection. -/
def gateMatrix (weights : GatedWeights V K H) (i : Channels K) (j : H) : ℝ :=
  weights.ofLp (.inr (.inl (i, j)))

/-- SwiGLU's value projection U. -/
def valueMatrix (weights : GatedWeights V K H) (i : Channels K) (j : H) : ℝ :=
  weights.ofLp (.inr (.inr (.inl (i, j))))

/-- SwiGLU's output projection D back to the residual width. -/
def outputMatrix (weights : GatedWeights V K H) (i : H) (j : Channels K) : ℝ :=
  weights.ofLp (.inr (.inr (.inr (i, j))))

/-- The normalized gated branch, added to the unnormalized original skip
(Activation comparisons, full three-projection variant). -/
noncomputable def gatedState (eps : ℝ) (weights : GatedWeights V K H) (a b : V)
    (i : Channels K) : ℝ :=
  let v := tiedInput (gatedEmbedding weights) a b
  let r := rmsNorm eps v
  v i + dense (outputMatrix weights)
    (fun j => swiglu (dense (gateMatrix weights) r j) (dense (valueMatrix weights) r j)) i

/-- Final normalized logits use the original shared E, including the zero
answer class (Model and protocol, tied readout). -/
noncomputable def gatedLogit (eps : ℝ) (weights : GatedWeights V K H) (a b q : V) : ℝ :=
  tiedReadout (gatedEmbedding weights) (rmsNorm eps (gatedState eps weights a b)) q

/-- Full cross-entropy on a fixed split; no answer-competition term is
removed to obtain analyticity (M9.2 of the research plan). -/
noncomputable def gatedLoss (eps : ℝ) (rows : Finset (V × V)) (target : V × V → V)
    (weights : GatedWeights V K H) : ℝ :=
  meanCrossEntropy rows (fun r q => gatedLogit eps weights r.1 r.2 q) target

/-- Full SwiGLU has `|V| W + 3 W |H|` actual scalar parameters. The budget
change is explicit rather than hidden in an activation (Activation comparisons). -/
theorem finrank_gatedWeights : Module.finrank ℝ (GatedWeights V K H) =
    Fintype.card V * (2 * Fintype.card K) + 3 * (2 * Fintype.card K) * Fintype.card H := by
  change Module.finrank ℝ (EuclideanSpace ℝ (GatedIndex V K H)) = _
  rw [finrank_euclideanSpace]
  simp only [GatedIndex, Channels, Fintype.card_sum, Fintype.card_prod, Fintype.card_fin]
  ring

/-- The complete SwiGLU residual state is analytic jointly in E, G, U and D.
Every coordinate is an actual trainable parameter (M9.2). -/
theorem analyticAt_gatedState {eps : ℝ} (heps : 0 < eps) (x : GatedWeights V K H)
    (a b : V) (i : Channels K) :
    AnalyticAt ℝ (fun y => gatedState eps y a b i) x := by
  have hE : ∀ q i, AnalyticAt ℝ (fun y : GatedWeights V K H => gatedEmbedding y q i) x :=
    fun q i => by simpa only [gatedEmbedding] using
      (analyticAt_parameter (I := GatedIndex V K H) (.inl (q, i)) x)
  have hG : ∀ i j, AnalyticAt ℝ (fun y : GatedWeights V K H => gateMatrix y i j) x :=
    fun i j => by simpa only [gateMatrix] using
      (analyticAt_parameter (I := GatedIndex V K H) (.inr (.inl (i, j))) x)
  have hU : ∀ i j, AnalyticAt ℝ (fun y : GatedWeights V K H => valueMatrix y i j) x :=
    fun i j => by simpa only [valueMatrix] using
      (analyticAt_parameter (I := GatedIndex V K H) (.inr (.inr (.inl (i, j)))) x)
  have hD : ∀ i j, AnalyticAt ℝ (fun y : GatedWeights V K H => outputMatrix y i j) x :=
    fun i j => by simpa only [outputMatrix] using
      (analyticAt_parameter (I := GatedIndex V K H) (.inr (.inr (.inr (i, j)))) x)
  have hv := fun i => analyticAt_tiedInput hE a b i
  have hr := fun i => analyticAt_rmsNorm heps hv i
  exact (hv i).fun_add (analyticAt_dense hD (fun j =>
    analyticAt_swiglu_comp (analyticAt_dense hG hr j) (analyticAt_dense hU hr j)) i)

/-- Analyticity survives the final normalization and the third occurrence
of E at the readout (M9.2, shared-parameter bridge). -/
theorem analyticAt_gatedLogit {eps : ℝ} (heps : 0 < eps) (x : GatedWeights V K H)
    (a b q : V) : AnalyticAt ℝ (fun y => gatedLogit eps y a b q) x := by
  have hE : ∀ r i, AnalyticAt ℝ (fun y : GatedWeights V K H => gatedEmbedding y r i) x :=
    fun r i => by simpa only [gatedEmbedding] using
      (analyticAt_parameter (I := GatedIndex V K H) (.inl (r, i)) x)
  exact analyticAt_tiedReadout hE
    (fun i => analyticAt_rmsNorm heps (analyticAt_gatedState heps x a b) i) q

/-- This concrete gated classifier has an analytic loss on its full
Euclidean parameter space, with arbitrary fixed labels and split (M9.2). -/
theorem analyticOnNhd_gatedLoss {eps : ℝ} (heps : 0 < eps)
    (rows : Finset (V × V)) (target : V × V → V) :
    AnalyticOnNhd ℝ (gatedLoss (K := K) (H := H) eps rows target) Set.univ := by
  intro x hx
  exact analyticAt_meanCrossEntropy rows
    (fun r _ q => analyticAt_gatedLogit heps x r.1 r.2 q) target

/-- The existing proved Lojasiewicz theorem discharges its local-inequality
hypothesis at every checkpoint of the gated network. Landscape, optimizer
and held-out guarantees remain separate obligations (M9.2). -/
theorem exists_isLojasiewiczAt_gatedLoss {eps : ℝ} (heps : 0 < eps)
    (rows : Finset (V × V)) (target : V × V → V) (x : GatedWeights V K H) :
    ∃ a, 0 ≤ a ∧ a < 1 ∧
      SaddleAvoidance.IsLojasiewiczAt (gatedLoss eps rows target) x a := by
  exact SaddleAvoidance.exists_isLojasiewiczAt_of_analyticAt
    (analyticOnNhd_gatedLoss heps rows target x (Set.mem_univ x))

/- Positive epsilon, nonempty three-class data, two residual channels and
two intermediate channels witness every explicit hypothesis. -/

example : (0 : ℝ) < 1 / 100000 ∧
    AnalyticAt ℝ (fun y : GatedWeights (Fin 3) (Fin 1) (Fin 2) =>
      gatedState (1 / 100000) y 0 1 (0, 0)) 0 := by
  exact ⟨by norm_num, analyticAt_gatedState (by norm_num) 0 0 1 (0, 0)⟩

example : (0 : ℝ) < 1 / 100000 ∧
    AnalyticAt ℝ (fun y : GatedWeights (Fin 3) (Fin 1) (Fin 2) =>
      gatedLogit (1 / 100000) y 0 1 2) 0 := by
  exact ⟨by norm_num, analyticAt_gatedLogit (by norm_num) 0 0 1 2⟩

example : (0 : ℝ) < 1 / 100000 ∧
    AnalyticOnNhd ℝ (gatedLoss (K := Fin 1) (H := Fin 2) (1 / 100000)
      (Finset.univ : Finset (Fin 3 × Fin 3)) (fun r => r.1)) Set.univ := by
  exact ⟨by norm_num, analyticOnNhd_gatedLoss (by norm_num) _ _⟩

example : (0 : ℝ) < 1 / 100000 ∧
    ∃ a, 0 ≤ a ∧ a < 1 ∧ SaddleAvoidance.IsLojasiewiczAt
      (gatedLoss (K := Fin 1) (H := Fin 2) (1 / 100000)
        (Finset.univ : Finset (Fin 3 × Fin 3)) (fun r => r.1)) 0 a := by
  exact ⟨by norm_num, exists_isLojasiewiczAt_gatedLoss (by norm_num) _ _ 0⟩

-- Width 128 with 43 intermediate channels is close to the original budget.
example : Module.finrank ℝ (GatedWeights (Fin 97) (Fin 64) (Fin 43)) = 28928 := by
  rw [finrank_gatedWeights]
  norm_num

end Transformer.Landscape.Neural
