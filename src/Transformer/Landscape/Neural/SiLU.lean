/-
# The tied Rotate90 classifier with SiLU

Source: `grokking_abmod/web/own-channel/README.md`, Architecture and
initialization, revision f49969393a7482490d45483d8a4fd3046b1adede; the exact
formula is recorded in `landscape_grokking_plan.md`, Model and protocol.

The sole architectural change is ReLU -> SiLU. All E and A entries are free
Euclidean coordinates. Input lookup and readout share the very same E.
The loss retains both normalizations, the skip and every competing class.
The labels and split are arbitrary fixed data, so the result applies to
the browser's division table without dropping its zero residue.
-/

import Transformer.Landscape.Neural.Residual
import Transformer.Landscape.SaddleAvoidance.LocalInequality

namespace Transformer.Landscape.Neural

/-- The distinct scalar entries of E and A. -/
abbrev SiluIndex (V K : Type*) := (V × Channels K) ⊕ (Channels K × Channels K)

/-- Exactly one trainable E and one trainable A, with the standard Euclidean
parameter metric used by gradient descent. -/
abbrev SiluWeights (V K : Type*) := EuclideanSpace ℝ (SiluIndex V K)

variable {V K : Type*} [Fintype V] [Fintype K]

/-- E's entries are coordinates of the complete parameter vector. -/
def siluEmbedding (weights : SiluWeights V K) (q : V) (i : Channels K) : ℝ :=
  weights.ofLp (.inl (q, i))

/-- A's entries are distinct coordinates of that same parameter vector. -/
def siluMatrix (weights : SiluWeights V K) (i j : Channels K) : ℝ :=
  weights.ofLp (.inr (i, j))

/-- The original skip plus the normalized dense branch, replacing ReLU
with SiLU (Model and protocol, `x1`). -/
noncomputable def siluState (eps : ℝ) (weights : SiluWeights V K) (a b : V)
    (i : Channels K) : ℝ :=
  let v := tiedInput (siluEmbedding weights) a b
  v i + silu (dense (siluMatrix weights) (rmsNorm eps v) i)

/-- Final RMSNorm followed by the transposed shared E, without a new
readout parameter (Model and protocol, `logit[q]`). -/
noncomputable def siluLogit (eps : ℝ) (weights : SiluWeights V K) (a b q : V) : ℝ :=
  tiedReadout (siluEmbedding weights) (rmsNorm eps (siluState eps weights a b)) q

/-- Mean cross-entropy on a specified fixed split. Every answer is included
in each softmax, regardless of which rows occur in that split. -/
noncomputable def siluLoss (eps : ℝ) (rows : Finset (V × V)) (target : V × V → V)
    (weights : SiluWeights V K) : ℝ :=
  meanCrossEntropy rows (fun r q => siluLogit eps weights r.1 r.2 q) target

/-- The standard Euclidean parameter dimension is exactly `|V| W + W^2`,
where `W = 2|K|` (Model and protocol, one E and one A). -/
theorem finrank_siluWeights : Module.finrank ℝ (SiluWeights V K) =
    Fintype.card V * (2 * Fintype.card K) + (2 * Fintype.card K) ^ 2 := by
  change Module.finrank ℝ (EuclideanSpace ℝ (SiluIndex V K)) = _
  rw [finrank_euclideanSpace]
  simp only [SiluIndex, Channels, Fintype.card_sum, Fintype.card_prod, Fintype.card_fin]
  ring

/-- The concrete network's empirical loss has a global lower bound, as
required when studying finite loss limits (M9.2, full cross-entropy). -/
theorem siluLoss_nonneg (eps : ℝ) (rows : Finset (V × V)) (target : V × V → V)
    (weights : SiluWeights V K) : 0 ≤ siluLoss eps rows target weights := by
  exact meanCrossEntropy_nonneg rows _ target

/-- Analyticity of the actual residual state in every E and A coordinate;
no analytic-network hypothesis is assumed (M9.2 of the research plan). -/
theorem analyticAt_siluState {eps : ℝ} (heps : 0 < eps) (x : SiluWeights V K)
    (a b : V) (i : Channels K) :
    AnalyticAt ℝ (fun y => siluState eps y a b i) x := by
  have hE : ∀ q i, AnalyticAt ℝ (fun y : SiluWeights V K => siluEmbedding y q i) x :=
    fun q i => by simpa only [siluEmbedding] using
      (analyticAt_parameter (I := SiluIndex V K) (.inl (q, i)) x)
  have hA : ∀ i j, AnalyticAt ℝ (fun y : SiluWeights V K => siluMatrix y i j) x :=
    fun i j => by simpa only [siluMatrix] using
      (analyticAt_parameter (I := SiluIndex V K) (.inr (i, j)) x)
  have hv := fun i => analyticAt_tiedInput hE a b i
  have hr := fun i => analyticAt_rmsNorm heps hv i
  exact (hv i).fun_add (analyticAt_silu_comp (analyticAt_dense hA hr i))

/-- The final normalized tied readout is analytic in those same weights
(M9.2; both input occurrences and the readout occurrence of E remain live). -/
theorem analyticAt_siluLogit {eps : ℝ} (heps : 0 < eps) (x : SiluWeights V K)
    (a b q : V) : AnalyticAt ℝ (fun y => siluLogit eps y a b q) x := by
  have hE : ∀ r i, AnalyticAt ℝ (fun y : SiluWeights V K => siluEmbedding y r i) x :=
    fun r i => by simpa only [siluEmbedding] using
      (analyticAt_parameter (I := SiluIndex V K) (.inl (r, i)) x)
  exact analyticAt_tiedReadout hE
    (fun i => analyticAt_rmsNorm heps (analyticAt_siluState heps x a b) i) q

/-- The SiLU variant's empirical loss is analytic everywhere in its actual
Euclidean parameters, for any fixed labels and split (M9.2). -/
theorem analyticOnNhd_siluLoss {eps : ℝ} (heps : 0 < eps)
    (rows : Finset (V × V)) (target : V × V → V) :
    AnalyticOnNhd ℝ (siluLoss (K := K) eps rows target) Set.univ := by
  intro x hx
  exact analyticAt_meanCrossEntropy rows
    (fun r _ q => analyticAt_siluLogit heps x r.1 r.2 q) target

/-- The proved Lojasiewicz theorem now applies at every parameter vector of
this concrete network. This discharges the local inequality, not a claim
that its empirical minima generalize (M9.2 and Existing evidence). -/
theorem exists_isLojasiewiczAt_siluLoss {eps : ℝ} (heps : 0 < eps)
    (rows : Finset (V × V)) (target : V × V → V) (x : SiluWeights V K) :
    ∃ a, 0 ≤ a ∧ a < 1 ∧
      SaddleAvoidance.IsLojasiewiczAt (siluLoss eps rows target) x a := by
  exact SaddleAvoidance.exists_isLojasiewiczAt_of_analyticAt
    (analyticOnNhd_siluLoss heps rows target x (Set.mem_univ x))

/- Witnesses use positive epsilon, three answer classes, nonempty equation
sets and two actual channels. All trainable weights are still variables of
the analytic functions, including when evaluated at the zero checkpoint. -/

example : (0 : ℝ) < 1 / 100000 ∧
    AnalyticAt ℝ (fun y : SiluWeights (Fin 3) (Fin 1) =>
      siluState (1 / 100000) y 0 1 (0, 0)) 0 := by
  exact ⟨by norm_num, analyticAt_siluState (by norm_num) 0 0 1 (0, 0)⟩

example : (0 : ℝ) < 1 / 100000 ∧
    AnalyticAt ℝ (fun y : SiluWeights (Fin 3) (Fin 1) =>
      siluLogit (1 / 100000) y 0 1 2) 0 := by
  exact ⟨by norm_num, analyticAt_siluLogit (by norm_num) 0 0 1 2⟩

example : (0 : ℝ) < 1 / 100000 ∧
    AnalyticOnNhd ℝ (siluLoss (K := Fin 1) (1 / 100000)
      (Finset.univ : Finset (Fin 3 × Fin 3)) (fun r => r.1)) Set.univ := by
  exact ⟨by norm_num, analyticOnNhd_siluLoss (by norm_num) _ _⟩

example : (0 : ℝ) < 1 / 100000 ∧
    ∃ a, 0 ≤ a ∧ a < 1 ∧ SaddleAvoidance.IsLojasiewiczAt
      (siluLoss (K := Fin 1) (1 / 100000) (Finset.univ : Finset (Fin 3 × Fin 3))
        (fun r => r.1)) 0 a := by
  exact ⟨by norm_num, exists_isLojasiewiczAt_siluLoss (by norm_num) _ _ 0⟩

-- At the browser's width 128, SiLU retains the original 28,800 parameters.
example : Module.finrank ℝ (SiluWeights (Fin 97) (Fin 64)) = 28800 := by
  rw [finrank_siluWeights]
  norm_num

example : 0 ≤ siluLoss (K := Fin 1) (1 / 100000)
    (Finset.univ : Finset (Fin 3 × Fin 3)) (fun r => r.1) 0 := by
  exact siluLoss_nonneg _ _ _ _

-- A zero shared embedding produces zero logits; there is no hidden bias.
example : siluLogit (1 / 100000) (0 : SiluWeights (Fin 3) (Fin 1)) 0 1 2 = 0 := by
  simp [siluLogit, tiedReadout, siluEmbedding]

end Transformer.Landscape.Neural
