module

public import Transformer.Residual.Basic
public import Mathlib.Analysis.SpecialFunctions.Exp

/-!
# Full Attention Residuals

arXiv:2603.15031v1, §3.1, Eqs. (2)–(4); Kimi K3 technical report,
§2.2, Eqs. (8)–(9), downloaded as `papers/Kimi-K3/k3_tech_report.pdf`.
Keys are RMS-normalized, values are the unnormalized layer outputs, and
the query is a learned vector independent of the current hidden state.
The statements concern aggregation, with no assumptions about training.
-/

@[expose] public section

open scoped BigOperators
open Transformer.Residual

namespace Transformer.AttentionResiduals

variable {ι : Type*} [Fintype ι]
variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- Depth-wise softmax normalizer. Source: arXiv:2603.15031v1, Eq. (2). -/
noncomputable def depthPartition (scores : ι → ℝ) : ℝ := ∑ i, Real.exp (scores i)

/-- Content-dependent weight on a preceding layer or block.
Source: arXiv:2603.15031v1, Eq. (2). -/
noncomputable def depthWeight (scores : ι → ℝ) (i : ι) : ℝ :=
  Real.exp (scores i) / depthPartition scores

/-- An attention aggregate for supplied scores and unnormalized values.
Source: arXiv:2603.15031v1, Eq. (4). -/
noncomputable def attend (scores : ι → ℝ) (values : ι → E) : E :=
  weightedSum (depthWeight scores) values

/-- RMS scale used only for keys. The gain is the learned normalization
weight; positive `ε` is the numerical stabilizer. Sources:
arXiv:2603.15031v1, §3.1; Kimi K3 `_apply_attn_res` in the pinned
`papers/Kimi-K3/modeling_kimi_linear.py`. -/
noncomputable def keyRMS {d : ℕ} (ε : ℝ) (v : Fin d → ℝ) : ℝ :=
  Real.sqrt ((∑ k, v k ^ 2) / (d : ℝ) + ε)

/-- The score is the learned query dotted with an RMS-normalized key.
Source: arXiv:2603.15031v1, §3.1, kernel `φ` before Eq. (2).
The query and gain may differ for attention and MLP sublayers. -/
noncomputable def depthScore {d : ℕ} (ε : ℝ) (gain query v : Fin d → ℝ) : ℝ :=
  ∑ k, query k * (gain k * v k / keyRMS ε v)

/-- Full AttnRes, with the embedding included among `values`.
Sources: arXiv:2603.15031v1, Eqs. (3)–(4); Kimi K3, §2.2, Eq. (9).
Only keys are normalized; the original values are aggregated. -/
noncomputable def attentionResidual {d : ℕ} (ε : ℝ) (gain query : Fin d → ℝ)
    (values : ι → (Fin d → ℝ)) : Fin d → ℝ :=
  attend (fun i => depthScore ε gain query (values i)) values

/-- A positive stabilizer makes every key denominator positive, even for
a zero key. This is the RMSNorm convention of Kimi K3's exported code. -/
theorem keyRMS_pos {d : ℕ} (ε : ℝ) (hε : 0 < ε) (v : Fin d → ℝ) :
    0 < keyRMS ε v := by
  unfold keyRMS
  positivity

/-- The key-normalization hypothesis holds for the concrete stabilizer one. -/
example : (0 : ℝ) < 1 := by norm_num

/-- The softmax normalizer is strictly positive whenever an embedding or
preceding output is present. Source: arXiv:2603.15031v1, Eq. (2). -/
theorem depthPartition_pos [Nonempty ι] (scores : ι → ℝ) :
    0 < depthPartition scores := by
  exact Finset.sum_pos (fun i _ => Real.exp_pos (scores i)) Finset.univ_nonempty

/-- All finite-score source weights are strictly positive.
Source: arXiv:2603.15031v1, Eq. (2). -/
theorem depthWeight_pos [Nonempty ι] (scores : ι → ℝ) (i : ι) :
    0 < depthWeight scores i := by
  exact div_pos (Real.exp_pos _) (depthPartition_pos scores)

/-- The attention weights sum to one, rather than to the source count.
Source: arXiv:2603.15031v1, §3, normalized `α`. -/
theorem depthWeight_sum [Nonempty ι] (scores : ι → ℝ) :
    ∑ i, depthWeight scores i = 1 := by
  simp only [depthWeight]
  rw [← Finset.sum_div]
  exact div_self (depthPartition_pos scores).ne'

/-- Aggregation cannot exceed a common bound on its source norms.
Source: arXiv:2603.15031v1, §3.1, convex attention formula. This does not
bound norms of the transformations producing the sources. -/
theorem attend_norm_le [Nonempty ι] (scores : ι → ℝ) (values : ι → E)
    (R : ℝ) (hv : ∀ i, ‖values i‖ ≤ R) : ‖attend scores values‖ ≤ R := by
  exact weightedSum_convex_norm_le _ values R
    (fun i => (depthWeight_pos scores i).le) (depthWeight_sum scores) hv

/-- A nonzero source family realizes the aggregation norm hypothesis. -/
example : ∀ i : Fin 3, ‖(fun _ : Fin 3 => (2 : ℝ)) i‖ ≤ 2 := by
  norm_num

/-- Normalized attention on identical sources returns that source exactly.
Source: arXiv:2603.15031v1, Eq. (4), normalization consequence. -/
theorem attend_const [Nonempty ι] (scores : ι → ℝ) (v : E) :
    attend scores (fun _ => v) = v := by
  unfold attend
  rw [weightedSum_const, depthWeight_sum, one_smul]

/-- Zero logits give uniform averaging, not the standard residual sum.
Source: arXiv:2603.15031v1, Eq. (2), specialized to equal logits. -/
theorem depthWeight_zero (i : ι) :
    depthWeight (fun _ : ι => 0) i = 1 / (Fintype.card ι : ℝ) := by
  simp [depthWeight, depthPartition]

/-- Subtracting a common logit maximum leaves attention unchanged. This
connects the source's mathematical softmax to the shifted evaluation in
Algorithm 1. Source: arXiv:2603.15031v1, §§3.1 and 4.2. -/
theorem depthWeight_add_const (scores : ι → ℝ) (c : ℝ) (i : ι) :
    depthWeight (fun j => scores j + c) i = depthWeight scores i := by
  have hp : depthPartition (fun j => scores j + c) =
      depthPartition scores * Real.exp c := by
    unfold depthPartition
    simp only [Real.exp_add, Finset.sum_mul]
  unfold depthWeight
  rw [Real.exp_add, hp]
  exact mul_div_mul_right _ _ (Real.exp_pos c).ne'

/-- A zero pseudo-query gives zero scores regardless of key magnitudes.
Source: arXiv:2603.15031v1, §3.1, learned pseudo-query. -/
theorem depthScore_zero_query {d : ℕ} (ε : ℝ) (gain v : Fin d → ℝ) :
    depthScore ε gain (fun _ => 0) v = 0 := by
  simp [depthScore]

/-- The zero-query specialization is an average over all sources, retaining
the actual unnormalized values. Source: arXiv:2603.15031v1, Eqs. (2)–(4). -/
theorem attentionResidual_zero_query {d : ℕ} (ε : ℝ) (gain : Fin d → ℝ)
    (values : ι → (Fin d → ℝ)) :
    attentionResidual ε gain (fun _ => 0) values =
      (1 / (Fintype.card ι : ℝ)) • ∑ i, values i := by
  simp only [attentionResidual, depthScore_zero_query, attend,
    weightedSum, depthWeight_zero, Finset.smul_sum]

/-- The norm bound for the actual query/RMSNorm formula. The softmax
weights are input-dependent; no global Lipschitz claim is made.
Sources: arXiv:2603.15031v1, §3.1; Kimi K3, §2.2, Eq. (9). -/
theorem attentionResidual_norm_le [Nonempty ι] {d : ℕ} (ε : ℝ)
    (gain query : Fin d → ℝ) (values : ι → (Fin d → ℝ)) (R : ℝ)
    (hv : ∀ i, ‖values i‖ ≤ R) : ‖attentionResidual ε gain query values‖ ≤ R := by
  exact attend_norm_le _ values R hv

/-- The actual-query theorem has a nonzero concrete source witness. -/
example : ∀ i : Fin 3, ‖(fun _ : Fin 3 => (fun _ : Fin 1 => (1 : ℝ))) i‖ ≤ 1 := by
  intro i
  norm_num

end Transformer.AttentionResiduals
