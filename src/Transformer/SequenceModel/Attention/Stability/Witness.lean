module

public import Transformer.SequenceModel.Attention.Stability.Stack

/-!
# Concrete nonempty witnesses for the stack certificate

Source: `review/flat_spectrum.md`, *Full-stack rounding certificates*, B9.
This synthetic source-shaped model has three distinct unit embeddings and
two identity residual blocks. RMS epsilon 3/4 is chosen to make its input
normalization exact; this example does not use the float32 dtype epsilon.
Exact source RoPE is enabled throughout; p=8 and s=5.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Stability

open CRASP GPTMini

/-- Three distinct raw tied embeddings in a four-dimensional residual. -/
def exampleEmbedding (v : Fin witnessConfig.vocab_size) : EucSpace witnessConfig.d_model :=
  WithLp.toLp 2 (fun c => if c.val = v.val then 1 else 0)

/-- Zero branch projections give identity residual blocks, retaining all source sites. -/
def exampleBlock : BlockParams witnessConfig := ⟨0, 0, 0, 0, 0⟩

/-- A nonconstant token-identity classifier, with three output classes and two blocks. -/
def exampleModel : ModelParams witnessConfig := ⟨exampleEmbedding, fun _ => exampleBlock⟩

/-- B9: all three embeddings have unit norm. -/
theorem example_embedding_norm (v : Fin witnessConfig.vocab_size) : ‖exampleEmbedding v‖ = 1 := by
  fin_cases v <;>
    norm_num [exampleEmbedding, witnessConfig, EuclideanSpace.norm_eq, Fin.sum_univ_succ] <;> decide

/-- B9: the specified positive RMS epsilon fixes these unit vectors. -/
theorem example_embedding_rms (v : Fin witnessConfig.vocab_size) :
    rmsNormEps (3 / 4) (exampleEmbedding v) = exampleEmbedding v := by
  rw [rmsNormEps, example_embedding_norm]
  norm_num [witnessConfig]

/-- B9: all coordinates lie strictly inside the fixed-point range. -/
theorem example_embedding_range (v : Fin witnessConfig.vocab_size) (c : Fin witnessConfig.d_model) :
    Fx.NoSaturation 8 5 (exampleEmbedding v c) := by
  simp only [exampleEmbedding, PiLp.toLp_apply]
  split <;> norm_num [Fx.NoSaturation]

/-- B9: the integer-valued embeddings are represented exactly by these bits. -/
theorem example_embedding_round (v : Fin witnessConfig.vocab_size) :
    Normalized.vector (fun c => Fx.round 8 5 (exampleEmbedding v c)) = exampleEmbedding v := by
  ext c
  change (Fx.round 8 5 (exampleEmbedding v c)).val = exampleEmbedding v c
  simp only [exampleEmbedding, PiLp.toLp_apply]
  split <;> norm_num [Fx.round, Fx.clamp, Fx.val]

/-- B9: every Q/K/V component vanishes, irrespective of the residual. -/
theorem example_component (eps : ℝ)
    (part : Fin witnessConfig.d_model → Fin (3 * witnessConfig.d_model))
    (x : EucSpace witnessConfig.d_model) (h : Fin witnessConfig.n_heads) :
    Normalized.component witnessConfig exampleBlock eps part x h = 0 := by
  ext c
  simp [Normalized.component, exampleBlock]

/-- B9: each exponential coefficient is one, at arbitrary exact phases. -/
theorem example_weight (eps : ℝ) (a : ℕ → Fin (witnessConfig.head_dim / 2) → ℝ)
    (r q : Rotary.State witnessConfig 8 5) (h : Fin witnessConfig.n_heads) (i j : ℕ) :
    Rotary.sourceWeight witnessConfig exampleBlock eps a r q h i j = 1 := by
  have hz (b : Fin (witnessConfig.head_dim / 2) → ℝ) :
      rotateHead witnessConfig.head_dim b 0 = 0 :=
    norm_eq_zero.mp (by rw [rotateHead_isometry, norm_zero])
  simp only [Rotary.sourceWeight, example_component]
  simp [headScore, rmsNormEps, hz]

/-- B9: coefficient and quotient range hypotheses hold at every finite prefix. -/
theorem example_aggregate_range (eps : ℝ) (a : ℕ → Fin (witnessConfig.head_dim / 2) → ℝ)
    (q : ℕ → Rotary.State witnessConfig 8 5) (i : ℕ) :
    AggregateInRange witnessConfig exampleBlock eps 8 5 a q i := by
  intro c
  simp only [example_weight, example_component, PiLp.zero_apply, mul_zero]
  constructor
  · intro j hj
    norm_num [Fx.NoSaturation]
  · simp [roundedRatio, Fx.round_zero, Fx.NoSaturation]

/-- B9: the two residual branches are identically zero, including their normalizers. -/
theorem example_local (eps xeps : ℝ) (first : Bool) (x : EucSpace witnessConfig.d_model)
    (avg : Fin witnessConfig.n_heads → EucSpace witnessConfig.head_dim) :
    Normalized.localUpdate witnessConfig exampleBlock eps xeps first x avg = x := by
  simp [Normalized.localUpdate, exampleBlock, feedForward, rmsNormEps]

/-- B9: storing an already represented residual introduces no new error in this model. -/
theorem example_step (eps xeps : ℝ) (a : ℕ → Fin (witnessConfig.head_dim / 2) → ℝ)
    (q : ℕ → Rotary.State witnessConfig 8 5) (i : ℕ) :
    (Rotary.sourceLayer witnessConfig exampleBlock eps xeps 8 5 a).step q i = q i := by
  apply Prod.ext
  · rfl
  · funext c
    change Fx.round 8 5 (Normalized.localUpdate witnessConfig exampleBlock eps xeps
      (q i).1 (Normalized.vector (q i).2) _ c) = (q i).2 c
    rw [example_local]
    exact Fx.round_val 8 5 _

/-- B9: the configured stack keeps its actual input state at every position. -/
theorem example_counted {T : ℕ} (tokens : Fin T → Fin witnessConfig.vocab_size) (n i : ℕ) :
    counted witnessConfig exampleModel (3 / 4) 1 8 5 (fun _ => true) tokens n i =
      Rotary.input (Rotary.embed witnessConfig exampleModel (3 / 4) 8 5)
        (false, fun _ => 0) (List.ofFn tokens) i := by
  induction n with
  | zero => rfl
  | succ n ih =>
      change (Rotary.sourceLayer witnessConfig exampleBlock (3 / 4) 1 8 5 _).step
        (counted witnessConfig exampleModel (3 / 4) 1 8 5 (fun _ => true) tokens n) i = _
      rw [example_step]
      exact ih

/-- B9: the complete actual rounded trajectory satisfies every range premise. -/
theorem example_trace {T : ℕ} (tokens : Fin T → Fin witnessConfig.vocab_size) :
    TraceInRange witnessConfig exampleModel (3 / 4) 1 8 5 (fun _ => true) tokens := by
  constructor
  · intro i c
    change Fx.NoSaturation 8 5 (rmsNormEps (3 / 4) (exampleEmbedding (tokens i)) c)
    rw [example_embedding_rms]
    exact example_embedding_range _ _
  · intro l i
    constructor
    · exact example_aggregate_range _ _ _ _
    · intro c
      change Fx.NoSaturation 8 5 (Normalized.localUpdate witnessConfig exampleBlock _ _ _ _ _ c)
      rw [example_local, example_counted, input_ofFn]
      change Fx.NoSaturation 8 5 (Normalized.vector
        (fun c => Fx.round 8 5 (rmsNormEps (3 / 4) (exampleEmbedding (tokens i)) c)) c)
      rw [example_embedding_rms, example_embedding_round]
      exact example_embedding_range _ _

/-- B9: five fractional bits satisfy the source attention underflow safeguard. -/
theorem example_precision :
    1 / (2 : ℝ) ^ 5 < Real.exp (-Real.sqrt (witnessConfig.head_dim : ℝ)) := by
  have he : Real.exp 2 < (32 : ℝ) := by
    rw [show (2 : ℝ) = 1 + 1 by norm_num, Real.exp_add]
    nlinarith [Real.exp_one_lt_three, Real.exp_pos 1]
  have hi := one_div_lt_one_div_of_lt (Real.exp_pos 2) he
  convert hi using 1 <;> norm_num [witnessConfig, Config.head_dim, Real.exp_neg]

end Transformer.SequenceModel.Attention.Stability
