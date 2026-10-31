module

public import Transformer.CRASP.Normalized.Program
public import Transformer.SequenceModel.Attention.Model

/-!
# A rounded counting model of the normalized source block

Source: `diff_gpt/domain/model/{block,mha,feed_forward,rope}.py` at `b98c665b`,
and `review/flat_spectrum.md`, *Depth, width and precision*, R4.
This extends arXiv:2506.16055v3, B.2/F to multiple heads, half-split RoPE,
RMS Q/K, XSA, four branch normalizers and SwiGLU. Deviations are explicit:
round exponential coefficients and weighted value coordinates before summing,
round the attention quotient, and round the stored residual after each block.
Local maps use real arithmetic. This is not an IEEE floating-point emulator.
Positions lie in a fixed finite context; the sentinel contributes zero.
-/

@[expose] public section

noncomputable section

namespace Transformer.CRASP.Normalized

open GPTMini SequenceModel.Attention

/-- Stored residuals, with finite position tags; `none` is an excluded sentinel. -/
abbrev State (cfg : Config) (T p s : ℕ) :=
  Option (Fin T × (Fin cfg.d_model → Fx p s))

/-- Every head coordinate has its own rounded numerator and denominator table. -/
abbrev Coordinate (cfg : Config) := Fin cfg.n_heads × Fin cfg.head_dim

/-- Interpret the stored fixed-point residual in the source's Euclidean space. -/
def vector {d p s : ℕ} (x : Fin d → Fx p s) : EucSpace d :=
  WithLp.toLp 2 (fun c => (x c).val)

/-- The source's pre-normalized fused-QKV projection and head reshape. -/
def component (cfg : Config) (B : BlockParams cfg) (eps : ℝ)
    (part : Fin cfg.d_model → Fin (3 * cfg.d_model))
    (x : EucSpace cfg.d_model) (h : Fin cfg.n_heads) :
    EucSpace cfg.head_dim :=
  headSlice cfg (qkvSlice cfg part (B.qkv (rmsNormEps eps x))) h

/-- All local source operations after the softmax-weighted value averages. -/
def localUpdate (cfg : Config) (B : BlockParams cfg) (eps xeps : ℝ)
    (first : Bool) (x : EucSpace cfg.d_model)
    (average : Fin cfg.n_heads → EucSpace cfg.head_dim) : EucSpace cfg.d_model :=
  let heads := fun h =>
    let u := normL2 xeps (component cfg B eps (qkvV cfg) x h)
    if first then 0 else average h - (inner (𝕜 := ℝ) (average h) u) • u
  let x1 := x + rmsNormEps eps (B.projection (headMerge cfg heads))
  x1 + rmsNormEps eps (feedForward cfg B (rmsNormEps eps x1))

/-- Unrounded averages from the actual source logits; XSA is applied afterwards. -/
def sourceAverage (cfg : Config) (B : BlockParams cfg) (eps : ℝ) {T : ℕ}
    (angle : Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (x : Fin T → EucSpace cfg.d_model) (i : Fin T) (h : Fin cfg.n_heads) :
    EucSpace cfg.head_dim :=
  ∑ j, causalWeight (fun a b => headScore cfg.head_dim eps (angle a) (angle b)
    (component cfg B eps (qkvQ cfg) (x a) h)
    (component cfg B eps (qkvK cfg) (x b) h)) i j •
      component cfg B eps (qkvV cfg) (x j) h

/-- Factoring the actual source block leaves every normalizer and XSA's first
position rule in place. This is a definitional identity, not a rounding claim. -/
theorem local_sourceAverage (cfg : Config) (B : BlockParams cfg) (eps xeps : ℝ)
    {T : ℕ} (angle : Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (x : Fin T → EucSpace cfg.d_model) (i : Fin T) :
    localUpdate cfg B eps xeps (decide (i.val = 0)) (x i)
      (sourceAverage cfg B eps angle x i) = block cfg B eps xeps angle x i := by
  simp only [localUpdate, sourceAverage, component, block, attention, headOutput,
    decide_eq_true_eq]

/-- Positive softmax weights use the source RMS Q/K and arbitrary half-split phases. -/
def weight (cfg : Config) (B : BlockParams cfg) (eps : ℝ) {T p s : ℕ}
    (angle : Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (q a : Fin T × (Fin cfg.d_model → Fx p s)) (h : Fin cfg.n_heads) : ℝ :=
  Real.exp (headScore cfg.head_dim eps (angle q.1) (angle a.1)
    (component cfg B eps (qkvQ cfg) (vector q.2) h)
    (component cfg B eps (qkvK cfg) (vector a.2) h))

/-- A concrete counted layer retaining the source local computation, with the
fixed-point rounding sites specified in this module's docstring. -/
def sourceLayer (cfg : Config) (B : BlockParams cfg) (eps xeps : ℝ) {T : ℕ}
    (p s : ℕ) (angle : Fin T → Fin (cfg.head_dim / 2) → ℝ) :
    Layer (State cfg T p s) (Coordinate cfg) p s where
  denominator q c a := match q, a with
    | some q, some a => Fx.round p s (weight cfg B eps angle q a c.1)
    | _, _ => 0
  numerator q c a := match q, a with
    | some q, some a => Fx.round p s (weight cfg B eps angle q a c.1 *
        component cfg B eps (qkvV cfg) (vector a.2) c.1 c.2)
    | _, _ => 0
  finish q y := match q with
    | none => none
    | some q => some (q.1, fun c => Fx.round p s
        (localUpdate cfg B eps xeps (decide (q.1.val = 0)) (vector q.2)
          (fun h => vector (fun j => y (h, j))) c))

/-- The required counting-cell hypothesis follows from the source exponential,
even with normalization, arbitrary phases and negative value coordinates. -/
theorem sourceLayer_nonnegative (cfg : Config) (B : BlockParams cfg) (eps xeps : ℝ)
    {T : ℕ} (p s : ℕ) (angle : Fin T → Fin (cfg.head_dim / 2) → ℝ) :
    (sourceLayer cfg B eps xeps p s angle).Nonnegative := by
  intro q c a
  cases q with
  | none => simp [sourceLayer]
  | some q =>
      cases a with
      | none => simp [sourceLayer]
      | some a => exact Fx.m_round_nonneg _ (Real.exp_nonneg _)

/-- A stack of the rounded source blocks has counting depth at most its layer
count above its initial state predicates. This includes all heads in parallel. -/
theorem source_stack_mem (cfg : Config) (B : ℕ → BlockParams cfg) (eps xeps : ℝ)
    {T : ℕ} (p s : ℕ) (angle : ℕ → Fin T → Fin (cfg.head_dim / 2) → ℝ)
    {σ : Type*} (initial : State cfg T p s → FormP σ) {k : ℕ}
    (hinit : ∀ r, initial r ∈ TLClMod σ k) (n : ℕ) (r : State cfg T p s) :
    formula (fun l => sourceLayer cfg (B l) eps xeps p s (angle l)) none initial n r ∈
      TLClMod σ (k + n) := formula_mem _ _ _ hinit n r

/-- Source stack simulation is exact for this specified rounding model, with
no representability hypothesis on the real weights or local nonlinearities. -/
theorem source_stack_sat (cfg : Config) (B : ℕ → BlockParams cfg) (eps xeps : ℝ)
    {T : ℕ} (p s : ℕ) (angle : ℕ → Fin T → Fin (cfg.head_dim / 2) → ℝ)
    {σ : Type*} [DecidableEq σ] (initial : State cfg T p s → FormP σ)
    (q : ℕ → State cfg T p s) (w : List σ) (n : ℕ) {i : ℕ} (hi : 0 < i)
    (hinit : ∀ j ∈ Finset.Icc 1 i, ∀ a, (initial a).sat w j = decide (q j = a))
    (r : State cfg T p s) :
    (formula (fun l => sourceLayer cfg (B l) eps xeps p s (angle l)) none initial n r).sat w i =
      decide (run (fun l => sourceLayer cfg (B l) eps xeps p s (angle l)) none q n i = r) :=
  sat_formula _ _ (fun l => sourceLayer_nonnegative cfg (B l) eps xeps p s (angle l))
    initial q w n hi hinit r

/-- Initial hypotheses can use the actual tokenwise normalized, rounded
embedding; finite tags carry arbitrary source phase tables at fixed context. -/
example (E : Fin witnessConfig.vocab_size → EucSpace witnessConfig.d_model) :
    ∀ r : State witnessConfig 8 8 4,
      initialFormula (fun a : Fin 8 × Fin witnessConfig.vocab_size =>
        some (a.1, fun c => Fx.round 8 4 (rmsNormEps (1 / 8388608) (E a.2) c))) r ∈
        TLClMod (Fin 8 × Fin witnessConfig.vocab_size) 0 := initialFormula_mem _

/-- Width and stored precision explicitly determine the number of residual
states, in addition to the finite position tag and excluded sentinel. -/
theorem card_state (cfg : Config) (T p s : ℕ) :
    Fintype.card (State cfg T p s) = T * (2 * 2 ^ (p - 1)) ^ cfg.d_model + 1 := by
  simp only [State, Fintype.card_option, Fintype.card_prod, Fintype.card_fin,
    Fintype.card_fun, Fx.card_eq]

/-- With a positive bit budget, W stored coordinates have exactly 2^(pW)
states. This is state size, not a bound on the number of learned functions. -/
theorem card_state_bits (cfg : Config) (T p s : ℕ) (hp : 0 < p) :
    Fintype.card (State cfg T p s) = T * 2 ^ (p * cfg.d_model) + 1 := by
  rw [card_state, Nat.pow_mul]
  have h : 2 * 2 ^ (p - 1) = 2 ^ p := by
    rw [mul_comm, ← pow_succ]
    congr 1
    omega
  rw [h]

/-- Concrete state counts expose the width/precision cost in the source witness. -/
example : Fintype.card (State witnessConfig 8 8 4) = 8 * 2 ^ 32 + 1 :=
  card_state_bits witnessConfig 8 8 4 (by decide)


end Transformer.CRASP.Normalized
