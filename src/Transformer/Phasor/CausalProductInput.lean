module

public import Transformer.Phasor.EmbeddingInput

/-!
# A causal product at the embedding input

Source: the causal product at the embedding input in
`experiments/phasor_value_transport/README.md`. Before expanding a normalized
embedding into the four streams, add a learned low-rank product of current
and preceding vectors. All four projections are shared across the sequence.
The initial lag is zero at the start of each sequence.

The mixed finite difference isolates the interaction between neighboring
vectors. The adjoint at any finite length includes a contribution from the
next position to the lag projection. A zero output projection recovers
the original backbone at every depth.
-/

@[expose] public section

namespace Transformer.Phasor

open Transformer Transformer.MHC.PeriRMS

variable {D m d H F : ℕ}

/-- The preceding row within a sequence, with zero left padding. -/
def causalLag {E : Type*} [Zero E] (x : ℕ → E) : ℕ → E
  | 0 => 0
  | i + 1 => x i

/-- Three learned input projections and a learned output projection.
The rank is independent of the sequence length and vocabulary. -/
structure CausalProductInput (D m : ℕ) where
  query : EucSpace D →ₗ[ℝ] (Fin m → ℝ)
  current : EucSpace D →ₗ[ℝ] (Fin m → ℝ)
  lag : EucSpace D →ₗ[ℝ] (Fin m → ℝ)
  output : (Fin m → ℝ) →ₗ[ℝ] EucSpace D

/-- A pointwise product of two projected features, added back to the
current normalized vector before it enters the first layer. -/
noncomputable def CausalProductInput.pair (P : CausalProductInput D m)
    (previous x : EucSpace D) : EucSpace D :=
  x + P.output (P.query x * (P.current x + P.lag previous))

/-- The difference of the responses to two independent perturbations, with
all constant, linear and single-position terms removed. It is the learned
bilinear interaction between the current and preceding embeddings. -/
theorem CausalProductInput.mixed_difference (P : CausalProductInput D m)
    (previous x u v : EucSpace D) :
    P.pair (previous + u) (x + v) - P.pair previous (x + v) -
      P.pair (previous + u) x + P.pair previous x =
      P.output (P.query v * P.lag u) := by
  simp only [CausalProductInput.pair, map_add, add_mul, mul_add]
  abel

/-- Exact simultaneous perturbation of the three scalar activations.
The coefficient of the first power gives the row derivative. -/
theorem causalProduct_perturb (a b c da db dc t : ℝ) :
    (a + t*da) * ((b+t*db) + (c+t*dc)) = a*(b+c) +
      t*(da*(b+c) + a*(db+dc)) + t^2*da*(db+dc) := by ring

/-- Shifting a zero-padded input back by one shifts its cotangent forward
and drops the last input coordinate, including for lengths zero and one. -/
theorem causalLag_adjoint (x g : ℕ → ℝ) (n : ℕ) :
    ∑ i ∈ Finset.range n, g i * causalLag x i =
      ∑ i ∈ Finset.range (n-1), g (i+1) * x i := by
  cases n with
  | zero => simp
  | succ n =>
    rw [Finset.sum_range_succ']
    simp [causalLag]

/-- The finite-sequence adjoint of the product. Each current input
projection receives its own row's gradient. The lag projection receives
the gradient from the next row of the same sequence. -/
theorem causalProduct_adjoint (a b c da db dc g : ℕ → ℝ) (n : ℕ) :
    ∑ i ∈ Finset.range n, g i *
        (da i * (b i + causalLag c i) + a i * (db i + causalLag dc i)) =
      (∑ i ∈ Finset.range n, (g i * (b i + causalLag c i)) * da i) +
      (∑ i ∈ Finset.range n, (g i * a i) * db i) +
      ∑ i ∈ Finset.range (n-1), (g (i+1) * a (i+1)) * dc i := by
  rw [← causalLag_adjoint dc (fun i => g i * a i), ← Finset.sum_add_distrib,
    ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro i _
  ring

/-- Phasor on an arbitrary embedding sequence. The input residual acts
after the original parameter-free RMS and before the four streams are
copied. All subsequent layers are the original contextual layers. -/
noncomputable def CausalProductInput.states (P : CausalProductInput D m)
    (A : EmbeddingBackbone D d H F) (eps scale : ℝ) (x : ℕ → EucSpace D) :
    ℕ → ℕ → State D
  | 0 => fun i _ =>
      let z := fun j => GPTMini.rmsNormEps eps (x j)
      P.pair (causalLag z i) (z i)
  | l + 1 => fun i => (A.layer l).apply eps scale A.alpha (P.states A eps scale x l) i

/-- Initialization with a zero output matrix reproduces the original
Phasor states for every real sequence and every depth. -/
theorem CausalProductInput.states_zero_output (P : CausalProductInput D m)
    (A : EmbeddingBackbone D d H F) (eps scale : ℝ) (x : ℕ → EucSpace D) (l : ℕ) :
    ({P with output := 0} : CausalProductInput D m).states A eps scale x l =
      A.states eps scale x l := by
  induction l with
  | zero =>
    funext i s
    simp [CausalProductInput.states, CausalProductInput.pair, EmbeddingBackbone.states,
      embeddingState]
  | succ l ih =>
    funext i
    change (A.layer l).apply _ _ _ _ i = (A.layer l).apply _ _ _ _ i
    rw [ih]

/-- Later embeddings have no effect on any earlier state, for arbitrary
learned parameters and any number of attention and feed-forward layers. -/
theorem CausalProductInput.states_congr_prefix (P : CausalProductInput D m)
    (A : EmbeddingBackbone D d H F) (eps scale : ℝ) (x y : ℕ → EucSpace D)
    (l i : ℕ) (h : ∀ j ≤ i, x j = y j) :
    P.states A eps scale x l i = P.states A eps scale y l i := by
  induction l generalizing i with
  | zero =>
    cases i with
    | zero => simp [CausalProductInput.states, causalLag, h 0 (by omega)]
    | succ i =>
      simp [CausalProductInput.states, causalLag, h i (by omega), h (i+1) (by omega)]
  | succ l ih =>
    change (A.layer l).apply _ _ _ _ i = (A.layer l).apply _ _ _ _ i
    apply Attention.block_congr
    intro j hj
    exact ih j fun k hk => h k (Nat.le_trans hk hj)

example (P : CausalProductInput 64 32) (A : EmbeddingBackbone 64 32 2 86)
    (x y : ℕ → EucSpace 64) (l i : ℕ) :
    P.states A 1 1 x l i = P.states A 1 1 (fun j => if j ≤ i then x j else y j) l i :=
  P.states_congr_prefix _ _ _ _ _ _ _ (fun j hj => by simp [hj])

end Transformer.Phasor
