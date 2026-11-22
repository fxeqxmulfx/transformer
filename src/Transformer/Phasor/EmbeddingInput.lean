module

public import Transformer.Phasor.LearnedStreamRead

/-!
# Phasor with an embedding-sequence input

Source: the embedding-input interface in `experiments/phasor_value_transport/README.md`.
A backbone contains the rotary parameters and layers. Its forward arguments are
vectors, including vectors outside any token embedding table. The embedding lookup
precedes this map. The tied classifier follows it. Training targets enter the loss.

The state at any depth is a function of the vector prefix. Factoring the earlier
token-level forward through this interface preserves every logit, including when
distinct token identities map to identical vectors.
-/

@[expose] public section

namespace Transformer.Phasor

open Transformer Transformer.MHC.PeriRMS

variable {D d H F V : ℕ}

/-- Rotary scales and layers of a Phasor backbone. Its type has no vocabulary
parameter, embedding table or target space. -/
structure EmbeddingBackbone (D d H F : ℕ) where
  alpha : Fin (d / 2) → ℝ
  layer : ℕ → Layer D d H F

/-- Forward states on an arbitrary sequence of real embedding vectors. -/
noncomputable def EmbeddingBackbone.states (A : EmbeddingBackbone D d H F)
    (eps scale : ℝ) (input : ℕ → EucSpace D) : ℕ → ℕ → State D
  | 0 => fun i => embeddingState eps (input i)
  | l + 1 => fun i => (A.layer l).apply eps scale A.alpha (A.states eps scale input l) i

/-- A sequence of vectors, a position and a depth determine the hidden feature
read by the external classifier. -/
noncomputable def EmbeddingBackbone.feature (A : EmbeddingBackbone D d H F)
    (θ : StreamReadWeights D) (eps scale : ℝ) (input : ℕ → EucSpace D) (l i : ℕ) : EucSpace D :=
  GPTMini.rmsNormEps eps (learnedCollapse θ (A.states eps scale input l i))

/-- If embedding prefixes coincide, all four hidden streams at the selected
position coincide after any number of layers. Later vectors may differ. -/
theorem EmbeddingBackbone.states_congr_prefix (A : EmbeddingBackbone D d H F)
    (eps scale : ℝ) (x y : ℕ → EucSpace D) (l i : ℕ) (h : ∀ j ≤ i, x j = y j) :
    A.states eps scale x l i = A.states eps scale y l i := by
  induction l generalizing i with
  | zero => simp [EmbeddingBackbone.states, h i (by omega)]
  | succ l ih =>
    change (A.layer l).apply _ _ _ _ i = (A.layer l).apply _ _ _ _ i
    apply Attention.block_congr
    intro j hj
    exact ih j fun k hk => h k (Nat.le_trans hk hj)

/-- Replacing the entire suffix after position seven leaves its states equal,
for arbitrary real vectors and a width-128 backbone. -/
example (A : EmbeddingBackbone 128 32 4 171) (x fill : ℕ → EucSpace 128) (l : ℕ) :
    A.states (1 / 8388608) 1 x l 7 =
      A.states (1 / 8388608) 1 (fun j => if j ≤ 7 then x j else fill j) l 7 :=
  A.states_congr_prefix _ _ _ _ _ _ (fun j hj => by simp [hj])

/-- Remove the external vocabulary-dependent tables from a trained Phasor. -/
def Model.embeddingBackbone (M : Model D d H F V) : EmbeddingBackbone D d H F :=
  ⟨M.alpha, M.layer⟩

/-- Every layer of the earlier token-level forward factors exactly through the
vector-input backbone applied to the lookup of the tokens. -/
theorem Model.states_factor_embeddings (M : Model D d H F V) (eps scale : ℝ)
    (tokens : ℕ → Fin V) (l : ℕ) :
    M.states eps scale tokens l =
      M.embeddingBackbone.states eps scale (M.embedding ∘ tokens) l := by
  induction l with
  | zero => rfl
  | succ l ih =>
    funext i
    change (M.layer l).apply _ _ _ (M.states _ _ _ l) i = _
    rw [ih]
    rfl

/-- The tied classification score is computed outside the backbone from the
input-indexed embeddings and a class-indexed output vector. -/
theorem Model.streamReadLogit_factor_embeddings (M : Model D d H F V) (θ : StreamReadWeights D)
    (eps scale : ℝ) (tokens : ℕ → Fin V) (l i : ℕ) (v : Fin V) :
    M.streamReadLogit θ eps scale tokens l i v = inner ℝ (M.embedding v)
      (M.embeddingBackbone.feature θ eps scale (M.embedding ∘ tokens) l i) := by
  rw [Model.streamReadLogit, Model.states_factor_embeddings]
  rfl

/-- Even nonbijective aliases of the input names are invisible at any depth
when their embedding vectors agree throughout the observed prefix. -/
theorem Model.streamReadLogit_congr_embedded_prefix (M : Model D d H F V) (θ : StreamReadWeights D)
    (eps scale : ℝ) (t u : ℕ → Fin V) (l i : ℕ) (v : Fin V)
    (h : ∀ j ≤ i, M.embedding (t j) = M.embedding (u j)) :
    M.streamReadLogit θ eps scale t l i v = M.streamReadLogit θ eps scale u l i v := by
  rw [Model.streamReadLogit_factor_embeddings, Model.streamReadLogit_factor_embeddings]
  unfold EmbeddingBackbone.feature
  rw [M.embeddingBackbone.states_congr_prefix eps scale (M.embedding ∘ t) (M.embedding ∘ u) l i h]

/-- With two names for the same vector the example sequences may have
completely different identity and repetition patterns. -/
example (A : EmbeddingBackbone 64 32 2 86) (e : EucSpace 64) (θ : StreamReadWeights 64)
    (t u : ℕ → Fin 2) (l i : ℕ) :
    let M : Model 64 32 2 86 2 := ⟨fun _ => e, A.alpha, A.layer⟩
    M.streamReadLogit θ 1 1 t l i 0 = M.streamReadLogit θ 1 1 u l i 0 := by
  dsimp only
  apply Model.streamReadLogit_congr_embedded_prefix
  intro _ _
  rfl

end Transformer.Phasor
