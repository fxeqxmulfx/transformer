module

public import Transformer.Phasor.PairValues

/-!
# A causal value residual shared by every Phasor configuration

Source: the single-model update in `experiments/phasor_value_transport/README.md`.
At every position the same Phasor feature receives a normalized value read
from its preceding and current tokens. The initial token is its own predecessor.
There is one learned real gain for the update and one tied output projection.
-/

@[expose] public section

namespace Transformer.Phasor

open scoped RealInnerProductSpace
open Transformer

variable {D d H F p : ℕ}

/-- Same hidden update for every vocabulary and position. In a composite
vocabulary the quotient address is the total power map specified by
`PairKey.address`; the update does not require a field. -/
noncomputable def Model.sharedFeature [NeZero p] (M : Model D d H F p) (T : PairTables D p)
    (gain eps scale : ℝ) (tokens : ℕ → Fin p) (L i : ℕ) : EucSpace D :=
  let a := tokens (i - 1)
  let b := tokens i
  let w := Perspective.softmaxWeight (M.logits eps scale tokens L (i - 1))
  M.feature eps scale tokens L i + gain • GPTMini.rmsNormEps eps
    (transportedValue M.embedding w (a*b) + T.read a b)

/-- The single tied projection of the modified hidden state. -/
noncomputable def Model.sharedLogit [NeZero p] (M : Model D d H F p) (T : PairTables D p)
    (gain eps scale : ℝ) (tokens : ℕ → Fin p) (L i : ℕ) (c : Fin p) : ℝ :=
  ⟪M.embedding c, M.sharedFeature T gain eps scale tokens L i⟫

/-- Exact score decomposition for the shared hidden update, including position
zero, long prefixes and composite vocabularies. -/
theorem Model.sharedLogit_eq [NeZero p] (M : Model D d H F p) (T : PairTables D p)
    (gain eps scale : ℝ) (tokens : ℕ → Fin p) (L i : ℕ) (c : Fin p) :
    M.sharedLogit T gain eps scale tokens L i c = M.logits eps scale tokens L i c +
      gain * pairValueLogit M.embedding (Perspective.softmaxWeight (M.logits eps scale tokens L (i-1)))
        eps (tokens (i-1) * tokens i) c (T.read (tokens (i-1)) (tokens i)) := by
  simp only [Model.sharedLogit, Model.sharedFeature, Model.logits, pairValueLogit,
    inner_add_right, real_inner_smul_right]

/-- At each position the residual's projected advantage must overcome the
backbone's adverse score gap. This decision rule uses the full contextual
Phasor feature and the same gain for all classes. -/
theorem Model.sharedLogit_lt_iff [NeZero p] (M : Model D d H F p) (T : PairTables D p)
    (gain eps scale : ℝ) (tokens : ℕ → Fin p) (L i : ℕ) (c y : Fin p) :
    let q := fun v => pairValueLogit M.embedding
      (Perspective.softmaxWeight (M.logits eps scale tokens L (i-1))) eps
      (tokens (i-1) * tokens i) v (T.read (tokens (i-1)) (tokens i))
    M.sharedLogit T gain eps scale tokens L i c < M.sharedLogit T gain eps scale tokens L i y ↔
      M.logits eps scale tokens L i c - M.logits eps scale tokens L i y < gain * (q y - q c) := by
  simp only [Model.sharedLogit_eq, mul_sub]
  constructor <;> intro h <;> linarith

/-- A finite collection of strict value advantages can overcome all of its
contextual score gaps with a single positive scalar. -/
theorem exists_shared_gain {ι : Type*} [Fintype ι] (gap advantage : ι → ℝ)
    (h : ∀ j, 0 < advantage j) : ∃ g : ℝ, 0 < g ∧ ∀ j, gap j < g * advantage j := by
  refine ⟨1 + ∑ j, |gap j / advantage j|, ?_, ?_⟩
  · have := Finset.sum_nonneg (s := Finset.univ) fun j _ => abs_nonneg (gap j / advantage j)
    linarith
  · intro j
    apply (div_lt_iff₀ (h j)).mp
    have hle := Finset.single_le_sum (fun k _ => abs_nonneg (gap k / advantage k)) (Finset.mem_univ j)
    linarith [le_abs_self (gap j / advantage j)]

/-- One gain can overcome adverse margins of opposite signs. -/
example : ∃ g : ℝ, 0 < g ∧ ∀ j : Fin 2, (if j = 0 then 7 else -4) < g * (if j = 0 then 1 else 2) :=
  exists_shared_gain _ _ (fun j => by split <;> norm_num)

/-- One table and one gain separate the entire zero row in the shared model
at fixed depth, using the actual causal score distributions and contextual
features. Every rival on every input uses the same parameters. -/
theorem Model.exists_shared_zero_row [NeZero p] (M : Model D d H F p)
    (hp : Nat.Prime p) (hE : Orthonormal ℝ M.embedding) (hD : 0 < D)
    {eps : ℝ} (heps : 0 < eps) (scale : ℝ) (L : ℕ) :
    let T : PairTables D p := ⟨fun b => (2 : ℝ) • M.embedding b, 0⟩
    let tokens := fun (b : Fin p) (i : ℕ) => if i = 0 then (0 : Fin p) else b
    ∃ g : ℝ, 0 < g ∧ ∀ b c : Fin p, c ≠ b →
      M.sharedLogit T g eps scale (tokens b) L 1 c < M.sharedLogit T g eps scale (tokens b) L 1 b := by
  let T : PairTables D p := ⟨fun b => (2 : ℝ) • M.embedding b, 0⟩
  let xs := fun (b : Fin p) (i : ℕ) => if i = 0 then (0 : Fin p) else b
  let gap := fun s : {z : Fin p × Fin p // z.2 ≠ z.1} =>
    M.logits eps scale (xs s.val.1) L 1 s.val.2 - M.logits eps scale (xs s.val.1) L 1 s.val.1
  let q := fun (b c : Fin p) => M.pairLogits T 2 eps scale (xs b) L 1 c
  let advantage := fun s : {z : Fin p × Fin p // z.2 ≠ z.1} => q s.val.1 s.val.1 - q s.val.1 s.val.2
  have pos (s) : 0 < advantage s := by
    apply sub_pos.mpr
    exact M.pairLogits_zero_left hp hE hD heps scale (xs s.val.1) rfl L s.val.2 s.property
  obtain ⟨g, hg, hall⟩ := exists_shared_gain gap advantage pos
  refine ⟨g, hg, fun b c hc => ?_⟩
  apply (M.sharedLogit_lt_iff T g eps scale (xs b) L 1 c b).mpr
  simpa [gap, advantage, q, Model.pairLogits, hp, xs] using hall ⟨(b,c), hc⟩

/-- The experimental width fits all 97 codes for any choices of the attention,
MLP and router parameters. -/
example (M : Model 128 32 4 171 97) (L : ℕ) :
    let A := { M with embedding := TwoCopies.coordCodes (by decide) }
    let T : PairTables 128 97 := ⟨fun b => (2 : ℝ) • A.embedding b, 0⟩
    let xs := fun (b : Fin 97) (i : ℕ) => if i = 0 then (0 : Fin 97) else b
    ∃ g : ℝ, 0 < g ∧ ∀ b c : Fin 97, c ≠ b →
      A.sharedLogit T g 1 1 (xs b) L 1 c < A.sharedLogit T g 1 1 (xs b) L 1 b := by
  let A : Model 128 32 4 171 97 := { M with embedding := TwoCopies.coordCodes (by decide) }
  exact A.exists_shared_zero_row (by norm_num) (TwoCopies.orthonormal_coordCodes _)
    (by decide) (eps := 1) (by norm_num) 1 L

end Transformer.Phasor
