module

public import Transformer.Phasor.PrefixValues

/-!
# A causal successor address for the shared value residual

Source: the successor-read trial in `experiments/phasor_value_transport/README.md`.
The right-operand table can read the successor of the most recent completed
occurrence of its key. The state accepts arbitrary tokens and stream lengths.
-/

@[expose] public section

namespace Transformer.Phasor

open scoped RealInnerProductSpace
open Transformer

universe u v

variable {α : Type u} {β : Type v}

/-- The first `n` transitions are the pairs at positions `(0,1), …, (n-1,n)`. -/
def latestSuccessor [DecidableEq α] (xs : ℕ → α) : ℕ → α → Option α
  | 0, _ => none
  | n + 1, k => if xs n = k then some (xs (n+1)) else latestSuccessor xs n k

/-- Arbitrarily many intervening writes to other keys preserve the last write
to the queried key, including when the same key had earlier, different values. -/
theorem latestSuccessor_lookup [DecidableEq α] (xs : ℕ → α) (k : α) (j n : ℕ)
    (hj : j < n) (hk : xs j = k) (hlast : ∀ r, j < r → r < n → xs r ≠ k) :
    latestSuccessor xs n k = some (xs (j+1)) := by
  induction n with
  | zero => omega
  | succ n ih =>
    by_cases h : j = n
    · subst n
      simp [latestSuccessor, hk]
    · have earlier : j < n := by omega
      have other := hlast n earlier (by omega)
      simpa [latestSuccessor, other] using ih earlier (fun r hjr hr => hlast r hjr (by omega))

/-- The lookup commutes with any injective renaming, also between vocabularies
of different sizes. Only equality of the key tokens matters. -/
theorem latestSuccessor_map [DecidableEq α] [DecidableEq β] (f : α → β)
    (hf : Function.Injective f) (xs : ℕ → α) (n : ℕ) (k : α) :
    latestSuccessor (fun i => f (xs i)) n (f k) = (latestSuccessor xs n k).map f := by
  induction n with
  | zero => rfl
  | succ n ih =>
    by_cases h : xs n = k <;> simp [latestSuccessor, hf.eq_iff, h, ih]

/-- At position `i` only transitions with both endpoints strictly before `i`
are eligible. A previously unwritten key reads itself. -/
def successorAddress [DecidableEq α] (xs : ℕ → α) (i : ℕ) : α :=
  (latestSuccessor xs (i-1) (xs i)).getD (xs i)

/-- The same read retrieves a value for any query position and alphabet once
its most recent completed matching transition is specified. -/
theorem successorAddress_eq [DecidableEq α] (xs : ℕ → α) (i j : ℕ)
    (hj : j+1 < i) (hk : xs j = xs i)
    (hlast : ∀ r, j < r → r < i-1 → xs r ≠ xs i) :
    successorAddress xs i = xs (j+1) := by
  unfold successorAddress
  rw [latestSuccessor_lookup xs (xs i) j (i-1) (by omega) hk hlast]
  rfl

/-- With no available transition the right-table read has its original address. -/
theorem successorAddress_start [DecidableEq α] (xs : ℕ → α) (i : ℕ) (hi : i ≤ 1) :
    successorAddress xs i = xs i := by
  have empty : i - 1 = 0 := by omega
  simp [successorAddress, empty, latestSuccessor]

/-- Contextualize the existing right-operand table at every position. The
parameter tensors and all power, difference and quotient tables stay shared. -/
noncomputable def PairTables.withSuccessor {D p : ℕ} (T : PairTables D p)
    (xs : ℕ → Fin p) (i : ℕ) : PairTables D p :=
  { T with right := fun k => T.right ((latestSuccessor xs (i-1) k).getD k) }

/-- The composed architecture reads the successor table and the XOR-prefix
linear sum in the same hidden residual at all positions. -/
noncomputable def Model.successorLogit {D d H F p : ℕ} [NeZero p] (M : Model D d H F p)
    (T : PairTables D p) (gain eps scale : ℝ) (xs : ℕ → Fin p) (L i : ℕ) (c : Fin p) : ℝ :=
  M.prefixLogit (T.withSuccessor xs i) .xor (0,0) gain eps scale xs L i c

/-- At arbitrary depth and position, changing the gain by `g-h` changes a
class logit by at most the embedding norm times `|g-h| √D`. The lookup and
the rest of the learned parameters are fixed. This bounds the effect of the
quarter- versus half-gain initializations in the successor experiment. -/
theorem Model.successorLogit_gain_bound {D d H F p : ℕ} [NeZero p] (M : Model D d H F p)
    (T : PairTables D p) (g h : ℝ) {eps : ℝ} (heps : 0 < eps) (hD : 0 < D)
    (scale : ℝ) (xs : ℕ → Fin p) (L i : ℕ) (c : Fin p) :
    |M.successorLogit T g eps scale xs L i c - M.successorLogit T h eps scale xs L i c| ≤
      |g-h| * (‖M.embedding c‖ * Real.sqrt D) := by
  let raw := transportedValue M.embedding (Perspective.softmaxWeight
    (M.logits eps scale xs L (i-1))) (xs (i-1) * xs i) +
    (T.withSuccessor xs i).prefixRead .xor (0,0) xs i
  let y := GPTMini.rmsNormEps eps raw
  change |⟪M.embedding c, M.feature eps scale xs L i + g • y⟫ -
    ⟪M.embedding c, M.feature eps scale xs L i + h • y⟫| ≤ _
  simp only [inner_add_right, real_inner_smul_right, add_sub_add_left_eq_sub, ← sub_mul, abs_mul]
  exact mul_le_mul_of_nonneg_left ((abs_real_inner_le_norm _ _).trans
    (mul_le_mul_of_nonneg_left (GPTMini.rmsNormEps_norm_le eps heps hD raw) (norm_nonneg _)))
    (abs_nonneg _)

/-- The composed score still agrees with the incumbent for every choice of
the learned parameters at the first two positions. -/
theorem Model.successorLogit_eq_shared {D d H F p : ℕ} [NeZero p] (M : Model D d H F p)
    (T : PairTables D p) (gain eps scale : ℝ) (xs : ℕ → Fin p) (L i : ℕ) (hi : i ≤ 1)
    (c : Fin p) : M.successorLogit T gain eps scale xs L i c = M.sharedLogit T gain eps scale xs L i c := by
  have empty : i - 1 = 0 := by omega
  have tables : T.withSuccessor xs i = T := by simp [PairTables.withSuccessor, empty, latestSuccessor]
  rw [Model.successorLogit, tables]
  exact M.prefixLogit_eq_shared T .xor (0,0) gain eps scale xs L i hi c

/-- An overwrite followed by an unrelated transition retains the latest value. -/
example : successorAddress (fun i => (![0, 1, 0, 2, 3, 0] : Fin 6 → Fin 4) (Fin.ofNat 6 i)) 5 = 2 := by decide

example (xs : ℕ → ℕ) (n k : ℕ) :
    latestSuccessor (fun i => xs i + 7) n (k+7) = (latestSuccessor xs n k).map (·+7) :=
  latestSuccessor_map (fun x : ℕ => x+7) (by intro a b h; dsimp at h; omega) xs n k

example (xs : ℕ → Fin 32) : successorAddress xs 1 = xs 1 :=
  successorAddress_start xs 1 (by decide)

example (M : Model 128 32 4 171 97) (T : PairTables 128 97) (xs : ℕ → Fin 97)
    (L : ℕ) (c : Fin 97) :
    M.successorLogit T (1/4) 1 1 xs L 1 c = M.sharedLogit T (1/4) 1 1 xs L 1 c :=
  M.successorLogit_eq_shared T _ _ _ xs L 1 (by decide) c

example (M : Model 64 16 4 86 68) (T : PairTables 64 68) (xs : ℕ → Fin 68)
    (L i : ℕ) (c : Fin 68) :
    |M.successorLogit T (1/2) 1 1 xs L i c - M.successorLogit T (1/4) 1 1 xs L i c| ≤
      |(1/2 : ℝ)-1/4| * (‖M.embedding c‖ * Real.sqrt 64) :=
  M.successorLogit_gain_bound T _ _ (by norm_num) (by decide) _ xs L i c

end Transformer.Phasor
