import Transformer.GPTMini.Sparsemax.ProjectionUpdate

/-!
# Separate anchor channels without independent ordinary inputs

Derived matrix accessibility construction for arXiv:1602.02068v2,
§2.5, and shared linear projections in `Attention.forward` at `73f8a0b`.
Only the `A` anchor inputs get dedicated coordinate channels. Arbitrary
ordinary embeddings occupy the remaining `B` channels. A continuous
linear decoder selects the anchor channels and kills all ordinary inputs,
regardless of the number, rank or repetitions of ordinary tokens.

The decoder and its identities are constructed and proved. No full-row
independence is assumed: contexts longer than the input width are proved
dependent and still satisfy the anchor decoder identities. This is an
explicit input architecture, adding separate anchor channels rather than
asserting that unconstrained learned embeddings have this property.
All identities concern actual finite shared matrix-vector multiplication.
The construction needs `A + B` input coordinates, independently of `N`.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall
open scoped BigOperators

/-- Prefix anchor coordinates followed by arbitrary ordinary embedding channels.
Source: the derived partial accessibility restriction for §2.5 of
arXiv:1602.02068v2 and shared projections at `73f8a0b`. -/
def prefixInputs {A B N : ℕ} (ordinary : Fin N → (Fin B → ℝ)) :
    Fin (A + N) → (Fin (A + B) → ℝ) :=
  Fin.addCases (fun a => basis (Fin.castAdd B a))
    (fun n => Fin.addCases (fun _ => 0) (ordinary n))

/-- Select only the dedicated anchor channels of an actual input vector.
Source: the derived coordinate decoder for §2.5 of arXiv:1602.02068v2;
ordinary embedding channels remain outside the decoder's image. -/
def prefixInputDecoder (A B : ℕ) : (Fin (A + B) → ℝ) →L[ℝ] (Fin A → ℝ) :=
  ContinuousLinearMap.pi fun a => ContinuousLinearMap.proj (Fin.castAdd B a)

/-- Every anchor input is decoded to its own coordinate vector.
Source: the derived partial input lift for §2.5 of arXiv:1602.02068v2;
no condition on the ordinary embeddings or their count is imposed. -/
theorem prefixInputDecoder_anchor {A B N : ℕ} (ordinary : Fin N → (Fin B → ℝ))
    (a : Fin A) : prefixInputDecoder A B (prefixInputs ordinary (Fin.castAdd N a)) = basis a := by
  funext b
  simp [prefixInputDecoder, prefixInputs, basis, ContinuousLinearMap.pi_apply]

/-- Every ordinary input is killed by the anchor decoder.
Source: the derived partial accessibility restriction for §2.5 of
arXiv:1602.02068v2, valid also for dependent ordinary embeddings. -/
theorem prefixInputDecoder_ordinary {A B N : ℕ} (ordinary : Fin N → (Fin B → ℝ))
    (n : Fin N) : prefixInputDecoder A B (prefixInputs ordinary (Fin.natAdd A n)) = 0 := by
  funext a
  simp only [prefixInputDecoder, ContinuousLinearMap.pi_apply, ContinuousLinearMap.proj_apply,
    prefixInputs, Fin.addCases_right, Fin.addCases_left, Pi.zero_apply]

/-- An anchor cannot coincide with any ordinary token, even when ordinary tokens repeat.
Source: the derived dedicated input channels for §2.5 of arXiv:1602.02068v2;
the decoder distinguishes one nonzero anchor coordinate from zero. -/
theorem prefixInputs_anchor_ne_ordinary {A B N : ℕ} (ordinary : Fin N → (Fin B → ℝ))
    (a : Fin A) (n : Fin N) :
    prefixInputs ordinary (Fin.castAdd N a) ≠ prefixInputs ordinary (Fin.natAdd A n) := by
  intro heq
  have hd := congrArg (prefixInputDecoder A B) heq
  rw [prefixInputDecoder_anchor, prefixInputDecoder_ordinary] at hd
  have ha := congrFun hd a
  norm_num [basis] at ha

/-- The first anchor remains distinct from one hundred repeated nonzero ordinary tokens.
Source context: the partial input accessibility construction for §2.5. -/
example : prefixInputs (fun _ : Fin 100 => fun _ : Fin 1 => (1 : ℝ)) (Fin.castAdd 100 (0 : Fin 2)) ≠
    prefixInputs (fun _ : Fin 100 => fun _ : Fin 1 => (1 : ℝ)) (Fin.natAdd 2 0) :=
  prefixInputs_anchor_ne_ordinary _ _ _

/-- An actual partial decoder exists for every ordinary input family.
Source: the derived anchor-channel construction for §2.5 of
arXiv:1602.02068v2; its existence is witnessed, not assumed. -/
theorem prefixInputDecoder_exists {A B N : ℕ} (ordinary : Fin N → (Fin B → ℝ)) :
    ∃ decoder : (Fin (A + B) → ℝ) →L[ℝ] (Fin A → ℝ),
      (∀ a, decoder (prefixInputs ordinary (Fin.castAdd N a)) = basis a) ∧
      (∀ n, decoder (prefixInputs ordinary (Fin.natAdd A n)) = 0) :=
  ⟨prefixInputDecoder A B, prefixInputDecoder_anchor ordinary, prefixInputDecoder_ordinary ordinary⟩

/-- The anchor subfamily is independent even when ordinary inputs repeat.
Source: the derived partial decoder restriction for §2.5 of
arXiv:1602.02068v2; independence concerns only the anchor inputs. -/
theorem prefixInputs_anchors_independent {A B N : ℕ} (ordinary : Fin N → (Fin B → ℝ)) :
    LinearIndependent ℝ (fun a : Fin A => prefixInputs ordinary (Fin.castAdd N a)) :=
  inputIndependent_of_decoder _ (prefixInputDecoder A B) (prefixInputDecoder_anchor ordinary)

/-- Long contexts can satisfy the partial decoder while failing full independence.
Source: the derived matrix-rank restriction for §2.5 of arXiv:1602.02068v2;
only `A + B` input coordinates are used for `A + N` tokens. -/
theorem prefixInputs_not_independent_of_long {A B N : ℕ}
    (ordinary : Fin N → (Fin B → ℝ)) (hlong : B < N) :
    ¬ LinearIndependent ℝ (prefixInputs (A := A) ordinary) := by
  intro hi
  have hw := inputIndependent_width_bound (prefixInputs (A := A) ordinary) hi
  omega

/-- One hundred ordinary tokens inhabit the long-context rank obstruction.
Source context: arXiv:1602.02068v2, §2.5, partial anchor accessibility. -/
example : ¬ LinearIndependent ℝ
    (prefixInputs (A := 2) (fun _ : Fin 100 => fun _ : Fin 1 => (1 : ℝ))) :=
  prefixInputs_not_independent_of_long _ (by norm_num)

/-- The same long, dependent family has a decoder for both anchors.
Source context: the derived anchor-channel restriction for §2.5. -/
example : (∀ a : Fin 2, prefixInputDecoder 2 1
      (prefixInputs (fun _ : Fin 100 => fun _ : Fin 1 => (1 : ℝ)) (Fin.castAdd 100 a)) = basis a) ∧
    (∀ n : Fin 100, prefixInputDecoder 2 1
      (prefixInputs (fun _ : Fin 100 => fun _ : Fin 1 => (1 : ℝ)) (Fin.natAdd 2 n)) = 0) :=
  ⟨prefixInputDecoder_anchor _, prefixInputDecoder_ordinary _⟩

/-- Actual shared projections on anchors select their dedicated matrix columns.
Source: the finite linear Q/K projections at `73f8a0b`, applied to the
derived prefix-channel construction for §2.5 of arXiv:1602.02068v2. -/
theorem projectionEvaluation_prefix_anchor {A B N h : ℕ}
    (ordinary : Fin N → (Fin B → ℝ)) (columns : Fin (A + B) → EucSpace h) (a : Fin A) :
    projectionEvaluation (prefixInputs ordinary) columns (Fin.castAdd N a) =
      columns (Fin.castAdd B a) := by
  rw [projectionEvaluation_apply, prefixInputs, Fin.addCases_left, projection_basis_column]

/-- Actual shared projections on ordinary inputs use only embedding channels.
Source: the finite matrix-vector sum at `73f8a0b`, before QKNorm and
§2.5's sparsemax score path; ordinary embeddings are unrestricted. -/
theorem projectionEvaluation_prefix_ordinary {A B N h : ℕ}
    (ordinary : Fin N → (Fin B → ℝ)) (columns : Fin (A + B) → EucSpace h) (n : Fin N) :
    projectionEvaluation (prefixInputs ordinary) columns (Fin.natAdd A n) =
      frozenValueReadout (fun b => columns (Fin.natAdd A b)) (ordinary n) := by
  rw [projectionEvaluation_apply, prefixInputs, Fin.addCases_right,
    frozenValueReadout_apply, Fin.sum_univ_add, frozenValueReadout_apply]
  simp only [Fin.addCases_left, Fin.addCases_right, zero_smul, Finset.sum_const_zero, zero_add]

/-- A decoder-kernel update preserves every ordinary key in this architecture.
Source: the derived anchor update for §2.5 of arXiv:1602.02068v2,
evaluated by the actual shared projection at `73f8a0b`. -/
theorem prefixProjectionUpdate_preserves_ordinary {A B N h : ℕ}
    (ordinary : Fin N → (Fin B → ℝ)) (columns : Fin (A + B) → EucSpace h)
    (keys : Fin A → EucSpace h) (n : Fin N) :
    frozenValueReadout (inputProjectionUpdate
      (fun a : Fin A => prefixInputs ordinary (Fin.castAdd N a))
      (prefixInputDecoder A B) columns keys) (prefixInputs ordinary (Fin.natAdd A n)) =
      frozenValueReadout columns (prefixInputs ordinary (Fin.natAdd A n)) :=
  inputProjectionUpdate_preserves_kernel _ _ _ _ _ (prefixInputDecoder_ordinary ordinary n)

end Transformer.GPTMini.Sparsemax
