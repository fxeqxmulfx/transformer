/-
# Blocks with injections do not collapse, one block or a stack of different ones

**Not a statement of any paper.**  `IsDrivenFlowOn.spread` for the drive of a
residual block (`Perspective.BlockDrive`): if two differences of injections
`z_k - z_l`, `z_m - z_p` are linearly independent throughout a window of
length `τ`, some time of the window has two tokens more than `δ` apart, and `δ`
is fixed by the two differences, the bounds of the block and `τ` alone
(`blockDrive_spread`).  For a stack of blocks, each on its own window and each
with its own weights, values, feed-forward term and injections, one `δ` serves
every block (`blockDrive_stack`).

The model is the stack of parameter-golf (github.com/openai/parameter-golf,
`records/track_10min_16mb/2026-04-29_SmearGateBOSFix_3Seed_1.06141/train_gpt.py`,
`Block.forward`), whose block `b` computes `x_in = mix[0] * x + mix[1] * x0`:
its drive carries `(mix^{(b)}[0] - 1) ⊙ x_i` in the feed-forward term and the
injection `z^{(b)}_i = mix^{(b)}[1] ⊙ x0_i`, whose differences
`mix^{(b)}[1] ⊙ (x0_k - x0_l)` stay independent when the embedding
differences are and `mix^{(b)}[1]` has no zero entry.  What the model still
leaves out: time is continuous, the tokens live on the sphere instead of being
RMS-normalised in `ℝ^d`, and the attention reads `x` rather than `x_in`.
-/

import Transformer.Perspective.BlockDrive

open scoped BigOperators

namespace Transformer
namespace Perspective

variable {d : ℕ}

/-- **Theorem (a block with injections does not collapse).**  Let `w₁`, `w₂` be
linearly independent, `H` a number of heads, `A, N, B, L_G, Z ≥ 0` and `τ > 0`.
There is `δ > 0` such that along every flow `ẋ_i = Proj_{x_i}(v_i(t))` driven by
a block (`blockDrive`) on a window `[t₀, t₀ + τ]`, with the bounds
`IsBoundedBlockOn` there and `z_k - z_l = w₁`, `z_m - z_p = w₂` throughout,
some time of the window has two tokens more than `δ` apart.  `δ` depends on
`w₁`, `w₂`, `H`, `A`, `N`, `B`, `L_G`, `Z` and `τ` only — not on the number of
tokens, the flow, the weights, the values, the feed-forward term or `t₀`.  For
softmax attention `A = 1`, whatever `β`, `Q` and `K`.

Source: none — posed here, as the counterpart of `lem: hemisphere.clustering`
(arXiv:2312.10794v5, §6.1) for `eq: albert` (§2.3) with an injection, through
`IsDrivenFlowOn.spread` at `L = 2 H A N + L_G` and `M = H A N + B + Z`. -/
theorem blockDrive_spread {w₁ w₂ : EucSpace d} (hw : LinearIndependent ℝ ![w₁, w₂])
    (H : ℕ) {A N B LG Z τ : ℝ} (hA : 0 ≤ A) (hN : 0 ≤ N) (hB : 0 ≤ B) (hLG : 0 ≤ LG)
    (hZ : 0 ≤ Z) (hτ : 0 < τ) :
    ∃ δ : ℝ, 0 < δ ∧ ∀ (n : ℕ) (X : ℝ → SphereTuple d n)
      (a : ℝ → Idx H → Idx n → Idx n → ℝ) (V : ℝ → Idx H → EucSpace d →L[ℝ] EucSpace d)
      (G : ℝ → EucSpace d → EucSpace d) (z : ℝ → Idx n → EucSpace d) (t₀ : ℝ)
      (k l m p : Idx n),
      IsDrivenFlowOn X (blockDrive a V G z X) t₀ (t₀ + τ) →
      IsBoundedBlockOn a V G z t₀ (t₀ + τ) A N B LG Z →
      (∀ t ∈ Set.Ioo t₀ (t₀ + τ), z t k - z t l = w₁ ∧ z t m - z t p = w₂) →
      ∃ t ∈ Set.Icc t₀ (t₀ + τ), ∃ i j : Idx n, δ < ‖(X t i : EucSpace d) - X t j‖ := by
  obtain ⟨δ, hδ, h⟩ := IsDrivenFlowOn.spread hw (L := 2 * H * A * N + LG)
    (M := H * (A * N) + B + Z) (by positivity) (by positivity) hτ
  refine ⟨δ, hδ, fun n X a V G z t₀ k l m p hX hb hz => h n X _ t₀ k l m p hX
    (fun t ht i => norm_blockDrive_le X (hb.abs_row_sum t ht) (hb.norm_value t ht)
      (hb.norm_ff t ht) (hb.norm_inj t ht) i) fun t ht ε hε hXε => ?_⟩
  have key := fun k l : Idx n => norm_blockDrive_sub_sub_le (z := z) (hb.row_sum t ht)
    (hb.abs_row_sum t ht) (hb.norm_value t ht) (hb.lipschitz_ff t ht) hLG hε hXε k l
  obtain ⟨h₁, h₂⟩ := hz t ht
  exact ⟨h₁ ▸ key k l, h₂ ▸ key m p⟩

/-- **Theorem (a stack of different blocks does not collapse).**  Let every
block `b` of `nb` come with linearly independent `w₁ b`, `w₂ b`, and let `H`,
`A, N, B, L_G, Z ≥ 0` and `τ > 0` bound them all.  There is one `δ > 0` such
that along every flow driven, on the window `[s_b, s_b + τ]` of each block `b`,
by a block with these bounds and with `z_{k_b} - z_{l_b} = w₁ b`,
`z_{m_b} - z_{p_b} = w₂ b` there, every window has a time at which two tokens
are more than `δ` apart.  The weights, values, feed-forward terms and
injections are functions of time, so each block has its own; for a stack run in
order, `s_b = b τ`.

Source: none — posed here; `blockDrive_spread` block by block, with the least
of the finitely many `δ`. -/
theorem blockDrive_stack {nb : ℕ} {w₁ w₂ : Fin nb → EucSpace d}
    (hw : ∀ b, LinearIndependent ℝ ![w₁ b, w₂ b]) (H : ℕ) {A N B LG Z τ : ℝ} (hA : 0 ≤ A)
    (hN : 0 ≤ N) (hB : 0 ≤ B) (hLG : 0 ≤ LG) (hZ : 0 ≤ Z) (hτ : 0 < τ) :
    ∃ δ : ℝ, 0 < δ ∧ ∀ (n : ℕ) (X : ℝ → SphereTuple d n)
      (a : ℝ → Idx H → Idx n → Idx n → ℝ) (V : ℝ → Idx H → EucSpace d →L[ℝ] EucSpace d)
      (G : ℝ → EucSpace d → EucSpace d) (z : ℝ → Idx n → EucSpace d) (s : Fin nb → ℝ)
      (k l m p : Fin nb → Idx n),
      (∀ b, IsDrivenFlowOn X (blockDrive a V G z X) (s b) (s b + τ)) →
      (∀ b, IsBoundedBlockOn a V G z (s b) (s b + τ) A N B LG Z) →
      (∀ b, ∀ t ∈ Set.Ioo (s b) (s b + τ),
        z t (k b) - z t (l b) = w₁ b ∧ z t (m b) - z t (p b) = w₂ b) →
      ∀ b, ∃ t ∈ Set.Icc (s b) (s b + τ), ∃ i j : Idx n,
        δ < ‖(X t i : EucSpace d) - X t j‖ := by
  choose δ hδ h using fun b => blockDrive_spread (hw b) H hA hN hB hLG hZ hτ
  obtain ⟨δ₀, hδ₀, hle⟩ : ∃ δ₀ : ℝ, 0 < δ₀ ∧ ∀ b, δ₀ ≤ δ b := by
    rcases isEmpty_or_nonempty (Fin nb) with hnb | hnb
    · exact ⟨1, one_pos, fun b => (IsEmpty.false b).elim⟩
    · obtain ⟨b₀, hb₀⟩ := Finite.exists_min δ
      exact ⟨δ b₀, hδ b₀, hb₀⟩
  refine ⟨δ₀, hδ₀, fun n X a V G z s k l m p hX hb hz b => ?_⟩
  obtain ⟨t, ht, i, j, hij⟩ :=
    h b n X a V G z (s b) (k b) (l b) (m b) (p b) (hX b) (hb b) (hz b)
  exact ⟨t, ht, i, j, (hle b).trans_lt hij⟩

/-- The hypotheses of `blockDrive_spread` are satisfiable, those inside its
conclusion included: in `ℝ²`, one head of weight `1` on every token with values
`0`, no feed-forward term, and three tokens `e₀`, `e₀`, `e₁` at rest under the
injections `0`, `-e₀`, `-e₁`, each normal to the sphere at its token, with
`z_0 - z_1 = e₀` and `z_0 - z_2 = e₁`; `A = 3`, `N = B = L_G = 0`, `Z = 1`. -/
example : ∃ (X : ℝ → SphereTuple 2 3) (a : ℝ → Idx 1 → Idx 3 → Idx 3 → ℝ)
    (V : ℝ → Idx 1 → EucSpace 2 →L[ℝ] EucSpace 2) (G : ℝ → EucSpace 2 → EucSpace 2)
    (z : ℝ → Idx 3 → EucSpace 2),
    LinearIndependent ℝ ![(EuclideanSpace.single 0 1 : EucSpace 2), EuclideanSpace.single 1 1] ∧
    IsDrivenFlowOn X (blockDrive a V G z X) 0 (0 + 1) ∧
    IsBoundedBlockOn a V G z 0 (0 + 1) 3 0 0 0 1 ∧
    ∀ t ∈ Set.Ioo (0 : ℝ) (0 + 1), z t 0 - z t 1 = EuclideanSpace.single 0 1 ∧
      z t 0 - z t 2 = EuclideanSpace.single 1 1 := by
  refine ⟨fun _ => ![basePoint 1, basePoint 1, ⟨EuclideanSpace.single 1 1, by simp⟩],
    fun _ _ _ _ => 1, fun _ _ => 0, fun _ _ => 0,
    fun _ => ![0, -EuclideanSpace.single 0 1, -EuclideanSpace.single 1 1], ?_,
    ⟨fun _ => continuousOn_const, fun t _ i => ?_⟩,
    ⟨fun _ _ _ _ _ => ?_, fun _ _ _ _ => ?_, fun _ _ _ => ?_, fun _ _ _ => ?_,
      fun _ _ _ _ => ?_, fun _ _ i => ?_⟩, fun _ _ => ?_⟩
  · refine linearIndependent_of_ne_zero_of_inner_eq_zero (fun i => ?_) fun i j hij => ?_
    · fin_cases i <;> simp
    · fin_cases i <;> fin_cases j <;> simp_all [EuclideanSpace.inner_single_left]
  · refine (hasDerivAt_const t _).congr_deriv ?_
    fin_cases i <;> simp [blockDrive, proj, basePoint]
  · rfl
  · simp
  · simp
  · simp
  · simp
  · fin_cases i <;> simp
  · simp

/-- The hypotheses of `blockDrive_stack` are satisfiable, those inside its
conclusion included, by two different blocks: the configuration of the example
above on the windows `[0, 1]` and `[1, 2]`, with the injections doubled on the
second, so that `w₁ = e₀, 2 e₀` and `w₂ = e₁, 2 e₁`; `Z = 2`. -/
example : ∃ (X : ℝ → SphereTuple 2 3) (a : ℝ → Idx 1 → Idx 3 → Idx 3 → ℝ)
    (V : ℝ → Idx 1 → EucSpace 2 →L[ℝ] EucSpace 2) (G : ℝ → EucSpace 2 → EucSpace 2)
    (z : ℝ → Idx 3 → EucSpace 2) (w₁ w₂ : Fin 2 → EucSpace 2) (s : Fin 2 → ℝ),
    (∀ b, LinearIndependent ℝ ![w₁ b, w₂ b]) ∧
    (∀ b, IsDrivenFlowOn X (blockDrive a V G z X) (s b) (s b + 1)) ∧
    (∀ b, IsBoundedBlockOn a V G z (s b) (s b + 1) 3 0 0 0 2) ∧
    ∀ b, ∀ t ∈ Set.Ioo (s b) (s b + 1), z t 0 - z t 1 = w₁ b ∧ z t 0 - z t 2 = w₂ b := by
  refine ⟨fun _ => ![basePoint 1, basePoint 1, ⟨EuclideanSpace.single 1 1, by simp⟩],
    fun _ _ _ _ => 1, fun _ _ => 0, fun _ _ => 0,
    fun t => (if t < 1 then (1 : ℝ) else 2) •
      ![0, -EuclideanSpace.single 0 1, -EuclideanSpace.single 1 1],
    ![EuclideanSpace.single 0 1, (2 : ℝ) • EuclideanSpace.single 0 1],
    ![EuclideanSpace.single 1 1, (2 : ℝ) • EuclideanSpace.single 1 1], ![0, 1],
    fun b => ?_, fun _ => ⟨fun _ => continuousOn_const, fun t _ i => ?_⟩,
    fun _ => ⟨fun _ _ _ _ _ => ?_, fun _ _ _ _ => ?_, fun _ _ _ => ?_, fun _ _ _ => ?_,
      fun _ _ _ _ => ?_, fun t _ i => ?_⟩, fun b t ht => ?_⟩
  · refine linearIndependent_of_ne_zero_of_inner_eq_zero (fun i => ?_) fun i j hij => ?_
    · fin_cases b <;> fin_cases i <;> simp
    · fin_cases b <;> fin_cases i <;> fin_cases j <;>
        simp_all [EuclideanSpace.inner_single_left, inner_smul_left, inner_smul_right]
  · refine (hasDerivAt_const t _).congr_deriv ?_
    by_cases h : t < 1 <;> fin_cases i <;> simp [blockDrive, proj, basePoint, h, inner_smul_right]
  · rfl
  · simp
  · simp
  · simp
  · simp
  · split_ifs <;> fin_cases i <;> simp [norm_smul]
  · fin_cases b
    · have h : t < 1 := by simpa using ht.2
      simp [h]
    · have h : ¬ t < 1 := by simpa using ht.1.le
      simp [h]

end Perspective
end Transformer
