module

public import Transformer.Grokking.FlatSpectrum.Recall.CausalKeyToken

/-!
# The phases of the causal induction head over the keys

Source: `review/flat_spectrum.md`, *Softmax attention escapes the cut* (added), item SA5, for
the attention scale `β_n = γ log n` of arXiv:2512.01868v4, §6 (*Long context Transformers*), on
the task of arXiv:2312.04927v1, §3.

`Recall.CausalRecall` and `Recall.CausalFading` settle each key by its copy
`w_q = copyWeight s₁ (b + 2q + 2)` at its value token: recall if `s₂ w_q > 1` and the key token
copies less, fading if `s₂ w_q < 1`. The copy is set by the position of the key, so the scales
place each key in a phase:

* below `s₂ = 1` every key fades, at every `s₁`, since `w_q ≤ 1`
  (`exists_lt_causalInductionHead_error_of_scale_lt_one`);
* below `s₁ = 1` the late keys fade at every `s₂` (`Recall.CausalFading`);
* at `s₁ = 1` every key has `w_q = L / (2L − 1) > 1/2`, `L = b + 2q + 2`, and from `s₁ = 1` on the
  value token outcopies the key token at every key but the first without a beginning-of-sequence
  token (`causalCopy_key_lt_copyWeight`). So `s₂ ≥ 2` recalls all of these
  (`exists_forall_causalInductionHead_recall_one`), and below it the late keys, with
  `w_q → 1/2`, fade (`exists_lt_causalInductionHead_error_one`);
* above `s₁ = 1` and `s₂ = 1` the late keys, with `w_q → 1`, are recalled
  (`exists_forall_causalInductionHead_recall_of_gt`).

The early keys are not always the ones lost. After a beginning-of-sequence token, at `s₁ = 2`
and `s₂ = 11/10` the first key fades while the fourth is recalled (`Recall.CausalRecall`); at
`s₁ = 1` and `s₂ = 17/10` the first key is recalled while the second fades.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Recall

open Filter Topology

/-- **From `s₁ = 1` on the value token outcopies the key token** (`review/flat_spectrum.md`,
SA5): at `s ≥ 1`, every key but the first without a beginning-of-sequence token is copied at its
key token with less than `copyWeight s (b + 2c + 2)`, its copy at the value token. -/
lemma causalCopy_key_lt_copyWeight {s : ℝ} (hs : 1 ≤ s) {b c : ℕ} (hc : 1 ≤ b + 2 * c) :
    causalCopy s b (b + 2 * c) c < copyWeight s (b + 2 * c + 2) := by
  have hK : (2 : ℝ) ≤ b + 2 * c + 1 := by
    have : (1 : ℝ) ≤ b + 2 * c := by exact_mod_cast hc
    linarith
  have hKs := Real.self_le_rpow_of_one_le (by linarith : (1 : ℝ) ≤ b + 2 * c + 1) hs
  have hLs := Real.self_le_rpow_of_one_le (by linarith : (1 : ℝ) ≤ b + 2 * c + 2) hs
  rw [causalCopy_key, copyWeight, div_lt_div_iff₀ (by linarith) (by linarith)]
  nlinarith [mul_le_mul hLs hKs (by linarith) (by linarith)]

/-- The first key after a beginning-of-sequence token at `s = 1`. -/
example := causalCopy_key_lt_copyWeight le_rfl (b := 1) (c := 0) (by norm_num)

/-- The copy at a value token is at most `1`. -/
lemma copyWeight_le_one (s : ℝ) {x : ℝ} (hx : 1 ≤ x) : copyWeight s x ≤ 1 := by
  have := Real.rpow_pos_of_pos (by linarith : (0 : ℝ) < x) s
  rw [copyWeight, div_le_one (by linarith)]
  linarith

/-- Three tokens at `s = 2`. -/
example := copyWeight_le_one 2 (x := 3) (by norm_num)

/-- **Below `s₂ = 1` every key fades** (`review/flat_spectrum.md`, SA5): at `0 ≤ s₂ < 1`, at every
`s₁`, for every `ε < 1`, from some number of keys on, every key has a dictionary of values `±1`
answered with an error above `ε`. -/
theorem exists_lt_causalInductionHead_error_of_scale_lt_one {s₂ ε : ℝ} (hs₂ : 0 ≤ s₂)
    (hs₂1 : s₂ < 1) (hε : ε < 1) (s₁ : ℝ) (b : ℕ) :
    ∃ M : ℕ, ∀ m ≥ M, ∀ q : Fin (m + 1), ∃ π : Fin (m + 1) → Bool,
      ε < |(if π q then (1 : ℝ) else -1) -
        causalInductionHead s₁ s₂ (fun x => if x then 1 else -1) b π q| := by
  obtain ⟨M, hM⟩ := exists_lt_causalInductionHead_error hs₂ hs₂1 hε s₁ b
  exact ⟨M, fun m hm q => hM m hm q (mul_le_of_le_one_right hs₂ (copyWeight_le_one s₁
    (by linarith [(by positivity : (0 : ℝ) ≤ b + 2 * (q : ℕ))])))⟩

/-- At `s₂ = 1/2` and `ε = 1/2`. -/
example := exists_lt_causalInductionHead_error_of_scale_lt_one (s₂ := 1 / 2) (ε := 1 / 2)
  (by norm_num) (by norm_num) (by norm_num) 2 1

/-- **At `s₁ = 1`, `s₂ ≥ 2` recalls every key** (`review/flat_spectrum.md`, SA5): every key but
the first without a beginning-of-sequence token is copied with `L / (2L − 1) > 1/2`, so for every
`ε > 0`, from some number of keys on, it is answered within `ε` in every dictionary. -/
theorem exists_forall_causalInductionHead_recall_one {s₂ : ℝ} (hs₂ : 2 ≤ s₂) {b q : ℕ}
    (hq : 1 ≤ b + 2 * q) {ν : Type*} {φ : ν → ℝ} {C : ℝ} (hφ : ∀ x, |φ x| ≤ C) {ε : ℝ}
    (hε : 0 < ε) :
    ∃ M : ℕ, ∀ m ≥ M, ∀ k : Fin (m + 1), (k : ℕ) = q → ∀ π : Fin (m + 1) → ν,
      |φ (π k) - causalInductionHead 1 s₂ φ b π k| ≤ ε := by
  refine exists_forall_causalInductionHead_recall zero_le_one b q
    (causalCopy_key_lt_copyWeight le_rfl hq) ?_ hφ hε
  have hL : (2 : ℝ) ≤ b + 2 * q + 2 := by linarith [(by positivity : (0 : ℝ) ≤ b + 2 * q)]
  rw [copyWeight, Real.rpow_one, ← mul_div_assoc, lt_div_iff₀ (by linarith)]
  nlinarith

/-- The first key after a beginning-of-sequence token at `s₂ = 2`. -/
example := exists_forall_causalInductionHead_recall_one le_rfl (b := 1) (q := 0) (by norm_num)
  (φ := fun x : Bool => if x then (1 : ℝ) else -1) (C := 1) (fun x => by cases x <;> norm_num)
  (ε := 1 / 10) (by norm_num)

/-- **At `s₁ = 1`, `s₂ < 2` loses the late keys** (`review/flat_spectrum.md`, SA5): the copies of
the late keys tend to `1/2`, so for every `ε < 1` there is a key from which on, from some number of
keys on, every key has a dictionary of values `±1` answered with an error above `ε`. -/
theorem exists_lt_causalInductionHead_error_one {s₂ ε : ℝ} (hs₂ : 0 ≤ s₂) (hs₂2 : s₂ < 2)
    (hε : ε < 1) (b : ℕ) :
    ∃ Q M : ℕ, ∀ m ≥ M, ∀ q : Fin (m + 1), Q ≤ (q : ℕ) → ∃ π : Fin (m + 1) → Bool,
      ε < |(if π q then (1 : ℝ) else -1) -
        causalInductionHead 1 s₂ (fun x => if x then 1 else -1) b π q| := by
  have h := (tendsto_causalCopy_value_one b).const_mul s₂
  obtain ⟨Q, hQ⟩ := eventually_atTop.1
    (h.eventually (ge_mem_nhds (by linarith : s₂ * (1 / 2) < (s₂ / 2 + 1) / 2)))
  obtain ⟨M, hM⟩ := exists_lt_causalInductionHead_error hs₂
    (by linarith : (s₂ / 2 + 1) / 2 < 1) hε 1 b
  refine ⟨Q, M, fun m hm q hq => hM m hm q ?_⟩
  have h' := hQ q hq
  rwa [causalCopy_value] at h'

/-- At `s₂ = 17/10`, the late keys after a beginning-of-sequence token. -/
example := exists_lt_causalInductionHead_error_one (s₂ := 17 / 10) (ε := 1 / 2) (by norm_num)
  (by norm_num) (by norm_num) 1

/-- With the same scales the first key is recalled: `s₂ w = 51/50`. -/
example := exists_forall_causalInductionHead_recall (s₁ := 1) (s₂ := 17 / 10) zero_le_one 1 0
  (causalCopy_key_lt_copyWeight le_rfl (by norm_num)) (by rw [copyWeight]; norm_num)
  (φ := fun x : Bool => if x then (1 : ℝ) else -1) (C := 1) (fun x => by cases x <;> norm_num)
  (ε := 1 / 10) (by norm_num)

/-- And the second fades: `s₂ w = 17/18`. -/
example : ∃ M : ℕ, ∀ m ≥ M, ∀ q : Fin (m + 1), (q : ℕ) = 1 → ∃ π : Fin (m + 1) → Bool,
    1 / 2 < |(if π q then (1 : ℝ) else -1) -
      causalInductionHead 1 (17 / 10) (fun x => if x then 1 else -1) 1 π q| := by
  obtain ⟨M, hM⟩ := exists_lt_causalInductionHead_error (s₂ := 17 / 10) (σ := 17 / 18)
    (ε := 1 / 2) (by norm_num) (by norm_num) (by norm_num) 1 1
  exact ⟨M, fun m hm q hq => hM m hm q (by rw [hq, copyWeight]; norm_num)⟩

/-- **Above `s₁ = 1` and `s₂ = 1` the late keys are recalled** (`review/flat_spectrum.md`, SA5):
the copies of the late keys tend to `1`, so there is a key from which on every key, for every
`ε > 0`, from some number of keys on, is answered within `ε` in every dictionary. -/
theorem exists_forall_causalInductionHead_recall_of_gt {s₁ s₂ : ℝ} (hs₁ : 1 < s₁) (hs₂ : 1 < s₂)
    (b : ℕ) {ν : Type*} {φ : ν → ℝ} {C : ℝ} (hφ : ∀ x, |φ x| ≤ C) :
    ∃ Q : ℕ, ∀ q ≥ Q, ∀ ε > 0, ∃ M : ℕ, ∀ m ≥ M, ∀ k : Fin (m + 1), (k : ℕ) = q →
      ∀ π : Fin (m + 1) → ν, |φ (π k) - causalInductionHead s₁ s₂ φ b π k| ≤ ε := by
  have h := (tendsto_causalCopy_value_of_gt hs₁ b).const_mul s₂
  rw [mul_one] at h
  obtain ⟨Q, hQ⟩ := eventually_atTop.1 (h.eventually (lt_mem_nhds hs₂))
  refine ⟨max Q 1, fun q hq ε hε => exists_forall_causalInductionHead_recall (by linarith) b q
    (causalCopy_key_lt_copyWeight hs₁.le (by have := le_of_max_le_right hq; omega)) ?_ hφ hε⟩
  have h' := hQ q (le_of_max_le_left hq)
  rwa [causalCopy_value] at h'

/-- At `s₁ = s₂ = 2` after a beginning-of-sequence token. -/
example := exists_forall_causalInductionHead_recall_of_gt (s₁ := 2) (s₂ := 2) (by norm_num)
  (by norm_num) 1 (φ := fun x : Bool => if x then (1 : ℝ) else -1) (C := 1)
  (fun x => by cases x <;> norm_num)

end Transformer.Grokking.FlatSpectrum.Recall
