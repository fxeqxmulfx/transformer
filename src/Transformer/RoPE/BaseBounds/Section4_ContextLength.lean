module

public import Mathlib.Analysis.Real.Pi.Irrational
public import Mathlib.Analysis.SpecialFunctions.Pow.Asymptotics
public import Transformer.RoPE.BaseBounds.Section4_CosineSums

/-!
# §4.3 The base of RoPE bounds the context length

Men et al., arXiv:2405.14591v1, §4.3, Eq. (10): "the context length `L_θ` that can be truly
obtained satisfies `L_θ = sup{L | B_{m,θ} ≥ 0, ∀ m ∈ [0, 1, ..., L]}`". §4.3 then concludes that
"the RoPE's base determines the upper bound of context length the model can truly obtain".

* `contextLength` is Eq. (10). `contextLength_spec`: once `B` turns negative the supremum is
  attained, `B ≥ 0` up to `L_θ` and `B(L_θ + 1) < 0`; `le_contextLength_iff` tests `L ≤ L_θ`;
* `ropeFrequencies` are the RoPE frequencies `θ_i = base^{-2i/d}`; for a base `> 1` they are
  distinct and lie in `(0, 1]`;
* `exists_ropeKernel_neg`: for every base `> 1` the RoPE kernel turns negative, so every base
  bounds the context length, not only the small ones of §4.2;
* `one_add_cos_nat_pos`: `cos m > -1` at every integer distance, used for the bases of Eq. (11)
  in `Section4_BaseBound`.
-/

@[expose] public section

open Filter Topology
open scoped BigOperators

namespace Transformer.RoPE.BaseBounds

/-- Eq. (10): the longest `L` with `B_{m,θ} ≥ 0` at every distance `m ≤ L`. -/
noncomputable def contextLength {n : ℕ} (omega : Fin n → ℝ) : ℕ :=
  sSup {L : ℕ | ∀ m ≤ L, 0 ≤ cosineKernel omega m}

theorem bddAbove_of_cosineKernel_neg {n : ℕ} {omega : Fin n → ℝ} {m₀ : ℕ}
    (hm₀ : cosineKernel omega m₀ < 0) :
    BddAbove {L : ℕ | ∀ m ≤ L, 0 ≤ cosineKernel omega m} :=
  ⟨m₀, fun L hL => by
    by_contra! hlt
    exact (hL m₀ hlt.le).not_gt hm₀⟩

example : BddAbove {L : ℕ | ∀ m ≤ L, 0 ≤ cosineKernel (fun _ : Fin 1 => (Real.pi : ℝ)) m} :=
  bddAbove_of_cosineKernel_neg (m₀ := 1) (by simp [cosineKernel])

/-- Eq. (10): once the kernel turns negative, the supremum is attained, and the kernel is
negative right after it. -/
theorem contextLength_spec {n : ℕ} {omega : Fin n → ℝ}
    (h : ∃ m : ℕ, cosineKernel omega m < 0) :
    (∀ m ≤ contextLength omega, 0 ≤ cosineKernel omega m) ∧
      cosineKernel omega (contextLength omega + 1 : ℕ) < 0 := by
  obtain ⟨m₀, hm₀⟩ := h
  have hbdd := bddAbove_of_cosineKernel_neg hm₀
  have hne : {L : ℕ | ∀ m ≤ L, 0 ≤ cosineKernel omega m}.Nonempty := ⟨0, fun m hm => by
    obtain rfl := Nat.le_zero.mp hm
    simp [cosineKernel]⟩
  have hmem : contextLength omega ∈ {L : ℕ | ∀ m ≤ L, 0 ≤ cosineKernel omega m} :=
    Nat.sSup_mem hne hbdd
  refine ⟨hmem, ?_⟩
  by_contra! hnn
  have hsucc : contextLength omega + 1 ∈ {L : ℕ | ∀ m ≤ L, 0 ≤ cosineKernel omega m} := by
    intro m hm
    rcases Nat.lt_or_eq_of_le hm with hlt | rfl
    · exact hmem m (Nat.lt_succ_iff.mp hlt)
    · exact hnn
  have hle : contextLength omega + 1 ≤ contextLength omega := le_csSup hbdd hsucc
  omega

example : (∀ m ≤ contextLength (fun _ : Fin 1 => (Real.pi : ℝ)),
      0 ≤ cosineKernel (fun _ : Fin 1 => (Real.pi : ℝ)) m) ∧
    cosineKernel (fun _ : Fin 1 => (Real.pi : ℝ))
      (contextLength (fun _ : Fin 1 => (Real.pi : ℝ)) + 1 : ℕ) < 0 :=
  contextLength_spec ⟨1, by simp [cosineKernel]⟩

/-- Eq. (10) as a test: `L ≤ L_θ` exactly when `B ≥ 0` up to `L`. -/
theorem le_contextLength_iff {n : ℕ} {omega : Fin n → ℝ}
    (h : ∃ m : ℕ, cosineKernel omega m < 0) (L : ℕ) :
    L ≤ contextLength omega ↔ ∀ m ≤ L, 0 ≤ cosineKernel omega m := by
  obtain ⟨m₀, hm₀⟩ := h
  exact ⟨fun hL m hm => (contextLength_spec ⟨m₀, hm₀⟩).1 m (hm.trans hL),
    fun hL => le_csSup (bddAbove_of_cosineKernel_neg hm₀) hL⟩

example : 0 ≤ contextLength (fun _ : Fin 1 => (Real.pi : ℝ)) ↔
    ∀ m ≤ (0 : ℕ), 0 ≤ cosineKernel (fun _ : Fin 1 => (Real.pi : ℝ)) m :=
  le_contextLength_iff (omega := fun _ : Fin 1 => (Real.pi : ℝ)) ⟨1, by simp [cosineKernel]⟩ 0

/-- The RoPE frequencies `θ_i = base^{-2i/d}` of a head with `d = 2 · pairs` (§2.1). -/
noncomputable def ropeFrequencies (base : ℝ) (pairs : ℕ) (i : Fin pairs) : ℝ :=
  frequency base (2 * pairs) i

theorem ropeFrequencies_strictAnti {base : ℝ} (hbase : 1 < base) (pairs : ℕ) :
    StrictAnti (ropeFrequencies base pairs) := by
  intro i j hij
  unfold ropeFrequencies frequency
  rw [Real.rpow_lt_rpow_left_iff hbase, div_lt_div_iff_of_pos_right
    (by have := i.pos; positivity), neg_lt_neg_iff]
  have : (i : ℝ) < j := by exact_mod_cast hij
  linarith

example : StrictAnti (ropeFrequencies 10000 64) := ropeFrequencies_strictAnti (by norm_num) 64

theorem ropeFrequencies_mem {base : ℝ} (hbase : 1 < base) {pairs : ℕ} (i : Fin pairs) :
    0 < ropeFrequencies base pairs i ∧ ropeFrequencies base pairs i ≤ 1 := by
  refine ⟨Real.rpow_pos_of_pos (by linarith) _,
    Real.rpow_le_one_of_one_le_of_nonpos hbase.le (div_nonpos_of_nonpos_of_nonneg ?_ ?_)⟩
  · have : (0 : ℝ) ≤ i := Nat.cast_nonneg _
    linarith
  · positivity

example : 0 < ropeFrequencies 10000 64 7 ∧ ropeFrequencies 10000 64 7 ≤ 1 :=
  ropeFrequencies_mem (by norm_num) 7

/-- §4.3: every base `> 1` bounds the context length. The RoPE kernel turns negative, so
`L_θ` is finite and attained (`contextLength_spec`). -/
theorem exists_ropeKernel_neg {base : ℝ} (hbase : 1 < base) (pairs : ℕ) [NeZero pairs] :
    ∃ m : ℕ, cosineKernel (ropeFrequencies base pairs) m < 0 :=
  exists_cosineKernel_neg (ropeFrequencies_strictAnti hbase pairs).injective
    (fun i => (ropeFrequencies_mem hbase i).1)
    (fun i => (ropeFrequencies_mem hbase i).2.trans_lt (by linarith [Real.pi_gt_three]))

example : ∃ m : ℕ, cosineKernel (ropeFrequencies 500000 64) m < 0 :=
  exists_ropeKernel_neg (by norm_num) 64

/-- `1 + cos m > 0` at every integer distance: `cos m = -1` would make `π` rational. -/
theorem one_add_cos_nat_pos (m : ℕ) : 0 < 1 + Real.cos m := by
  rcases (Real.neg_one_le_cos m).lt_or_eq with h | h
  · linarith
  · obtain ⟨k, hk⟩ := Real.cos_eq_neg_one_iff.mp h.symm
    have hk' : ((2 * k + 1 : ℤ) : ℝ) * Real.pi = m := by push_cast; linarith
    exact absurd hk' ((irrational_pi.intCast_mul (by omega)).ne_nat m)

end Transformer.RoPE.BaseBounds
