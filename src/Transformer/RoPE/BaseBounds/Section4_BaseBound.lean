module

public import Transformer.RoPE.BaseBounds.Section4_ContextLength

/-!
# §4.3 The lower bound of the base

Men et al., arXiv:2405.14591v1, §4.3, Eq. (11): "if we follow the setting that
`θ_i = base^{-2i/d}`, in order to get the expected context length `L`, there is a lower bound of
the base value `base_L = inf{base | B_{m,θ} ≥ 0, ∀ m ∈ [0, 1, ..., L]}`".

* `tendsto_ropeKernel`: as the base grows, `θ_0 = 1` stays and the other frequencies tend to `0`,
  so `B_{m,θ} → cos m + (d/2 − 1)`;
* `eventually_ropeKernel_pos`, `exists_base`: with at least two pairs, large bases keep `B > 0` up
  to any `L`, so the set of Eq. (11) is not empty;
* `baseBound` is Eq. (11) over the bases `> 1`. The source leaves the range implicit, and the
  restriction is needed: at `base = 0` every frequency but `θ_0` is `0^x = 0`, and `B ≥ 0` at
  every distance;
* `baseBound_le_of_le_contextLength`: a base whose context length reaches `L` is at least
  `base_L`, the source's "lower bound"; `baseBound_mono`: `base_L` grows with `L`;
* `contextLength_one_pair`: with one pair (`d = 2`) the context length is `1` for every base, so
  the set of Eq. (11) is empty from `L = 2` on.

`base_L` is only necessary: nothing here puts the bases above it in the set of Eq. (11). The
values of Table 1 are numerical and are not formalized.
-/

@[expose] public section

open Filter Topology
open scoped BigOperators

namespace Transformer.RoPE.BaseBounds

/-- As the base grows, `θ_0 = 1` stays and every other RoPE frequency tends to `0`. -/
theorem tendsto_ropeKernel {pairs : ℕ} (i₀ : Fin pairs) (hi₀ : (i₀ : ℕ) = 0) (m : ℝ) :
    Tendsto (fun base => cosineKernel (ropeFrequencies base pairs) m) atTop
      (𝓝 (∑ i, if i = i₀ then Real.cos m else 1)) := by
  unfold cosineKernel
  refine tendsto_finsetSum _ fun i _ => ?_
  split_ifs with hi
  · have h1 : ∀ base, ropeFrequencies base pairs i = 1 := fun base => by
      simp [ropeFrequencies, frequency, hi, hi₀]
    simp only [h1, one_mul]
    exact tendsto_const_nhds
  · have hi' : (0 : ℝ) < (i : ℕ) := by
      have : (i : ℕ) ≠ 0 := fun h => hi (Fin.ext (h.trans hi₀.symm))
      exact_mod_cast Nat.pos_of_ne_zero this
    have hd : (0 : ℝ) < ((2 * pairs : ℕ) : ℝ) := by
      have := i.pos
      positivity
    have h0 : Tendsto (fun base => ropeFrequencies base pairs i) atTop (𝓝 0) :=
      (tendsto_rpow_neg_atTop (y := 2 * ((i : ℕ) : ℝ) / ((2 * pairs : ℕ) : ℝ))
        (div_pos (by linarith) hd)).congr fun base => by
        simp only [ropeFrequencies, frequency, neg_div]
    simpa [Function.comp_def] using (Real.continuous_cos.tendsto _).comp (h0.mul_const m)

example : Tendsto (fun base => cosineKernel (ropeFrequencies base 64) 7) atTop
    (𝓝 (∑ i, if i = (0 : Fin 64) then Real.cos 7 else 1)) :=
  tendsto_ropeKernel 0 rfl 7

/-- With at least two pairs, large bases keep `B > 0` at every distance up to `L`. -/
theorem eventually_ropeKernel_pos {pairs : ℕ} (hpairs : 2 ≤ pairs) (L : ℕ) :
    ∀ᶠ base in atTop, ∀ m ≤ L, 0 < cosineKernel (ropeFrequencies base pairs) m := by
  have hm : ∀ m ∈ Finset.range (L + 1),
      ∀ᶠ base in atTop, 0 < cosineKernel (ropeFrequencies base pairs) m := by
    intro m _
    refine (tendsto_ropeKernel ⟨0, by omega⟩ rfl m).eventually_const_lt ?_
    have hsplit : ∀ i : Fin pairs, (if i = ⟨0, by omega⟩ then Real.cos m else 1) =
        1 + (if i = ⟨0, by omega⟩ then Real.cos m - 1 else 0) := by
      intro i
      split_ifs <;> ring
    simp only [hsplit, Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ,
      Fintype.card_fin, nsmul_eq_mul, mul_one, Finset.sum_ite_eq', Finset.mem_univ, ite_true]
    have : (2 : ℝ) ≤ pairs := by exact_mod_cast hpairs
    linarith [one_add_cos_nat_pos m]
  filter_upwards [(eventually_all_finset _).2 hm] with base hb m hmL
  exact hb m (Finset.mem_range.mpr (by omega))

example : ∀ᶠ base in atTop, ∀ m ≤ (32768 : ℕ), 0 < cosineKernel (ropeFrequencies base 64) m :=
  eventually_ropeKernel_pos (by norm_num) 32768

/-- §4.3: with at least two pairs, some base reaches any context length `L`, so the set of
Eq. (11) is not empty. -/
theorem exists_base {pairs : ℕ} (hpairs : 2 ≤ pairs) (L : ℕ) :
    ∃ base, 1 < base ∧ ∀ m ≤ L, 0 ≤ cosineKernel (ropeFrequencies base pairs) m := by
  obtain ⟨base, h1, h2⟩ :=
    ((eventually_gt_atTop 1).and (eventually_ropeKernel_pos hpairs L)).exists
  exact ⟨base, h1, fun m hm => (h2 m hm).le⟩

example : ∃ base, 1 < base ∧ ∀ m ≤ (1000000 : ℕ), 0 ≤ cosineKernel (ropeFrequencies base 64) m :=
  exists_base (by norm_num) 1000000

/-- Eq. (11) over the bases `> 1`. -/
noncomputable def baseBound (pairs L : ℕ) : ℝ :=
  sInf {base : ℝ | 1 < base ∧ ∀ m ≤ L, 0 ≤ cosineKernel (ropeFrequencies base pairs) m}

/-- The first distance never breaks the kernel: every RoPE frequency is in `(0, 1]`. -/
theorem one_le_contextLength {base : ℝ} (hbase : 1 < base) (pairs : ℕ) [NeZero pairs] :
    1 ≤ contextLength (ropeFrequencies base pairs) := by
  rw [le_contextLength_iff (exists_ropeKernel_neg hbase pairs)]
  intro m hm
  refine Finset.sum_nonneg fun i _ => (Real.cos_nonneg_of_mem_Icc ⟨?_, ?_⟩)
  all_goals
    have h := ropeFrequencies_mem hbase i
    have hm' : (m : ℝ) ≤ 1 := by exact_mod_cast hm
    have : (0 : ℝ) ≤ m := Nat.cast_nonneg m
    nlinarith [Real.pi_gt_three]

example : 1 ≤ contextLength (ropeFrequencies 10000 64) := one_le_contextLength (by norm_num) 64

/-- Eq. (11): to reach the context length `L`, a base must be at least `base_L`. -/
theorem baseBound_le_of_le_contextLength {pairs L : ℕ} [NeZero pairs] {base : ℝ}
    (hbase : 1 < base) (hL : L ≤ contextLength (ropeFrequencies base pairs)) :
    baseBound pairs L ≤ base :=
  csInf_le ⟨1, fun _ hb => hb.1.le⟩
    ⟨hbase, (le_contextLength_iff (exists_ropeKernel_neg hbase pairs) L).mp hL⟩

example : baseBound 64 1 ≤ 10000 :=
  baseBound_le_of_le_contextLength (by norm_num) (one_le_contextLength (by norm_num) 64)

/-- `base_L` grows with the expected context length `L`. -/
theorem baseBound_mono {pairs : ℕ} (hpairs : 2 ≤ pairs) {L L' : ℕ} (h : L ≤ L') :
    baseBound pairs L ≤ baseBound pairs L' :=
  csInf_le_csInf ⟨1, fun _ hb => hb.1.le⟩ (exists_base hpairs L')
    fun _ hb => ⟨hb.1, fun m hm => hb.2 m (hm.trans h)⟩

example : baseBound 64 4096 ≤ baseBound 64 32768 := baseBound_mono (by norm_num) (by norm_num)

theorem one_le_baseBound {pairs : ℕ} (hpairs : 2 ≤ pairs) (L : ℕ) : 1 ≤ baseBound pairs L :=
  le_csInf (exists_base hpairs L) fun _ hb => hb.1.le

example : 1 ≤ baseBound 64 32768 := one_le_baseBound (by norm_num) 32768

/-- With one pair (`d = 2`) the kernel is `cos m` for every base, and `cos 2 < 0`. -/
theorem ropeKernel_one_pair (base : ℝ) : cosineKernel (ropeFrequencies base 1) (2 : ℕ) < 0 := by
  have h : ropeFrequencies base 1 0 = 1 := by simp [ropeFrequencies, frequency]
  simp only [cosineKernel, Fin.sum_univ_one, h, one_mul, Nat.cast_ofNat]
  exact Real.cos_neg_of_pi_div_two_lt_of_lt (by linarith [Real.pi_lt_d2])
    (by linarith [Real.pi_gt_three])

/-- With one pair the context length is `1` for every base `> 1`: the set of Eq. (11) is empty
from `L = 2` on, and the source's bound needs `d ≥ 4`. -/
theorem contextLength_one_pair {base : ℝ} (hbase : 1 < base) :
    contextLength (ropeFrequencies base 1) = 1 := by
  refine le_antisymm ?_ (one_le_contextLength hbase 1)
  by_contra! h
  have := (le_contextLength_iff (exists_ropeKernel_neg hbase 1) 2).mp h 2 le_rfl
  exact this.not_gt (ropeKernel_one_pair base)

example : contextLength (ropeFrequencies 10000 1) = 1 := contextLength_one_pair (by norm_num)

end Transformer.RoPE.BaseBounds
