module

public import Transformer.Grokking.FlatSpectrum.AddReadoutSpectrum
public import Transformer.Grokking.FlatSpectrum.RealKernels

/-!
# The least effective rank of a centered readout of `(a + b)²` is `2`

Source: `review/flat_spectrum.md`, Readout-free version: "E chooses the readout, so the right
invariant is the minimum of r_eff(φ∘f) over readouts φ. For (a+b)², gradient search finds 2.0",
and the torch check "2.00 at p = 47, 61, 97". The kernel `φ((a + b)²)` is `g(a + b)` with
`g(s) = φ(s²)`, so by `AddReadoutSpectrum` its singular values are the `|ĝ(k)|`. As `g` is even,
so is `ĝ`, and a centered readout, `∑_s φ(s²) = 0`, has `ĝ(0) = 0`. For odd prime `p` the
nonzero `k` pair off with `−k ≠ k`, so `|ĝ(k)| ∑_l |ĝ(l)| ≥ |ĝ(k)| (|ĝ(k)| + |ĝ(−k)|) = 2 |ĝ(k)|²`
and `(∑_k |ĝ(k)|)² ≥ 2 ∑_k |ĝ(k)|²`: `r_eff ≥ 2` for every centered readout not zero on the
squares. A readout with `φ(s²) = cos(2π r s/p)`, `r ≠ 0`, has `ĝ = (p/2)([k = r] + [k = −r])`
and `r_eff = 2`. So the minimum is exactly `2` for every odd prime `p`. Centering is needed: a
constant readout has `r_eff = 1` (`effectiveRank_taskKernel_add_const`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset Real

/-- If `x ≥ 0` vanishes at `0`, `x(−k) = x(k)` and `k ≠ −k` for `k ≠ 0`, then
`2 ∑ x_k² ≤ (∑ x_k)²`. -/
lemma two_mul_sum_sq_le_sq_sum {N : ℕ} [NeZero N] {x : ZMod N → ℝ} (hx : ∀ k, 0 ≤ x k)
    (h0 : x 0 = 0) (hneg : x.Even) (hN : ∀ k : ZMod N, k ≠ 0 → k ≠ -k) :
    2 * ∑ k, x k ^ 2 ≤ (∑ k, x k) ^ 2 := by
  have hk (k : ZMod N) : 2 * x k ^ 2 ≤ x k * ∑ l, x l := by
    rcases eq_or_ne k 0 with rfl | hk
    · simp [h0]
    · have hpair : x k + x (-k) ≤ ∑ l, x l := by
        rw [← sum_pair (hN k hk)]
        exact sum_le_sum_of_subset_of_nonneg (subset_univ _) fun l _ _ => hx l
      rw [hneg k] at hpair
      nlinarith [hx k]
  calc 2 * ∑ k, x k ^ 2 = ∑ k, 2 * x k ^ 2 := by rw [mul_sum]
    _ ≤ ∑ k, x k * ∑ l, x l := sum_le_sum fun k _ => hk k
    _ = (∑ k, x k) ^ 2 := by rw [← sum_mul, sq]

example := two_mul_sum_sq_le_sq_sum (N := 3) (x := fun _ => 0) (fun _ => le_rfl) rfl
  (fun _ => rfl) (by decide)

variable {p : ℕ} [Fact p.Prime]

/-- `r_eff [g(a + b)] ≥ 2` for every even `g ≠ 0` with `∑_s g(s) = 0`, `p` an odd prime. -/
theorem two_le_effectiveRank_taskKernel_add_of_even (hp : p ≠ 2) {g : ZMod p → ℂ}
    (heven : g.Even) (hsum : ∑ s, g s = 0) (hne : g ≠ 0) :
    2 ≤ effectiveRank (taskKernel (· + ·) g) := by
  rw [effectiveRank_taskKernel_add_eq_dft]
  obtain ⟨k₀, hk₀⟩ := Function.ne_iff.mp (ZMod.dft.map_ne_zero_iff.mpr hne)
  have hQ : 0 < ∑ k, ‖ZMod.dft g k‖ ^ 2 :=
    sum_pos' (fun k _ => sq_nonneg _) ⟨k₀, mem_univ _, pow_pos (norm_pos_iff.mpr hk₀) 2⟩
  rw [le_div_iff₀ hQ]
  refine two_mul_sum_sq_le_sq_sum (x := fun k => ‖ZMod.dft g k‖) (fun k => norm_nonneg _) ?_
    (fun k => ?_) fun k hk h => hk ((eq_neg_iff_of_ne_two hp).mp h)
  · simp only [ZMod.dft_apply_zero, hsum, norm_zero]
  · simp only [ZMod.dft_even_iff.mpr heven k]

/-- `s ↦ cos(2π r s/N)` is even. -/
lemma cos_readout_even {N : ℕ} [NeZero N] (r : ZMod N) :
    Function.Even fun s => (ZMod.stdAddChar (r * s) + ZMod.stdAddChar (-(r * s))) / 2 :=
  fun s => by simp only [mul_neg, neg_neg, add_comm]

/-- `∑_s cos(2π r s/N) = 0` for `r ≠ 0`: it is the transform at `0`. -/
lemma sum_cos_readout {N : ℕ} [NeZero N] {r : ZMod N} (hr : r ≠ 0) :
    ∑ s, (ZMod.stdAddChar (r * s) + ZMod.stdAddChar (-(r * s))) / 2 = 0 := by
  have h := dft_cos r 0
  rw [ZMod.dft_apply_zero, ite_eq_right hr, ite_eq_right (neg_ne_zero.mpr hr)] at h
  simpa using h

example := two_le_effectiveRank_taskKernel_add_of_even (p := 3) (by decide) (cos_readout_even 1)
  (sum_cos_readout one_ne_zero) fun h => by simpa using congrFun h 0

/-- `r_eff [cos(2π r (a + b)/p)] = 2` for `r ≠ 0`, `p` an odd prime. -/
theorem effectiveRank_taskKernel_add_cos (hp : p ≠ 2) {r : ZMod p} (hr : r ≠ 0) :
    effectiveRank (taskKernel (· + ·)
      fun s => (ZMod.stdAddChar (r * s) + ZMod.stdAddChar (-(r * s))) / 2) = 2 := by
  have hrr : r ≠ -r := fun h => hr ((eq_neg_iff_of_ne_two hp).mp h)
  have hP : (p : ℝ) ≠ 0 := by exact_mod_cast (Fact.out : p.Prime).ne_zero
  have hsq (k : ZMod p) : ((if r = k then (1 : ℝ) else 0) + if -r = k then 1 else 0) ^ 2 =
      (if r = k then 1 else 0) + if -r = k then 1 else 0 := by
    split_ifs with h1 h2 <;> first | exact absurd (h1.trans h2.symm) hrr | norm_num
  have hnorm (k : ZMod p) : ‖ZMod.dft
      (fun s => (ZMod.stdAddChar (r * s) + ZMod.stdAddChar (-(r * s))) / 2) k‖ =
      p / 2 * ((if r = k then 1 else 0) + if -r = k then 1 else 0) := by
    rw [dft_cos, Complex.norm_of_nonneg (by positivity)]
  rw [effectiveRank_taskKernel_add_eq_dft]
  simp only [hnorm, mul_pow, hsq, ← mul_sum, sum_add_distrib, sum_ite_eq, mem_univ, ite_true]
  field_simp
  ring

example := effectiveRank_taskKernel_add_cos (p := 3) (by decide) one_ne_zero

/-- The readout `t ↦ cos(2π r √t/p)` of `(a + b)²`, for a chosen square root `√t` of `t`, and
`0` off the squares. As `cos` is even, its value at `s²` does not depend on the choice. -/
noncomputable def cosSqReadout (r t : ZMod p) : ℂ :=
  if h : ∃ s, s ^ 2 = t then (cos (2 * π * (r * h.choose).val / p) : ℂ) else 0

/-- `φ_r(s²) = cos(2π r s/p) = (ψ(r s) + ψ(−r s))/2`. -/
lemma cosSqReadout_sq (r s : ZMod p) :
    cosSqReadout r (s ^ 2) = (ZMod.stdAddChar (r * s) + ZMod.stdAddChar (-(r * s))) / 2 := by
  have h : ∃ s', s' ^ 2 = s ^ 2 := ⟨s, rfl⟩
  rw [cosSqReadout, dite_eq_left h, ofReal_cos_eq]
  rcases sq_eq_sq_iff_eq_or_eq_neg.mp h.choose_spec with h' | h'
  · rw [h']
  · rw [h', mul_neg, neg_neg, add_comm]

/-- `φ_r` is centered for `r ≠ 0`: `∑_s φ_r(s²) = ∑_s cos(2π r s/p) = 0`. -/
lemma sum_cosSqReadout {r : ZMod p} (hr : r ≠ 0) : ∑ s, cosSqReadout r (s ^ 2) = 0 := by
  simp only [cosSqReadout_sq]
  exact sum_cos_readout hr

example := sum_cosSqReadout (p := 3) one_ne_zero

/-- `r_eff [φ_r((a + b)²)] = r_eff [cos(2π r (a + b)/p)] = 2` for `r ≠ 0`, `p` an odd prime. -/
theorem effectiveRank_sq_add_cosSqReadout (hp : p ≠ 2) {r : ZMod p} (hr : r ≠ 0) :
    effectiveRank (taskKernel (fun a b => (a + b) ^ 2) (cosSqReadout r)) = 2 := by
  have hK : taskKernel (fun a b => (a + b) ^ 2) (cosSqReadout r) = taskKernel (· + ·)
      fun s => (ZMod.stdAddChar (r * s) + ZMod.stdAddChar (-(r * s))) / 2 := by
    ext a b
    simp only [taskKernel, of_apply, cosSqReadout_sq]
  rw [hK, effectiveRank_taskKernel_add_cos hp hr]

example := effectiveRank_sq_add_cosSqReadout (p := 3) (by decide) one_ne_zero

/-- **The `(a + b)²` readouts**, `review/flat_spectrum.md`, Readout-free version: "For (a+b)²,
gradient search finds 2.0". For odd prime `p`, every centered readout `φ`, `∑_s φ(s²) = 0`, not
zero on the squares, has `r_eff [φ((a + b)²)] ≥ 2`. -/
theorem two_le_effectiveRank_sq_add (hp : p ≠ 2) {φ : ZMod p → ℂ} (hφ : ∑ s, φ (s ^ 2) = 0)
    (hne : ∃ s, φ (s ^ 2) ≠ 0) :
    2 ≤ effectiveRank (taskKernel (fun a b => (a + b) ^ 2) φ) := by
  obtain ⟨s, hs⟩ := hne
  exact two_le_effectiveRank_taskKernel_add_of_even (g := fun s => φ (s ^ 2)) hp
    (fun s => by simp only [neg_sq]) hφ fun h => hs (congrFun h s)

example := two_le_effectiveRank_sq_add (p := 3) (by decide) (sum_cosSqReadout one_ne_zero)
  ⟨0, by rw [cosSqReadout_sq]; simp⟩

/-- A constant readout `c ≠ 0` of `(a + b)²` has `r_eff = 1`: over all readouts, the minimum the
note asks for is not `2`. -/
theorem effectiveRank_sq_add_const {c : ℂ} (hc : c ≠ 0) :
    effectiveRank (taskKernel (fun a b => (a + b) ^ 2) fun _ : ZMod p => c) = 1 :=
  effectiveRank_taskKernel_add_const hc

example := effectiveRank_sq_add_const (p := 3) one_ne_zero

/-- **The `(a + b)²` readout minimum**, `review/flat_spectrum.md`, Readout-free version: "the
minimum of r_eff(φ∘f) over readouts φ. For (a+b)², gradient search finds 2.0", torch check
"2.00 at p = 47, 61, 97". Changed: the minimum runs over the centered readouts, `∑_s φ(s²) = 0`,
not zero on the squares, since a constant readout gives `1` (`effectiveRank_sq_add_const`).
For every odd prime `p` this minimum is `2`, attained by `φ(s²) = cos(2π s/p)`. -/
theorem isLeast_effectiveRank_sq_add (hp : p ≠ 2) :
    IsLeast {e | ∃ φ : ZMod p → ℂ, ∑ s, φ (s ^ 2) = 0 ∧ (∃ s, φ (s ^ 2) ≠ 0) ∧
      effectiveRank (taskKernel (fun a b => (a + b) ^ 2) φ) = e} 2 :=
  ⟨⟨cosSqReadout 1, sum_cosSqReadout one_ne_zero, ⟨0, by rw [cosSqReadout_sq]; simp⟩,
    effectiveRank_sq_add_cosSqReadout hp one_ne_zero⟩,
    fun _ ⟨_, hφ, hne, he⟩ => he ▸ two_le_effectiveRank_sq_add hp hφ hne⟩

example := isLeast_effectiveRank_sq_add (p := 3) (by decide)

end Transformer.Grokking.FlatSpectrum
