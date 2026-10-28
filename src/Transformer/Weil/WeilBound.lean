module

public import Transformer.Weil.ArtinSchreier
public import Transformer.Weil.FieldSum
public import Transformer.Weil.Rationality
public import Transformer.Weil.Transfer

/-!
# Weil's bound for `∑_x ψ(f(x))`

For a nontrivial additive character `ψ` of `𝔽_p` and `f ∈ 𝔽_p[X]` of degree `m ≥ 1` prime to
`p`, `|∑_{x ∈ 𝔽_p} ψ(f(x))| ≤ (m − 1) √p`:

* over the field with `p^d` elements, `d ≥ 3`, Stepanov's bound gives
  `|S_d| = |∑_x ψ(Tr f(x))| ≤ 4 p m √p^d`;
* `S_d = −∑_ω ω^d` over the `m − 1` roots `ω` of `L^*`, so the transfer gives `|ω| ≤ √p`;
* `∑_{x ∈ 𝔽_p} ψ(f(x)) = S_1 = −∑_ω ω`.
-/

@[expose] public section

namespace Transformer.Weil

open Polynomial Finset

theorem sqrt_pow_eq_pow_sqrt {x : ℝ} (hx : 0 ≤ x) (d : ℕ) : √(x ^ d) = √x ^ d := by
  rw [show x ^ d = (√x ^ d) ^ 2 by rw [← pow_mul, mul_comm, pow_mul, Real.sq_sqrt hx],
    Real.sqrt_sq (pow_nonneg (Real.sqrt_nonneg x) d)]

example := sqrt_pow_eq_pow_sqrt (x := 4) (by norm_num) 2

variable {p : ℕ} [Fact p.Prime]

/-- `|S_d| ≤ 4 p m √p^d` for `d ≥ 3`: Stepanov's bound over the field with `p^d` elements. -/
theorem norm_lPowerSum_le {ψ : AddChar (ZMod p) ℂ} (hψ : ψ ≠ 1) {f : (ZMod p)[X]}
    (hm : 1 ≤ f.natDegree) (hpm : ¬p ∣ f.natDegree) {d : ℕ} (hd : 3 ≤ d) :
    ‖lPowerSum (addWeight ψ f) d‖ ≤ 4 * p * f.natDegree * √p ^ d := by
  have hd0 : d ≠ 0 := by omega
  let _ : Fintype (GaloisField p d) := Fintype.ofFinite _
  have hcard : Fintype.card (GaloisField p d) = p ^ d := by
    rw [← Nat.card_eq_fintype_card, GaloisField.card p d hd0]
  have h5 : 5 ≤ Fintype.card (GaloisField p d) := by
    rw [hcard]
    calc 5 ≤ 2 ^ 3 := by norm_num
      _ ≤ p ^ 3 := Nat.pow_le_pow_left (Fact.out : p.Prime).two_le 3
      _ ≤ p ^ d := Nat.pow_le_pow_right (Fact.out : p.Prime).pos hd
  have hS := sum_addChar_trace_aeval (L := GaloisField p d) ψ f
  rw [GaloisField.finrank p hd0] at hS
  have hb := norm_sum_addChar_trace_le hψ (f := f.map (algebraMap (ZMod p) (GaloisField p d)))
    (by rwa [natDegree_map]) (by rwa [natDegree_map]) h5
  simp only [eval_map_algebraMap, natDegree_map] at hb
  rw [← hS]
  refine hb.trans_eq ?_
  rw [hcard, Nat.cast_pow, sqrt_pow_eq_pow_sqrt (Nat.cast_nonneg _)]

example := norm_lPowerSum_le (p := 2) stdAddChar_ne_one (f := X) (by rw [natDegree_X])
  (by rw [natDegree_X]; norm_num) le_rfl

/-- `c_n = 0` for `n ≥ m`, in the form the transfer takes. -/
theorem lCoeff_addWeight_eq_zero {ψ : AddChar (ZMod p) ℂ} (hψ : ψ ≠ 1) {f : (ZMod p)[X]}
    (hm : 1 ≤ f.natDegree) (hpm : ¬p ∣ f.natDegree) (n : ℕ) (hn : f.natDegree ≤ n) :
    lCoeff (addWeight ψ f) n = 0 :=
  lCoeff_eq_zero hψ (by omega) (fun h => hpm ((ZMod.natCast_eq_zero_iff _ _).mp h)) hn

example := lCoeff_addWeight_eq_zero (p := 2) stdAddChar_ne_one (f := X) (by rw [natDegree_X])
  (by rw [natDegree_X]; norm_num) 1 (by rw [natDegree_X])

/-- The roots of `L^*` have absolute value at most `√p`. -/
theorem norm_le_sqrt_of_mem_roots {ψ : AddChar (ZMod p) ℂ} (hψ : ψ ≠ 1) {f : (ZMod p)[X]}
    (hm : 1 ≤ f.natDegree) (hpm : ¬p ∣ f.natDegree) {ω : ℂ}
    (hω : ω ∈ (lReverse (addWeight ψ f) f.natDegree).roots) : ‖ω‖ ≤ √p :=
  norm_le_of_mem_roots_lReverse (isMonicMul_addWeight ψ f) hm
    (lCoeff_addWeight_eq_zero hψ hm hpm) (Real.sqrt_nonneg _) (n₀ := 3)
    (fun _ hd => norm_lPowerSum_le hψ hm hpm hd) hω

/-- For `f = X²` over `𝔽₃`, `L^*` has degree `1`, so it has a root, and the root is bounded. -/
example : ∃ ω ∈ (lReverse (addWeight ZMod.stdAddChar (X ^ 2 : (ZMod 3)[X])) 2).roots,
    ‖ω‖ ≤ √3 := by
  have hm : 1 ≤ (X ^ 2 : (ZMod 3)[X]).natDegree := by rw [natDegree_X_pow]; norm_num
  obtain ⟨ω, hω⟩ := Multiset.card_pos_iff_exists_mem.mp (by
    rw [card_roots_lReverse (isMonicMul_addWeight _ _) (by norm_num : 1 ≤ 2)]; norm_num :
      0 < (lReverse (addWeight ZMod.stdAddChar (X ^ 2 : (ZMod 3)[X])) 2).roots.card)
  have hb := norm_le_sqrt_of_mem_roots stdAddChar_ne_one hm (by rw [natDegree_X_pow]; norm_num)
    (ω := ω) (by rwa [natDegree_X_pow])
  exact ⟨ω, hω, by exact_mod_cast hb⟩

/-- **Weil's bound**: for a nontrivial additive character `ψ` of `𝔽_p` and `f ∈ 𝔽_p[X]` of
degree `m ≥ 1` prime to `p`, `|∑_{x ∈ 𝔽_p} ψ(f(x))| ≤ (m − 1) √p`. -/
theorem norm_sum_addChar_eval_le {ψ : AddChar (ZMod p) ℂ} (hψ : ψ ≠ 1) {f : (ZMod p)[X]}
    (hm : 1 ≤ f.natDegree) (hpm : ¬p ∣ f.natDegree) :
    ‖∑ x : ZMod p, ψ (f.eval x)‖ ≤ ((f.natDegree : ℝ) - 1) * √p := by
  have h1 := sum_addChar_trace_aeval (L := ZMod p) ψ f
  rw [Module.finrank_self] at h1
  simp only [Algebra.trace_self, LinearMap.id_apply, coe_aeval_eq_eval] at h1
  rw [h1]
  exact norm_lPowerSum_one_le (isMonicMul_addWeight ψ f) hm (lCoeff_addWeight_eq_zero hψ hm hpm)
    (Real.sqrt_nonneg _) (n₀ := 3) fun _ hd => norm_lPowerSum_le hψ hm hpm hd

example := norm_sum_addChar_eval_le (p := 2) stdAddChar_ne_one (f := X) (by rw [natDegree_X])
  (by rw [natDegree_X]; norm_num)

end Transformer.Weil
