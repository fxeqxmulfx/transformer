module

public import Mathlib.RingTheory.AdjoinRoot
public import Transformer.Weil.PowWeil
public import Transformer.Weil.Rationality

/-!
# The extensions of `𝔽_p` of degree divisible by `deg P`

Three facts used by `PowWeilRoot` for `f = P^e Q`, `P` irreducible of degree `s`, `P ∤ Q`:

* `Rationality` along the multiples `k = s j`: if `|S_{s j}| ≤ M R^{s j}` for all large `j`,
  the inverse roots `ω` of the `L`-function have `|∑_ω (ω^s)^j| = |S_{s j}| ≤ M (R^s)^j`, so
  `|ω^s| ≤ R^s` (`Transfer`), `|ω| ≤ R`, and `|S_1| ≤ (m − 1) R`;
* on the field `L` with `p^k` elements, `χ_L = χ ∘ N_{L/𝔽_p}` has `χ_L(u) = 1 ⇒ u^c = 1` with
  `d c + 1 = p^k`, `d` the order of `χ` (`PowWeil`);
* for `s ∣ k`, `L` contains a copy of `𝔽_p[X]/(P)`, hence a root `α` of `P`, simple as `P` is
  separable, with `Q(α) ≠ 0` as `a P + b Q = 1`: so `f = (X − α)^e G` over `L`, `G(α) ≠ 0`.
-/

@[expose] public section

namespace Transformer.Weil

open Polynomial Finset

section Rational

variable {K : Type*} [Field K] [Fintype K] {w : K[X] → ℂ} (hw : IsMonicMul w) {m : ℕ}
  (hm : 1 ≤ m) (hvan : ∀ n, m ≤ n → lCoeff w n = 0) {M R : ℝ} (hR : 0 ≤ R) {s n₀ : ℕ}
  (hs : s ≠ 0) (hS : ∀ j, n₀ ≤ j → ‖lPowerSum w (s * j)‖ ≤ M * R ^ (s * j))

include hw hm hvan hR hs hS

/-- **The roots from the extensions of degree divisible by `s`**: if `|S_{s j}| ≤ M R^{s j}`
for all `j ≥ n₀`, every root of `L^*` has `|ω| ≤ R`. -/
theorem norm_le_of_mem_roots_lReverse_of_mul {ω : ℂ} (hω : ω ∈ (lReverse w m).roots) :
    ‖ω‖ ≤ R := by
  have h := norm_le_of_norm_sum_pow_le (s := (lReverse w m).roots.map (· ^ s))
    (pow_nonneg hR s) (M := M) (n₀ := n₀ + 1) (fun j hj => by
      have hmap : ((lReverse w m).roots.map (· ^ s)).map (· ^ j) =
          (lReverse w m).roots.map (· ^ (s * j)) := by
        rw [Multiset.map_map]
        exact Multiset.map_congr rfl fun x _ => (pow_mul x s j).symm
      rw [hmap, ← norm_neg, ← lPowerSum_eq_neg_sum_roots hw hm hvan (Nat.mul_ne_zero hs
        (by omega)), ← pow_mul]
      exact hS j (by omega)) (Multiset.mem_map_of_mem _ hω)
  rw [norm_pow] at h
  exact (pow_le_pow_iff_left₀ (norm_nonneg ω) hR hs).mp h

/-- **The bound over `K` from the extensions of degree divisible by `s`**: if
`|S_{s j}| ≤ M R^{s j}` for all `j ≥ n₀`, then `|S_1| ≤ (m − 1) R`. -/
theorem norm_lPowerSum_one_le_of_mul : ‖lPowerSum w 1‖ ≤ ((m : ℝ) - 1) * R := by
  rw [lPowerSum_eq_neg_sum_roots hw hm hvan one_ne_zero, norm_neg]
  simp only [pow_one, Multiset.map_id']
  calc ‖(lReverse w m).roots.sum‖ ≤ ((lReverse w m).roots.map (‖·‖)).sum :=
        norm_multiset_sum_le _
    _ ≤ ((lReverse w m).roots.map (‖·‖)).card • R :=
        Multiset.sum_le_card_nsmul _ _ fun x hx => by
          obtain ⟨ω, hω, rfl⟩ := Multiset.mem_map.mp hx
          exact norm_le_of_mem_roots_lReverse_of_mul hw hm hvan hR hs hS hω
    _ = ((m : ℝ) - 1) * R := by
        rw [Multiset.card_map, card_roots_lReverse hw hm, nsmul_eq_mul, Nat.cast_sub hm,
          Nat.cast_one]

end Rational

variable {p : ℕ} [Fact p.Prime]

/-- `d c + 1 = p^k` and `χ_L(u) = 1 ⇒ u^c = 1` for `χ_L = χ ∘ N` on the field with `p^k`
elements, `d` the order of `χ`. -/
theorem exists_normMulChar_eq_one_imp_pow_eq_one (χ : MulChar (ZMod p) ℂ) {k : ℕ}
    (hk : k ≠ 0) : ∃ c : ℕ, orderOf χ * c + 1 = p ^ k ∧
      ∀ u : GaloisField p k, normMulChar χ u = 1 → u ^ c = 1 := by
  obtain ⟨r, hr⟩ := MulChar.orderOf_dvd_card_sub_one (χ := χ)
  refine ⟨(∑ i ∈ range k, p ^ i) * r, ?_, fun u hu => ?_⟩
  · have hgeom := geom_sum_mul_add (p - 1) k
    rw [Nat.sub_add_cancel (Fact.out : p.Prime).one_lt.le,
      show p - 1 = orderOf χ * r by rw [← hr, ZMod.card]] at hgeom
    rw [← hgeom]
    ring
  · have h := pow_eq_one_of_normMulChar_eq_one hr hu
    rwa [GaloisField.finrank p hk, ZMod.card] at h

/-- In the field with `p^k` elements, `deg P ∣ k`, an irreducible `P` has a root `α`, and
`P^e Q` with `P ∤ Q` is `(X − α)^e G` with `G(α) ≠ 0`. -/
theorem exists_map_pow_mul_eq_X_sub_C_pow_mul {P Q : (ZMod p)[X]} (hP : Irreducible P)
    (hPQ : ¬P ∣ Q) (e : ℕ) {k : ℕ} (hk : k ≠ 0) (hs : P.natDegree ∣ k) :
    ∃ α : GaloisField p k, ∃ G : (GaloisField p k)[X],
      (P ^ e * Q).map (algebraMap (ZMod p) (GaloisField p k)) = (X - C α) ^ e * G ∧
        G.eval α ≠ 0 := by
  have : Fact (Irreducible P) := ⟨hP⟩
  obtain ⟨φ⟩ := FiniteField.nonempty_algHom_of_finrank_dvd (F := ZMod p) (K := AdjoinRoot P)
    (L := GaloisField p k) (by
      rwa [(AdjoinRoot.powerBasis hP.ne_zero).finrank, AdjoinRoot.powerBasis_dim,
        GaloisField.finrank p hk])
  have hPα : aeval (φ (AdjoinRoot.root P)) P = 0 := by
    rw [aeval_algHom_apply, AdjoinRoot.aeval_eq, AdjoinRoot.mk_self, map_zero]
  obtain ⟨P₁, hP₁⟩ := dvd_iff_isRoot.mpr (show IsRoot (P.map (algebraMap (ZMod p)
    (GaloisField p k))) (φ (AdjoinRoot.root P)) by rw [IsRoot, eval_map_algebraMap, hPα])
  have hP₁α : P₁.eval (φ (AdjoinRoot.root P)) ≠ 0 := by
    intro h0
    obtain ⟨P₂, hP₂⟩ := dvd_iff_isRoot.mpr h0
    exact not_isUnit_X_sub_C _ (((PerfectField.separable_of_irreducible hP).map
      (f := algebraMap (ZMod p) (GaloisField p k))).squarefree _
        ⟨P₂, by rw [hP₁, hP₂, mul_assoc]⟩)
  obtain ⟨a, b, hab⟩ := hP.coprime_iff_not_dvd.mpr hPQ
  have hQα : aeval (φ (AdjoinRoot.root P)) Q ≠ 0 := by
    intro h0
    have h1 := congrArg (aeval (φ (AdjoinRoot.root P))) hab
    rw [map_add, map_mul, map_mul, hPα, h0, mul_zero, mul_zero, add_zero, map_one] at h1
    exact zero_ne_one h1
  refine ⟨φ (AdjoinRoot.root P), P₁ ^ e * Q.map (algebraMap (ZMod p) (GaloisField p k)), ?_, ?_⟩
  · rw [Polynomial.map_mul, Polynomial.map_pow, hP₁, mul_pow, mul_assoc]
  · rw [eval_mul, eval_pow, eval_map_algebraMap]
    exact mul_ne_zero (pow_ne_zero _ hP₁α) hQα

section Examples

/-- For the trivial character of `𝔽₃` and `k = 1`. -/
example := exists_normMulChar_eq_one_imp_pow_eq_one (p := 3) (1 : MulChar (ZMod 3) ℂ)
  one_ne_zero

example := exists_map_pow_mul_eq_X_sub_C_pow_mul (p := 3) irreducible_X (Q := X + C 1)
  (by rw [X_dvd_iff]; simp) 2 one_ne_zero (by rw [natDegree_X])

example (ω : ℂ) (hω : ω ∈ (lReverse (addWeight ZMod.stdAddChar (X : (ZMod 2)[X])) 1).roots) :=
  norm_le_of_mem_roots_lReverse_of_mul isMonicMul_addWeight_X le_rfl
    (fun _ => lCoeff_addWeight_X_eq_zero) zero_le_one (M := 0) (s := 2) (n₀ := 1) two_ne_zero
    (fun j hj => by rw [lPowerSum_addWeight_X_eq_zero (by omega), norm_zero, zero_mul]) hω

example := norm_lPowerSum_one_le_of_mul isMonicMul_addWeight_X le_rfl
  (fun _ => lCoeff_addWeight_X_eq_zero) zero_le_one (M := 0) (s := 2) (n₀ := 1) two_ne_zero
  fun j hj => by rw [lPowerSum_addWeight_X_eq_zero (by omega), norm_zero, zero_mul]

end Examples

end Transformer.Weil
