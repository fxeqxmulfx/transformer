module

public import Mathlib.Analysis.SpecialFunctions.Complex.Arg
public import Transformer.RoPE.RoFormer.Section3_Rotary

/-!
# §3.4.1 Derivation of RoPE under 2D

Su et al., *RoFormer*, arXiv:2104.09864v5, §3.4.1, Eqs. (20)–(33). The source looks for `f_q, f_k`
with `⟨f_q(x_q, m), f_k(x_k, n)⟩ = g(x_q, x_k, n − m)` (Eq. (21)) and the initial condition
`q = f_q(x_q, 0)`, `k = f_k(x_k, 0)` (Eq. (22)). Splitting `g = R_g e^{iΘ_g}` gives
`R_q R_k = R_g` and `Θ_k − Θ_q = Θ_g` (Eq. (24)), so the product of Eq. (21) is the complex one,
`conj(f_q) f_k`; the positions are integers ("`φ(m)` with continuous integer inputs").

* `conj_rotate_mul_rotate`: the rotations `f(x, m) = x e^{imθ}` solve Eqs. (21)–(22);
* `eq_rotate_of_relative`: with the source's radial solution `R_q(x_q, m) = ‖q‖` (Eq. (27)), they
  are the only solutions: `f_q(x, m) = f_k(x, m) = x e^{imθ}` for one real `θ` (Eq. (33)). The
  initial condition already forces the `γ = 0` that the source sets by hand after Eq. (31);
* `exists_relative_not_rotate`: the radial solution is a choice, as the source's "a
  straightforward solution" says: `f_q(q, m) = q 2^m` and `f_k(k, n) = k 2^{−n}` also solve
  Eqs. (21)–(22), and they do not keep lengths.
-/

@[expose] public section

open scoped ComplexConjugate

namespace Transformer.RoPE.RoFormer

/-- §3.2.1, Eq. (12) as a solution of Eqs. (21)–(22): `conj(q e^{imθ}) k e^{inθ}` is
`conj(q) k e^{i(n−m)θ}`, a function of `n − m`. -/
theorem conj_rotate_mul_rotate (θ m n : ℝ) (q k : ℂ) :
    conj (rotate θ m q) * rotate θ n k =
      conj q * k * Complex.exp (((n - m) * θ : ℝ) * Complex.I) := by
  simp only [rotate, map_mul, ← Complex.exp_conj, Complex.conj_ofReal, Complex.conj_I]
  rw [show conj q * Complex.exp (↑(m * θ) * -Complex.I) * (k * Complex.exp (↑(n * θ) * Complex.I))
      = conj q * k * (Complex.exp (↑(m * θ) * -Complex.I) * Complex.exp (↑(n * θ) * Complex.I))
      by ring, ← Complex.exp_add]
  congr 2
  push_cast
  ring

/-- §3.4.1, Eqs. (21)–(33): if `conj(f_q(q, m)) f_k(k, n) = g(q, k, n − m)` (Eqs. (21), (24)),
`f_q(q, 0) = q` and `f_k(k, 0) = k` (Eq. (22)), and `‖f_q(q, m)‖ = ‖q‖` (Eq. (27)), then
`f_q(x, m) = f_k(x, m) = x e^{imθ}` for one real `θ` (Eq. (33)). -/
theorem eq_rotate_of_relative {fq fk : ℂ → ℕ → ℂ} {g : ℂ → ℂ → ℤ → ℂ}
    (hrel : ∀ q k m n, conj (fq q m) * fk k n = g q k ((n : ℤ) - m))
    (hq0 : ∀ q, fq q 0 = q) (hk0 : ∀ k, fk k 0 = k) (hq : ∀ q m, ‖fq q m‖ = ‖q‖) :
    ∃ θ : ℝ, ∀ x m, fq x m = rotate θ m x ∧ fk x m = rotate θ m x := by
  have hunit : ∀ m, conj (fq 1 m) * fq 1 m = 1 := fun m => by
    rw [mul_comm, Complex.mul_conj', hq]
    simp
  -- Eq. (26): on the diagonal the product is the one at position `0`
  have hdiag : ∀ q k m, conj (fq q m) * fk k m = conj q * k := fun q k m => by
    have h0 := hrel q k 0 0
    rw [hq0, hk0] at h0
    rw [hrel, h0]
    simp
  have hk1 : ∀ m, fk 1 m = fq 1 m := fun m => by
    have h := hdiag 1 1 m
    rw [map_one, mul_one] at h
    calc fk 1 m = fq 1 m * conj (fq 1 m) * fk 1 m := by rw [mul_comm (fq 1 m), hunit, one_mul]
      _ = fq 1 m * (conj (fq 1 m) * fk 1 m) := by ring
      _ = fq 1 m := by rw [h, mul_one]
  have hfq : ∀ q m, fq q m = q * fq 1 m := fun q m => by
    have h := congrArg conj (hdiag q 1 m)
    rw [hk1, mul_one, map_mul, Complex.conj_conj, Complex.conj_conj] at h
    calc fq q m = fq q m * (conj (fq 1 m) * fq 1 m) := by rw [hunit, mul_one]
      _ = fq q m * conj (fq 1 m) * fq 1 m := by ring
      _ = q * fq 1 m := by rw [h]
  have hfk : ∀ k m, fk k m = k * fq 1 m := fun k m => by
    have h := hdiag 1 k m
    rw [map_one, one_mul] at h
    calc fk k m = fq 1 m * conj (fq 1 m) * fk k m := by rw [mul_comm (fq 1 m), hunit, one_mul]
      _ = fq 1 m * (conj (fq 1 m) * fk k m) := by ring
      _ = k * fq 1 m := by rw [h, mul_comm]
  -- Eqs. (29)–(30): consecutive positions differ by the factor `f_q(1, 1)`
  have hstep : ∀ m, fq 1 (m + 1) = fq 1 m * fq 1 1 := fun m => by
    have h := hrel 1 1 m (m + 1)
    have h1 := hrel 1 1 0 1
    rw [hk1] at h h1
    rw [hq0, map_one, one_mul] at h1
    rw [show ((m + 1 : ℕ) : ℤ) - m = ((1 : ℕ) : ℤ) - ((0 : ℕ) : ℤ) by push_cast; ring, ← h1] at h
    calc fq 1 (m + 1) = fq 1 m * conj (fq 1 m) * fq 1 (m + 1) := by
          rw [mul_comm (fq 1 m), hunit, one_mul]
      _ = fq 1 m * (conj (fq 1 m) * fq 1 (m + 1)) := by ring
      _ = fq 1 m * fq 1 1 := by rw [h]
  have hpow : ∀ m, fq 1 m = fq 1 1 ^ m := fun m => by
    induction m with
    | zero => rw [hq0, pow_zero]
    | succ m ih => rw [hstep, ih, pow_succ]
  have hc : fq 1 1 = Complex.exp ((Complex.arg (fq 1 1) : ℂ) * Complex.I) := by
    have h := Complex.norm_mul_exp_arg_mul_I (fq 1 1)
    rw [hq] at h
    simpa using h.symm
  have hrot : ∀ x (m : ℕ), x * fq 1 1 ^ m = rotate (Complex.arg (fq 1 1)) m x := fun x m => by
    rw [rotate]
    congr 1
    conv_lhs => rw [hc]
    rw [← Complex.exp_nat_mul]
    congr 1
    push_cast
    ring
  refine ⟨Complex.arg (fq 1 1), fun x m => ?_⟩
  exact ⟨by rw [hfq, hpow, hrot], by rw [hfk, hpow, hrot]⟩

example : ∃ θ : ℝ, ∀ (x : ℂ) (m : ℕ), rotate 1 m x = rotate θ m x ∧ rotate 1 m x = rotate θ m x :=
  eq_rotate_of_relative (fq := fun x m => rotate 1 m x) (fk := fun x m => rotate 1 m x)
    (g := fun q k d => conj q * k * Complex.exp (((d : ℝ) * 1 : ℝ) * Complex.I))
    (fun q k m n => by rw [conj_rotate_mul_rotate]; push_cast; ring_nf)
    (fun q => by simp [rotate]) (fun k => by simp [rotate])
    (fun q m => by rw [rotate, norm_mul, Complex.norm_exp_ofReal_mul_I, mul_one])

/-- §3.4.1: Eqs. (21)–(22) do not force the radial solution of Eq. (27). `f_q(q, m) = q 2^m` and
`f_k(k, n) = k 2^{−n}` give `conj(f_q(q, m)) f_k(k, n) = conj(q) k 2^{−(n−m)}`, and
`‖f_q(1, 1)‖ = 2`. -/
theorem exists_relative_not_rotate :
    ∃ (fq fk : ℂ → ℕ → ℂ) (g : ℂ → ℂ → ℤ → ℂ),
      (∀ q k m n, conj (fq q m) * fk k n = g q k ((n : ℤ) - m)) ∧
        (∀ q, fq q 0 = q) ∧ (∀ k, fk k 0 = k) ∧ ‖fq 1 1‖ ≠ 1 := by
  refine ⟨fun q m => q * 2 ^ m, fun k n => k * (2 ^ n)⁻¹, fun q k d => conj q * k * 2 ^ (-d),
    fun q k m n => ?_, fun q => by simp, fun k => by simp, by norm_num⟩
  dsimp only
  rw [neg_sub, zpow_sub₀ two_ne_zero, zpow_natCast, zpow_natCast, map_mul, map_pow,
    map_ofNat]
  ring

end Transformer.RoPE.RoFormer
