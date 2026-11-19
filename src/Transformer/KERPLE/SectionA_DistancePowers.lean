module

public import Transformer.KERPLE.Section4_BiasKernel
public import Transformer.KERPLE.SectionA_PDFunctions
public import Transformer.KERPLE.SectionA_ShiftLemma

/-!
# Appendix A.2: Fact 6 and Proposition 2

Chi, Fan, Ramadge, Rudnicky, arXiv:2205.09921v2, Appendix A.2.

* Fact 6 (cited as Theorem 4 of Schoenberg 1938): "`f(x)` is bounded away from zero (`f(x) > 0`)
  and its positive powers `f(x)^λ` (`λ > 0`) are all positive definite if and only if `f(x)` is
  of the form `f(x) = exp(c + ψ(x))`, where `ψ(x)` is positive definite and `c` is a real
  constant." Read with the paper's gloss `f(x) > 0`, which its proof of Proposition 2 needs
  (`exp(−|x|^p)` has infimum `0`), the "if" holds (`fact6_of_isPDFunction`) and the "only if" is
  false (`not_fact6`): `exp(−x²)` is positive with PD powers `exp(−λx²)`, but
  `exp(−x²) = exp(c + ψ)` forces `ψ(x − y) = −c − (x − y)²`, PD for no `c`. With `ψ` CPD instead
  of PD the equivalence holds (`fact6_iff`): the powers `f^λ` give the CPD kernels
  `(f^λ − 1)/λ`, whose limit as `λ → 0⁺` is `log f`. The usual meaning of "bounded away from
  zero", `inf f > 0`, excludes the counterexample; that reading is not formalized here.
* Proposition 2 (Kernel from Distance Powers): "For any `p ∈ (0, 2]`, there exists `c_min ∈ ℝ`
  such that for any `c ≥ c_min`, `k(m, n) = c − |m − n|^p` is a positive definite kernel. When
  `p > 2`, there is no `c` to make `k(m, n)` positive definite." The first part is false for
  every `p > 0` (`not_prop2_part1`): `c − |x − y|^p` is PD for no `c` on `ℝ`, nor on `ℕ`
  (`not_isPDKernel_const_sub_abs_rpow`); its proof applies the false direction of Fact 6 to
  `exp(−|x|^p)`, positive with infimum `0`. The
  second part holds, by the paper's proof through Fact 5 (`prop2_part2`), and for every `p > 0`.
  At `p = 2` three points already defeat every `c` (`not_isPDKernel_const_sub_sq`).
* §1 says "ALiBi implicitly admits a PD kernel of the form `c − |m − n|`": no real `c` makes it
  PD on the positions `ℕ` (`not_isPDKernel_alibi`).
-/

@[expose] public section

namespace Transformer.KERPLE

open Filter Topology

/-- **Fact 6, corrected**: `f` is positive with all positive powers `f^λ` PD iff
`f(x) = exp(c + ψ(x))` for a real `c` and a `ψ` whose kernel `ψ(x − y)` is CPD (in Fact 6, PD).
"Only if" takes `ψ = log f`, the limit of the CPD kernels `(f^λ − 1)/λ` as `λ → 0⁺`; "if" is
Schoenberg's theorem. -/
theorem fact6_iff {G : Type*} [Sub G] {f : G → ℝ} :
    ((∀ x, 0 < f x) ∧ ∀ l : ℝ, 0 < l → IsPDFunction fun x => f x ^ l) ↔
      ∃ (c : ℝ) (ψ : G → ℝ), IsCPDKernel (fun x y => ψ (x - y)) ∧
        ∀ x, f x = Real.exp (c + ψ x) := by
  constructor
  · rintro ⟨hf, hpow⟩
    refine ⟨0, fun x => Real.log (f x), ?_, fun x => by rw [zero_add, Real.exp_log (hf x)]⟩
    refine IsCPDKernel.of_tendsto (l := 𝓝[>] (0 : ℝ))
      (K := fun l x y => l⁻¹ * (-1 + f (x - y) ^ l)) ?_ fun x y => ?_
    · filter_upwards [self_mem_nhdsWithin] with l (hl : 0 < l)
      exact ((hpow l hl).isCPDKernel.add_const (-1)).smul (inv_nonneg.mpr hl.le)
    · have h :=
        (Real.hasStrictDerivAt_const_rpow (hf (x - y)) 0).hasDerivAt.tendsto_slope_zero_right
      simp only [zero_add, Real.rpow_zero, one_mul, smul_eq_mul] at h
      exact h.congr fun l => by ring
  · rintro ⟨c, ψ, hψ, hf⟩
    refine ⟨fun x => hf x ▸ Real.exp_pos _, fun l hl => ?_⟩
    refine IsPDKernel.congr ((hψ.add_const c).isPDKernel_exp_mul hl.le) fun x y => ?_
    simp only [hf, ← Real.exp_mul]
    ring_nf

/-- **Fact 6, "if"**: `f = exp(c + ψ)` with `ψ` a PD function is positive, and all its positive
powers are PD. -/
theorem fact6_of_isPDFunction {G : Type*} [Sub G] {f ψ : G → ℝ} {c : ℝ} (hψ : IsPDFunction ψ)
    (hf : ∀ x, f x = Real.exp (c + ψ x)) :
    (∀ x, 0 < f x) ∧ ∀ l : ℝ, 0 < l → IsPDFunction fun x => f x ^ l :=
  fact6_iff.mpr ⟨c, ψ, IsPDKernel.isCPDKernel hψ, hf⟩

/-- The hypotheses of `fact6_of_isPDFunction` hold for `ψ = exp(−x²)`, `c = 0`. -/
example : (∀ x : ℝ, 0 < Real.exp (0 + Real.exp (-|x| ^ (2 : ℝ)))) ∧
    ∀ l : ℝ, 0 < l → IsPDFunction fun x : ℝ => Real.exp (0 + Real.exp (-|x| ^ (2 : ℝ))) ^ l :=
  fact6_of_isPDFunction (isPDFunction_exp_neg_rpow two_pos le_rfl) fun _ => rfl

/-- **Fact 6, "only if", is false** with the paper's gloss of "bounded away from zero" as
`f(x) > 0`: `f(x) = exp(−x²)` is positive and all its positive powers `exp(−λx²)` are PD, but it
is not `exp(c + ψ(x))` for a real `c` and a PD function `ψ`. Its infimum is `0`. -/
theorem not_fact6 :
    ¬∀ f : ℝ → ℝ, ((∀ x, 0 < f x) ∧ ∀ l : ℝ, 0 < l → IsPDFunction fun x => f x ^ l) →
      ∃ (c : ℝ) (ψ : ℝ → ℝ), IsPDFunction ψ ∧ ∀ x, f x = Real.exp (c + ψ x) := fun h => by
  obtain ⟨c, ψ, hψ, hf⟩ := h (fun x => Real.exp (-x ^ 2)) ⟨fun x => Real.exp_pos _, fun l hl =>
    IsPDKernel.congr ((isCPDKernel_neg_norm_sub_sq (E := ℝ)).isPDKernel_exp_mul hl.le)
      fun x y => by
        simp only [← Real.exp_mul, Real.norm_eq_abs, sq_abs]
        ring_nf⟩
  refine not_isPDKernel_const_sub_sq (-c) (IsPDKernel.congr hψ fun x y => ?_)
  have : -(x - y) ^ 2 = c + ψ (x - y) := Real.exp_injective (hf (x - y))
  linarith

/-- No real `c` makes `c − |x − y|^p` PD on `ℝ`, for `p > 0`: already on the positions `ℕ ⊆ ℝ`
it is not (`not_isPDKernel_power_bias`). -/
theorem not_isPDKernel_const_sub_abs_rpow {p : ℝ} (hp : 0 < p) (c : ℝ) :
    ¬IsPDKernel fun x y : ℝ => c - |x - y| ^ p := fun h =>
  not_isPDKernel_power_bias one_pos hp c ((h.comp fun n : ℕ => (n : ℝ)).congr fun m n => by
    simp only [biasKernel, powerKernel, neg_one_mul, sub_eq_add_neg])

/-- The hypothesis of `not_isPDKernel_const_sub_abs_rpow` holds for `p = 1`. -/
example : ¬IsPDKernel fun x y : ℝ => 10 - |x - y| ^ (1 : ℝ) :=
  not_isPDKernel_const_sub_abs_rpow one_pos 10

/-- **Proposition 2, first part, is false**: for no `p > 0`, in particular for no `p ∈ (0, 2]`,
is there a `c_min` such that `c − |m − n|^p` is PD for every `c ≥ c_min`. -/
theorem not_prop2_part1 {p : ℝ} (hp : 0 < p) :
    ¬∃ cmin : ℝ, ∀ c ≥ cmin, IsPDKernel fun x y : ℝ => c - |x - y| ^ p := fun ⟨cmin, h⟩ =>
  not_isPDKernel_const_sub_abs_rpow hp cmin (h cmin le_rfl)

/-- The hypothesis of `not_prop2_part1` holds for ALiBi's `p = 1`. -/
example : ¬∃ cmin : ℝ, ∀ c ≥ cmin, IsPDKernel fun x y : ℝ => c - |x - y| ^ (1 : ℝ) :=
  not_prop2_part1 one_pos

/-- **Proposition 2, second part**: for `p > 2` no `c` makes `c − |m − n|^p` PD. The paper's proof:
`exp(c − |x − y|^p)` would be PD, hence `exp(−|x − y|^p)`, against Fact 5. -/
theorem prop2_part2 {p : ℝ} (hp : 2 < p) (c : ℝ) : ¬IsPDKernel fun x y : ℝ => c - |x - y| ^ p :=
  fun h => not_isPDFunction_exp_neg_rpow hp <| IsPDKernel.congr
    (h.exp.smul (Real.exp_pos (-c)).le) fun x y => by
      rw [← Real.exp_add]
      ring_nf

/-- The hypothesis of `prop2_part2` holds for `p = 3`. -/
example : ¬IsPDKernel fun x y : ℝ => 10 - |x - y| ^ (3 : ℝ) := prop2_part2 (by norm_num) 10

/-- **ALiBi admits no PD kernel `c − |m − n|`** (§1: "ALiBi implicitly admits a PD kernel of the
form `c − |m − n|`"): no real `c` makes it PD on the positions `ℕ`. -/
theorem not_isPDKernel_alibi (c : ℝ) : ¬IsPDKernel fun m n : ℕ => c - |(m : ℝ) - n| := fun h =>
  not_isPDKernel_power_bias one_pos one_pos c (h.congr fun m n => by
    simp only [biasKernel, powerKernel, Real.rpow_one, neg_one_mul, sub_eq_add_neg])

end Transformer.KERPLE
