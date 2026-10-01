/-
# What the position encodings buy, and what they do not

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), Appendix F (`sec:sinusoidal_pes`, `sec:rope_pes`,
`sec:alibi_pes`) and §4.5 (`thm:rtfr_pes_depth_hierarchy`).

Each of the three encodings is simulated by one of the two fragments of
`TL[◁#]^pos`: the periodic ones — sinusoidal and RoPE — by `TL[◁#, MOD]`,
and ALiBi by `TL[◁#, Y]`. Its rounded coefficients are eventually constant;
state counts describe this tail and `Y` corrects a bounded recent window.
Since `thm:tlclpos_depth_hierarchy` separates the depths of both fragments,
none of the three collapses the transformer depth hierarchy — which is
`thm:rtfr_pes_depth_hierarchy`, the statement the paper puts in the main
text, in `CRASP.EncodingHierarchy`.  The three unnamed theorems of Appendix F,
one per encoding, are its three instances.

Appendix F also carries `thm:mnf`, `thm:tlmod_to_rtfr` and
`thm:TLCmod_to_rtfr`, the converse simulations for `MOD`.  They sit inside an
`\iffalse` block in the source and so are not part of the paper; like
`lem:bb` they are deliberately left out here.
-/

import Transformer.CRASP.PositionalTransformers
import Transformer.CRASP.PeriodicEncodingsToLogic
import Transformer.CRASP.ZeroModulus
import Transformer.CRASP.AlibiToLogic
import Transformer.CRASP.PositionalDepth

namespace Transformer
namespace CRASP

/-- The three position encodings named in `thm:rtfr_pes_depth_hierarchy`. -/
def PosEnc.IsStandard : PosEnc → Prop
  | .plain => False
  | .sinusoidal _ => True
  | .rope _ => True
  | .alibi _ => True

/-- **Lemma `lem:alibi_window`.**  *ALiBi has a finite attention window.*

For a fixed-precision representation `𝔽` and a slope `a > 0` there is a
`Δ_a` such that `round(exp(x - a(i - j))) = 0` whenever `j ≤ i - Δ_a` and
`x ∈ 𝔽`.  The attention logits are bounded above, and the smallest positive
element of `𝔽` is `2^{-s}`, so a large enough linear bias pushes the rounded
weight to zero.

Source: arXiv:2506.16055v3, Appendix F, `lem:alibi_window`. -/
theorem alibi_window (p s : ℕ) (a : ℝ) (ha : 0 < a) :
    ∃ Δ : ℕ, ∀ i j : ℕ, j + Δ ≤ i → ∀ x : Fx p s,
      Fx.round p s (Real.exp (x.val - a * ((i : ℝ) - (j : ℝ)))) = 0 := by
  have hs : (0 : ℝ) < 2 ^ s := by positivity
  set M : ℝ := (2 : ℝ) ^ (p - 1) / 2 ^ s with hM
  refine ⟨⌈(M + s * Real.log 2) / a⌉₊ + 1, fun i j hij x => ?_⟩
  have hxlt : x.val < M := by
    rw [hM, Fx.val]
    gcongr
    exact_mod_cast x.hi
  have hd : (M + s * Real.log 2) / a + 1 ≤ (i : ℝ) - (j : ℝ) := by
    have hceil : (M + s * Real.log 2) / a ≤ (⌈(M + s * Real.log 2) / a⌉₊ : ℝ) := Nat.le_ceil _
    have hcast : (j : ℝ) + (⌈(M + s * Real.log 2) / a⌉₊ : ℝ) + 1 ≤ (i : ℝ) := by
      exact_mod_cast hij
    linarith
  have hmul : M + s * Real.log 2 + a ≤ a * ((i : ℝ) - (j : ℝ)) := by
    have h := mul_le_mul_of_nonneg_left hd ha.le
    rwa [mul_add, mul_div_cancel₀ _ (ne_of_gt ha), mul_one] at h
  have hexp : Real.exp (x.val - a * ((i : ℝ) - (j : ℝ))) < ((2 : ℝ) ^ s)⁻¹ := by
    rw [← Real.lt_log_iff_exp_lt (by positivity), Real.log_inv, Real.log_pow]
    linarith
  refine Fx.ext ?_
  have hfloor : ⌊Real.exp (x.val - a * ((i : ℝ) - (j : ℝ))) * 2 ^ s⌋ = 0 := by
    rw [Int.floor_eq_zero_iff, Set.mem_Ico]
    refine ⟨by positivity, ?_⟩
    have h := mul_lt_mul_of_pos_right hexp hs
    rwa [inv_mul_cancel₀ (ne_of_gt hs)] at h
  rw [Fx.m_round, hfloor, Fx.m_zero, Fx.clamp]
  have hpos : (0 : ℤ) < 2 ^ (p - 1) := by positivity
  omega

/-- The hypothesis of `alibi_window` is satisfiable: `a = 1` is a slope. -/
example : (0 : ℝ) < 1 := one_pos

/- The former unrestricted sinusoidal equivalence admitted `MOD_0^r` and
was false. `sinusoidal_equivalence_zero_modulus_counterexample` proves the
counterexample. The reverse simulation is proved in
`exists_mem_TLClMod_of_sinusoidal`; the paper's forward periodic equivalence
needs a syntax requiring positive moduli. -/

end CRASP
end Transformer
