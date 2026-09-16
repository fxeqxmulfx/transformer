/-
# What the position encodings buy, and what they do not

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), Appendix E (`sec:sinusoidal_pes`, `sec:rope_pes`,
`sec:alibi_pes`) and §5.2 (`thm:rtfr_pes_depth_hierarchy`).

Each of the three encodings is simulated by one of the two fragments of
`TL[◁#]^pos`: the periodic ones — sinusoidal and RoPE — by `TL[◁#, MOD]`,
and ALiBi by `TL[◁#, Y]`, because past a fixed distance `Δ_a` the linear bias
rounds the attention weight to zero and only a bounded window survives.
Since `thm:tlclpos_depth_hierarchy` separates the depths of both fragments,
none of the three collapses the transformer depth hierarchy — which is
`thm:rtfr_pes_depth_hierarchy`, the statement the paper puts in the main
text.  The three unnamed theorems of Appendix E, one per encoding, are its
three instances.

Appendix E also carries `thm:mnf`, `thm:tlmod_to_rtfr` and
`thm:TLCmod_to_rtfr`, the converse simulations for `MOD`.  They sit inside an
`\iffalse` block in the source and so are not part of the paper; like
`lem:bb` they are deliberately left out here.
-/

import Transformer.CRASP.PositionalTransformers
import Transformer.CRASP.PositionalDepth

namespace Transformer
namespace CRASP

universe u

variable {σ : Type u}

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

Source: arXiv:2506.16055v3, Appendix E, `lem:alibi_window`. -/
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

variable [DecidableEq σ]

/-- **Theorem `thm:rtfr_eq_tlclmod`.**  *Sinusoidal encodings are `MOD`.*

A language `L` is defined by a `TL[◁#, MOD]` formula of depth `k ≥ 1` exactly
when `⊲ · L` is recognized by a depth-`k` transformer with sinusoidal
position encoding.  The angles are assumed rational, so that the encoding is
periodic in the position; that is what `MOD` can express.

Source: arXiv:2506.16055v3, Appendix E, `thm:rtfr_eq_tlclmod`. -/
theorem definableMod_iff_recognizes_sinusoidal (L : Set (List σ)) (k : ℕ) (hk : 1 ≤ k) :
    DefinableMod L k ↔
      ∃ (p s d : ℕ) (θ : ℕ → ℝ) (T : PTfr (Option σ) p s d k),
        T.pe = .sinusoidal θ ∧ (PosEnc.sinusoidal θ).RationalAngles ∧ T.Recognizes L :=
  sorry

/-- The hypothesis of `definableMod_iff_recognizes_sinusoidal` is satisfiable:
depth `1` is a depth. -/
example : 1 ≤ 1 := le_rfl

/-- **Proposition `thm:rtfr_to_TLClmod`.**  *RoPE is `MOD` too.*

A depth-`k` transformer with RoPE, at rational angles, is simulated by a
depth-`k` formula of `TL[◁#, MOD]`: the rotations `R(θ)^i` and `R(θ)^j` are
periodic in the position, hence computable in fixed precision from `MOD`
predicates by `lem:finite_function`, and the rest is `thm:rtfr_to_TLCl`.

Source: arXiv:2506.16055v3, Appendix E, `thm:rtfr_to_TLClmod`. -/
theorem exists_mem_TLClMod_of_rope {p s d k : ℕ} (θ : ℕ → ℝ)
    (hθ : (PosEnc.rope θ).RationalAngles) (T : PTfr (Option σ) p s d k)
    (hT : T.pe = .rope θ) :
    ∃ φ ∈ TLClMod σ k, φ.lang = {w : List σ | T.Accepts (bos w)} :=
  sorry

/-- The hypotheses of `exists_mem_TLClMod_of_rope` are satisfiable: the zero
transformer with all angles `0` uses RoPE at rational angles. -/
example :
    ∃ (θ : ℕ → ℝ) (T : PTfr (Option Bool) 1 0 0 0),
      (PosEnc.rope θ).RationalAngles ∧ T.pe = .rope θ := by
  refine ⟨fun _ => 0,
    { E := fun _ _ => 0, WQ := fun _ _ => 0, WK := fun _ _ => 0,
      WV := fun _ _ => 0, ff := fun _ _ => 0, Wout := fun _ => 0,
      pe := .rope fun _ => 0 },
    ⟨fun _ => 0, fun c => ?_⟩, rfl⟩
  simp

/-- **Proposition `thm:rtfr_to_TLCly`.**  *ALiBi is `Y`.*

A depth-`k` transformer with ALiBi is simulated by a depth-`k` formula of
`TL[◁#, Y]`.  For slope `0` this is `thm:rtfr_to_TLCl`; for a positive slope
`lem:alibi_window` confines the attention to the window
`[i - Δ_a, i]`, whose keys are read off by `Δ_a` nested applications of `Y`.

Source: arXiv:2506.16055v3, Appendix E, `thm:rtfr_to_TLCly`. -/
theorem exists_mem_TLClY_of_alibi {p s d k : ℕ} (a : ℝ)
    (T : PTfr (Option σ) p s d k) (hT : T.pe = .alibi a) :
    ∃ φ ∈ TLClY σ k, φ.lang = {w : List σ | T.Accepts (bos w)} :=
  sorry

/-- The hypothesis of `exists_mem_TLClY_of_alibi` is satisfiable: the zero
transformer with slope `1` uses ALiBi. -/
example : ∃ T : PTfr (Option Bool) 1 0 0 0, T.pe = .alibi 1 :=
  ⟨{ E := fun _ _ => 0, WQ := fun _ _ => 0, WK := fun _ _ => 0,
     WV := fun _ _ => 0, ff := fun _ _ => 0, Wout := fun _ => 0,
     pe := .alibi 1 }, rfl⟩

/-- **Theorem `thm:rtfr_pes_depth_hierarchy`.**  *The depth hierarchy survives
the position encodings.*

A depth-`(k+1)` transformer can recognize `E_{k+1}` but no depth-`k` one can,
whether the transformers use sinusoidal position embeddings, RoPE or ALiBi.
The positive half needs no position encoding at all; the negative half is
`thm:rtfr_eq_tlclmod`, `thm:rtfr_to_TLClmod` or `thm:rtfr_to_TLCly` followed
by `thm:tlclpos_depth_hierarchy`, which denies `E_{k+1}` to both
`TL[◁#, MOD]_k` and `TL[◁#, Y]_k`.

The rational-angle assumption is the one under which the periodic encodings
are simulated at all; it is vacuous for ALiBi.

Source: arXiv:2506.16055v3, §5.2, `thm:rtfr_pes_depth_hierarchy`, and the
three unnamed theorems of Appendix E. -/
theorem rtfr_pes_depth_hierarchy (k : ℕ) (hk : 0 < k)
    (pe : PosEnc) (hpe : pe.IsStandard) (hrat : pe.RationalAngles) :
    (∃ (p s d : ℕ) (T : PTfr (Option (Option Bool)) p s d (k + 1)),
        T.pe = pe ∧ T.Recognizes (altPlusNeutral (k + 1))) ∧
      ∀ (p s d : ℕ) (T : PTfr (Option (Option Bool)) p s d k),
        T.pe = pe → ¬ T.Recognizes (altPlusNeutral (k + 1)) :=
  sorry

/-- The hypotheses of `rtfr_pes_depth_hierarchy` are satisfiable: `k = 1` is
positive, and ALiBi with slope `1` is one of the three encodings and carries
no angles to be rational. -/
example : 0 < 1 ∧ (PosEnc.alibi 1).IsStandard ∧ (PosEnc.alibi 1).RationalAngles :=
  ⟨one_pos, trivial, trivial⟩

end CRASP
end Transformer
