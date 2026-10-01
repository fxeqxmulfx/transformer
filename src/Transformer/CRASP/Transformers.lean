/-
# The two transformer simulations and their depth hierarchy

arXiv:2506.16055v3, Appendix B.2, `thm:TLCl_to_rtfr`,
`thm:rtfr_to_TLCl`, and `thm:transformer_equivalence`.
The rounded model is defined in `CRASP.TransformerModel`.
-/

import Transformer.CRASP.TransformerToLogic
import Transformer.CRASP.LogicToTransformer
import Transformer.CRASP.Depth
import Transformer.CRASP.Frame

namespace Transformer.CRASP

universe u
variable {σ : Type u} [DecidableEq σ]


/-- The `b`-th bit of the `i`-th entry of a length-preserving fixed-precision
map, at the logic's one-based positions; `false` off the string.

`lem:finite_function`, which postcomposes such a map with an arbitrary
`g : 𝔽 → 𝔽` at no cost in depth, is `Transformer.CRASP.FiniteFunction`. -/
noncomputable def bitAt {p s : ℕ} (F : List σ → List (Fx p s)) (w : List σ) (i b : ℕ) : Bool :=
  ((F w)[i - 1]?).elim false fun x => x.bit b

/-- **Theorem `thm:transformer_equivalence`.** Over the finite alphabet
fixed in Section 2.3, depth-`k` past-counting formulas and future-masked
rounded transformers recognize exactly the same languages.

Source: arXiv:2506.16055v3, §3.1 and Appendix B.2,
`thm:transformer_equivalence`. -/
theorem definableL_iff_recognizes [Fintype σ] (L : Set (List σ)) (k : ℕ) :
    DefinableL L k ↔ ∃ (p s d : ℕ) (T : RTfr (Option σ) p s d k), T.Recognizes L := by
  constructor
  · rintro ⟨φ, hφ, rfl⟩
    exact exists_rtfr_of_mem_TLCl k φ hφ
  · rintro ⟨p, s, d, T, hT⟩
    obtain ⟨φ, hφ, hlang⟩ := exists_mem_TLCl_of_rtfr T
    exact ⟨φ, hφ, hlang.trans (Set.ext hT)⟩

/-- **`L_1 = a⁺` is not definable at depth `0`.**  The case `k = 0` of the
negative half of `thm:TLCl_depth`, which the paper states for `k > 0` only: a
PNP-free formula of depth `0` reads nothing but the last letter, and `a ∈ L_1`
while `ba ∉ L_1`. -/
theorem not_definableL_altPlus_one_zero : ¬ DefinableL (altPlus false 1) 0 := by
  rintro ⟨φ, ⟨-, hf, hd⟩, hlang⟩
  have ha : [false] ∈ altPlus false 1 := by
    rw [altPlus_one]; exact ⟨1, one_pos, rfl⟩
  have hba : [true, false] ∉ altPlus false 1 := by
    rw [altPlus_one]
    rintro ⟨m, -, hm⟩
    have h0 := congrArg (·[0]?) hm
    cases m <;> simp at h0
  rw [← hlang] at ha hba
  apply hba
  have h := Form.sat_eq_of_pnpFree_depth_eq_zero (w := [false]) (w' := [true, false])
    (i := 1) (i' := 2) rfl φ hf (Nat.le_zero.mp hd)
  show φ.sat [true, false] 2 = true
  rw [← h]; exact ha

/-- **Theorem `thm:rtfr_depth_hierarchy`.** A depth-`(k+1)` transformer
recognizes `L_{k+1}`, and no depth-`k` transformer does.

The two now-proved simulations transfer the logic's depth hierarchy.
The manuscript derives the result from a logic theorem stated for `k > 0`;
the additional case `k = 0` is `not_definableL_altPlus_one_zero`.

Source: arXiv:2506.16055v3, §4, `thm:rtfr_depth_hierarchy`. -/
theorem rtfr_depth_hierarchy (k : ℕ) :
    (∃ (p s d : ℕ) (T : RTfr (Option Bool) p s d (k + 1)),
        T.Recognizes (altPlus false (k + 1))) ∧
      ∀ (p s d : ℕ) (T : RTfr (Option Bool) p s d k),
        ¬ T.Recognizes (altPlus false (k + 1)) := by
  have hpos : DefinableL (altPlus false (k + 1)) (k + 1) :=
    definableL_of_kPiecewiseTestable (k + 1) _
      (kPiecewiseTestable_altPlus (k + 1) k.succ_pos)
  have hneg : ¬ DefinableL (altPlus false (k + 1)) k := by
    rcases Nat.eq_zero_or_pos k with rfl | hk
    · exact not_definableL_altPlus_one_zero
    · exact (definableL_altPlus k hk).2
  exact ⟨(definableL_iff_recognizes _ _).mp hpos,
    fun p s d T hT => hneg ((definableL_iff_recognizes _ _).mpr ⟨p, s, d, T, hT⟩)⟩

end Transformer.CRASP
