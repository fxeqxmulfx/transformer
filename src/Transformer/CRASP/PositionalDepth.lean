/-
# `TL[◁#]^pos`: reduction to `TL[◁#]`, and the depth hierarchy

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), Appendix E (`app:tlclpos`, "Depth Hierarchy"):
`lem:tlclpos_reduction`, `thm:tlclpos_depth_hierarchy`.

The reduction pulls a `TL[◁#]^pos_k` formula over `Σ ∪ {e}` back along the
string map `w ↦ e^r w₁ e^{r−1} ⋯ wₙ e^{r−1}` to a plain `TL[◁#]_k` formula
over `Σ`: the `Y`-normal form of `Transformer.CRASP.YNormalFormEquiv`, put through
the translation of `Transformer.CRASP.PositionalReductionEquiv`.  This turns the
hierarchy of `thm:TLCl_depth` into a hierarchy for `TL[◁#]^pos` with the
separating language `E_{k+1}`.
-/

import Transformer.CRASP.NeutralLetter
import Transformer.CRASP.PositionalEmbedding
import Transformer.CRASP.PositionalReductionEquiv

namespace Transformer
namespace CRASP

universe u

variable {σ : Type u} [DecidableEq σ]

/-- **Lemma `lem:tlclpos_reduction`.**  A `TL[◁#]^pos_k` formula over
`Σ ∪ {e}` is pulled back along `spread r`, for a suitable `r ≥ 1`, to a plain
`TL[◁#]_k` formula over `Σ`.

As in the paper, the formula is `T_r⟦ψ⟧` for the `Y`-normal form `ψ` of `φ`.
The paper's `r = M(Y+1)`, for `M` the lcm of the moduli and `Y` the `Y`-depth,
is too small once `w = ε` (`tr_paperBlockSize_unsound`).  Here
`r = period ψ · (reach ψ + 1)`: a multiple of every modulus, as `M(Y+1)` is,
and at least `reach ψ`, which exceeds every `Y`-nesting by two. -/
theorem exists_form_of_formP (k : ℕ) (φ : FormP (Option σ)) (hφ : φ ∈ TLClPos (Option σ) k) :
    ∃ r : ℕ, 1 ≤ r ∧ ∃ φ' ∈ TLCl σ k, ∀ w : List σ, FormP.models (spread r w) φ ↔ w ∈ φ'.lang := by
  obtain ⟨ψ, hψk, hψ, hsat⟩ := exists_yNormal k φ hφ
  have hr : 1 ≤ ψ.period * (ψ.reach + 1) := Nat.mul_pos ψ.period_pos (Nat.succ_pos _)
  have hreach : ψ.reach ≤ ψ.period * (ψ.reach + 1) := by
    have := Nat.le_mul_of_pos_left (ψ.reach + 1) ψ.period_pos
    omega
  have hper : ψ.period ∣ ψ.period * (ψ.reach + 1) := Nat.dvd_mul_right _ _
  generalize ψ.period * (ψ.reach + 1) = r at hr hreach hper
  refine ⟨r, hr, ψ.tr r r, FormP.tr_mem_TLCl r r ψ hψk, fun w => ?_⟩
  rw [FormP.models, hsat]
  exact FormP.models_spread_iff hr w hψ hreach hper

/-- The hypothesis is satisfiable: `Q_{some a}` lies in `TL[◁#]^pos_k`. -/
example (a : σ) (k : ℕ) : (FormP.sym (some a) : FormP (Option σ)) ∈ TLClPos (Option σ) k :=
  Nat.zero_le k

/-- `E_k`, the language `altPlus` with a neutral letter: the strings over
`Σ ∪ {e}` that lie in `A_k` once every `e` is deleted (Appendix E,
`app:tlclpos`). -/
def altPlusNeutral (k : ℕ) : Set (List (Option Bool)) := List.reduceOption ⁻¹' altPlus false k

/-- **Theorem `thm:tlclpos_depth_hierarchy`.**  `E_{k+1}` is definable in
`TL[◁#]^pos_{k+1}` but not in `TL[◁#]^pos_k`: a depth-`k` definition would
reduce along `spread` to a `TL[◁#]_k` definition of `A_{k+1}`, contradicting
`thm:TLCl_depth`.

The negative half is proved in those words: `spread` is a section of
`List.reduceOption`, so pulling `E_{k+1}` back along it gives `A_{k+1}` on the
nose.  The proof in the paper starts from `φ ∈ TL[◁#]^pos_{k+1}`, which has to
read `TL[◁#]^pos_k` for the reduction to land in `TL[◁#]_k`.

The paper does not argue the positive half.  It holds without `MOD` and `Y`:
`E_{k+1}` is `(k+1)`-piecewise testable because `A_{k+1}` is and a neutral
letter does not change that, `lem:piecewise_testable_depth` puts it in
`TL[◁#]_{k+1}`, and `TL[◁#]` sits inside `TL[◁#]^pos`.

Source: arXiv:2506.16055v3, Appendix E, `thm:tlclpos_depth_hierarchy`. -/
theorem definablePos_altPlusNeutral (k : ℕ) (hk : 0 < k) :
    DefinablePos (altPlusNeutral (k + 1)) (k + 1) ∧
      ¬ DefinablePos (altPlusNeutral (k + 1)) k := by
  refine ⟨(definableL_of_kPiecewiseTestable (k + 1) _
    (kPiecewiseTestable_altPlus (k + 1) k.succ_pos).preimage_reduceOption).definablePos, ?_⟩
  rintro ⟨φ, hφ, hlang⟩
  obtain ⟨r, -, φ', hφ', hiff⟩ := exists_form_of_formP (σ := Bool) k φ hφ
  refine (definableL_altPlus k hk).2 ⟨φ', hφ', ?_⟩
  ext w
  rw [← hiff w]
  have hmem : FormP.models (spread r w) φ ↔ spread r w ∈ altPlusNeutral (k + 1) := by
    rw [← hlang]
    exact Iff.rfl
  rw [hmem, altPlusNeutral, Set.mem_preimage, reduceOption_spread]

end CRASP
end Transformer
