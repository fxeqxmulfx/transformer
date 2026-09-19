/-
# Future-masked fixed-precision transformers

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), Appendix B.1, `def:transformer`.

A *future-masked rounded transformer* of depth `k` is the map `Σ* → 𝔽` given by

    h⁰ᵢ = E(wᵢ),
    qᵢ = W_Q(hᵢ),  kᵢ = W_K(hᵢ),  vᵢ = W_V(hᵢ),  sᵢⱼ = qᵢ · kⱼ,
    cᵢ = round( Σ_{j ≤ i} round(exp sᵢⱼ · vⱼ) / Σ_{j ≤ i} round(exp sᵢⱼ) ),
    hᵢ = f(cᵢ + h^{prev}ᵢ),     𝒯(w) = W_out(h^k_{|w|}),

with the attention reading as the average of all `vⱼ` when the denominator
rounds to zero — the device that keeps attention informative even when
`i ≫ 2^s`.  The transformer *accepts* `w` when `𝒯(w) > 0`.

Two modelling notes.  The weight families are indexed by `ℕ` rather than by
`[k]`, with only the first `k` layers reachable from `RTfr.out`; and the
residual connection `cᵢ + h^{prev}ᵢ` is rounded, since `𝔽` is not closed under
addition and the paper does not say what happens on overflow.
-/

import Transformer.CRASP.Depth
import Transformer.CRASP.Fixed

namespace Transformer
namespace CRASP

universe u

variable {σ : Type u}

/-- A future-masked rounded transformer with `p` bits of precision, `s` of them
fractional, model dimension `d` and depth `k` (Definition `def:transformer`). -/
structure RTfr (σ : Type u) (p s d k : ℕ) where
  /-- The word embedding `E`. -/
  E : σ → Fin d → Fx p s
  /-- The query projections `W_Q^{(ℓ)}`. -/
  WQ : ℕ → (Fin d → Fx p s) → (Fin d → Fx p s)
  /-- The key projections `W_K^{(ℓ)}`. -/
  WK : ℕ → (Fin d → Fx p s) → (Fin d → Fx p s)
  /-- The value projections `W_V^{(ℓ)}`. -/
  WV : ℕ → (Fin d → Fx p s) → (Fin d → Fx p s)
  /-- The feed-forward networks `f^{(ℓ)}`. -/
  ff : ℕ → (Fin d → Fx p s) → (Fin d → Fx p s)
  /-- The output projection `W_out`. -/
  Wout : (Fin d → Fx p s) → Fx p s

namespace RTfr

variable {p s d k : ℕ}

/-- The positions a future-masked query at `i` may attend to: `j ≤ i`. -/
def masked {n : ℕ} (i : Fin n) : Finset (Fin n) := Finset.univ.filter fun j => j ≤ i

theorem self_mem_masked {n : ℕ} (i : Fin n) : i ∈ masked i := by simp [masked]

/-- One layer: attention (Equation `eq:att`) followed by the residual
connection and the feed-forward network. -/
noncomputable def layer (T : RTfr σ p s d k) (ℓ : ℕ) {n : ℕ}
    (h : Fin n → Fin d → Fx p s) : Fin n → Fin d → Fx p s := fun i =>
  let score : Fin n → ℝ := fun j =>
    ∑ c : Fin d, (T.WQ ℓ (h i) c).val * (T.WK ℓ (h j) c).val
  let den : ℝ := ∑ j ∈ masked i, (Fx.round p s (Real.exp (score j))).val
  let att : Fin d → Fx p s := fun c =>
    if den = 0 then
      Fx.round p s
        ((∑ j ∈ masked i, (T.WV ℓ (h j) c).val) / ((masked i).card : ℝ))
    else
      Fx.round p s
        ((∑ j ∈ masked i, (Fx.round p s (Real.exp (score j) * (T.WV ℓ (h j) c).val)).val) / den)
  T.ff ℓ fun c => Fx.add (att c) (h i c)

/-- The activations `h^{(ℓ)}` at every position (Equation `eq:emb` onwards). -/
noncomputable def act (T : RTfr σ p s d k) (w : List σ) :
    ℕ → Fin w.length → Fin d → Fx p s
  | 0 => fun i c => T.E w[i] c
  | ℓ + 1 => T.layer ℓ (T.act w ℓ)

/-- `𝒯(w) = W_out(h^{(k)}_{|w|})`, the output read off the last position; the
empty string, which has no last position, is sent to `0`. -/
noncomputable def out (T : RTfr σ p s d k) (w : List σ) : Fx p s :=
  if h : 0 < w.length then T.Wout (T.act w k ⟨w.length - 1, by omega⟩) else 0

/-- "We say that `𝒯` *accepts* `w` if `𝒯(w) > 0`." -/
def Accepts (T : RTfr σ p s d k) (w : List σ) : Prop := 0 < (T.out w).val

end RTfr

/-- `⊲ · w`, the input with the beginning-of-sequence symbol prepended.  The
transformer works over `Σ ∪ {⊲}`, which is `Option σ` with `⊲ = none`. -/
def bos (w : List σ) : List (Option σ) := none :: w.map some

@[simp] theorem length_bos (w : List σ) : (bos w).length = w.length + 1 := by
  simp [bos]

/-- A transformer *recognizes* `L` when it accepts exactly the strings `⊲ · w`
with `w ∈ L` (Appendix B.2). -/
def RTfr.Recognizes {p s d k : ℕ} (T : RTfr (Option σ) p s d k) (L : Set (List σ)) : Prop :=
  ∀ w : List σ, T.Accepts (bos w) ↔ w ∈ L

variable [DecidableEq σ]

/-- **Proposition `thm:TLCl_to_rtfr`.**  Every `TL[◁#]` formula of depth `k` is
simulated by a future-masked rounded transformer of depth `k`. -/
theorem exists_rtfr_of_mem_TLCl (k : ℕ) (φ : Form σ) (hφ : φ ∈ TLCl σ k) :
    ∃ (p s d : ℕ) (T : RTfr (Option σ) p s d k), T.Recognizes φ.lang :=
  sorry

/-- The hypothesis of `exists_rtfr_of_mem_TLCl` is satisfiable: `Q_a` is a
PNP-free past-only formula of depth `0`. -/
example (a : σ) (k : ℕ) : (Form.sym a : Form σ) ∈ TLCl σ k :=
  ⟨rfl, rfl, Nat.zero_le k⟩

/-- The `b`-th bit of the `i`-th entry of a length-preserving fixed-precision
map, at the logic's one-based positions; `false` off the string.

`lem:finite_function`, which postcomposes such a map with an arbitrary
`g : 𝔽 → 𝔽` at no cost in depth, is `Transformer.CRASP.FiniteFunction`. -/
noncomputable def bitAt {p s : ℕ} (F : List σ → List (Fx p s)) (w : List σ) (i b : ℕ) : Bool :=
  ((F w)[i - 1]?).elim false fun x => x.bit b

/-- **Proposition `thm:rtfr_to_TLCl`.**  Every future-masked rounded
transformer of depth `k` is simulated by a `TL[◁#]` formula of depth `k`. -/
theorem exists_mem_TLCl_of_rtfr {p s d k : ℕ} (T : RTfr (Option σ) p s d k) :
    ∃ φ ∈ TLCl σ k, φ.lang = {w : List σ | T.Accepts (bos w)} :=
  sorry

/-- **Theorem `thm:transformer_equivalence`.**  A language is defined by a
`TL[◁#]` formula of depth `k` exactly when `⊲ · L` is recognized by a
future-masked rounded transformer of depth `k`.

The paper states it as the conjunction of the two simulations, and that is how
it is proved here: the first hypothesis is `exists_rtfr_of_mem_TLCl`, the
second `exists_mem_TLCl_of_rtfr`, and what is proved is that together they give
the equivalence — nothing left to do but rewrite the language along
`RTfr.Recognizes`.

Neither simulation is proved, so neither is used: both are hypotheses, and
nothing beyond them is assumed here. -/
theorem definableL_iff_recognizes (L : Set (List σ)) (k : ℕ) :
    (∀ (j : ℕ) (φ : Form σ), φ ∈ TLCl σ j →
        ∃ (p s d : ℕ) (T : RTfr (Option σ) p s d j), T.Recognizes φ.lang) →
    (∀ (p s d j : ℕ) (T : RTfr (Option σ) p s d j),
        ∃ φ ∈ TLCl σ j, φ.lang = {w : List σ | T.Accepts (bos w)}) →
    (DefinableL L k ↔ ∃ (p s d : ℕ) (T : RTfr (Option σ) p s d k), T.Recognizes L) := by
  intro hfwd hbwd
  constructor
  · rintro ⟨φ, hφ, rfl⟩
    exact hfwd k φ hφ
  · rintro ⟨p, s, d, T, hT⟩
    obtain ⟨φ, hφ, hlang⟩ := hbwd p s d k T
    exact ⟨φ, hφ, hlang.trans (Set.ext hT)⟩

/-- **Theorem `thm:rtfr_depth_hierarchy`.**  A depth-`(k+1)` transformer
recognizes `L_{k+1}`, and no depth-`k` transformer does.

A corollary of the depth hierarchy `thm:TLCl_depth` for the logic — which *is*
proved here — and of the equivalence `definableL_iff_recognizes`, exactly as
the paper derives it: the transformer hierarchy carries no combinatorics of its
own.  The two simulations are the same hypotheses as there, passed through. -/
theorem rtfr_depth_hierarchy (k : ℕ) (hk : 0 < k) :
    (∀ (j : ℕ) (φ : Form Bool), φ ∈ TLCl Bool j →
        ∃ (p s d : ℕ) (T : RTfr (Option Bool) p s d j), T.Recognizes φ.lang) →
    (∀ (p s d j : ℕ) (T : RTfr (Option Bool) p s d j),
        ∃ φ ∈ TLCl Bool j, φ.lang = {w : List Bool | T.Accepts (bos w)}) →
    ((∃ (p s d : ℕ) (T : RTfr (Option Bool) p s d (k + 1)),
        T.Recognizes (altPlus false (k + 1))) ∧
      ∀ (p s d : ℕ) (T : RTfr (Option Bool) p s d k),
        ¬ T.Recognizes (altPlus false (k + 1))) := by
  intro hfwd hbwd
  obtain ⟨hpos, hneg⟩ := definableL_altPlus k hk
  exact ⟨(definableL_iff_recognizes _ _ hfwd hbwd).mp hpos,
    fun p s d T hT => hneg ((definableL_iff_recognizes _ _ hfwd hbwd).mpr ⟨p, s, d, T, hT⟩)⟩

/-- The hypothesis of `rtfr_depth_hierarchy` is satisfiable: `k = 1`. -/
example : (0 : ℕ) < 1 := Nat.one_pos

end CRASP
end Transformer
