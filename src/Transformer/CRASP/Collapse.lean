/-
# Representation collapse caps the useful depth

**Not a statement of any paper.**  A conjecture posed here to join two lines
of this development that no paper connects: the depth hierarchy of
arXiv:2506.16055v3 (`thm:transformer_equivalence`, `thm:rtfr_depth_hierarchy`)
and the clustering of token representations with depth in the self-attention
dynamics of arXiv:2312.10794v5 (§§3–5), arXiv:2410.06833v1 and
arXiv:2512.01868v4.

The clustering papers say that, in their regime, the representations after
many layers lie in an `ε`-cluster.  The hierarchy says that a transformer of
depth `k` recognizes no more than a `TL[◁#]` formula of depth `k`.  The
conjecture joins them: **once every position sits in an `ε`-cluster after `L`
layers, the layers above add nothing** — the whole transformer, however deep,
recognizes a language of `TL[◁#]` depth `L` (`exists_mem_TLCl_of_clustered`),
and so, by the hierarchy, not `L_{L+1}` (`not_recognizes_altPlus_of_clustered`).

It is stated for the future-masked rounded transformers of
`Transformer.CRASP.Transformers`, the one model where "depth of a formula" is
defined.  There `ε` below one grid step `2^{-s}` makes the cluster a single
point, and the reason to expect the conjecture is then exact: identical
activations give identical attention scores, an average of identical values
is that value whatever the prefix length, so every later layer computes one
function of the common vector, which is the last activation of a depth-`L`
transformer.  The continuous reading — softmax dynamics on the sphere, `ε`
above the resolution, an output margin paid for with the Lipschitz constants
of `Transformer.GPTMini` — is the harder and more interesting one, and is not
stated: rounding is not Lipschitz, and no model in this development carries
both a margin and a formula depth.
-/

import Transformer.CRASP.ConstantLayer

namespace Transformer
namespace CRASP

universe u

variable {σ : Type u}

namespace RTfr

variable {p s d k : ℕ}

/-- The representations after layer `ℓ` lie in an `ε`-cluster on every input
`⊲ · w`: any two positions differ by at most `ε` in every coordinate. -/
def Clustered (T : RTfr (Option σ) p s d k) (ℓ : ℕ) (ε : ℝ) : Prop :=
  ∀ (w : List σ) (i j : Fin (bos w).length) (c : Fin d),
    |(T.act (bos w) ℓ i c).val - (T.act (bos w) ℓ j c).val| ≤ ε

end RTfr

variable [DecidableEq σ]

/-- **Collapse after `L` layers caps the depth at `L`.**  A future-masked
rounded transformer of any depth `L + M` whose representations lie in an
`ε`-cluster after `L` layers, `ε` below one grid step, recognizes a language
of `TL[◁#]` depth `L`.

Posed here as a conjecture — see the module docstring — and proved for the
regime in which the docstring says the reason is exact: below one grid step an
`ε`-cluster is a single point (`Fx.eq_of_abs_val_sub_lt`), the layers above it
compute one vector whatever the prefix length
(`RTfr.layer_of_const`), and `RTfr.collapse` is the depth-`L` transformer that
reads the output off that vector.

What the conjecture still needs is `thm:rtfr_to_TLCl`, the depth-`k`
equivalence `exists_mem_TLCl_of_rtfr`, which is unproved here; it is therefore
carried as the explicit hypothesis `hequiv`, for the depth `L` and the shape
`p, s, d` of `T` alone, rather than used.

Source: none — posed here; it joins `thm:transformer_equivalence`
(arXiv:2506.16055v3, Appendix B) with the clustering theorems of
arXiv:2312.10794v5, §§3–5. -/
theorem exists_mem_TLCl_of_clustered {p s d L M : ℕ} (T : RTfr (Option σ) p s d (L + M))
    (hequiv : ∀ T' : RTfr (Option σ) p s d L,
      ∃ φ ∈ TLCl σ L, φ.lang = {w : List σ | T'.Accepts (bos w)})
    {ε : ℝ} (hε : ε < 2⁻¹ ^ s) (hT : T.Clustered L ε) :
    ∃ φ ∈ TLCl σ L, φ.lang = {w : List σ | T.Accepts (bos w)} := by
  obtain ⟨φ, hφ, hlang⟩ := hequiv (T.collapse : RTfr (Option σ) p s d L)
  refine ⟨φ, hφ, hlang.trans ?_⟩
  ext w
  have hconst : ∀ i j : Fin (bos w).length,
      T.act (bos w) L i = T.act (bos w) L j := by
    intro i j
    funext c
    exact Fx.eq_of_abs_val_sub_lt (lt_of_le_of_lt (hT w i j c) hε)
  simp only [Set.mem_ofPred_eq, RTfr.Accepts,
    RTfr.out_collapse T (bos w) hconst]

/-- The hypotheses of `exists_mem_TLCl_of_clustered` are satisfiable: a
transformer whose embedding is `0` is a single point at layer `0`. -/
example {p s d M : ℕ} (T : RTfr (Option σ) p s d (0 + M)) (hE : T.E = fun _ _ => 0) :
    (0 : ℝ) < 2⁻¹ ^ s ∧ T.Clustered 0 0 := by
  refine ⟨by positivity, fun w i j c => ?_⟩
  simp [RTfr.act, hE]

/-- And the equivalence hypothesis is satisfiable too — not by an empty class
of transformers, but by a degenerate one: at model dimension `0` every
activation is the empty tuple, so the output does not depend on the input and
the language recognized is `∅` or every string.  Both are defined at every
depth, by `1 < 1` and its negation. -/
example (p s L : ℕ) (T' : RTfr (Option σ) p s 0 L) :
    ∃ φ ∈ TLCl σ L, φ.lang = {w : List σ | T'.Accepts (bos w)} := by
  have hsub : ∀ f g : Fin 0 → Fx p s, f = g := fun f g => funext fun c => c.elim0
  have hout : ∀ w : List σ, T'.out (bos w) = T'.Wout (fun c => c.elim0) := by
    intro w
    rw [RTfr.out]
    split_ifs with hlen
    · exact congrArg _ (hsub _ _)
    · simp at hlen
  have htop : ∀ w : List σ,
      (Form.neg (Form.lt Term.one Term.one) : Form σ).models w := by
    intro w
    simp [Form.models, Form.sat, Term.val]
  have hbot : ∀ w : List σ,
      ¬ (Form.lt Term.one Term.one : Form σ).models w := by
    intro w
    simp [Form.models, Form.sat, Term.val]
  by_cases hpos : 0 < (T'.Wout (fun c => (c.elim0 : Fx p s))).val
  · refine ⟨Form.neg (Form.lt Term.one Term.one),
      ⟨rfl, rfl, Nat.zero_le L⟩, ?_⟩
    ext w
    simp only [Form.lang, Set.mem_ofPred_eq, RTfr.Accepts, hout w]
    exact iff_of_true (htop w) hpos
  · refine ⟨Form.lt Term.one Term.one, ⟨rfl, rfl, Nat.zero_le L⟩, ?_⟩
    ext w
    simp only [Form.lang, Set.mem_ofPred_eq, RTfr.Accepts, hout w]
    exact iff_of_false (hbot w) hpos

/-- **Corollary: a transformer that collapses after `L` layers does not
recognize `L_{L+1}`,** however many layers sit above the collapse.  This is
`exists_mem_TLCl_of_clustered` against `thm:TLCl_depth` (arXiv:2506.16055v3),
the lower half of `thm:rtfr_depth_hierarchy` with the depth of the transformer
replaced by the depth at which it collapses.

As there, the unproved depth-`L` equivalence `thm:rtfr_to_TLCl` is carried as
a hypothesis; the depth hierarchy `definableL_altPlus` it is played against is
proved. -/
theorem not_recognizes_altPlus_of_clustered {p s d L M : ℕ} (hL : 0 < L)
    (T : RTfr (Option Bool) p s d (L + M))
    (hequiv : ∀ T' : RTfr (Option Bool) p s d L,
      ∃ φ ∈ TLCl Bool L, φ.lang = {w : List Bool | T'.Accepts (bos w)})
    {ε : ℝ} (hε : ε < 2⁻¹ ^ s) (hT : T.Clustered L ε) :
    ¬ T.Recognizes (altPlus false (L + 1)) := by
  intro hrec
  obtain ⟨φ, hφ, hlang⟩ := exists_mem_TLCl_of_clustered T hequiv hε hT
  exact (definableL_altPlus L hL).2 ⟨φ, hφ, hlang.trans (Set.ext hrec)⟩

/-- The hypotheses of `not_recognizes_altPlus_of_clustered` are satisfiable: a
transformer whose first feed-forward network is `0` is a single point at layer
`1`. -/
example {p s d M : ℕ} (T : RTfr (Option Bool) p s d (1 + M))
    (hf : T.ff 0 = fun _ _ => 0) : 0 < 1 ∧ (0 : ℝ) < 2⁻¹ ^ s ∧ T.Clustered 1 0 := by
  refine ⟨Nat.one_pos, by positivity, fun w i j c => ?_⟩
  simp [RTfr.act, RTfr.layer, hf]

end CRASP
end Transformer
