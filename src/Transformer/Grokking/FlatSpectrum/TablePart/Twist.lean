module

public import Mathlib.Combinatorics.SimpleGraph.Connectivity.Connected
public import Mathlib.LinearAlgebra.Dual.Lemmas
public import Transformer.Grokking.FlatSpectrum.PatternGraph
public import Transformer.Grokking.FlatSpectrum.TablePart.FlatClass

/-!
# PT3: the twist of a closed walk

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT3. A step of a walk on
the rows and the columns has a gain: `u vᵀ` from the row `u` to the column `v`, and `−u vᵀ` back
(`gain`). The twist of a walk is the sum of its gains (`twist`). On a closed walk
`u₁ v₁ u₂ v₂ … u_r v_r u₁` the steps pair up into `Σᵢ (uᵢ − uᵢ₊₁) vᵢᵀ` (`twist_row_step`), the `Z`
of the source. `twistSpan Ω` is the span of the twists of the closed walks from a row, as in the
source, and a closed walk from a column has its twist in it too (`twist_mem_twistSpan`).

`⟨M, Z⟩ = Σ_ij M_ij Z_ij` is `pair`, with `⟨M, u vᵀ⟩ = uᵀ M v` (`pair_vecMulVec`), and the matrices
orthogonal to a space `S` have dimension `k l − dim S` (`finrank_ann`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart

open Matrix Module SimpleGraph

variable {F ι κ : Type*} [Field F]

/-- The gain of a step (`review/flat_spectrum.md`, PT3): `u vᵀ` from the row `u` to the column `v`,
`−u vᵀ` back, and `0` between two rows or two columns, where the graph of a part has no edge. -/
def gain : (ι → F) ⊕ (κ → F) → (ι → F) ⊕ (κ → F) → Matrix ι κ F
  | .inl u, .inr v => vecMulVec u v
  | .inr v, .inl u => -vecMulVec u v
  | _, _ => 0

/-- Walking a step backwards negates its gain. -/
lemma gain_swap (a b : (ι → F) ⊕ (κ → F)) : gain b a = -gain a b := by
  rcases a with u | v <;> rcases b with u' | v' <;> simp [gain]

variable {G : SimpleGraph ((ι → F) ⊕ (κ → F))}

/-- The twist of a walk (`review/flat_spectrum.md`, PT3): the sum of the gains of its steps. -/
def twist : ∀ {a b : (ι → F) ⊕ (κ → F)}, G.Walk a b → Matrix ι κ F
  | _, _, .nil => 0
  | a, _, @Walk.cons _ _ _ c _ _ p => gain a c + twist p

@[simp] lemma twist_nil {a : (ι → F) ⊕ (κ → F)} : twist (Walk.nil : G.Walk a a) = 0 := rfl

@[simp] lemma twist_cons {a b c : (ι → F) ⊕ (κ → F)} (h : G.Adj a b) (p : G.Walk b c) :
    twist (Walk.cons h p) = gain a b + twist p := rfl

/-- The twist adds along a concatenation. -/
lemma twist_append {a b c : (ι → F) ⊕ (κ → F)} (p : G.Walk a b) (q : G.Walk b c) :
    twist (p.append q) = twist p + twist q := by
  induction p with
  | nil => simp
  | cons h p ih => simp [ih, add_assoc]

/-- The reverse walk has the opposite twist. -/
lemma twist_reverse {a b : (ι → F) ⊕ (κ → F)} (p : G.Walk a b) :
    twist p.reverse = -twist p := by
  induction p with
  | nil => simp
  | cons h p ih =>
    rw [Walk.reverse_cons, twist_append, ih, twist_cons, twist_cons, twist_nil, gain_swap]
    abel

/-- Copying a walk along equalities of its ends keeps its twist. -/
lemma twist_copy {a a' b b' : (ι → F) ⊕ (κ → F)} (p : G.Walk a b) (ha : a = a') (hb : b = b') :
    twist (p.copy ha hb) = twist p := by
  subst ha hb
  rfl

/-- Two steps through a column contribute `(u − u') vᵀ`, the summand of the source's `Z`. -/
lemma twist_row_step {u u' : ι → F} {v : κ → F} {b : (ι → F) ⊕ (κ → F)}
    (h : G.Adj (.inl u) (.inr v)) (h' : G.Adj (.inr v) (.inl u')) (p : G.Walk (.inl u') b) :
    twist (Walk.cons h (Walk.cons h' p)) = vecMulVec (u - u') v + twist p := by
  rw [twist_cons, twist_cons, ← add_assoc]
  congr 1
  ext i j
  simp [gain, vecMulVec_apply, sub_eq_add_neg]
  ring

/-- The span of the twists of the closed walks of the graph of `Ω`
(`review/flat_spectrum.md`, PT3). A closed walk there starts from a row. -/
def twistSpan (Ω : Set ((ι → F) × (κ → F))) : Submodule F (Matrix ι κ F) :=
  Submodule.span F {Z | ∃ (u : ι → F) (w : (patternGraph Ω).Walk (.inl u) (.inl u)), twist w = Z}

/-- A closed walk from a column has its twist in the span too: rotated by one step it starts from
a row, with the same gains. -/
lemma twist_mem_twistSpan {Ω : Set ((ι → F) × (κ → F))} {a : (ι → F) ⊕ (κ → F)}
    (w : (patternGraph Ω).Walk a a) : twist w ∈ twistSpan Ω := by
  rcases a with u | v
  · exact Submodule.subset_span ⟨u, w, rfl⟩
  · cases w with
    | nil => simp
    | @cons _ c _ h p =>
      rcases c with u | v'
      · refine Submodule.subset_span ⟨u, p.append (Walk.cons h Walk.nil), ?_⟩
        rw [twist_append, twist_cons, twist_cons, twist_nil, add_zero, add_comm]
      · exact absurd h not_patternGraph_adj_inr_inr

section Pairing

variable [Fintype ι] [Fintype κ]

/-- The pairing `⟨M, Z⟩ = Σ_ij M_ij Z_ij` of the source. -/
noncomputable def pair : Matrix ι κ F →ₗ[F] Matrix ι κ F →ₗ[F] F :=
  LinearMap.mk₂ F (fun M Z => ∑ i, ∑ j, M i j * Z i j)
    (fun M M' Z => by simp [add_mul, Finset.sum_add_distrib])
    (fun t M Z => by simp [Finset.mul_sum, mul_assoc])
    (fun M Z Z' => by simp [mul_add, Finset.sum_add_distrib])
    (fun t M Z => by simp [Finset.mul_sum, mul_left_comm])

lemma pair_apply (M Z : Matrix ι κ F) : pair M Z = ∑ i, ∑ j, M i j * Z i j := rfl

/-- `⟨M, u vᵀ⟩ = uᵀ M v`. -/
lemma pair_vecMulVec (M : Matrix ι κ F) (u : ι → F) (v : κ → F) :
    pair M (vecMulVec u v) = u ⬝ᵥ M *ᵥ v := by
  simp only [pair_apply, vecMulVec_apply, dotProduct, mulVec, Finset.mul_sum]
  refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
  ring

/-- The matrices orthogonal to `S`. -/
noncomputable def ann (S : Submodule F (Matrix ι κ F)) : Submodule F (Matrix ι κ F) :=
  S.dualAnnihilator.comap pair

lemma mem_ann {S : Submodule F (Matrix ι κ F)} {M : Matrix ι κ F} :
    M ∈ ann S ↔ ∀ Z ∈ S, pair M Z = 0 := by
  simp [ann, Submodule.mem_dualAnnihilator]

/-- The pairing has no kernel: `⟨M, e_i e_jᵀ⟩ = M_ij`. -/
lemma pair_injective [DecidableEq ι] [DecidableEq κ] :
    Function.Injective (pair : Matrix ι κ F →ₗ[F] Matrix ι κ F →ₗ[F] F) := by
  refine (injective_iff_map_eq_zero _).2 fun M hM => ?_
  ext i j
  have h := LinearMap.congr_fun hM (vecMulVec (Pi.single i 1) (Pi.single j 1))
  simpa [pair_vecMulVec, mulVec_single_one] using h

/-- The matrices orthogonal to `S` have dimension `k l − dim S`, written without subtraction. -/
theorem finrank_ann [DecidableEq ι] [DecidableEq κ] (S : Submodule F (Matrix ι κ F)) :
    finrank F (ann S) + finrank F S = Fintype.card ι * Fintype.card κ := by
  have hsurj : Function.Surjective (pair : Matrix ι κ F →ₗ[F] Matrix ι κ F →ₗ[F] F) :=
    (LinearMap.injective_iff_surjective_of_finrank_eq_finrank
      (Subspace.dual_finrank_eq (K := F) (V := Matrix ι κ F)).symm).1 pair_injective
  let e := LinearEquiv.ofBijective pair ⟨pair_injective, hsurj⟩
  have hcomap : ann S = S.dualAnnihilator.map (e.symm : _ →ₗ[F] _) :=
    Submodule.comap_equiv_eq_map_symm e S.dualAnnihilator
  have hdual := Subspace.finrank_add_finrank_dualAnnihilator_eq S
  rw [hcomap, LinearEquiv.finrank_map_eq]
  simp only [finrank_matrix, finrank_self, mul_one] at hdual
  omega

example : finrank ℚ (ann (⊥ : Submodule ℚ (Matrix (Fin 1) (Fin 1) ℚ))) +
    finrank ℚ (⊥ : Submodule ℚ (Matrix (Fin 1) (Fin 1) ℚ)) = 1 := by
  simpa using finrank_ann (⊥ : Submodule ℚ (Matrix (Fin 1) (Fin 1) ℚ))

end Pairing

end Transformer.Grokking.FlatSpectrum.TablePart
