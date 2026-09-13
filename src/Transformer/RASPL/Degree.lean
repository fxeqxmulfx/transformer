/-
# Degree profiles, and what restricting a coordinate does to one

Zhou et al. — arXiv:2310.16028v1, "What Algorithms can Transformers Learn?",
§4.1 and Appendix E.

The degree profile of a boolean function is "the tuple of its Fourier weights
at each level, with the natural total ordering which refines the standard
polynomial degree" (footnote to §4.1, following Abbe, Boix-Adserà,
Misiakiewicz).  The ordering that refines polynomial degree is the
lexicographic one read from the top level down: a function is smaller when,
at the highest level where the two differ, it carries less weight.  That is
`DegPLt` below, written out rather than routed through a lexicographic order
on tuples, which is the same thing and less readable.

`degP_restrict_lt` is Lemma E.1 ("Technical Lemma"): restricting a coordinate
the function actually depends on strictly reduces the degree profile.  The
paper's proof is a sketch; the level that witnesses the drop is the top level
carrying weight on that coordinate, and above it nothing changes at all.
-/

import Transformer.RASPL.Restrict

namespace Transformer
namespace RASPL

variable {n : ℕ}

/-- The Fourier weight of `f` at level `k`. -/
noncomputable def levelWeight (f : Cube n → ℝ) (k : ℕ) : ℝ :=
  ∑ S ∈ Finset.univ.filter (fun S : Finset (Fin n) => S.card = k), coeff f S ^ 2

/-- `DegP g < DegP f`: at the highest level at which their weights differ,
`g` carries less.  This is the lexicographic order on the weight tuple read
from the top level down. -/
def DegPLt (g f : Cube n → ℝ) : Prop :=
  ∃ d : ℕ, levelWeight g d < levelWeight f d ∧
    ∀ k, d < k → levelWeight g k = levelWeight f k

/-- `f` depends on coordinate `i`: flipping it changes some value.  This is
the hypothesis "`∃ z : f(1 ∘ z) ≠ f(-1 ∘ z)`" of Lemma E.1. -/
def DependsOn (f : Cube n → ℝ) (i : Fin n) : Prop :=
  ∃ x : Cube n, f (Function.update x i false) ≠ f (Function.update x i true)

/-- Depending on a coordinate is carrying Fourier weight on it. -/
theorem dependsOn_iff (f : Cube n → ℝ) (i : Fin n) :
    DependsOn f i ↔ ∃ S : Finset (Fin n), i ∈ S ∧ coeff f S ≠ 0 := by
  classical
  constructor
  · intro ⟨x, hx⟩
    by_contra hc
    have hc' : ∀ S : Finset (Fin n), i ∈ S → coeff f S = 0 := fun S hiS => by
      by_contra h0
      exact hc ⟨S, hiS, h0⟩
    refine hx ?_
    have : restrict i false f = restrict i true f := by
      refine eq_of_coeff_eq fun S => ?_
      by_cases hi : i ∈ S
      · rw [coeff_restrict_of_mem hi, coeff_restrict_of_mem hi]
      · rw [coeff_restrict_of_notMem hi, coeff_restrict_of_notMem hi,
          hc' _ (Finset.mem_insert_self i S)]
        ring
    exact congrFun this x
  · intro ⟨S, hiS, hS⟩
    by_contra hc
    have hfe : restrict i false f = restrict i true f := by
      funext x
      by_contra hne
      exact hc ⟨x, hne⟩
    have hi : i ∉ S.erase i := Finset.notMem_erase i S
    have := congrArg (fun g => coeff g (S.erase i)) hfe
    rw [coeff_restrict_of_notMem hi, coeff_restrict_of_notMem hi,
      Finset.insert_erase hiS] at this
    simp only [bitSign_false, bitSign_true, one_mul, neg_one_mul] at this
    exact hS (by linarith)

/-- One level of the weight, split by whether the set contains `i`. -/
lemma levelWeight_split (f : Cube n → ℝ) (i : Fin n) (k : ℕ) :
    levelWeight f k
      = (∑ S ∈ (Finset.univ.filter fun S : Finset (Fin n) => S.card = k).filter
            (fun S => i ∈ S), coeff f S ^ 2)
        + ∑ S ∈ (Finset.univ.filter fun S : Finset (Fin n) => S.card = k).filter
            (fun S => i ∉ S), coeff f S ^ 2 :=
  (Finset.sum_filter_add_sum_filter_not _ _ _).symm

/-- At a level above which `f` carries no weight on `i`, restricting `i`
leaves only the sets avoiding `i` — with their coefficients untouched. -/
lemma levelWeight_restrict (i : Fin n) (b : Bool) (f : Cube n → ℝ) (k : ℕ)
    (hz : ∀ S : Finset (Fin n), i ∈ S → S.card = k + 1 → coeff f S = 0) :
    levelWeight (restrict i b f) k
      = ∑ S ∈ (Finset.univ.filter fun S : Finset (Fin n) => S.card = k).filter
          (fun S => i ∉ S), coeff f S ^ 2 := by
  classical
  rw [levelWeight_split _ i k, Finset.sum_eq_zero, zero_add]
  · refine Finset.sum_congr rfl fun S hS => ?_
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hS
    rw [coeff_restrict_of_notMem hS.2,
      hz _ (Finset.mem_insert_self i S) (by rw [Finset.card_insert_of_notMem hS.2, hS.1]),
      mul_zero, add_zero]
  · intro S hS
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hS
    rw [coeff_restrict_of_mem hS.2]
    ring

/-- **Lemma E.1.**  Restricting a coordinate that `f` depends on strictly
reduces its degree profile, whichever of the two values it is held at. -/
theorem degP_restrict_lt {i : Fin n} {f : Cube n → ℝ} (h : DependsOn f i) (b : Bool) :
    DegPLt (restrict i b f) f := by
  classical
  obtain ⟨S₀, hiS₀, hS₀⟩ := (dependsOn_iff f i).1 h
  set A : Finset ℕ :=
    (Finset.univ.filter fun S : Finset (Fin n) => i ∈ S ∧ coeff f S ≠ 0).image Finset.card
    with hA
  have hne : A.Nonempty := ⟨S₀.card, by simp [hA, Finset.mem_image]; exact ⟨S₀, ⟨hiS₀, hS₀⟩, rfl⟩⟩
  set d := A.max' hne with hd
  have hmem : d ∈ A := A.max'_mem hne
  obtain ⟨S₁, hS₁, hcard⟩ := Finset.mem_image.1 hmem
  simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hS₁
  have hzero : ∀ S : Finset (Fin n), i ∈ S → d < S.card → coeff f S = 0 := by
    intro S hiS hlt
    by_contra hne'
    exact absurd (A.le_max' S.card (Finset.mem_image.2 ⟨S, by simp [hiS, hne'], rfl⟩)) (by omega)
  refine ⟨d, ?_, fun k hk => ?_⟩
  · rw [levelWeight_restrict i b f d (fun S hiS hc => hzero S hiS (by omega)),
      levelWeight_split f i d]
    have hpos : 0 < ∑ S ∈ (Finset.univ.filter fun S : Finset (Fin n) => S.card = d).filter
        (fun S => i ∈ S), coeff f S ^ 2 := by
      refine Finset.sum_pos' (fun S _ => sq_nonneg _) ⟨S₁, ?_, ?_⟩
      · simp only [Finset.mem_filter, Finset.mem_univ, true_and]
        exact ⟨hcard, hS₁.1⟩
      · exact pow_pos (abs_pos.2 hS₁.2) 2 |>.trans_le (le_of_eq (sq_abs _))
    linarith
  · have hz0 : (∑ S ∈ (Finset.univ.filter fun S : Finset (Fin n) => S.card = k).filter
        (fun S => i ∈ S), coeff f S ^ 2) = 0 := by
      refine Finset.sum_eq_zero fun S hS => ?_
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hS
      obtain ⟨hc1, hc2⟩ := hS
      subst hc1
      rw [hzero S hc2 hk]
      ring
    rw [levelWeight_restrict i b f k (fun S hiS hc => hzero S hiS (by omega)),
      levelWeight_split f i k, hz0, zero_add]

/-- The hypothesis is satisfiable: the single coordinate of `bitSign` on the
one-dimensional cube. -/
example : DependsOn (fun x : Cube 1 => bitSign (x 0)) 0 :=
  ⟨fun _ => false, by norm_num⟩

end RASPL
end Transformer
