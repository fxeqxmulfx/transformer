/-
# Attention's forward pass and Frank-Wolfe — `thm: exp.fast.polytope`

§4 of arXiv:2508.09628v1: for `B ≻ 0` and a generic initial polytope, the
dynamics `(SA_∞)` is solved exactly.  As in the source's proof, the vertices
never move, so the hull `𝒦^t = 𝒦` is constant; a particle in the cell of
`v_{σ(i)}` has `v_{σ(i)}` as its only maximizer and moves along the segment
towards it, never meeting another cell.

The two geometric facts the source uses in passing — the maximizer is the
vertex, and the segment to it stays in one cell — are in
`Section4_Faces.lean`.

Source: arXiv:2508.09628v1, §4, `thm: exp.fast.polytope` and its proof in
`sec: proof.exp.fast`.
-/

import Transformer.FrankWolfe.Section4_Polytope
import Transformer.FrankWolfe.Section4_Faces

open scoped BigOperators
open Set

namespace Transformer
namespace FrankWolfe

variable {d n κ : ℕ}

/-- An empty running product is `1`. -/
theorem gammaProd_self (γ : ℕ → ℝ) (a : ℕ) : gammaProd γ a a = 1 := by
  simp [gammaProd]

/-- One more step multiplies the running product by `1 - γ^b`. -/
theorem gammaProd_succ (γ : ℕ → ℝ) {a b : ℕ} (hab : a ≤ b) :
    gammaProd γ a (b + 1) = gammaProd γ a b * (1 - γ b) := by
  rw [gammaProd, gammaProd, Finset.prod_Ico_succ_top hab]

/-- The two coefficients of the solution formula add up to `1`. -/
theorem sum_gammaProd (γ : ℕ → ℝ) (t : ℕ) :
    ∑ τ ∈ Finset.range t, γ τ * gammaProd γ (τ + 1) t = 1 - gammaProd γ 0 t := by
  induction t with
  | zero => simp [gammaProd]
  | succ t ih =>
    rw [Finset.sum_range_succ, gammaProd_self, gammaProd_succ γ (Nat.zero_le t)]
    have hs : ∀ τ ∈ Finset.range t, γ τ * gammaProd γ (τ + 1) (t + 1) =
        γ τ * gammaProd γ (τ + 1) t * (1 - γ t) := fun τ hτ => by
      rw [gammaProd_succ γ (Finset.mem_range.1 hτ), mul_assoc]
    rw [Finset.sum_congr rfl hs, ← Finset.sum_mul, ih]
    ring

/-- **Theorem (thm: exp.fast.polytope) — the dynamics, solved.**

Let `B^t ≡ B` and `γ^t ∈ (0, 1)`.  For an initial configuration
satisfying `GenericPolytope` and an assignment `σ` with `x_i^0 ∈ 𝒞_{σ(i)}(v)`,
particles evolving according to `(SA_∞)` satisfy

  `x_i^t = (∏_{τ=0}^{t-1}(1 - γ^τ)) x_i^0
            + (Σ_{τ=0}^{t-1} γ^τ ∏_{s=τ+1}^{t-1}(1 - γ^s)) v_{σ(i)}`.

**What the source says and what is changed here.**  The source lets `σ` be
produced by the statement ("the map `σ` … is well-defined"); here it is taken
as data, with `x_i^0 ∈ 𝒞_{σ(i)}(v)` as a hypothesis, which `uniqueCell`
makes available and determines uniquely.

The `argmax` of `(SA_∞)` is a relation, not a function, so the theorem covers
every selection of maximizers.  That costs nothing: `ownCell` forces the
maximizing face of `⟨B x, ·⟩` at `x = x_i^t` to be the single vertex
`v_{σ(i)}`.

The source assumes `B ≻ 0`; the proof does not use it, and it is dropped.
Positive-definiteness is what makes the cells tile `𝒦` with disjoint
interiors (`lem: cells`), which the source invokes for `σ` to be well-defined;
here that is `GenericPolytope.uniqueCell`, assumed directly.

Source: arXiv:2508.09628v1, §4, `thm: exp.fast.polytope`. -/
theorem exp_fast_polytope (B : ParamMatrix d) (γ : ℕ → ℝ)
    (x : ℕ → Idx n → EucSpace d) (v : Idx κ → EucSpace d) (σ : Idx n → Idx κ)
    (hγ : ∀ t : ℕ, γ t ∈ Set.Ioo (0 : ℝ) 1)
    (hgen : GenericPolytope B v (x 0))
    (hσ : ∀ i : Idx n, x 0 i ∈ cell B (configHull (x 0)) v (σ i))
    (hflow : HardmaxFlow (fun _ : ℕ => B) γ x) :
    ∀ (t : ℕ) (i : Idx n),
      x t i = gammaProd γ 0 t • x 0 i
        + (∑ τ ∈ Finset.range t, γ τ * gammaProd γ (τ + 1) t) • v (σ i) := by
  set K := configHull (x 0)
  have hKv : Convex ℝ K := convex_convexHull ℝ _
  have hKc : IsCompact K := (finite_range _).isCompact_convexHull (𝕜 := ℝ)
  have hext : range v = K.extremePoints ℝ := hgen.isVertexList.2
  have hvK : ∀ j, v j ∈ K := fun j => (hext ▸ mem_range_self j : v j ∈ K.extremePoints ℝ).1
  have hx0K : ∀ i, x 0 i ∈ K := fun i => subset_convexHull ℝ _ (mem_range_self i)
  have hvcell : ∀ k j, v k ∈ cell B K v j ↔ j = k := fun k j =>
    ⟨fun h => by_contra fun hjk => (hgen.ownCell k).2 j hjk h, fun h => h ▸ (hgen.ownCell k).1⟩
  have hx0cell : ∀ i j, x 0 i ∈ cell B K v j ↔ j = σ i := fun i j => by
    obtain ⟨k, -, hk⟩ := hgen.uniqueCell i
    exact ⟨fun h => (hk j h).trans (hk _ (hσ i)).symm, fun h => h ▸ hσ i⟩
  -- The vertices are particles, and those particles are assigned to themselves.
  have hvert : ∀ j, ∃ i, x 0 i = v j ∧ σ i = j := fun j => by
    obtain ⟨i, hi⟩ := extremePoints_convexHull_subset (hext ▸ mem_range_self j :
      v j ∈ K.extremePoints ℝ)
    exact ⟨i, hi, ((hvcell j (σ i)).1 (hi ▸ hσ i))⟩
  set a : ℕ → ℝ := gammaProd γ 0
  have ha : ∀ t, 0 ≤ a t ∧ a t ≤ 1 := fun t =>
    ⟨Finset.prod_nonneg fun s _ => by linarith [(hγ s).2],
      Finset.prod_le_one₀ (fun s _ => by linarith [(hγ s).2]) fun s _ => by linarith [(hγ s).1]⟩
  -- Along the formula the hull does not change.
  have hhull : ∀ t, (∀ i, x t i = a t • x 0 i + (1 - a t) • v (σ i)) → configHull (x t) = K := by
    intro t ht
    refine (convexHull_min (range_subset_iff.2 fun i => ?_) hKv).antisymm ?_
    · rw [ht i]
      exact hKv (hx0K i) (hvK _) (ha t).1 (by linarith [(ha t).2]) (by ring)
    · rw [show K = _ from configHull_eq_convexHull hgen.isVertexList]
      refine convexHull_mono (range_subset_iff.2 fun j => ?_)
      obtain ⟨i, hi, hσi⟩ := hvert j
      refine ⟨i, ?_⟩
      rw [ht i, hi, hσi, ← add_smul]
      simp
  have key : ∀ t i, x t i = a t • x 0 i + (1 - a t) • v (σ i) := by
    intro t
    induction t with
    | zero => intro i; simp [a, gammaProd_self]
    | succ t ih =>
      intro i
      obtain ⟨y, hy, hstep⟩ := hflow t i
      rw [hhull t ih] at hy
      have hcell : ∀ j, x t i ∈ cell B K v j ↔ j = σ i := fun j => by
        rw [ih i]
        exact mem_cell_segment_iff B hKv hvK (hx0cell i) (hvcell (σ i)) (ha t).1 (ha t).2 j
      have hxK : x t i ∈ K := ((hcell (σ i)).2 rfl).1
      have hyv : y = v (σ i) :=
        eq_of_isMaximizerOn hKc hKv hext hy fun j hj => (hcell j).1 ⟨hxK, hj⟩
      rw [hstep, hyv, ih i, show a (t + 1) = a t * (1 - γ t) from gammaProd_succ γ (Nat.zero_le t)]
      module
  intro t i
  rw [key t i, sum_gammaProd]

/-- The hypotheses of `exp_fast_polytope` are satisfiable: `B = I` on `ℝ^1`,
one particle sitting at the single vertex `0`, `γ^t ≡ 1/2`. -/
example :
    (∀ _ : ℕ, (1 / 2 : ℝ) ∈ Set.Ioo (0 : ℝ) 1) ∧
    GenericPolytope (ContinuousLinearMap.id ℝ (EucSpace 1))
      (fun _ : Idx 1 => (0 : EucSpace 1)) (fun _ : Idx 1 => (0 : EucSpace 1)) ∧
    HardmaxFlow (fun _ : ℕ => ContinuousLinearMap.id ℝ (EucSpace 1))
      (fun _ : ℕ => (1 / 2 : ℝ)) (fun _ : ℕ => fun _ : Idx 1 => (0 : EucSpace 1)) := by
  have hhull : configHull (fun _ : Idx 1 => (0 : EucSpace 1)) = {(0 : EucSpace 1)} := by
    rw [configHull, Set.range_const, convexHull_singleton]
  have hcell : ∀ j : Idx 1, (0 : EucSpace 1) ∈
      cell (ContinuousLinearMap.id ℝ (EucSpace 1)) (configHull (fun _ : Idx 1 => (0 : EucSpace 1)))
        (fun _ : Idx 1 => (0 : EucSpace 1)) j := by
    intro j
    refine ⟨by rw [hhull]; rfl, fun y hy => ?_⟩
    simp
  refine ⟨fun _ => by norm_num,
    ⟨⟨?_, ?_⟩, fun j => ⟨hcell j, fun i hij => absurd (Subsingleton.elim i j) hij⟩,
      fun i => ⟨0, hcell 0, fun j _ => Subsingleton.elim j 0⟩⟩,
    fun t i => ⟨0, ⟨?_, ?_⟩, by simp⟩⟩
  · intro a b _
    exact Subsingleton.elim a b
  · rw [Set.range_const, hhull, extremePoints_singleton]
  · exact subset_convexHull ℝ _ ⟨i, rfl⟩
  · intro z _
    simp

/-- **Theorem (thm: exp.fast.polytope), the convergence statement.**  Under the
hypotheses of `exp_fast_polytope` and a uniform lower bound `c ≤ γ^t`,

  `x_i^t → v_{σ(i)}`   exponentially fast.

**What the source says and what is changed here.**  The source concludes "in
particular, `x_i^t` converges to `v_{σ(i)}` at least exponentially fast" from
`γ^t ∈ (0,1)` alone, and that is false: for `γ^t = 2^{-t-1}` the product
`∏_{τ<t}(1 - γ^τ)` decreases to a strictly positive limit, so the solution
formula leaves `x_i^t` at a nontrivial convex combination of `x_i^0` and
`v_{σ(i)}` forever.  A uniform lower bound on the step-size repairs it, and is
what "exponentially fast" needs in any case.  As in `exp_fast_polytope`,
`B ≻ 0` is not needed.

Source: arXiv:2508.09628v1, §4, `thm: exp.fast.polytope`, final sentence. -/
theorem exp_fast_polytope_tendsto (B : ParamMatrix d) (γ : ℕ → ℝ)
    (x : ℕ → Idx n → EucSpace d) (v : Idx κ → EucSpace d) (σ : Idx n → Idx κ) (c : ℝ)
    (hc : 0 < c) (hγ : ∀ t : ℕ, γ t ∈ Set.Ico c 1)
    (hgen : GenericPolytope B v (x 0))
    (hσ : ∀ i : Idx n, x 0 i ∈ cell B (configHull (x 0)) v (σ i))
    (hflow : HardmaxFlow (fun _ : ℕ => B) γ x) (i : Idx n) :
    ∃ C > 0, ∀ t : ℕ, ‖x t i - v (σ i)‖ ≤ C * (1 - c) ^ t := by
  have hγ' : ∀ t, γ t ∈ Set.Ioo (0 : ℝ) 1 := fun t => ⟨hc.trans_le (hγ t).1, (hγ t).2⟩
  refine ⟨‖x 0 i - v (σ i)‖ + 1, by positivity, fun t => ?_⟩
  have hdiff : x t i - v (σ i) = gammaProd γ 0 t • (x 0 i - v (σ i)) := by
    rw [exp_fast_polytope B γ x v σ hγ' hgen hσ hflow t i, sum_gammaProd]
    module
  have h0 : 0 ≤ gammaProd γ 0 t := Finset.prod_nonneg fun s _ => by linarith [(hγ s).2]
  have hle : gammaProd γ 0 t ≤ (1 - c) ^ t := by
    calc gammaProd γ 0 t ≤ ∏ _s ∈ Finset.Ico 0 t, (1 - c) := by
          unfold gammaProd
          gcongr with s
          · exact fun s _ => by linarith [(hγ s).2]
          · exact (hγ s).1
      _ = (1 - c) ^ t := by simp
  rw [hdiff, norm_smul, Real.norm_of_nonneg h0, mul_comm]
  have hpow : 0 ≤ (1 - c) ^ t := h0.trans hle
  nlinarith [norm_nonneg (x 0 i - v (σ i))]

/-- The hypotheses of `exp_fast_polytope_tendsto` are satisfiable: the witness
of `exp_fast_polytope`, with `c = 1/2`. -/
example : (0 : ℝ) < 1 / 2 ∧ ∀ _ : ℕ, (1 / 2 : ℝ) ∈ Set.Ico (1 / 2 : ℝ) 1 :=
  ⟨by norm_num, fun _ => by norm_num⟩

end FrankWolfe
end Transformer
