/-
# Attention's forward pass and Frank-Wolfe — positive-definite key-query

Formalization of Alcalde, Geshkovski, Ruiz-Balet — arXiv:2508.09628v1,
*Attention's forward pass and Frank-Wolfe*, §4.

For `B ≻ 0` the Frank-Wolfe objective is concave.  The generic literature
bound on the duality gap is `prop: trash`; under genericity assumptions on the
initial polytope (`GenericPolytope`, here) the dynamics is instead solved
exactly (`thm: exp.fast.polytope`, in `Section4_ExpFast.lean`), and the same
geometry gives well-posedness of the singular ODE `eq: hardmax.ode`
(`thm: ode`, in `Section4_ODEWellPosed.lean`).
-/

import Transformer.FrankWolfe.Section4_Cells

open scoped BigOperators
open Real

namespace Transformer
namespace FrankWolfe

variable {d n κ : ℕ}

/-- The objective of `(SA_∞)` at `B^t = β B` is `β` times the one at `B`. -/
theorem quadForm_smul (β : ℝ) (B : ParamMatrix d) (y : EucSpace d) :
    quadForm (β • B) y = β * quadForm B y := by
  simp only [quadForm, smul_apply, real_inner_smul_left]; ring

/-- **One Frank-Wolfe step against the concave objective.**  If `y` maximizes
`⟨β B x, ·⟩` over `K` and `x' = x + γ (y - x)` with `β = c γ`, `c ≥ 0`, then the
Frank-Wolfe gap at `x` is at most `c (J(x') - J(x))` for `J = ½ ⟨B ·, ·⟩`:
the first display of the proof of `prop: trash`, where concavity of `-J` is
`⟨B u, u⟩ ≥ 0`.

Source: arXiv:2508.09628v1, the proof of `prop: trash` (a commented-out
subsection of the appendix). -/
theorem gap_le_of_step {B : ParamMatrix d} (hB : ContinuousLinearMap.IsPositive B) {β γ c : ℝ}
    (hc : 0 ≤ c) (hβ : β = c * γ) {K : Set (EucSpace d)} {x x' y : EucSpace d}
    (hy : IsMaximizerOn ((β • B) x) K y) (hx' : x' = x + γ • (y - x)) :
    sSup {r : ℝ | ∃ z ∈ K, r = inner (𝕜 := ℝ) (-(β • B) x) (x - z)}
      ≤ c * (quadForm B x' - quadForm B x) := by
  refine csSup_le ⟨_, y, hy.1, rfl⟩ ?_
  rintro r ⟨z, hz, rfl⟩
  have hzy := hy.2 z hz
  have hsym : inner (𝕜 := ℝ) (B (y - x)) x = inner (𝕜 := ℝ) (B x) (y - x) := by
    exact (hB.1 (y - x) x).trans (real_inner_comm _ _)
  have hu := hB.inner_nonneg_left (y - x)
  have hexp : quadForm B x' - quadForm B x =
      γ * inner (𝕜 := ℝ) (B x) (y - x) + γ ^ 2 / 2 * inner (𝕜 := ℝ) (B (y - x)) (y - x) := by
    simp only [hx', quadForm, map_add, map_smul, inner_add_left, inner_add_right,
      real_inner_smul_left, real_inner_smul_right, hsym]
    ring
  simp only [smul_apply, real_inner_smul_left, inner_neg_left,
    inner_sub_right] at hzy ⊢
  rw [hexp, inner_sub_right]
  subst hβ
  nlinarith [mul_nonneg (mul_nonneg hc (sq_nonneg γ)) hu]

/-- The hypotheses of `gap_le_of_step` are satisfiable: `B = 0`, `K = {0}`,
everything at the origin. -/
example : ContinuousLinearMap.IsPositive (0 : ParamMatrix 1) ∧ (0 : ℝ) ≤ 1 ∧
    (1 : ℝ) = 1 * 1 ∧ IsMaximizerOn (((1 : ℝ) • (0 : ParamMatrix 1)) 0) {0} 0 ∧
    (0 : EucSpace 1) = 0 + (1 : ℝ) • (0 - 0) :=
  ⟨ContinuousLinearMap.isPositive_zero, zero_le_one, (one_mul 1).symm,
    ⟨rfl, fun z _ => by simp⟩, by simp⟩

/-- **Proposition (prop: trash).**  Suppose `B^t = β^t B` for `B ≽ 0`, with
`β^t/β^{t+1} = γ^t/γ^{t+1}` for all `t ≥ 0`.  Then particles evolving according
to `(SA_∞)` satisfy

  `min_{τ ∈ [1,t]} max_{y ∈ 𝒦^τ} ⟨∇J^τ(x_i^τ), x_i^τ - y⟩
      ≤ (1/t) (J^1(x_i^1)/γ^1 - inf_{y ∈ 𝒦^t} J^t(y)/γ^t)`,

where `J^τ(y) = -½ ⟨B^τ y, y⟩`, so `∇J^τ(x) = -B^τ x`.

**What the source says and what is changed here.**  The ratio condition is
written as the cross-product `β^t γ^{t+1} = β^{t+1} γ^t`, which says the same
and needs no division.  Two standing assumptions of §4 that the statement
leaves implicit are made explicit: `γ^t ∈ (0, 1)` (as `γ^t = h^t/(1+h^t)` with
`h^t > 0`, `(SA_∞)`), and `β^t ≥ 0` (§4 treats `B^t ≽ 0`, and `J^t` is concave
only then).  The source's `t ≥ 0` is `t ≥ 1`, the right side being `1/t` times
something.

The source omits the proof in the text, quoting Yurtsever-Sra, `Lemma 2.1`; a
proof is in its commented-out appendix, and it is the one given here: each gap
is bounded by one step of the telescoping sum of `J^s/γ^s = J/c`
(`gap_le_of_step`), and the minimum by the average.

Source: arXiv:2508.09628v1, §4, `prop: trash`. -/
theorem duality_gap (B : ParamMatrix d) (β γ : ℕ → ℝ) (x : ℕ → Idx n → EucSpace d)
    (hB : ContinuousLinearMap.IsPositive B) (hβ : ∀ t, 0 ≤ β t)
    (hγ : ∀ t, γ t ∈ Set.Ioo (0 : ℝ) 1)
    (hratio : ∀ t : ℕ, β t * γ (t + 1) = β (t + 1) * γ t)
    (hflow : ∀ t : ℕ, IsHardmaxStep (β t • B) (γ t) (x t) (x (t + 1)))
    (i : Idx n) (t : ℕ) (ht : 1 ≤ t) :
    (Finset.Icc 1 t).inf' (Finset.nonempty_Icc.mpr ht)
        (fun τ => sSup {r : ℝ | ∃ y ∈ configHull (x τ),
          r = inner (𝕜 := ℝ) (-(β τ • B) (x τ i)) (x τ i - y)})
      ≤ (1 / (t : ℝ)) * (-quadForm (β 1 • B) (x 1 i) / γ 1
          - sInf {r : ℝ | ∃ y ∈ configHull (x t), r = -quadForm (β t • B) y} / γ t) := by
  set c := β 0 / γ 0
  have hc : 0 ≤ c := div_nonneg (hβ 0) (hγ 0).1.le
  have hcβ : ∀ s, β s = c * γ s := by
    intro s
    induction s with
    | zero => exact (div_mul_cancel₀ _ (hγ 0).1.ne').symm
    | succ s ih =>
      have h := hratio s
      rw [ih] at h
      exact (mul_right_cancel₀ (hγ s).1.ne' (by linarith [h] : β (s + 1) * γ s = c * γ (s + 1) * γ s))
  set g : ℕ → ℝ := fun τ => sSup {r : ℝ | ∃ y ∈ configHull (x τ),
    r = inner (𝕜 := ℝ) (-(β τ • B) (x τ i)) (x τ i - y)}
  have hstep : ∀ s, g s ≤ c * (quadForm B (x (s + 1) i) - quadForm B (x s i)) := fun s => by
    obtain ⟨y, hy, hxy⟩ := hflow s i
    exact gap_le_of_step hB hc (hcβ s) hy hxy
  have hsum : ∀ T, ∑ s ∈ Finset.Icc 1 T, g s ≤ c * (quadForm B (x (T + 1) i) - quadForm B (x 1 i)) := by
    intro T
    induction T with
    | zero => simp
    | succ T ih =>
      rw [Finset.sum_Icc_succ_top (by omega)]
      nlinarith [hstep (T + 1)]
  have hmin := Finset.card_nsmul_le_sum (Finset.Icc 1 t) g
    ((Finset.Icc 1 t).inf' (Finset.nonempty_Icc.mpr ht) g) fun s hs => Finset.inf'_le g hs
  rw [Nat.card_Icc, Nat.add_sub_cancel, nsmul_eq_mul] at hmin
  -- The last iterate lies in `𝒦^t`, which bounds the infimum.
  have hmem : x (t + 1) i ∈ configHull (x t) := by
    obtain ⟨y, ⟨hyK, -⟩, hxy⟩ := hflow t i
    have hXi : x t i ∈ configHull (x t) := subset_convexHull ℝ _ ⟨i, rfl⟩
    rw [hxy, show x t i + γ t • (y - x t i) = (1 - γ t) • x t i + γ t • y by module]
    exact (convex_convexHull ℝ _) hXi hyK (by linarith [(hγ t).2]) (hγ t).1.le (by ring)
  have hbdd : BddBelow {r : ℝ | ∃ y ∈ configHull (x t), r = -quadForm (β t • B) y} := by
    have hK : IsCompact (configHull (x t)) := (Set.finite_range _).isCompact_convexHull (𝕜 := ℝ)
    have hcont : Continuous fun y => -quadForm (β t • B) y := by unfold quadForm; fun_prop
    convert (hK.image hcont).bddBelow using 1
    ext r; simp [eq_comm]
  have hinf : sInf {r : ℝ | ∃ y ∈ configHull (x t), r = -quadForm (β t • B) y}
      ≤ -(c * γ t * quadForm B (x (t + 1) i)) :=
    (csInf_le hbdd ⟨x (t + 1) i, hmem, rfl⟩).trans_eq (by rw [quadForm_smul, hcβ])
  have e1 : -quadForm (β 1 • B) (x 1 i) / γ 1 = -(c * quadForm B (x 1 i)) := by
    rw [quadForm_smul, hcβ]; field_simp [(hγ 1).1.ne']
  have e2 : sInf {r : ℝ | ∃ y ∈ configHull (x t), r = -quadForm (β t • B) y} / γ t
      ≤ -(c * quadForm B (x (t + 1) i)) := by
    rw [div_le_iff₀ (hγ t).1]; linarith
  have ht0 : (0 : ℝ) < t := by exact_mod_cast ht
  rw [e1, one_div, ← div_eq_inv_mul, le_div_iff₀ ht0]
  nlinarith [hsum t]

/-- The hypotheses of `duality_gap` are satisfiable: `B = 0`, one particle
resting at the origin, `β^t ≡ 1`, `γ^t ≡ 1/2`, `t = 1`. -/
example :
    ContinuousLinearMap.IsPositive (0 : ParamMatrix 1) ∧ (∀ _ : ℕ, (0 : ℝ) ≤ 1) ∧
    (∀ _ : ℕ, (1 / 2 : ℝ) ∈ Set.Ioo (0 : ℝ) 1) ∧
    (∀ _ : ℕ, (1 : ℝ) * (1 / 2) = 1 * (1 / 2)) ∧
    (∀ t : ℕ, IsHardmaxStep ((1 : ℝ) • (0 : ParamMatrix 1)) (1 / 2)
      ((fun _ : ℕ => fun _ : Idx 1 => (0 : EucSpace 1)) t)
      ((fun _ : ℕ => fun _ : Idx 1 => (0 : EucSpace 1)) (t + 1))) ∧
    (1 : ℕ) ≤ 1 := by
  refine ⟨ContinuousLinearMap.isPositive_zero, fun _ => zero_le_one, fun _ => by norm_num,
    fun _ => rfl, fun t i => ⟨0, ⟨?_, ?_⟩, by simp⟩, le_rfl⟩
  · exact subset_convexHull ℝ _ ⟨i, rfl⟩
  · intro z _
    simp

/-- The genericity conditions `thm: exp.fast.polytope` puts on the initial
configuration: `v` lists the vertices of `𝒦 = conv{x_i^0}`, each vertex lies
in its own cell and in no other, and every particle lies in exactly one cell.

**What the source says and what is changed here.**  The source's second
condition reads "if `x_i^0` is not a vertex, then it doesn't lie on any face
of two adjacent cells".  It is recorded here as "`x_i^0` lies in exactly one
cell", which is what makes the assignment map `σ` well-defined and is the only
consequence the proof uses; for a vertex it follows from the first condition.

Source: arXiv:2508.09628v1, §4, `thm: exp.fast.polytope`, conditions 1 and 2. -/
structure GenericPolytope (B : ParamMatrix d) (v : Idx κ → EucSpace d)
    (X : Idx n → EucSpace d) : Prop where
  /-- `v` enumerates the vertices of `𝒦 = conv{x_i^0}`. -/
  isVertexList : IsVertexList (configHull X) v
  /-- `eq: vertices.own.cell`: `v_j ∈ 𝒞_j(v) \ ⋃_{i ≠ j} 𝒞_i(v)`. -/
  ownCell : ∀ j : Idx κ, v j ∈ cell B (configHull X) v j ∧
    ∀ i : Idx κ, i ≠ j → v j ∉ cell B (configHull X) v i
  /-- Every particle lies in exactly one cell. -/
  uniqueCell : ∀ i : Idx n, ∃! j : Idx κ, X i ∈ cell B (configHull X) v j

/-- The running products `∏_{s = a}^{b-1} (1 - γ^s)` of the solution formula of
`thm: exp.fast.polytope`. -/
noncomputable def gammaProd (γ : ℕ → ℝ) (a b : ℕ) : ℝ :=
  ∏ s ∈ Finset.Ico a b, (1 - γ s)

end FrankWolfe
end Transformer
