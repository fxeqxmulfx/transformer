module

public import Transformer.Landscape.Uniformization.Comparable
public import Transformer.Landscape.Uniformization.Principal
public import Transformer.Landscape.Uniformization.Ramification

/-!
# Local uniformization: examples for germs, normal crossings and monomial charts

The coordinate `x₀` of `ℝ¹` and its powers satisfy the hypotheses of the lemmas on
normal crossings, on the domain of germs, on division by a coordinate, on factors of
normal crossings and on comparable exponents. The coordinate has monomial charts at every
point and satisfies the hypotheses of the lemmas that shrink, transport and glue them.
Small boxes, ramifications and the principalizing charts of `(y, t)` satisfy those of the
lemmas on coordinates. Auxiliary examples for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2.
-/

@[expose] public section

open Filter Set
open scoped BigOperators Topology

namespace Transformer.Landscape.Uniformization

open Transformer.Landscape.AnalyticPreparation (Base)

/-- The coordinate `x₀` has normal crossings at the origin, both as a unit times a
monomial and as a translate of `x₀ - 0`; so do its square and the constant `1`, and the
property spreads to nearby points. Appendix D.1 of arXiv:2510.22026v2. -/
example : NormalCrossingAt (fun x : Base 1 => x 0) 0 ∧ AnalyticAt ℝ (fun x : Base 1 => x 0) 0 ∧
    ¬ (fun x : Base 1 => x 0) =ᶠ[𝓝 0] 0 ∧ NormalCrossingAt (fun x : Base 1 => x 0 * x 0) 0 ∧
    (∀ᶠ w in 𝓝 (0 : Base 1), NormalCrossingAt (fun x : Base 1 => x 0) w) ∧
    NormalCrossingAt (fun _ : Base 1 => (1 : ℝ)) 0 ∧
    NormalCrossingAt (fun x : Base 1 => x 0 ^ 2) 0 := by
  have h : NormalCrossingAt (fun x : Base 1 => x 0) 0 :=
    (normalCrossingAt_coord 0 0).congr (Eventually.of_forall fun x => by simp)
  have h2 : NormalCrossingAt (fun x : Base 1 => x 0 ^ 2) 0 :=
    normalCrossingAt_of_eq_mul_monomial 0 (fun _ => 2) analyticAt_const one_ne_zero
      (Eventually.of_forall fun x => by simp [monomial])
  exact ⟨h, h.analyticAt, h.not_eventuallyEq_zero, h.mul h, h.eventually,
    normalCrossingAt_of_ne_zero analyticAt_const one_ne_zero, h2⟩

/-- The square `x₀ · x₀` is a product of two analytic factors, and of the factors
indexed by `Fin 2`; each factor recovers normal crossings, also through the induction
on the degree of the monomial `x₀²`. Appendix D.1 of arXiv:2510.22026v2. -/
example : NormalCrossingAt (fun x : Base 1 => x 0) 0 ∧
    NormalCrossingAt (fun x : Base 1 => x 0) 0 ∧ NormalCrossingAt (fun x : Base 1 => x 0) 0 ∧
    (∀ y : Base 1, monomial (fun _ => 2) y = y 0 * monomial ((fun _ => 2) - Pi.single 0 1) y) ∧
    ∑ j, ((fun _ => 2) - Pi.single 0 1 : Fin 1 → ℕ) j + 1 = ∑ _j : Fin 1, 2 := by
  have h : NormalCrossingAt (fun x : Base 1 => x 0) 0 :=
    (normalCrossingAt_coord 0 0).congr (Eventually.of_forall fun x => by simp)
  have hA : AnalyticAt ℝ (fun x : Base 1 => x 0) 0 := h.analyticAt
  have hprod : NormalCrossingAt (fun x : Base 1 => ∏ _i ∈ (Finset.univ : Finset (Fin 2)), x 0) 0 :=
    (h.mul h).congr (Eventually.of_forall fun x => by simp [sq])
  have h2 : (fun _ => 2 : Fin 1 → ℕ) 0 ≠ 0 := two_ne_zero
  exact ⟨NormalCrossingAt.of_mul hA hA (h.mul h),
    NormalCrossingAt.of_prod Finset.univ (fun _ _ => hA) hprod (Finset.mem_univ 0),
    normalCrossingAt_of_mul_aux 2 _ _ (fun _ => 1) (fun _ => 2) (by simp) hA hA
      analyticAt_const one_ne_zero (Eventually.of_forall fun x => by simp [monomial, sq]),
    monomial_eq_coord_mul h2, sum_sub_single h2⟩

/-- The germs of `x₀` and `0` have product zero, the two factors of `x₀ · x₀` are
nonzero, and `x₀` cancels from `x₀ · x₀ = x₀ · x₀`; the zero exponent is the only one
whose monomial does not vanish at the origin. Appendix D.1 of arXiv:2510.22026v2. -/
example : ((fun x : Base 1 => x 0) =ᶠ[𝓝 0] 0 ∨ (fun _ : Base 1 => (0 : ℝ)) =ᶠ[𝓝 0] 0) ∧
    ¬ (fun x : Base 1 => ∏ _i ∈ (Finset.univ : Finset (Fin 2)), x 0) =ᶠ[𝓝 0] 0 ∧
    (fun x : Base 1 => x 0) =ᶠ[𝓝 0] (fun x : Base 1 => x 0) ∧
    monomial (fun _ : Fin 1 => 1) 0 = 0 := by
  have h : NormalCrossingAt (fun x : Base 1 => x 0) 0 :=
    (normalCrossingAt_coord 0 0).congr (Eventually.of_forall fun x => by simp)
  exact ⟨eventuallyEq_zero_or_of_mul h.analyticAt analyticAt_const
      (Eventually.of_forall fun x => mul_zero _),
    not_eventuallyEq_zero_prod Finset.univ (fun _ _ => h.analyticAt)
      fun _ _ => h.not_eventuallyEq_zero,
    eventuallyEq_of_coord_mul 0 h.analyticAt h.analyticAt (Eventually.of_forall fun _ => rfl),
    monomial_zero_point _ fun h1 => one_ne_zero (congrFun h1 0)⟩

/-- `x₀²` vanishes on the hyperplane `x₀ = 0` and is `x₀` times the analytic `x₀`.
Appendix D.1 of arXiv:2510.22026v2. -/
example : ∃ G : Base 1 → ℝ, AnalyticAt ℝ G 0 ∧
    ∀ᶠ x in 𝓝 (0 : Base 1), x 0 ^ 2 = (x 0 - (0 : Base 1) 0) * G x :=
  exists_eq_coord_mul (F := fun x : Base 1 => x 0 ^ 2) 0
    (((ContinuousLinearMap.proj 0 : Base 1 →L[ℝ] ℝ).analyticAt 0).fun_pow 2)
    (Eventually.of_forall fun x => by simp)

/-- The test vectors of `{0}` and of `∅` evaluate the monomial `x₀`, and some nonzero
test vector stays in the unit ball. Appendix D.1 of arXiv:2510.22026v2. -/
example : monomial (fun _ : Fin 1 => 1) (testVec {0} 1) = 0 ∧
    monomial (fun _ : Fin 1 => 1) (testVec ∅ 2) = 2 ^ ∑ _l : Fin 1, 1 ∧
    ∃ s : ℝ, s ≠ 0 ∧ (0 : Base 1) + testVec {0} s ∈ Metric.ball (0 : Base 1) 1 :=
  ⟨monomial_testVec_of_mem (Finset.mem_singleton_self 0) one_ne_zero 1,
    monomial_testVec_of_forall (fun l hl => absurd hl (Finset.notMem_empty l)) 2,
    exists_testVec {0} (Metric.ball_mem_nhds 0 one_pos)⟩

/-- `2 x₀ - x₀` has normal crossings, so the exponents of `2 x₀` and `x₀` are
comparable; a single index has a minimal ratio. Appendix D.1 of arXiv:2510.22026v2. -/
example : ((Pi.single 0 1 : Fin 1 → ℕ) ≤ Pi.single 0 1 ∨
      (Pi.single 0 1 : Fin 1 → ℕ) ≤ Pi.single 0 1) ∧
    ∃ k ∈ ({0} : Finset (Fin 1)), ∀ i ∈ ({0} : Finset (Fin 1)),
      (fun _ => 1 : Fin 1 → ℕ) i • (fun _ _ => 1 : Fin 1 → Fin 1 → ℕ) k ≤
        (fun _ => 1 : Fin 1 → ℕ) k • (fun _ _ => 1 : Fin 1 → Fin 1 → ℕ) i := by
  have hNC : NormalCrossingAt (fun x : Base 1 =>
      (fun _ => (2 : ℝ)) x * monomial (Pi.single 0 1) (x - 0) -
        (fun _ => (1 : ℝ)) x * monomial (Pi.single 0 1) (x - 0)) 0 :=
    normalCrossingAt_of_eq_mul_monomial 0 (Pi.single 0 1) analyticAt_const one_ne_zero
      (Eventually.of_forall fun x => by ring)
  exact ⟨le_or_le_of_normalCrossingAt_sub analyticAt_const two_ne_zero analyticAt_const
    one_ne_zero hNC, exists_min_ratio {0} (Finset.singleton_nonempty 0) _
      (fun _ _ => one_ne_zero) _ fun _ _ _ _ => Or.inl le_rfl⟩

/-- The coordinate `x₀` has monomial charts at every point of `ℝ¹`: by its normal
crossings, shrunk into the unit ball, through the identity chart, from the cover by the
closed unit ball, after an eventual equality and after a translation by `1`.
Appendix D.1 of arXiv:2510.22026v2. -/
example : (∃ S : Set ((Base 1 → Base 1) × Set (Base 1)), S.Finite ∧
      (∀ c ∈ S, IsMonomialChart (fun x : Base 1 => x 0) c ∧
        c.1 '' c.2 ⊆ Metric.ball 0 1) ∧ (⋃ c ∈ S, c.1 '' c.2) ∈ 𝓝 (0 : Base 1)) ∧
    (∃ S : Set ((Base 1 → Base 1) × Set (Base 1)), S.Finite ∧
      (∀ c ∈ S, IsMonomialChart (fun x : Base 1 => x 0) c) ∧
      ∃ N ∈ 𝓝 (0 : Base 1), id '' N ⊆ ⋃ c ∈ S, c.1 '' c.2) ∧
    (∃ T : Set ((Base 1 → Base 1) × Set (Base 1)), T.Finite ∧
      (∀ d ∈ T, IsMonomialChart (fun x : Base 1 => x 0) d) ∧
      id '' Metric.closedBall (0 : Base 1) 1 ⊆ ⋃ d ∈ T, d.1 '' d.2) ∧
    HasMonomialCharts (fun x : Base 1 => x 0) 0 ∧
    HasMonomialCharts (fun x : Base 1 => x 0 + 0) 0 ∧
    HasMonomialCharts (fun x : Base 1 => x 0) (id 0) ∧
    HasMonomialCharts (fun x : Base 1 => x 0) 1 := by
  have hA : ∀ w : Base 1, AnalyticAt ℝ (fun x : Base 1 => x 0) w :=
    fun w => (ContinuousLinearMap.proj 0 : Base 1 →L[ℝ] ℝ).analyticAt w
  have hNC : ∀ w : Base 1, NormalCrossingAt (fun x : Base 1 => x 0) w := by
    intro w
    by_cases hw : w 0 = 0
    · exact (normalCrossingAt_coord 0 w).congr (Eventually.of_forall fun x => by simp [hw])
    · exact normalCrossingAt_of_ne_zero (hA w) hw
  have hH : ∀ w : Base 1, HasMonomialCharts (fun x : Base 1 => x 0) w :=
    fun w => hasMonomialCharts_of_normalCrossingAt (hNC w)
  have hid : ∀ᶠ x in 𝓝 (0 : Base 1), AnalyticAt ℝ (id : Base 1 → Base 1) x :=
    Eventually.of_forall fun _ => analyticAt_id
  have hcover : HasMonomialCharts (fun x : Base 1 => x 0) 0 := by
    refine hasMonomialCharts_of_cover {(id, Metric.closedBall 0 1)} (Set.finite_singleton _)
      (fun c hc => ?_) ?_
    · rw [Set.mem_singleton_iff.1 hc]
      exact ⟨isCompact_closedBall _ _, fun _ _ => analyticAt_id, fun w _ => hH w⟩
    · rw [Set.biUnion_singleton, Set.image_id]
      exact Metric.closedBall_mem_nhds 0 one_pos
  have hshift : HasMonomialCharts (fun x : Base 1 => x 0) (0 + 1) := by
    refine hasMonomialCharts_of_add (hasMonomialCharts_of_normalCrossingAt
      (normalCrossingAt_of_ne_zero ((hA 0).add analyticAt_const) ?_))
    simp
  exact ⟨(hH 0).shrink (Metric.ball_mem_nhds 0 one_pos), HasMonomialCharts.exists_comp (hH 0) hid,
    exists_monomialCharts_of_compact (id, Metric.closedBall 0 1) (isCompact_closedBall _ _)
      (fun _ _ => analyticAt_id) fun w _ => hH w, hcover,
    (hH 0).congr (Eventually.of_forall fun x => (add_zero (x 0)).symm),
    HasMonomialCharts.comp (T' := id) (hH 0) hid continuousAt_id rfl
      (Eventually.of_forall fun _ => rfl), by simpa only [zero_add] using hshift⟩

/-- The pair `(x, x₀)` is analytic, a function of the first `n` coordinates keeps its
normal crossings, boxes are compact neighbourhoods, and the zero vector has norm at most
`1`. Appendix D.1 of arXiv:2510.22026v2. -/
example : AnalyticAt ℝ (fun y : Base 1 => (Fin.snoc y (y 0) : Base 2)) 0 ∧
    NormalCrossingAt (fun x : Base 2 => (Fin.init x : Base 1) 0) 0 ∧
    IsCompact (prodIcc (Metric.closedBall (0 : Base 1) 1) (-1) 1) ∧
    prodIcc (univ : Set (Base 1)) ((0 : Base 2) (Fin.last 1) - 1)
      ((0 : Base 2) (Fin.last 1) + 1) ∈ 𝓝 (0 : Base 2) ∧
    ‖(Fin.snoc (0 : Base 1) 0 : Base 2)‖ ≤ 1 := by
  have hA : AnalyticAt ℝ (fun x : Base 1 => x 0) 0 :=
    (ContinuousLinearMap.proj 0 : Base 1 →L[ℝ] ℝ).analyticAt 0
  have hNC : NormalCrossingAt (fun x : Base 1 => x 0) (Fin.init (0 : Base 2)) :=
    (normalCrossingAt_coord 0 0).congr (Eventually.of_forall fun x => by simp)
  exact ⟨AnalyticAt.snoc analyticAt_id hA, NormalCrossingAt.comp_init hNC,
    IsCompact.prodIcc (isCompact_closedBall _ _) _ _, prodIcc_mem_nhds univ_mem one_pos,
    norm_snoc_le (by simp) (by simp)⟩

/-- The ramification `t ↦ w' ± t^q` moves `0` by at most `0^q`, and reaches every point
within `0^1` of `w'`. Appendix D.1 of arXiv:2510.22026v2. -/
example : ‖ram (0 : Base 1) 2 (fun _ => true) 0 - 0‖ ≤ 0 ^ 2 ∧
    ∃ b t, ram (0 : Base 1) 1 b t = 0 ∧ ∀ j, |t j| ≤ 0 :=
  ⟨norm_ram_sub_le 0 2 _ le_rfl fun j => by simp,
    exists_ram_eq 0 0 one_ne_zero le_rfl fun j => by simp⟩

/-- The exponent `p = (1)` of `t` satisfies the hypotheses of the blow-up identities, and
of the principalization of `(y, t)` for the bound `c = 1`. Appendix D.1 of
arXiv:2510.22026v2. -/
example : |monomial (fun _ : Fin 1 => 1) (fun _ => 1)| ≤ monomial (fun _ : Fin 1 => 1) 1 ∧
    (∀ s : Base 2, monomial (fun _ : Fin 1 => 1) (Fin.init (monomialMap (blowTwo 0) s)) =
      monomialMap (blowTwo 0) s (Fin.last 1) * monomial (Fin.snoc (fun _ => 1) 0) s) ∧
    monomial (Fin.snoc (fun _ : Fin 1 => 1) 0 : Fin 2 → ℕ) (blowTwoBound 1 0) = 1 ∧
    (∃ C, PrincipalCharts 1 (fun _ : Fin 1 => 1) C) ∧
    ∃ C, PrincipalCharts 1 (0 : Fin 1 → ℕ) C :=
  ⟨abs_monomial_le fun i => le_of_eq (abs_one), monomial_blowTwo one_ne_zero,
    monomial_blowTwoBound one_pos one_ne_zero,
    exists_principalCharts_aux one_pos 1 _ (by simp), exists_principalCharts one_pos 0⟩

end Transformer.Landscape.Uniformization
