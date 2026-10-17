/-
# Clustering one law on the circle

ArXiv:2411.04551v3, §2.2, proof of `prop: compression`, Steps 1–3, for
one law on the circle (`d = 2`): its mass in an open region closed under
geodesic hulls, either the whole circle or inside an open half-plane, is
gathered near prescribed distinct points of the region with prescribed
weights by phases local in the region. A window of angles carries the
law and the points; the law is collapsed into a small arc around the
first point in angle order and then swept along the window. The masses
near the points bound the distance to the atomic target.
-/

import Transformer.Interpolation.Section2_CircleSweep
import Transformer.Interpolation.Section2_ClusterCoupling
import Mathlib.Data.Fin.Tuple.Sort

open scoped ENNReal NNReal Topology
open Real MeasureTheory Set Filter

namespace Transformer.Interpolation

/-- **Clustering one law, `d = 2`.** On the circle, a law without atoms
carried by an open region closed under geodesic hulls that is the whole
circle or lies in an open half-plane, distinct points of the region, and
weights summing to one: for every `ε > 0`, phases local in the region
bring the law within `ε` of the atomic law of the points with these
weights in `W_2`. Source: arXiv:2411.04551v3, §2.2, proof of
`prop: compression`, Steps 1–3, for one law on the circle, with the
region in place of the geodesic convex hull of its support. -/
theorem exists_cluster_phases_of_eq_two (ν : Perspective.ProbSphere 2)
    (hatom : ∀ y : SSphere 2, (ν : Measure (SSphere 2)) {y} = 0) (U : Set (SSphere 2))
    (hU : IsOpen U) (hconv : ∀ A ⊆ U, convG 2 A ⊆ U)
    (hcap : U = univ ∨ ∃ f : StrongDual ℝ (EucSpace 2), ∀ x ∈ U, 0 < f x)
    (hνU : (ν : Measure (SSphere 2)) U = 1) {M : ℕ} (x : Fin M → SSphere 2)
    (hxinj : Function.Injective x) (hx : ∀ k, x k ∈ U) (α : Fin M → ℝ≥0)
    (hα : ∑ k, α k = 1) (ε : ℝ) (hε : 0 < ε) :
    ∃ L : List (AutonomousPhase 2), AllLocalIn L U ∧
      W2 2 (phaseSequenceLaw L ν : Measure (SSphere 2)) (∑ k, α k • Measure.dirac (x k)) ≤
        ε := by
  classical
  obtain ⟨M, rfl⟩ : ∃ M', M = M' + 1 := Nat.exists_eq_succ_of_ne_zero fun h => by
    subst h
    simp at hα
  set ε₁ := min ε 1
  have hε₁ : 0 < ε₁ := lt_min hε one_pos
  have hε₁1 : ε₁ ≤ 1 := min_le_right _ _
  have hM : (0 : ℝ) < M + 1 := by positivity
  set e := ε₁ ^ 2 / (16 * (M + 1)) with he_def
  have he : 0 < e := by positivity
  have he1 : e ≤ 1 := div_le_one_of_le₀ (by nlinarith) (by positivity)
  set ε' := e / (M + 1) with hε'_def
  have hε' : 0 < ε' := by positivity
  have hε'1 : ε' ≤ 1 := (div_le_self he.le (by linarith [(M.cast_nonneg : (0 : ℝ) ≤ M)])).trans he1
  have hsum : ε' + M * ε' = e := by
    rw [hε'_def]
    field_simp
    ring
  obtain ⟨a, b, hab, h2π, hWU, hνW, hT⟩ := exists_circle_window (ν : Measure (SSphere 2)) hatom U
    hU hconv hcap hνU (Finset.univ.image x) fun y hy => by
      obtain ⟨k, -, rfl⟩ := Finset.mem_image.mp hy
      exact hx k
  choose θ' hθ' hθ'x using fun k => hT (x k) (Finset.mem_image_of_mem x (Finset.mem_univ k))
  have hθinj : Function.Injective θ' := fun k k' h => hxinj (by rw [← hθ'x k, ← hθ'x k', h])
  set σ := Tuple.sort θ'
  obtain ⟨θ, hθ⟩ : ∃ θ : Fin (M + 1) → ℝ, θ = θ' ∘ σ := ⟨_, rfl⟩
  have hθj (j : Fin (M + 1)) : θ j = θ' (σ j) := by rw [hθ]; rfl
  have hθmono : StrictMono θ := by
    rw [hθ]
    exact (Tuple.monotone_sort θ').strictMono_of_injective (hθinj.comp σ.injective)
  have hlt {c : ℝ} (hc : 0 < c) : ∀ᶠ r in 𝓝[>] (0 : ℝ), r < c :=
    eventually_nhdsWithin_of_eventually_nhds (eventually_lt_nhds hc)
  have hgap : ∀ᶠ r in 𝓝[>] (0 : ℝ), ∀ j j' : Fin (M + 1), j < j' → r < (θ j' - θ j) / 2 := by
    refine eventually_all.mpr fun j => eventually_all.mpr fun j' => ?_
    by_cases hjj : j < j'
    · exact (hlt (by linarith [hθmono hjj])).mono fun _ h _ => h
    · exact Eventually.of_forall fun _ h => absurd h hjj
  have hleft : ∀ᶠ r in 𝓝[>] (0 : ℝ), ∀ j, r < θ j - a :=
    eventually_all.mpr fun j => hlt (by linarith [hθj j, (hθ' (σ j)).1])
  have hright : ∀ᶠ r in 𝓝[>] (0 : ℝ), ∀ j, r < b - θ j :=
    eventually_all.mpr fun j => hlt (by linarith [hθj j, (hθ' (σ j)).2])
  obtain ⟨s, hs, hsgap, hsl, hsr, hsε⟩ := (eventually_mem_nhdsWithin.and
    (hgap.and (hleft.and (hright.and (hlt (half_pos hε₁)))))).exists
  have hs0 : 0 < s := hs
  have hW := image_Ioo_eq_ball hab.le h2π
  have hA₀ := ball_circlePoint (θ 0) s hs0.le (by linarith [hsl 0, hsr 0])
  obtain ⟨P₀, Φ₀, hP₀, hmass₀⟩ := exists_collapse_phase (0 : Fin 2) 1 (by decide)
    (circlePoint ((a + b) / 2)) (circlePoint (θ 0)) (chord ((b - a) / 2))
    (chord_pos (by linarith) (by linarith)) (chord_le_two _)
    (Metric.ball (circlePoint (θ 0)) (chord s)) Metric.isOpen_ball
    (by rw [hA₀, ← hW]; exact image_mono (Ioo_subset_Ioo (by linarith [hsl 0])
      (by linarith [hsr 0])))
    (Metric.mem_ball_self (chord_pos hs0 (by linarith [hsl 0, hsr 0]))) ε' hε' hε'1 ν
  rw [← hW] at hP₀ hmass₀
  rw [hA₀, hνW, mul_one] at hmass₀
  have hρ₀ : 1 - ε' ≤
      (P₀.2 ν 1 : Measure (SSphere 2)).real (circlePoint '' Ioo (θ 0 - s) (θ 0 + s)) := by
    have h := ENNReal.toReal_mono (measure_ne_top _ _) hmass₀
    rwa [ENNReal.toReal_ofReal (by linarith)] at h
  obtain ⟨L', E', hL', hE'm, hE'sub, hE'mass⟩ := exists_circle_sweep (b := b) hs0 hε' hε'1 M θ
    (fun j => (α (σ j) : ℝ)) (P₀.2 ν 1) ε'
    (fun k => by linarith [hsgap k.castSucc k.succ Fin.castSucc_lt_succ])
    (fun j => by linarith [hsr j]) (by linarith [hsl 0]) (fun j => (α (σ j)).coe_nonneg)
    hε'.le (hP₀.map_singleton_eq_zero ν hatom) (by
      have h1 : ∑ j, (α (σ j) : ℝ) = 1 := by
        rw [Equiv.sum_comp σ (fun k => (α k : ℝ)), ← NNReal.coe_sum, hα, NNReal.coe_one]
      linarith)
  refine ⟨P₀ :: L', fun Q hQ => ?_, ?_⟩
  · rcases List.mem_cons.mp hQ with rfl | hQ
    · exact ⟨Φ₀, hP₀.mono hWU⟩
    · obtain ⟨Ψ, hΨ⟩ := hL' Q hQ
      exact ⟨Ψ, hΨ.mono ((image_mono (Ioo_subset_Ioo_left (by linarith [hsl 0]))).trans hWU)⟩
  have hdisjE : Pairwise (Function.onFun Disjoint fun k => E' (σ.symm k)) := fun k k' hkk' => by
    have hne : σ.symm k ≠ σ.symm k' := fun h => hkk' (σ.symm.injective h)
    have key (j j' : Fin (M + 1)) (hjj' : j < j') : Disjoint (E' j) (E' j') :=
      (disjoint_image_circlePoint h2π (I := Icc (θ j - s) (θ j + s))
        (J := Icc (θ j' - s) (θ j' + s))
        (fun t ht => ⟨by linarith [ht.1, hsl j], by linarith [ht.2, hsr j]⟩)
        (fun t ht => ⟨by linarith [ht.1, hsl j'], by linarith [ht.2, hsr j']⟩)
        (Set.disjoint_left.mpr fun t h₁ h₂ => by
          linarith [h₁.2, h₂.1, hsgap j j' hjj'])).mono (hE'sub j) (hE'sub j')
    rcases lt_or_gt_of_ne hne with h | h
    · exact key _ _ h
    · exact (key _ _ h).symm
  have hEs : ∀ k, ∀ y ∈ E' (σ.symm k), dist y (x k) ≤ ε₁ / 2 := fun k y hy => by
    obtain ⟨t, ht, rfl⟩ := hE'sub (σ.symm k) hy
    have hθk : θ (σ.symm k) = θ' k := by rw [hθj, Equiv.apply_symm_apply]
    rw [← hθ'x k]
    calc dist (circlePoint t) (circlePoint (θ' k)) ≤ |t - θ' k| := dist_circlePoint_le _ _
      _ ≤ s := abs_le.mpr ⟨by linarith [ht.1], by linarith [ht.2]⟩
      _ ≤ ε₁ / 2 := hsε.le
  have hmassE : ∀ k, (α k : ℝ≥0∞) ≤
      (phaseSequenceLaw (P₀ :: L') ν : Measure (SSphere 2)) (E' (σ.symm k)) +
        ENNReal.ofReal e := fun k => by
    have h := hE'mass (σ.symm k)
    rw [Equiv.apply_symm_apply] at h
    rw [phaseSequenceLaw_cons, ← ENNReal.ofReal_coe_nnreal,
      ← ENNReal.ofReal_toReal (measure_ne_top
        (phaseSequenceLaw L' (P₀.2 ν 1) : Measure (SSphere 2)) (E' (σ.symm k))),
      ← ENNReal.ofReal_add ENNReal.toReal_nonneg he.le]
    refine ENNReal.ofReal_le_ofReal ?_
    rw [← measureReal_def, ← hsum]
    linarith
  refine (W2_le_of_cluster_masses (phaseSequenceLaw (P₀ :: L') ν : Measure (SSphere 2)) x α hα
    (fun k => E' (σ.symm k)) (fun k => hE'm _) hdisjE (ε₁ / 2) e he.le hEs hmassE).trans ?_
  have h4 : 4 * ((M + 1 : ℕ) : ℝ) * e = ε₁ ^ 2 / 4 := by
    rw [he_def]
    push_cast
    field_simp
    ring
  rw [Real.sqrt_le_iff, h4]
  refine ⟨hε.le, ?_⟩
  have hε₁ε : ε₁ ≤ ε := min_le_left _ _
  nlinarith

/-- The hypotheses of `exists_cluster_phases_of_eq_two` are satisfiable: a
law without atoms on the circle, the whole circle, and one point of
weight one. -/
example : ∃ ν : Perspective.ProbSphere 2,
    (∀ y : SSphere 2, (ν : Measure (SSphere 2)) {y} = 0) ∧
    (ν : Measure (SSphere 2)) univ = 1 ∧ IsOpen (univ : Set (SSphere 2)) ∧
    (∀ A ⊆ (univ : Set (SSphere 2)), convG 2 A ⊆ univ) ∧
    ((univ : Set (SSphere 2)) = univ ∨
      ∃ f : StrongDual ℝ (EucSpace 2), ∀ x ∈ (univ : Set (SSphere 2)), 0 < f x) ∧
    Function.Injective (fun _ : Fin 1 => axisX) ∧ ∑ _k : Fin 1, (1 : ℝ≥0) = 1 := by
  obtain ⟨ν, hν, -⟩ := exists_diffuse_probSphere 2 le_rfl isClosed_univ (by
    have := MeanField.isProbabilityMeasure_uniformLaw 2 (by norm_num)
    rw [measure_univ]
    exact one_pos)
  exact ⟨ν, hν, measure_univ, isOpen_univ, fun _ _ => subset_univ _, Or.inl rfl,
    fun a b _ => Subsingleton.elim a b, by simp⟩

end Transformer.Interpolation
