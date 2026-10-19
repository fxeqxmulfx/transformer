module

public import Transformer.Interpolation.Section5_BallRadius

/-!
# Equal-mass balls for separated small carriers

ArXiv:2411.04551v3, §5, `cl: balls`. The actual common-radius
quantiles of each original probability provide its ordered centers.
Their differences have mass `1/M` by nesting on the small carrier.
The proved parking construction gives a point outside every later ball.

Each ball stays within `3k/8` of its own carrier center, while the
other carriers are at distance at least `2κ`, with `k ≤ κ`.
Thus the original measures see none of one another's actual balls.
This explicitly uses both source inputs: separation and support
smallness. Absolute continuity is used only through null spheres.
-/

@[expose] public section

open Real MeasureTheory Metric Set

namespace Transformer.Interpolation

variable {d : ℕ}

/-- The full geometric and measure construction in `cl: balls` for
actual separated small carriers. Source: arXiv:2411.04551v3, §5,
`cl: balls`, `eq: boundary.ball`, `eq: bound` and `eq: min.dist`.
The explicit scale `k/1000` is a valid choice of the source's
"small enough" support radius; all three original conclusions are retained. -/
theorem equal_mass_balls_of_small_carriers (hd : 2 ≤ d) (N M : ℕ) (hM : 0 < M)
    (ν : Fin N → Perspective.ProbSphere d) (S : Fin N → Set (SSphere d))
    (p : Fin N → SSphere d) (κ k : ℝ) (hk : 0 < k) (hk1 : k ≤ 1) (hkκ : k ≤ κ)
    (hboundary : ∀ i, ∀ y : SSphere d, ∀ r : ℝ, (ν i : Measure (SSphere d)) (sphere y r) = 0)
    (hcarrier : ∀ i, (ν i : Measure (SSphere d)) (S i)ᶜ = 0) (hp : ∀ i, p i ∈ S i)
    (hsmall : ∀ i, ∀ x ∈ S i, dist x (p i) ≤ k / 1000)
    (hsep : ∀ i j, i ≠ j → ∀ x ∈ S i, ∀ y ∈ S j, 2 * κ ≤ dist x y) :
    ∃ (x : Fin N → ℕ → SSphere d) (r : Fin N → ℝ),
      (∀ i, 0 < r i) ∧
      (∀ i, (ν i : Measure (SSphere d)) (ball (x i 0) (r i)) = ENNReal.ofReal ((M : ℝ)⁻¹)) ∧
      (∀ i, ∀ m : ℕ, 1 ≤ m → m < M →
        (ν i : Measure (SSphere d)) (ball (x i m) (r i) \ ball (x i (m - 1)) (r i)) =
          ENNReal.ofReal ((M : ℝ)⁻¹)) ∧
      (∀ i, ∀ m : ℕ, m + 1 < M → ∃ z ∈ ball (x i m) (r i),
        ∀ m' : ℕ, m < m' → m' < M → z ∉ ball (x i m') (r i)) ∧
      ∀ i j, i ≠ j → ∀ m : ℕ, m < M →
        (ν i : Measure (SSphere d)) (ball (x j m) (r j)) = 0 := by
  choose v hperp using fun i => exists_unit_tangent hd (p i)
  have hex (i : Fin N) := exists_small_cap_sweep_quantiles (ν i : Measure (SSphere d))
    (p i) (v i) (hperp i) (S i) (hcarrier i) (hboundary i) k hk hk1 (hsmall i) M hM
  choose r t hrlo hrhi ht hanti hmass using hex
  let x : Fin N → ℕ → SSphere d := fun i m =>
    if hm : m < M then sweepPoint (p i) (v i) (hperp i) (t i ⟨m, hm⟩) else p i
  have hxeq (i : Fin N) (m : Fin M) : x i m.val = sweepPoint (p i) (v i) (hperp i) (t i m) := by
    simp only [x, dite_eq_left m.isLt]
  have hr (i : Fin N) : 0 < r i := lt_trans (by positivity : 0 < k / 1000) (hrlo i)
  have hr1 (i : Fin N) : r i ≤ 1 := by linarith [hrhi i]
  have hcap (i : Fin N) (y : SSphere d) (hy : y ∈ S i) : y ∈ ball (p i) (r i) :=
    lt_of_le_of_lt (hsmall i y hy) (hrlo i)
  have hcenter (i : Fin N) (m : Fin M) : dist (x i m.val) (p i) ≤ k / 8 := by
    rw [hxeq]
    have h := dist_sweepPoint_le_abs (p i) (v i) (hperp i) (t i m) 0
    rw [sweepPoint_zero, sub_zero, abs_of_nonneg (ht i m).1] at h
    exact h.trans (ht i m).2
  refine ⟨x, r, hr, ?_, ?_, ?_, ?_⟩
  · intro i
    rw [hxeq i ⟨0, hM⟩]
    simpa only [Nat.cast_zero, zero_add, one_div] using hmass i ⟨0, hM⟩
  · intro i m hm hmM
    have hprev : m - 1 < M := by omega
    let a : Fin M := ⟨m, hmM⟩
    let b : Fin M := ⟨m - 1, hprev⟩
    have hab : t i a < t i b := hanti i (by change m - 1 < m; omega)
    have hB : (ν i : Measure (SSphere d)) (ball (sweepPoint (p i) (v i) (hperp i) (t i b)) (r i)) =
        ENNReal.ofReal ((m : ℝ) / M) := by
      simpa only [b, Nat.cast_sub hm, Nat.cast_one, sub_add_cancel] using hmass i b
    rw [hxeq i a, hxeq i b]
    exact sweep_shell_mass (ν i : Measure (SSphere d)) (p i) (v i) (hperp i) (S i) (hcarrier i)
      (hr i) (hr1 i) (ht i a).1 hab.le (by linarith [(ht i b).2, Real.pi_gt_three])
      (hcap i) M m (hmass i a) hB
  · intro i m hm
    have hmM : m < M := by omega
    let a : Fin M := ⟨m, hmM⟩
    let b : Fin M := ⟨m + 1, hm⟩
    obtain ⟨z, hz, hrest⟩ := exists_sweep_parking (p i) (v i) (hperp i) (hr i) (hr1 i)
      (a := k / 8) (ht i b).1 (hanti i (by change m < m + 1; omega)) (ht i a).2
      (by linarith [Real.pi_gt_three])
    refine ⟨z, ?_, fun m' hmm' hm'M => ?_⟩
    · rwa [hxeq i a]
    · let c : Fin M := ⟨m', hm'M⟩
      rw [hxeq i c]
      exact hrest (t i c) (ht i c).1 ((hanti i).antitone (by change m + 1 ≤ m'; omega))
  · intro i j hij m hm
    apply measure_zero_of_disjoint_carrier (ν i : Measure (SSphere d)) (S i) (hcarrier i)
    intro y hy hball
    have hgap := hsep i j hij y hy (p j) (hp j)
    have hclose := hcenter j ⟨m, hm⟩
    have htri := dist_triangle y (x j m) (p j)
    have hball' : dist y (x j m) < r j := hball
    linarith [hrhi j]

/-- The source's "small enough" support scale can be chosen before
the actual measures and carriers. Source: arXiv:2411.04551v3, §5,
`cl: balls`, `eq: smallness.two`, `eq: boundary.ball` and `eq: bound`.
We choose `δ = min(κ,1)/1000`; the original count, separated carriers,
all requested shell masses, parking points and cross-measure zeros remain.
The positive count is the manuscript's number of target atoms. -/
theorem exists_smallness_equal_mass_balls (hd : 2 ≤ d) (N M : ℕ) (hM : 0 < M)
    (κ : ℝ) (hκ : 0 < κ) :
    ∃ δ : ℝ, 0 < δ ∧
      ∀ (ν : Fin N → Perspective.ProbSphere d) (S : Fin N → Set (SSphere d)),
      (∀ i, ∀ y : SSphere d, ∀ r : ℝ, (ν i : Measure (SSphere d)) (sphere y r) = 0) →
      (∀ i, (ν i : Measure (SSphere d)) (S i)ᶜ = 0) →
      (∀ i j, i ≠ j → ∀ x ∈ S i, ∀ y ∈ S j, 2 * κ ≤ dist x y) →
      (∀ i, ∃ p ∈ S i, ∀ x ∈ S i, dist x p ≤ δ) →
      ∃ (x : Fin N → ℕ → SSphere d) (r : Fin N → ℝ),
        (∀ i, 0 < r i) ∧
        (∀ i, (ν i : Measure (SSphere d)) (ball (x i 0) (r i)) = ENNReal.ofReal ((M : ℝ)⁻¹)) ∧
        (∀ i, ∀ m : ℕ, 1 ≤ m → m < M →
          (ν i : Measure (SSphere d)) (ball (x i m) (r i) \ ball (x i (m - 1)) (r i)) =
            ENNReal.ofReal ((M : ℝ)⁻¹)) ∧
        (∀ i, ∀ m : ℕ, m + 1 < M → ∃ z ∈ ball (x i m) (r i),
          ∀ m' : ℕ, m < m' → m' < M → z ∉ ball (x i m') (r i)) ∧
        ∀ i j, i ≠ j → ∀ m : ℕ, m < M →
          (ν i : Measure (SSphere d)) (ball (x j m) (r j)) = 0 := by
  let k := min κ 1
  have hk : 0 < k := lt_min hκ one_pos
  refine ⟨k / 1000, by positivity, fun ν S hboundary hcarrier hsep hsmall => ?_⟩
  choose p hp hsmall using hsmall
  exact equal_mass_balls_of_small_carriers hd N M hM ν S p κ k hk
    (min_le_right _ _) (min_le_left _ _) hboundary hcarrier hp hsmall hsep

/-- The full source claim's dimension, positive atom count and
positive separation scale are simultaneously realized on the circle.
The nonempty diffuse family is supplied below. -/
example : (2 : ℕ) ≤ 2 ∧ (0 : ℕ) < 2 ∧ (0 : ℝ) < 1 :=
  ⟨le_rfl, by decide, one_pos⟩

/-- A nonempty original family with two requested masses realizes
all geometric, probability and diffuse-boundary inputs at once.
Source: arXiv:2411.04551v3, §5, `cl: balls`; the law is an actual
normalized uniform restriction to a small closed circle cap. -/
example : ∃ (ν : Fin 1 → Perspective.ProbSphere 2) (S : Fin 1 → Set (SSphere 2))
    (p : Fin 1 → SSphere 2),
    (2 : ℕ) ≤ 2 ∧ (0 : ℕ) < 2 ∧ (0 : ℝ) < 1 ∧ (1 : ℝ) ≤ 1 ∧ (1 : ℝ) ≤ 1 ∧
      (∀ i, ∀ y : SSphere 2, ∀ r : ℝ, (ν i : Measure (SSphere 2)) (sphere y r) = 0) ∧
      (∀ i, (ν i : Measure (SSphere 2)) (S i)ᶜ = 0) ∧ (∀ i, p i ∈ S i) ∧
      (∀ i, ∀ x ∈ S i, dist x (p i) ≤ (1 : ℝ) / 1000) ∧
      ∀ i j, i ≠ j → ∀ x ∈ S i, ∀ y ∈ S j, 2 * (1 : ℝ) ≤ dist x y := by
  obtain ⟨ν, hν, hcap⟩ := exists_small_cap_ball_diffuse le_rfl (basePoint 1) (1 / 1000) (by norm_num)
  refine ⟨fun _ => ν, fun _ => closedBall (basePoint 1) (1 / 1000), fun _ => basePoint 1,
    le_rfl, by decide, one_pos, le_rfl, le_rfl, fun _ => hν, fun _ => hcap,
    fun _ => by simp, fun _ x hx => hx, ?_⟩
  intro i j hij
  exact (hij (Subsingleton.elim i j)).elim

end Transformer.Interpolation
