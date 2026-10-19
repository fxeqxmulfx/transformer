module

public import Transformer.Interpolation.Section4_TransferPhases

/-!
# Stability of original protected transfers for concentrated laws

ArXiv:2411.04551v3, §1, `thm: targets.atoms`, its proof in §5,
and §3.2, attention gathering and restoration. A law concentrated
near a selected atom undergoes a controlled perturbation of the
exact protected atomic transfer. Its initial axial mean need not
vanish: the proved nonlinear attention radius estimate controls
both the forward and reversed attention phases.

The middle original ridge flow is Lipschitz on the actual sphere.
Transporting supports through these three actual law maps therefore
gives a radius constant independent of the input law and its initial
radius. In particular a sufficiently small initial support reaches
any prescribed positive final radius around the exact target.

These are full-network estimates. They use the actual rank-one
attention laws in the first and last phases, retaining their
dependence on the evolving original probability law.
-/

@[expose] public section

open Real MeasureTheory Set
open scoped NNReal

namespace Transformer.Interpolation

variable {d : ℕ}

/-- An original ridge law carries a concentrated support into the
Lipschitz enlargement of the corresponding actual particle endpoint.
Source auxiliary: arXiv:2411.04551v3, §1 and §5, the stability
step following exact atomic interpolation. -/
theorem support_ridgeLaw_closedBall (a u : EucSpace d) (b T : ℝ)
    (K : ℝ≥0) (hK : LipschitzWith K (fun x : SSphere d => ridgeFlow a u b x T))
    (p : SSphere d) (ν : Perspective.ProbSphere d) (δ : ℝ)
    (hν : (ν : Measure (SSphere d)).support ⊆ Metric.closedBall p δ) :
    (ridgeLaw a u b ν T : Measure (SSphere d)).support ⊆
      Metric.closedBall (ridgeFlow a u b p T) ((K : ℝ) * δ) := by
  rw [coe_ridgeLaw]
  intro y hy
  obtain ⟨x, hx, rfl⟩ := support_map_subset_image _ _
    (continuous_ridgeFlow_initial a u b T) hy
  exact (hK.dist_le_mul x p).trans
    (mul_le_mul_of_nonneg_left (hν hx) K.2)

/-- A concentrated original law has a quantitative final support
radius under the complete transfer sequence, without a zero-mean
hypothesis. Source auxiliary: arXiv:2411.04551v3, §1 and §5,
stability of matching after concentration. -/
theorem transferPhaseList_support_radius (i : Fin d) (a m : EucSpace d)
    (w p q : SSphere d) (c t₁ T δ : ℝ) (ht₁ : 0 ≤ t₁)
    (hwp : inner ℝ (w : EucSpace d) (p : EucSpace d) = 0)
    (hwq : inner ℝ (w : EucSpace d) (q : EucSpace d) = 0)
    (hend : ridgeFlow a (-m) (-c) p T = q)
    (K : ℝ≥0) (hK : LipschitzWith K (fun x : SSphere d => ridgeFlow a (-m) (-c) x T))
    (ν : Perspective.ProbSphere d)
    (hν : (ν : Measure (SSphere d)).support ⊆ Metric.closedBall p δ) :
    (phaseSequenceLaw (transferPhaseList i a m w c t₁ T) ν : Measure (SSphere d)).support ⊆
      Metric.closedBall q (((K : ℝ) * (δ * exp t₁)) * exp t₁) := by
  have hpre : (meanAxisLaw w 1 ν t₁ : Measure (SSphere d)).support ⊆
      Metric.closedBall p (δ * exp t₁) := by
    simpa only [abs_one, one_mul] using
      support_meanAxisLaw_radius w p 1 ν δ hwp (fun _ hx => hν hx) t₁ ht₁
  have hmid := support_ridgeLaw_closedBall a (-m) (-c) T K hK p
    (meanAxisLaw w 1 ν t₁) (δ * exp t₁) hpre
  rw [hend] at hmid
  rw [phaseSequenceLaw_transfer]
  simpa only [abs_neg, abs_one, one_mul] using
    support_meanAxisLaw_radius w q (-1) (ridgeLaw a (-m) (-c) (meanAxisLaw w 1 ν t₁) T)
      ((K : ℝ) * (δ * exp t₁)) hwq (fun _ hx => hmid hx) t₁ ht₁

/-- One positive radius constant controls every concentrated original
input law under a fixed actual protected transfer. Source auxiliary:
arXiv:2411.04551v3, §1 and §5, the error propagated after point matching. -/
theorem exists_transfer_phase_radius_scale (i : Fin d) (a m : EucSpace d)
    (w p q : SSphere d) (c t₁ T : ℝ) (ht₁ : 0 ≤ t₁)
    (hwp : inner ℝ (w : EucSpace d) (p : EucSpace d) = 0)
    (hwq : inner ℝ (w : EucSpace d) (q : EucSpace d) = 0)
    (hend : ridgeFlow a (-m) (-c) p T = q) :
    ∃ C : ℝ, 0 < C ∧ ∀ δ : ℝ, 0 ≤ δ → ∀ ν : Perspective.ProbSphere d,
      (ν : Measure (SSphere d)).support ⊆ Metric.closedBall p δ →
      (phaseSequenceLaw (transferPhaseList i a m w c t₁ T) ν : Measure (SSphere d)).support ⊆
        Metric.closedBall q (C * δ) := by
  obtain ⟨K, hK⟩ := exists_lipschitzWith_ridgeFlow a (-m) (-c) T
  let C : ℝ := ((K : ℝ) + 1) * (exp t₁) ^ 2
  have hC : 0 < C := by dsimp only [C]; positivity
  refine ⟨C, hC, ?_⟩
  intro δ hδ ν hν x hx
  have hh := transferPhaseList_support_radius i a m w p q c t₁ T δ ht₁ hwp hwq hend K hK ν hν hx
  change dist x q ≤ (K : ℝ) * (δ * exp t₁) * exp t₁ at hh
  change dist x q ≤ C * δ
  dsimp only [C]
  nlinarith [mul_nonneg hδ (sq_nonneg (exp t₁))]

/-- Before applying fixed actual transfer phases, one may choose a
positive concentration radius guaranteeing any positive final error.
The choice works for every original input probability with that
support bound. Source auxiliary: arXiv:2411.04551v3, §1 and §5. -/
theorem exists_transfer_phase_small_radius (i : Fin d) (a m : EucSpace d)
    (w p q : SSphere d) (c t₁ T : ℝ) (ht₁ : 0 ≤ t₁)
    (hwp : inner ℝ (w : EucSpace d) (p : EucSpace d) = 0)
    (hwq : inner ℝ (w : EucSpace d) (q : EucSpace d) = 0)
    (hend : ridgeFlow a (-m) (-c) p T = q) (ε : ℝ) (hε : 0 < ε) :
    ∃ δ : ℝ, 0 < δ ∧ ∀ ν : Perspective.ProbSphere d,
      (ν : Measure (SSphere d)).support ⊆ Metric.closedBall p δ →
      (phaseSequenceLaw (transferPhaseList i a m w c t₁ T) ν : Measure (SSphere d)).support ⊆
        Metric.closedBall q ε := by
  obtain ⟨C, hC, hbound⟩ := exists_transfer_phase_radius_scale i a m w p q c t₁ T ht₁ hwp hwq hend
  refine ⟨ε / C, div_pos hε hC, ?_⟩
  intro ν hν
  have he : C * (ε / C) = ε := by field_simp
  simpa only [he] using hbound (ε / C) (div_pos hε hC).le ν hν

/-- A nonzero exact gated transfer between transverse coordinate
atoms, a positive tolerance, a genuinely supported input Dirac and
an actual flow Lipschitz constant jointly realize every hypothesis. -/
example : ∃ (a m : EucSpace 3) (w p q : SSphere 3) (c T : ℝ)
    (K : ℝ≥0) (ν : Perspective.ProbSphere 3),
    (0 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 ∧
      inner ℝ (w : EucSpace 3) (p : EucSpace 3) = 0 ∧
      inner ℝ (w : EucSpace 3) (q : EucSpace 3) = 0 ∧
      ridgeFlow a (-m) (-c) p T = q ∧
      LipschitzWith K (fun x : SSphere 3 => ridgeFlow a (-m) (-c) x T) ∧
      (ν : Measure (SSphere 3)).support ⊆ Metric.closedBall p 1 := by
  let p : SSphere 3 := ⟨PiLp.single 2 (0 : Fin 3) (1 : ℝ), by simp⟩
  let q : SSphere 3 := ⟨PiLp.single 2 (1 : Fin 3) (1 : ℝ), by simp⟩
  let w : SSphere 3 := ⟨PiLp.single 2 (2 : Fin 3) (1 : ℝ), by simp⟩
  have hpq : p ≠ q := by
    intro he
    have hh := congrArg (fun z : SSphere 3 => (z : EucSpace 3) (0 : Fin 3)) he
    norm_num [p, q] at hh
  have hanti : q ≠ antipode 3 p := by
    intro he
    have hh := congrArg (fun z : SSphere 3 => (z : EucSpace 3) (1 : Fin 3)) he
    norm_num [p, q, antipode] at hh
  obtain ⟨a, c, T, _, _, hend, _⟩ := exists_exact_gated_transfer p q hpq hanti
  let m : EucSpace 3 := (p : EucSpace 3) + (q : EucSpace 3)
  obtain ⟨K, hK⟩ := exists_lipschitzWith_ridgeFlow (a : EucSpace 3) (-m) (-c) T
  refine ⟨a, m, w, p, q, c, T, K, Perspective.diracProb 3 p,
    zero_le_one, one_pos, ?_, ?_, hend, hK, ?_⟩
  · simp [w, p, EuclideanSpace.inner_single_left]
  · simp [w, q, EuclideanSpace.inner_single_left]
  · intro x hx
    rw [eq_of_mem_support_dirac hx]
    exact Metric.mem_closedBall.mpr
      (by simpa only [dist_self] using (zero_le_one : (0 : ℝ) ≤ 1))

end Transformer.Interpolation
