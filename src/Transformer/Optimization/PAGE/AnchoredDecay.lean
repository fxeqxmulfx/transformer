module

public import Transformer.Optimization.PAGE.AnchoredUpdate
public import Transformer.Optimization.PAGE.AnchorDynamics
public import Mathlib.Topology.MetricSpace.Contracting

/-! A constant-rate local convergence theorem for full-parameter PAGE with
weight decay about a reference. On the certified ball the data gradients
are Lipschitz, so a sufficiently large quadratic penalty makes the full
refresh map contract. Corrections also contract the preceding displacement. -/

@[expose] public section

namespace Transformer.Optimization.PAGE

open Filter Metric Finset Set
open scoped Topology NNReal

variable {I E : Type*} [Fintype I] [Nonempty I] [NormedAddCommGroup E]
  [InnerProductSpace ℝ E] [CompleteSpace E] [ProperSpace E]

omit [Nonempty I] [CompleteSpace E] [ProperSpace E] in
private theorem norm_mean_le (v : I → E) : ‖mean v‖ ≤ mean (fun i => ‖v i‖) := by
  simp only [mean, norm_smul, Real.norm_eq_abs, abs_of_nonneg (by positivity : 0 ≤ (Fintype.card I : ℝ)⁻¹),
    smul_eq_mul]
  exact mul_le_mul_of_nonneg_left (norm_sum_le _ _) (by positivity)

/-- From any bounded set of starts, every run enters an arbitrary open
region around c on its first update and converges inside it at a constant
positive learning rate. Infinitely many refreshes give the stationary
equation at the limit. A stationary center then equals that limit. -/
theorem anchored_page_converges (loss : I → E → ℝ) (hd : ∀ i, Differentiable ℝ (loss i))
    (hl : ∀ i, LocallyLipschitz (gradient (loss i))) {G : Set E} (hG : IsOpen G)
    {c : E} (hc : c ∈ G) (D : ℝ) :
    ∃ η₀ > 0, ∀ η, 0 < η → η ≤ η₀ → ∀ y ∈ closedBall c D, ∀ path : ℕ → Option I,
      let w := fun t => (run (anchoredLoss loss c η⁻¹) η y path t).weights
      (∀ t, w (t+1) ∈ G) ∧ ∃ star ∈ G, Tendsto w atTop (𝓝 star) ∧
        ((∃ᶠ t in atTop, path t = none) →
          star = c - η • full loss star ∧ (full loss c = 0 → star = c)) := by
  obtain ⟨R,hR,hball⟩ := Metric.nhds_basis_closedBall.mem_iff.mp (hG.mem_nhds hc)
  let B := closedBall c R
  have hcompact : IsCompact B := isCompact_closedBall c R
  choose l lip using fun i => (hl i).locallyLipschitzOn.exists_lipschitzOnWith_of_compact hcompact
  let L : ℝ := ∑ i, (l i : ℝ)
  have hL : 0 ≤ L := sum_nonneg fun _ _ => by positivity
  have component (i : I) {x y : E} (hx : x ∈ B) (hy : y ∈ B) :
      ‖gradient (loss i) x - gradient (loss i) y‖ ≤ L * dist x y := by
    have hi : (l i : ℝ) ≤ L := single_le_sum (f := fun j => (l j : ℝ)) (fun _ _ => by positivity) (mem_univ i)
    rw [dist_eq_norm]
    exact ((lip i).norm_sub_le hx hy).trans (mul_le_mul_of_nonneg_right hi (norm_nonneg _))
  have hmean {x y : E} (hx : x ∈ B) (hy : y ∈ B) : ‖full loss x - full loss y‖ ≤ L * dist x y := by
    change ‖mean (fun i => gradient (loss i) x) - mean (fun i => gradient (loss i) y)‖ ≤ _
    rw [← mean_sub]
    exact (norm_mean_le _).trans ((mean_mono fun i => component i hx hy).trans_eq (mean_const _))
  have hfull : Continuous (full loss) := (continuous_finsetSum _ (fun i _ => (hl i).continuous)).const_smul _
  let W := closedBall c (max D R)
  let size (x : E) := ‖full loss x‖ + ∑ i, ‖gradient (loss i) x‖
  have hsize : Continuous size := hfull.norm.add (continuous_finsetSum _ (fun i _ => (hl i).continuous.norm))
  obtain ⟨M,hM⟩ := (isCompact_closedBall c (max D R)).exists_bound_of_continuousOn hsize.continuousOn
  have gradBound {x : E} (hx : x ∈ W) : ‖full loss x‖ ≤ M ∧ ∀ i, ‖gradient (loss i) x‖ ≤ M := by
    have hs := hM x hx
    have hn : 0 ≤ ∑ i, ‖gradient (loss i) x‖ := sum_nonneg fun _ _ => norm_nonneg _
    rw [Real.norm_of_nonneg (add_nonneg (norm_nonneg _) hn : 0 ≤ size x)] at hs
    refine ⟨(le_add_of_nonneg_right hn).trans hs,fun i => ?_⟩
    exact (single_le_sum (f := fun j => ‖gradient (loss j) x‖) (fun j _ => norm_nonneg _) (mem_univ i)).trans
      ((le_add_of_nonneg_left (norm_nonneg _)).trans hs)
  have bigger : B ⊆ W := closedBall_subset_closedBall (le_max_right _ _)
  have hM₀ : 0 ≤ M := (norm_nonneg _).trans (gradBound (bigger (mem_closedBall_self hR.le))).1
  refine ⟨min (R/(32*(M+1))) (1/(8*(L+1))),lt_min (by positivity) (by positivity),?_⟩
  intro η hη hsmall y hy path
  have hηM : η*M ≤ R/32 := by
    have h := (le_div_iff₀ (by positivity : 0 < 32*(M+1))).mp (hsmall.trans (min_le_left _ _))
    linarith
  have hηL : η*L ≤ 1/8 := by
    have h := (le_div_iff₀ (by positivity : 0 < 8*(L+1))).mp (hsmall.trans (min_le_right _ _))
    linarith
  let F := fun x => c - η • full loss x
  have stepDist (x v : E) : dist (x - η • v) x = η * ‖v‖ := by
    rw [dist_eq_norm, show x - η • v - x = -(η • v) by abel, norm_neg, norm_smul, Real.norm_of_nonneg hη.le]
  have near {x : E} (hx : x ∈ W) : dist (F x) c ≤ R/32 := by
    rw [show dist (F x) c = _ from stepDist c (full loss x)]
    exact (mul_le_mul_of_nonneg_left (gradBound hx).1 hη.le).trans hηM
  have smallDiff {x y : E} (v : E) (h : ‖v‖ ≤ L*dist x y) : η*‖v‖ ≤ (1/8)*dist x y := by
    calc η*‖v‖ ≤ η*(L*dist x y) := mul_le_mul_of_nonneg_left h hη.le
         _ = (η*L)*dist x y := (mul_assoc _ _ _).symm
         _ ≤ _ := mul_le_mul_of_nonneg_right hηL dist_nonneg
  have contract {x y : E} (hx : x ∈ B) (hy : y ∈ B) : dist (F x) (F y) ≤ (1/8)*dist x y := by
    rw [dist_eq_norm, show F x - F y = -(η • (full loss x - full loss y)) by dsimp [F]; rw [smul_sub]; abel,
      norm_neg, norm_smul, Real.norm_of_nonneg hη.le]
    exact smallDiff _ (hmean hx hy)
  have maps : MapsTo F B B := fun x hx => (near (bigger hx)).trans (by linarith : R/32 ≤ R)
  let restricted : B → B := maps.restrict F B B
  let _ : Nonempty B := ⟨c,mem_closedBall_self hR.le⟩
  let _ : CompleteSpace B := isClosed_closedBall.isComplete.completeSpace_coe
  have hcontr : ContractingWith (1/8 : ℝ≥0) restricted := by
    refine ⟨by norm_num,LipschitzWith.of_dist_le_mul fun x y => ?_⟩
    change dist (F x.val) (F y.val) ≤ (↑(1/8 : ℝ≥0) : ℝ) * dist x.val y.val
    simpa using contract x.2 y.2
  let z := hcontr.fixedPoint restricted
  have hfix : F (z : E) = z := congrArg Subtype.val hcontr.fixedPoint_isFixedPt.eq
  have hz : dist (z : E) c ≤ R/32 := by simpa only [hfix] using near (bigger z.2)
  let w (t : ℕ) := (run (anchoredLoss loss c η⁻¹) η y path t).weights
  obtain ⟨_,first,moves⟩ := anchored_weight_updates loss hd c y hη path
  have hyW : y ∈ W := closedBall_subset_closedBall (le_max_left _ _) hy
  have hone : dist (w 1) c ≤ R/32 := by
    rw [show w 1 = F y from first]
    exact near hyW
  have h₁ : w 1 ∈ B := hone.trans (by linarith)
  have move₂ : dist (w 2) (w 1) ≤ R/16 := by
    cases hp : path 0 with
    | none =>
      have he : w 2 = F (w 1) := by simpa only [hp] using moves 0
      rw [he]
      linarith [dist_triangle_right (F (w 1)) (w 1) c,near (bigger h₁)]
    | some i =>
      have he : w 2 = w 1 - η • (gradient (loss i) (w 1) - gradient (loss i) y) := by
        change w 2 = w 1 - η • (gradient (loss i) (w 1) - gradient (loss i) (w 0))
        simpa only [hp] using moves 0
      rw [he,stepDist]
      have hb := (norm_sub_le (gradient (loss i) (w 1)) (gradient (loss i) y)).trans
        (add_le_add ((gradBound (bigger h₁)).2 i) ((gradBound hyW).2 i))
      linarith [mul_le_mul_of_nonneg_left hb hη.le]
  have base : dist (w 2) z + dist (w 2) (w 1)/2 ≤ R/2 := by
    linarith [dist_triangle (w 2) (w 1) z,dist_triangle_right (w 1) z c]
  obtain ⟨hw,star,hstar,ht⟩ := converges_of_refresh_or_correction (fun t => w (t+1)) c z hR
    (hz.trans (by linarith)) h₁ base (by
    intro t hx hy'
    cases hp : path (t+1) with
    | none =>
      left
      have hf : w (t+1+2) = F (w (t+1+1)) := by simpa only [hp] using moves (t+1)
      change dist (w (t+1+2)) z ≤ (1/8)*dist (w (t+1+1)) z
      rw [hf]
      simpa only [hfix] using (contract hy' z.2)
    | some i =>
      right
      have hf : w (t+1+2) = w (t+1+1) - η •
          (gradient (loss i) (w (t+1+1)) - gradient (loss i) (w (t+1))) := by
        simpa only [hp] using moves (t+1)
      change dist (w (t+1+2)) (w (t+1+1)) ≤ (1/8)*dist (w (t+1+1)) (w (t+1))
      rw [hf,stepDist]
      exact smallDiff _ (component i hy' hx))
  have hlim : Tendsto w atTop (𝓝 star) := (Filter.tendsto_add_atTop_iff_nat 1).mp ht
  refine ⟨fun t => hball (hw t),star,hball hstar,hlim,fun hrefresh => ?_⟩
  have hcont : Continuous F := continuous_const.sub (hfull.const_smul η)
  have fixed : star = F star := tendsto_nhds_unique_of_frequently_eq
    ((Filter.tendsto_add_atTop_iff_nat 2).mpr hlim) ((hcont.tendsto star).comp ht)
    (hrefresh.mono fun t hp => by
      change w (t+2) = F (w (t+1))
      simpa only [hp] using moves t)
  refine ⟨fixed,fun hg => ?_⟩
  have hcfix : F c = c := by simp [F,hg]
  have hdist := contract hstar (mem_closedBall_self hR.le)
  rw [← fixed,hcfix] at hdist
  exact dist_eq_zero.mp (by linarith [dist_nonneg (x := star) (y := c)])

end Transformer.Optimization.PAGE
