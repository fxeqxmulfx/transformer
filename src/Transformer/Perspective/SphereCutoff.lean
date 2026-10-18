/-
# Smooth cut-offs of tuples away from the zero vectors

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

`g = 𝖤_β ∘ normalize` and the rescalings `‖x_i‖²` of `USA` and `SA` are smooth only
off the tuples with a zero vector, and they are unbounded, while saddle avoidance asks
for a `C²` energy and a bounded, Lipschitz field on the whole space.  A cut-off mends
both.  On the shell `ε < ‖x_i‖² < K` (`blockShell`), `χ(x) = Π_i φ(‖x_i‖²)` is `1`
for a bump `φ` of `ℝ` equal to `1` on `[ε, K]` (`blockCutoff`); `χ` is smooth with
compact support away from the zero vectors (`hasCompactSupport_blockCutoff`,
`tsupport_blockCutoff_subset`).  A product `χ • φ` is then smooth wherever `φ` is
smooth near the support of `χ` (`Landscape.SaddleAvoidance.contDiff_smul_of_tsupport_subset`),
and block scalings depend linearly on their scales (`blockScaleCLM`).

Source: arXiv:2312.10794v5, Appendix A, `l:nosaddleconv` (the center-stable manifold
theorem, applied here on a shell), for `thm: boumal` of §6.
-/

import Transformer.Perspective.SphereConvergence
import Transformer.Landscape.SaddleAvoidance.Cutoff
import Mathlib.Analysis.Calculus.BumpFunction.InnerProduct

open Filter
open scoped Topology

namespace Transformer
namespace Perspective

variable {d n : ℕ}

/-- **Block scaling is linear in the scales:** `c ↦ blockScale c`. -/
noncomputable def blockScaleCLM :
    (Idx n → ℝ) →L[ℝ] (TupleSpace d n →L[ℝ] TupleSpace d n) :=
  LinearMap.toContinuousLinearMap
    { toFun := blockScale
      map_add' := fun c c' => ContinuousLinearMap.ext fun v => PiLp.ext fun i => by
        simp [add_smul]
      map_smul' := fun a c => ContinuousLinearMap.ext fun v => PiLp.ext fun i => by
        simp [mul_smul] }

@[simp] theorem blockScaleCLM_apply (c : Idx n → ℝ) :
    (blockScaleCLM c : TupleSpace d n →L[ℝ] TupleSpace d n) = blockScale c := rfl

/-- **The shell** of tuples whose vectors have squared norms in `(ε, K)`. -/
def blockShell (ε K : ℝ) : Set (TupleSpace d n) :=
  {x | ∀ i, ε < ‖x.ofLp i‖ ^ 2 ∧ ‖x.ofLp i‖ ^ 2 < K}

theorem isOpen_blockShell (ε K : ℝ) : IsOpen (blockShell (d := d) (n := n) ε K) := by
  simp only [blockShell]
  rw [Set.ofPred_forall]
  refine isOpen_iInter_of_finite fun i => ?_
  have hc : Continuous fun x : TupleSpace d n => ‖x.ofLp i‖ ^ 2 :=
    (continuous_norm.comp (PiLp.continuous_apply 2 _ i)).pow 2
  exact (isOpen_lt continuous_const hc).and (isOpen_lt hc continuous_const)

/-- The bump of `ℝ` equal to `1` on `[ε, K]`, supported in `[ε/2, K + ε/2]`. -/
noncomputable def shellBump {ε K : ℝ} (hε : 0 < ε) (hεK : ε < K) :
    ContDiffBump ((ε + K) / 2) :=
  ⟨(K - ε) / 2, K / 2, by linarith, by linarith⟩

/-- **The cut-off of the shell:** `χ(x) = Π_i φ(‖x_i‖²)` for the bump `φ` of `[ε, K]`. -/
noncomputable def blockCutoff {ε K : ℝ} (hε : 0 < ε) (hεK : ε < K) (x : TupleSpace d n) : ℝ :=
  ∏ i, shellBump hε hεK (‖x.ofLp i‖ ^ 2)

variable {ε K : ℝ} (hε : 0 < ε) (hεK : ε < K)
include hε hεK

theorem contDiff_blockCutoff {k : ℕ∞} :
    ContDiff ℝ k (blockCutoff (d := d) (n := n) hε hεK) :=
  contDiff_prod fun i _ => ((shellBump hε hεK).contDiff (n := k)).comp
    ((contDiff_norm_sq ℝ).comp (PiLp.proj 2 (𝕜 := ℝ) (fun _ : Idx n => EucSpace d) i).contDiff)

theorem blockCutoff_nonneg (x : TupleSpace d n) : 0 ≤ blockCutoff hε hεK x :=
  Finset.prod_nonneg fun _ _ => (shellBump hε hεK).nonneg

theorem blockCutoff_le_one (x : TupleSpace d n) : blockCutoff hε hεK x ≤ 1 :=
  Finset.prod_le_one₀ (fun _ _ => (shellBump hε hεK).nonneg) fun _ _ => (shellBump hε hεK).le_one

/-- **The cut-off is `1` on the shell.** -/
theorem blockCutoff_eq_one {x : TupleSpace d n} (hx : x ∈ blockShell ε K) :
    blockCutoff hε hεK x = 1 :=
  Finset.prod_eq_one fun i _ => (shellBump hε hεK).one_of_mem_closedBall (by
    rw [Metric.mem_closedBall, Real.dist_eq, abs_sub_le_iff]
    simp only [shellBump]
    constructor <;> linarith [(hx i).1, (hx i).2])

/-- Off the closed shell `ε/2 ≤ ‖x_i‖² ≤ K + ε/2` the cut-off vanishes. -/
theorem tsupport_blockCutoff_subset_closedShell :
    tsupport (blockCutoff (d := d) (n := n) hε hεK) ⊆
      {x | ∀ i, ε / 2 ≤ ‖x.ofLp i‖ ^ 2 ∧ ‖x.ofLp i‖ ^ 2 ≤ K + ε / 2} := by
  refine closure_minimal (fun x hx i => ?_) ?_
  · have hi := Finset.prod_ne_zero_iff.1 hx i (Finset.mem_univ i)
    have hb : ‖x.ofLp i‖ ^ 2 ∈ Metric.ball ((ε + K) / 2) (shellBump hε hεK).rOut := by
      rw [← (shellBump hε hεK).support_eq]
      exact hi
    rw [Metric.mem_ball, Real.dist_eq, abs_sub_lt_iff] at hb
    simp only [shellBump] at hb
    constructor <;> linarith [hb.1, hb.2]
  · simp only [Set.ofPred_forall]
    refine isClosed_iInter fun i => ?_
    have hc : Continuous fun x : TupleSpace d n => ‖x.ofLp i‖ ^ 2 :=
      (continuous_norm.comp (PiLp.continuous_apply 2 _ i)).pow 2
    exact (isClosed_le continuous_const hc).inter (isClosed_le hc continuous_const)

/-- **The cut-off vanishes near the tuples with a zero vector.** -/
theorem tsupport_blockCutoff_subset :
    tsupport (blockCutoff (d := d) (n := n) hε hεK) ⊆ {x | ∀ i, x.ofLp i ≠ 0} :=
  fun x hx i h => by
    have := ((tsupport_blockCutoff_subset_closedShell hε hεK) hx i).1
    rw [h, norm_zero] at this
    linarith

/-- **The cut-off has compact support.** -/
theorem hasCompactSupport_blockCutoff :
    HasCompactSupport (blockCutoff (d := d) (n := n) hε hεK) := by
  refine Metric.isCompact_of_isClosed_isBounded (isClosed_tsupport _)
    ((Metric.isBounded_closedBall (x := 0) (r := n * (K + ε / 2) + 1)).subset fun x hx => ?_)
  have hx := tsupport_blockCutoff_subset_closedShell hε hεK hx
  have hx2 : ‖x‖ ^ 2 ≤ n * (K + ε / 2) := by
    rw [PiLp.norm_sq_eq_of_L2 (fun _ : Idx n => EucSpace d) x]
    calc ∑ i, ‖x.ofLp i‖ ^ 2 ≤ ∑ _i : Idx n, (K + ε / 2) :=
          Finset.sum_le_sum fun i _ => (hx i).2
      _ = n * (K + ε / 2) := by simp [mul_add]
  rw [mem_closedBall_zero_iff]
  nlinarith [norm_nonneg x, sq_nonneg (‖x‖ - 1)]

omit hε hεK in
/-- The hypotheses `0 < ε < K` of the cut-off hold for `ε = 1/2`, `K = 2`. -/
example : (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) < 2 := by norm_num

end Perspective
end Transformer
