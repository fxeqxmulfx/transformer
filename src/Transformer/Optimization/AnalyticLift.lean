module

public import Mathlib.Analysis.Analytic.Constructions
public import Mathlib.Analysis.Normed.Module.FiniteDimension
public import Mathlib.Analysis.Calculus.FDeriv.Prod
public import Mathlib.Analysis.Calculus.FDeriv.Analytic
public import Mathlib.Topology.Sequences

/-! Locally analytic extensions along convergent subsequences. At the
lifted points the derivative ignores the auxiliary coordinates. This
allows an analytic gradient inequality to pass down to the original
function even when a lifted auxiliary coordinate has several possible
limits. -/

@[expose] public section

open Filter Function
open scoped Topology

namespace Transformer.Optimization

/-- A finite-dimensional real normed space for auxiliary coordinates. -/
structure AnalyticLiftSpace where
  carrier : Type
  [normedAdd : NormedAddCommGroup carrier]
  [normedSpace : NormedSpace ℝ carrier]
  [finiteDimensional : FiniteDimensional ℝ carrier]

attribute [instance] AnalyticLiftSpace.normedAdd AnalyticLiftSpace.normedSpace
  AnalyticLiftSpace.finiteDimensional

noncomputable def AnalyticLiftSpace.real : AnalyticLiftSpace := ⟨ℝ⟩

noncomputable def AnalyticLiftSpace.prod (U V : AnalyticLiftSpace) : AnalyticLiftSpace := ⟨U.carrier × V.carrier⟩

variable {X Y Z : Type*} [NormedAddCommGroup X] [NormedSpace ℝ X]
  [NormedAddCommGroup Y] [NormedSpace ℝ Y] [NormedAddCommGroup Z] [NormedSpace ℝ Z]

/-- The value and derivative of a function agree with an analytic lift
along a subsequence of each convergent sequence. -/
structure HasAnalyticLift (f : X → Y) : Prop where
  differentiable : Differentiable ℝ f
  lift : ∀ (x : ℕ → X) (z : X), Tendsto x atTop (𝓝 z) →
    ∃ (U : AnalyticLiftSpace) (q : X × U.carrier → Y) (b : ℕ → U.carrier)
      (b₀ : U.carrier) (u : ℕ → ℕ),
      StrictMono u ∧ Tendsto b atTop (𝓝 b₀) ∧ AnalyticAt ℝ q (z, b₀) ∧
      ∀ n, q (x (u n), b n) = f (x (u n)) ∧
        HasFDerivAt q ((fderiv ℝ f (x (u n))).comp (ContinuousLinearMap.fst ℝ X U.carrier))
          (x (u n), b n)

namespace HasAnalyticLift

theorem of_analytic {f : X → Y} (hf : ∀ x, AnalyticAt ℝ f x) : HasAnalyticLift f := by
  have hd : Differentiable ℝ f := fun x => (hf x).differentiableAt
  refine ⟨hd, fun x z _ => ?_⟩
  refine ⟨AnalyticLiftSpace.real, fun p => f p.1, fun _ => 0, 0, id, strictMono_id,
    tendsto_const_nhds, ?_, fun n => ⟨rfl, ?_⟩⟩
  · exact (hf z).comp_of_eq ((ContinuousLinearMap.fst ℝ X ℝ).analyticAt (z, 0)) rfl
  · exact (hd (x n)).hasFDerivAt.comp (x n, (0 : ℝ)) (ContinuousLinearMap.fst ℝ X ℝ).hasFDerivAt

/-- Treating a continuous linear map as an analytic lift introduces a
constant auxiliary coordinate. -/
theorem clm (L : X →L[ℝ] Y) : HasAnalyticLift L :=
  of_analytic L.analyticAt

theorem const (y : Y) : HasAnalyticLift (fun _ : X => y) :=
  of_analytic fun _ => analyticAt_const

theorem id : HasAnalyticLift (id : X → X) :=
  clm (ContinuousLinearMap.id ℝ X)

private theorem lift_value_at_limit {U : AnalyticLiftSpace} {f : X → Y}
    {q : X × U.carrier → Y} {x : ℕ → X} {z : X} {b : ℕ → U.carrier} {b₀ : U.carrier}
    {u : ℕ → ℕ} (hu : StrictMono u) (hx : Tendsto x atTop (𝓝 z)) (hb : Tendsto b atTop (𝓝 b₀))
    (hq : AnalyticAt ℝ q (z, b₀)) (hf : ContinuousAt f z)
    (hval : ∀ n, q (x (u n), b n) = f (x (u n))) :
    q (z, b₀) = f z := by
  have ht := (hx.comp hu.tendsto_atTop).prodMk_nhds hb
  exact tendsto_nhds_unique (hq.continuousAt.tendsto.comp ht)
    (((hf.tendsto.comp (hx.comp hu.tendsto_atTop))).congr fun n => (hval n).symm)

/-- Compose the lift with a function analytic at all image points.
This covers division by a positive sum and the logarithm of a positive
softmax probability without needing an analytic extension at zero. -/
theorem comp_analytic {f : X → Y} {g : Y → Z} (hf : HasAnalyticLift f)
    (hg : ∀ x, AnalyticAt ℝ g (f x)) : HasAnalyticLift (fun x => g (f x)) := by
  refine ⟨fun x => (hg x).differentiableAt.comp x (hf.differentiable x), fun x z hx => ?_⟩
  obtain ⟨U, q, b, b₀, u, hu, hb, hq, hstep⟩ := hf.lift x z hx
  have he := lift_value_at_limit hu hx hb hq (hf.differentiable z).continuousAt
    (fun n => (hstep n).1)
  refine ⟨U, g ∘ q, b, b₀, u, hu, hb, (hg z).comp_of_eq hq he, fun n => ?_⟩
  refine ⟨congrArg g (hstep n).1, ?_⟩
  have hd := (hg (x (u n))).differentiableAt.hasFDerivAt
  rw [← (hstep n).1] at hd
  have hchain := fderiv_fun_comp (x (u n)) (hg (x (u n))).differentiableAt (hf.differentiable _)
  rw [hchain, ContinuousLinearMap.comp_assoc]
  simpa only [(hstep n).1] using hd.comp (x (u n), b n) (hstep n).2

/-- Two lifts fit into a shared auxiliary space after taking a further
subsequence. -/
theorem prodMk {f : X → Y} {g : X → Z} (hf : HasAnalyticLift f)
    (hg : HasAnalyticLift g) : HasAnalyticLift (fun x => (f x, g x)) := by
  refine ⟨hf.differentiable.prodMk hg.differentiable, fun x z hx => ?_⟩
  obtain ⟨U, q, b, b₀, u, hu, hb, hq, hstep⟩ := hf.lift x z hx
  obtain ⟨V, r, c, c₀, v, hv, hc, hr, hnext⟩ := hg.lift (x ∘ u) z (hx.comp hu.tendsto_atTop)
  let W := U.prod V
  let α : X × W.carrier →L[ℝ] X × U.carrier :=
    (ContinuousLinearMap.fst ℝ X W.carrier).prod ((ContinuousLinearMap.fst ℝ U.carrier V.carrier).comp
      (ContinuousLinearMap.snd ℝ X W.carrier))
  let β : X × W.carrier →L[ℝ] X × V.carrier :=
    (ContinuousLinearMap.fst ℝ X W.carrier).prod ((ContinuousLinearMap.snd ℝ U.carrier V.carrier).comp
      (ContinuousLinearMap.snd ℝ X W.carrier))
  refine ⟨W, (fun p => (q (α p), r (β p))), (fun n => (b (v n), c n)), (b₀, c₀), u ∘ v,
    hu.comp hv, (hb.comp hv.tendsto_atTop).prodMk_nhds hc, ?_, fun n => ?_⟩
  · have hα := hq.comp_of_eq (α.analyticAt (z, (b₀, c₀))) (show α (z, (b₀, c₀)) = (z, b₀) from rfl)
    have hβ := hr.comp_of_eq (β.analyticAt (z, (b₀, c₀))) (show β (z, (b₀, c₀)) = (z, c₀) from rfl)
    exact hα.prod hβ
  refine ⟨Prod.ext (hstep (v n)).1 (hnext n).1, ?_⟩
  have h₁ := (hstep (v n)).2.comp (x (u (v n)), (b (v n), c n)) α.hasFDerivAt
  have h₂ := (hnext n).2.comp (x (u (v n)), (b (v n), c n)) β.hasFDerivAt
  have hF := h₁.prodMk h₂
  rw [(hf.differentiable _).fderiv_prodMk (hg.differentiable _)]
  have hEq : ((fderiv ℝ f (x (u (v n)))).prod (fderiv ℝ g (x (u (v n))))).comp
      (ContinuousLinearMap.fst ℝ X W.carrier) =
      (((fderiv ℝ f (x (u (v n)))).comp (ContinuousLinearMap.fst ℝ X U.carrier)).comp α).prod
      (((fderiv ℝ g (x (u (v n)))).comp (ContinuousLinearMap.fst ℝ X V.carrier)).comp β) := by
    ext p <;> rfl
  simp only [Function.comp_apply]
  rw [hEq]
  exact hF

end HasAnalyticLift

end Transformer.Optimization
