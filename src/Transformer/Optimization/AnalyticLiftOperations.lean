module

public import Transformer.Optimization.AnalyticLift
public import Mathlib.Analysis.Calculus.ContDiff.WithLp
public import Mathlib.Analysis.InnerProductSpace.Calculus
public import Mathlib.Analysis.SpecialFunctions.Complex.Analytic
public import Mathlib.Analysis.SpecialFunctions.ExpDeriv

/-! Composition and finite products of analytic lifts. -/

@[expose] public section

open Filter Function
open scoped Topology

namespace Transformer.Optimization.HasAnalyticLift

variable {X Y Z : Type*} [NormedAddCommGroup X] [NormedSpace ℝ X]
  [NormedAddCommGroup Y] [NormedSpace ℝ Y] [NormedAddCommGroup Z] [NormedSpace ℝ Z]

theorem comp {f : X → Y} {g : Y → Z} (hg : HasAnalyticLift g)
    (hf : HasAnalyticLift f) : HasAnalyticLift (fun x => g (f x)) := by
  refine ⟨hg.differentiable.comp hf.differentiable, fun x z hx => ?_⟩
  obtain ⟨U, q, b, b₀, u, hu, hb, hq, hstep⟩ := hf.lift x z hx
  let y (n : ℕ) := f (x (u n))
  have hy : Tendsto y atTop (𝓝 (f z)) :=
    (hf.differentiable z).continuousAt.tendsto.comp (hx.comp hu.tendsto_atTop)
  obtain ⟨V, r, c, c₀, v, hv, hc, hr, hnext⟩ := hg.lift y (f z) hy
  let W := U.prod V
  let α : X × W.carrier →L[ℝ] X × U.carrier :=
    (ContinuousLinearMap.fst ℝ X W.carrier).prod ((ContinuousLinearMap.fst ℝ U.carrier V.carrier).comp
      (ContinuousLinearMap.snd ℝ X W.carrier))
  let β : X × W.carrier →L[ℝ] V.carrier :=
    (ContinuousLinearMap.snd ℝ U.carrier V.carrier).comp (ContinuousLinearMap.snd ℝ X W.carrier)
  let Q (p : X × W.carrier) := (q (α p), β p)
  let p₀ : X × W.carrier := (z, (b₀, c₀))
  have hα : α p₀ = (z, b₀) := rfl
  have he : q (z, b₀) = f z := by
    have hpair := (hx.comp hu.tendsto_atTop).prodMk_nhds hb
    exact tendsto_nhds_unique (hq.continuousAt.tendsto.comp hpair)
      (hy.congr fun n => (hstep n).1.symm)
  have hQ : AnalyticAt ℝ Q p₀ := (hq.comp_of_eq (α.analyticAt p₀) hα).prod (β.analyticAt p₀)
  have hQ₀ : Q p₀ = (f z, c₀) := Prod.ext he rfl
  refine ⟨W, r ∘ Q, (fun n => (b (v n), c n)), (b₀, c₀), u ∘ v, hu.comp hv,
    (hb.comp hv.tendsto_atTop).prodMk_nhds hc, hr.comp_of_eq hQ hQ₀, fun n => ?_⟩
  let p : X × W.carrier := (x (u (v n)), (b (v n), c n))
  have hp : Q p = (y (v n), c n) := Prod.ext (hstep (v n)).1 rfl
  refine ⟨(congrArg r hp).trans (hnext n).1, ?_⟩
  have hd₁ := (hstep (v n)).2.comp p α.hasFDerivAt
  have hd₂ := hd₁.prodMk (β.hasFDerivAt (x := p))
  have hd₃ := (hnext n).2
  rw [← hp] at hd₃
  have hd := hd₃.comp p hd₂
  simp only [Function.comp_apply]
  rw [fderiv_fun_comp (x (u (v n))) (hg.differentiable _) (hf.differentiable _)]
  have heq :
      ((fderiv ℝ g (y (v n))).comp (ContinuousLinearMap.fst ℝ Y V.carrier)).comp
        ((((fderiv ℝ f (x (u (v n)))).comp (ContinuousLinearMap.fst ℝ X U.carrier)).comp α).prod β) =
      ((fderiv ℝ g (y (v n))).comp (fderiv ℝ f (x (u (v n))))).comp
        (ContinuousLinearMap.fst ℝ X W.carrier) := by
    apply ContinuousLinearMap.ext
    intro t
    rfl
  rwa [heq] at hd

theorem add {f g : X → Y} (hf : HasAnalyticLift f) (hg : HasAnalyticLift g) :
    HasAnalyticLift (fun x => f x + g x) :=
  (hf.prodMk hg).comp_analytic (fun _ => analyticAt_fst.add analyticAt_snd)

theorem neg {f : X → Y} (hf : HasAnalyticLift f) : HasAnalyticLift (fun x => -f x) :=
  hf.comp_analytic fun _ => analyticAt_id.neg

theorem sub {f g : X → Y} (hf : HasAnalyticLift f) (hg : HasAnalyticLift g) :
    HasAnalyticLift (fun x => f x - g x) := by
  simpa only [sub_eq_add_neg] using hf.add hg.neg

theorem mul {B : Type*} [NormedRing B] [NormedAlgebra ℝ B]
    {f g : X → B} (hf : HasAnalyticLift f) (hg : HasAnalyticLift g) :
    HasAnalyticLift (fun x => f x * g x) :=
  (hf.prodMk hg).comp_analytic (fun _ => analyticAt_fst.mul analyticAt_snd)

theorem smul {𝕜 : Type*} [NontriviallyNormedField 𝕜] [NormedAlgebra ℝ 𝕜]
    [NormedSpace 𝕜 Y] [IsScalarTower ℝ 𝕜 Y] {f : X → 𝕜} {g : X → Y}
    (hf : HasAnalyticLift f) (hg : HasAnalyticLift g) :
    HasAnalyticLift (fun x => f x • g x) :=
  (hf.prodMk hg).comp_analytic (fun _ => analyticAt_fst.smul analyticAt_snd)

theorem sum {ι : Type*} {s : Finset ι} {f : ι → X → Y}
    (h : ∀ i ∈ s, HasAnalyticLift (f i)) : HasAnalyticLift (fun x => ∑ i ∈ s, f i x) := by
  classical
  induction s using Finset.induction_on with
  | empty => simpa using (const (X := X) (0 : Y))
  | insert i s hi ih =>
    simp only [Finset.sum_insert hi]
    exact (h i (Finset.mem_insert_self _ _)).add (ih fun j hj => h j (Finset.mem_insert_of_mem hj))

theorem pi {ι : Type*} [Fintype ι] {f : X → ι → Y}
    (h : ∀ i, HasAnalyticLift (fun x => f x i)) : HasAnalyticLift f := by
  classical
  let L (i : ι) : Y →L[ℝ] (ι → Y) :=
    ContinuousLinearMap.pi fun j => if j = i then ContinuousLinearMap.id ℝ Y else 0
  have hL (i : ι) : HasAnalyticLift (fun x => L i (f x i)) :=
    (h i).comp_analytic (fun _ => (L i).analyticAt _)
  have he : (fun x => ∑ i, L i (f x i)) = f := by
    funext x j
    simp only [Finset.sum_apply, L, ContinuousLinearMap.pi_apply]
    rw [Finset.sum_eq_single j]
    · simp
    · intro b _ hb
      simp [Ne.symm hb]
    · simp
  rw [← he]
  exact sum fun i _ => hL i

theorem proj {ι : Type*} [Fintype ι] {f : X → ι → Y} (h : HasAnalyticLift f) (i : ι) :
    HasAnalyticLift (fun x => f x i) :=
  h.comp_analytic fun _ => (ContinuousLinearMap.proj (R := ℝ) (φ := fun _ : ι => Y) i).analyticAt _

theorem piLp {ι : Type*} [Fintype ι] {f : X → PiLp 2 (fun _ : ι => Y)}
    (h : ∀ i, HasAnalyticLift (fun x => f x i)) : HasAnalyticLift f :=
  (pi h).comp_analytic fun _ => (PiLp.continuousLinearEquiv 2 ℝ (fun _ : ι => Y)).symm.analyticAt _

theorem projLp {ι : Type*} [Fintype ι] {f : X → PiLp 2 (fun _ : ι => Y)}
    (h : HasAnalyticLift f) (i : ι) : HasAnalyticLift (fun x => f x i) :=
  (h.comp_analytic (fun _ => (PiLp.continuousLinearEquiv 2 ℝ (fun _ : ι => Y)).analyticAt _)).proj i

/-- The inner product is real-bilinear in both arguments. -/
theorem inner (𝕜 : Type*) [RCLike 𝕜] {E : Type*} [NormedAddCommGroup E] [InnerProductSpace 𝕜 E]
    [NormedSpace ℝ E] [IsScalarTower ℝ 𝕜 E] {f g : X → E}
    (hf : HasAnalyticLift f) (hg : HasAnalyticLift g) :
    HasAnalyticLift (fun x => Inner.inner 𝕜 (f x) (g x)) :=
  (hf.prodMk hg).comp_analytic fun x =>
    (isBoundedBilinearMap_inner (𝕜 := 𝕜) (E := E)).toContinuousLinearMap.analyticAt_bilinear (f x, g x)

theorem div {f g : X → ℝ} (hf : HasAnalyticLift f) (hg : HasAnalyticLift g) (hn : ∀ x, g x ≠ 0) :
    HasAnalyticLift (fun x => f x / g x) :=
  (hf.prodMk hg).comp_analytic (fun x => analyticAt_fst.div analyticAt_snd (hn x))

theorem div_const {f : X → ℝ} (hf : HasAnalyticLift f) (c : ℝ) :
    HasAnalyticLift (fun x => f x / c) := by simpa only [div_eq_mul_inv] using hf.mul (const c⁻¹)

theorem exp {f : X → ℝ} (hf : HasAnalyticLift f) :
    HasAnalyticLift (fun x => Real.exp (f x)) :=
  hf.comp_analytic fun _ => analyticAt_rexp

theorem log {f : X → ℝ} (hf : HasAnalyticLift f) (hn : ∀ x, 0 < f x) :
    HasAnalyticLift (fun x => Real.log (f x)) :=
  hf.comp_analytic fun x => analyticAt_id.log (hn x)

end Transformer.Optimization.HasAnalyticLift
