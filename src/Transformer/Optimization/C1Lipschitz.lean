module

public import Mathlib.Analysis.Calculus.ContDiff.RCLike
public import Mathlib.Analysis.Calculus.ContDiff.WithLp
public import Mathlib.Analysis.Calculus.FDeriv.CompCLM
public import Mathlib.Analysis.InnerProductSpace.Calculus
public import Mathlib.Analysis.Normed.Operator.Prod
public import Mathlib.Analysis.SpecialFunctions.Log.Deriv

/-! First derivatives that are locally Lipschitz. The composition rules
also apply to maps that are only once differentiable at zero, including
the RMS normalization with the offset outside the norm. -/

@[expose] public section

open Filter Set
open scoped Topology NNReal

namespace Transformer.Optimization

variable {X Y Z : Type*} [NormedAddCommGroup X] [NormedSpace ℝ X]
  [NormedAddCommGroup Y] [NormedSpace ℝ Y] [NormedAddCommGroup Z] [NormedSpace ℝ Z]

structure C1Lipschitz (f : X → Y) : Prop where
  contDiff : ContDiff ℝ 1 f
  fderiv_lipschitz : LocallyLipschitz (fderiv ℝ f)

namespace C1Lipschitz

theorem of_contDiff {f : X → Y} (hf : ContDiff ℝ 2 f) : C1Lipschitz f :=
  ⟨hf.of_le (by norm_num), (hf.fderiv_right (m := 1) (by norm_num)).locallyLipschitz⟩

set_option maxHeartbeats 3000000 in
theorem comp_at {f : X → Y} {g : Y → Z} (hf : C1Lipschitz f)
    (hg : ∀ x, ContDiffAt ℝ 1 g (f x))
    (hL : ∀ x, ∃ K, ∃ s ∈ 𝓝 (f x), LipschitzOnWith K (fderiv ℝ g) s) :
    C1Lipschitz (fun x => g (f x)) := by
  refine ⟨contDiff_iff_contDiffAt.mpr fun x => (hg x).comp x hf.contDiff.contDiffAt, ?_⟩
  have hF (x : X) : fderiv ℝ (fun x => g (f x)) x =
      (fderiv ℝ g (f x)).comp (fderiv ℝ f x) :=
    fderiv_comp x ((hg x).differentiableAt one_ne_zero)
      (hf.contDiff.differentiable one_ne_zero x)
  have hF' : fderiv ℝ (fun x => g (f x)) =
      (fun x => (fderiv ℝ g (f x)).comp (fderiv ℝ f x)) := funext hF
  rw [hF']
  have hgL : LocallyLipschitz (fun x => fderiv ℝ g (f x)) := fun x => by
    obtain ⟨K, s, hs, hKs⟩ := hL x
    obtain ⟨L, t, ht, hLt⟩ := hf.contDiff.locallyLipschitz x
    refine ⟨K * L, t ∩ f ⁻¹' s, inter_mem ht (hf.contDiff.continuous.continuousAt hs), ?_⟩
    exact hKs.comp (hLt.mono inter_subset_left) inter_subset_right
  have hbil : LocallyLipschitz (fun q : (Y →L[ℝ] Z) × (X →L[ℝ] Y) => q.1.comp q.2) :=
    (contDiff_fst.clm_comp contDiff_snd : ContDiff ℝ 1 _).locallyLipschitz
  have H : LocallyLipschitz (fun x => (fderiv ℝ g (f x), fderiv ℝ f x)) :=
    hgL.prodMk hf.fderiv_lipschitz
  exact LocallyLipschitz.comp (f := fun q : (Y →L[ℝ] Z) × (X →L[ℝ] Y) => q.1.comp q.2)
    (g := fun x => (fderiv ℝ g (f x), fderiv ℝ f x)) hbil H

theorem comp {f : X → Y} {g : Y → Z} (hg : C1Lipschitz g) (hf : C1Lipschitz f) :
    C1Lipschitz (fun x => g (f x)) :=
  hf.comp_at (fun _ => hg.contDiff.contDiffAt) (fun x => hg.fderiv_lipschitz (f x))

theorem comp_contDiffAt {f : X → Y} {g : Y → Z} (hf : C1Lipschitz f)
    (hg : ∀ x, ContDiffAt ℝ 2 g (f x)) : C1Lipschitz (fun x => g (f x)) :=
  hf.comp_at (fun x => (hg x).of_le (by norm_num)) fun x =>
    ((hg x).fderiv_right (m := 1) (by norm_num)).exists_lipschitzOnWith

set_option maxHeartbeats 3000000 in
theorem prodMk {f : X → Y} {g : X → Z} (hf : C1Lipschitz f) (hg : C1Lipschitz g) :
    C1Lipschitz (fun x => (f x, g x)) := by
  refine ⟨hf.contDiff.prodMk hg.contDiff, ?_⟩
  have hF (x : X) : fderiv ℝ (fun x => (f x, g x)) x =
      (fderiv ℝ f x).prod (fderiv ℝ g x) :=
    (hf.contDiff.differentiable one_ne_zero x).fderiv_prodMk (hg.contDiff.differentiable one_ne_zero x)
  have hF' : fderiv ℝ (fun x => (f x, g x)) =
      (fun x => (fderiv ℝ f x).prod (fderiv ℝ g x)) := funext hF
  rw [hF']
  let p (a : (X →L[ℝ] Y) × (X →L[ℝ] Z)) : X →L[ℝ] (Y × Z) := a.1.prod a.2
  have hP : LipschitzWith 1 p := by
    rw [lipschitzWith_iff_dist_le_mul]
    intro a b
    simp only [dist_eq_norm, NNReal.coe_one, one_mul]
    have he : p a - p b = p (a - b) := by ext x <;> rfl
    rw [he]
    exact (ContinuousLinearMap.opNorm_prod _ _).le
  have H : LocallyLipschitz (fun x => (fderiv ℝ f x, fderiv ℝ g x)) :=
    hf.fderiv_lipschitz.prodMk hg.fderiv_lipschitz
  exact LocallyLipschitz.comp (f := p) (g := fun x => (fderiv ℝ f x, fderiv ℝ g x))
    hP.locallyLipschitz H

theorem pi {ι : Type*} [Fintype ι] {M : ι → Type*} [∀ i, NormedAddCommGroup (M i)]
    [∀ i, NormedSpace ℝ (M i)] {f : X → (i : ι) → M i} (h : ∀ i, C1Lipschitz (fun x => f x i)) :
    C1Lipschitz f := by
  classical
  refine ⟨contDiff_pi.mpr fun i => (h i).contDiff, fun x => ?_⟩
  choose K s hs hL using fun i => (h i).fderiv_lipschitz x
  refine ⟨∑ i, K i, ⋂ i, s i, Filter.iInter_mem.mpr hs, ?_⟩
  rw [lipschitzOnWith_iff_dist_le_mul]
  intro u hu v hv
  have hF (z : X) : fderiv ℝ f z =
      ContinuousLinearMap.pi (fun i => fderiv ℝ (fun x => f x i) z) :=
    fderiv_pi fun i => (h i).contDiff.differentiable one_ne_zero z
  rw [dist_eq_norm, hF u, hF v, dist_eq_norm]
  have hsub :
      (ContinuousLinearMap.pi (fun i => fderiv ℝ (fun x => f x i) u) -
        ContinuousLinearMap.pi (fun i => fderiv ℝ (fun x => f x i) v)) =
      ContinuousLinearMap.pi (fun i => fderiv ℝ (fun x => f x i) u -
        fderiv ℝ (fun x => f x i) v) := by ext w i; rfl
  rw [hsub]
  refine ContinuousLinearMap.norm_pi_le_of_le (fun i => ?_) (by positivity)
  have hi : (K i : ℝ) ≤ ((∑ j, K j : ℝ≥0) : ℝ) := by
    exact_mod_cast (Finset.single_le_sum (fun j _ => (by positivity : 0 ≤ K j)) (Finset.mem_univ i))
  exact ((hL i).norm_sub_le (mem_iInter.mp hu i) (mem_iInter.mp hv i)).trans
    (mul_le_mul_of_nonneg_right hi (norm_nonneg _))

theorem proj {ι : Type*} [Fintype ι] {M : ι → Type*} [∀ i, NormedAddCommGroup (M i)]
    [∀ i, NormedSpace ℝ (M i)] {f : X → (i : ι) → M i} (h : C1Lipschitz f) (i : ι) :
    C1Lipschitz (fun x => f x i) :=
  (of_contDiff (contDiff_apply ℝ _ i)).comp h

theorem piLp {ι : Type*} [Fintype ι] {M : ι → Type*} [∀ i, NormedAddCommGroup (M i)]
    [∀ i, NormedSpace ℝ (M i)] {f : X → PiLp 2 M} (h : ∀ i, C1Lipschitz (fun x => f x i)) :
    C1Lipschitz f :=
  (of_contDiff (PiLp.continuousLinearEquiv 2 ℝ M).symm.contDiff).comp (pi h)

theorem projLp {ι : Type*} [Fintype ι] {M : ι → Type*} [∀ i, NormedAddCommGroup (M i)]
    [∀ i, NormedSpace ℝ (M i)] {f : X → PiLp 2 M} (h : C1Lipschitz f) (i : ι) :
    C1Lipschitz (fun x => f x i) :=
  ((of_contDiff (PiLp.continuousLinearEquiv 2 ℝ M).contDiff).comp h).proj i

theorem const (y : Y) : C1Lipschitz (fun _ : X => y) := of_contDiff contDiff_const

theorem add {f g : X → Y} (hf : C1Lipschitz f) (hg : C1Lipschitz g) :
    C1Lipschitz (fun x => f x + g x) := (of_contDiff contDiff_add).comp (hf.prodMk hg)

theorem sum {ι : Type*} {s : Finset ι} {f : ι → X → Y}
    (hf : ∀ i ∈ s, C1Lipschitz (f i)) : C1Lipschitz (fun x => ∑ i ∈ s, f i x) := by
  classical
  induction s using Finset.induction_on with
  | empty => simpa using (const (X := X) (0 : Y))
  | insert i s hi ih =>
    simp only [Finset.sum_insert hi]
    exact (hf i (Finset.mem_insert_self _ _)).add (ih fun j hj => hf j (Finset.mem_insert_of_mem hj))

end C1Lipschitz

end Transformer.Optimization
