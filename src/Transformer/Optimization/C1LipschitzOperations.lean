module

public import Transformer.Optimization.C1Lipschitz
public import Mathlib.Analysis.Complex.RealDeriv

/-! Scalar and bilinear operations on maps with locally Lipschitz
derivatives. The inverse and log rules are local on the positive or
nonzero range of their inputs. -/

@[expose] public section

namespace Transformer.Optimization.C1Lipschitz

variable {X Y : Type*} [NormedAddCommGroup X] [NormedSpace ℝ X]
  [NormedAddCommGroup Y] [NormedSpace ℝ Y]

theorem mul {𝔸 : Type*} [NormedRing 𝔸] [NormedAlgebra ℝ 𝔸]
    {f g : X → 𝔸} (hf : C1Lipschitz f) (hg : C1Lipschitz g) :
    C1Lipschitz (fun x => f x * g x) :=
  (of_contDiff contDiff_mul).comp (hf.prodMk hg)

theorem smul {𝕜 : Type*} [NontriviallyNormedField 𝕜] [NormedAlgebra ℝ 𝕜]
    [NormedSpace 𝕜 Y] [IsScalarTower ℝ 𝕜 Y]
    {f : X → 𝕜} {g : X → Y} (hf : C1Lipschitz f) (hg : C1Lipschitz g) :
    C1Lipschitz (fun x => f x • g x) :=
  (of_contDiff contDiff_smul).comp (hf.prodMk hg)

theorem inner (𝕜 : Type*) [RCLike 𝕜] {E : Type*} [NormedAddCommGroup E] [InnerProductSpace 𝕜 E]
    [NormedSpace ℝ E] [IsScalarTower ℝ 𝕜 E] {f g : X → E}
    (hf : C1Lipschitz f) (hg : C1Lipschitz g) :
    C1Lipschitz (fun x => Inner.inner 𝕜 (f x) (g x)) :=
  (of_contDiff (contDiff_inner (𝕜 := 𝕜))).comp (hf.prodMk hg)

theorem inv {f : X → ℝ} (hf : C1Lipschitz f) (hn : ∀ x, f x ≠ 0) :
    C1Lipschitz (fun x => (f x)⁻¹) :=
  hf.comp_contDiffAt (fun x => contDiffAt_id.inv (hn x))

theorem div {f g : X → ℝ} (hf : C1Lipschitz f) (hg : C1Lipschitz g) (hn : ∀ x, g x ≠ 0) :
    C1Lipschitz (fun x => f x / g x) := by
  simpa only [div_eq_mul_inv] using hf.mul (hg.inv hn)

theorem neg {f : X → Y} (hf : C1Lipschitz f) :
    C1Lipschitz (fun x => -(f x)) := (of_contDiff contDiff_id.neg).comp hf

theorem div_const {f : X → ℝ} (hf : C1Lipschitz f) (c : ℝ) :
    C1Lipschitz (fun x => f x / c) := by
  simpa only [div_eq_mul_inv] using hf.mul (const c⁻¹)

theorem exp {f : X → ℝ} (hf : C1Lipschitz f) :
    C1Lipschitz (fun x => Real.exp (f x)) := (of_contDiff Real.contDiff_exp).comp hf

theorem log {f : X → ℝ} (hf : C1Lipschitz f) (hn : ∀ x, f x ≠ 0) :
    C1Lipschitz (fun x => Real.log (f x)) :=
  hf.comp_contDiffAt (fun x => Real.contDiffAt_log.mpr (hn x))

end Transformer.Optimization.C1Lipschitz
