module

public import Transformer.RoPE.FrequencyMatching.Section3_Optimality
public import Transformer.RoPE.Scores
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Inverse

/-!
# Endpoint guarantees and a retention-constrained optimum

Derived results using the contrast/field model of Wu et al.,
arXiv:2607.07678v1, §3.2–3.3. These answer the worst-case endpoint question:
cosine retention is bounded below by its endpoint value throughout a field,
whereas positional contrast is bounded above by its endpoint value.

For a prescribed retention floor `rho`, the frequency `arccos(rho)/width`
maximizes contrast simultaneously at every distance subject to that floor
and half-period admissibility. This is a scalar two-criterion result;
arbitrary content-dependent rotary scores need not have their minimum at
the far endpoint, as the explicit counterexample shows.
-/

@[expose] public section

open Set

namespace Transformer.RoPE.FrequencyMatching

/-- Within a monotone field, cosine retention at every nearer distance is at
least its endpoint value. The contrast inequality has the opposite direction. -/
theorem endpoint_retention_bound {width omega distance : ℝ}
    (homega : Admissible width omega) (hr : distance ∈ Icc 0 width) :
    Real.cos (omega * width) ≤ Real.cos (omega * distance) ∧
      contrast omega distance ≤ contrast omega width := by
  have hnonneg : 0 ≤ width := hr.1.trans hr.2
  have hnear : omega * distance ∈ Icc 0 Real.pi :=
    ⟨mul_nonneg homega.1.le hr.1,
      (mul_le_mul_of_nonneg_left hr.2 homega.1.le).trans homega.2⟩
  have hfar : omega * width ∈ Icc 0 Real.pi :=
    ⟨mul_nonneg homega.1.le hnonneg, homega.2⟩
  have h := Real.strictAntiOn_cos.antitoneOn hnear hfar
    (mul_le_mul_of_nonneg_left hr.2 homega.1.le)
  exact ⟨h, by unfold contrast; linarith⟩

example : Real.cos (Real.pi * 1) ≤ Real.cos (Real.pi * (1 / 2)) ∧
    contrast Real.pi (1 / 2) ≤ contrast Real.pi 1 :=
  endpoint_retention_bound ⟨Real.pi_pos, by simp⟩ ⟨by norm_num, by norm_num⟩

/-- The two criteria move strictly in opposite directions at distinct positive
distances within an admissible field. Endpoint access cannot order both alike. -/
theorem endpoint_metric_conflict {width omega distance : ℝ}
    (homega : Admissible width omega) (hr : distance ∈ Ioo 0 width) :
    Real.cos (omega * width) < Real.cos (omega * distance) ∧
      contrast omega distance < contrast omega width := by
  have hnear : omega * distance ∈ Icc 0 Real.pi :=
    ⟨mul_nonneg homega.1.le hr.1.le,
      (mul_le_mul_of_nonneg_left hr.2.le homega.1.le).trans homega.2⟩
  have hfar : omega * width ∈ Icc 0 Real.pi :=
    ⟨mul_nonneg homega.1.le (hr.1.le.trans hr.2.le), homega.2⟩
  have h := Real.strictAntiOn_cos hnear hfar (mul_lt_mul_of_pos_left hr.2 homega.1)
  exact ⟨h, by unfold contrast; linarith⟩

example : Real.cos (Real.pi * 1) < Real.cos (Real.pi * (1 / 2)) ∧
    contrast Real.pi (1 / 2) < contrast Real.pi 1 :=
  endpoint_metric_conflict ⟨Real.pi_pos, by simp⟩ ⟨by norm_num, by norm_num⟩

/-- Requiring endpoint retention `rho` is exactly an upper bound on frequency
inside the monotone field. This gives an explicit design constraint. -/
theorem retention_cap_iff {width rho omega : ℝ} (hwidth : 0 < width)
    (hrho : rho ∈ Ioo (-1) 1) (homega : Admissible width omega) :
    rho ≤ Real.cos (omega * width) ↔ omega ≤ Real.arccos rho / width := by
  have hphase : omega * width ∈ Icc 0 Real.pi :=
    ⟨mul_nonneg homega.1.le hwidth.le, homega.2⟩
  have hacos : Real.arccos rho ∈ Icc 0 Real.pi :=
    ⟨Real.arccos_nonneg rho, Real.arccos_le_pi rho⟩
  have heq := Real.cos_arccos hrho.1.le hrho.2.le
  constructor
  · intro h
    by_contra hnot
    have hlt : Real.arccos rho < omega * width :=
      (div_lt_iff₀ hwidth).1 (lt_of_not_ge hnot)
    have hcos := Real.strictAntiOn_cos hacos hphase hlt
    rw [heq] at hcos
    linarith
  · intro h
    have hle : omega * width ≤ Real.arccos rho := (le_div_iff₀ hwidth).1 h
    have hcos := Real.strictAntiOn_cos.antitoneOn hphase hacos hle
    simpa only [heq] using hcos

example : (0 : ℝ) ≤ Real.cos (Real.pi * (1 / 2) * 1) ↔
    Real.pi * (1 / 2) ≤ Real.arccos 0 / 1 :=
  retention_cap_iff (by norm_num) ⟨by norm_num, by norm_num⟩
    ⟨by positivity, by nlinarith [Real.pi_pos]⟩

/-- The largest frequency respecting a fixed cosine-retention floor maximizes
contrast at every distance in the field simultaneously. This is the precise
scalar guarantee available without knowing the dependency distribution. -/
theorem retention_constrained_optimum {width rho : ℝ} (hwidth : 0 < width)
    (hrho : rho ∈ Ioo (-1) 1) :
    Admissible width (Real.arccos rho / width) ∧
      Real.cos ((Real.arccos rho / width) * width) = rho ∧
      (∀ distance ∈ Icc 0 width,
        rho ≤ Real.cos ((Real.arccos rho / width) * distance)) ∧
      ∀ omega, Admissible width omega → rho ≤ Real.cos (omega * width) →
        ∀ distance ∈ Icc 0 width,
          contrast omega distance ≤ contrast (Real.arccos rho / width) distance := by
  have hphase : (Real.arccos rho / width) * width = Real.arccos rho :=
    div_mul_cancel₀ _ hwidth.ne'
  have hbest : Admissible width (Real.arccos rho / width) :=
    ⟨div_pos (Real.arccos_pos.mpr hrho.2) hwidth,
      by rw [hphase]; exact Real.arccos_le_pi rho⟩
  have hretain : Real.cos ((Real.arccos rho / width) * width) = rho := by
    rw [hphase, Real.cos_arccos hrho.1.le hrho.2.le]
  refine ⟨hbest, hretain, ?_, ?_⟩
  · intro distance hr
    calc
      rho = Real.cos ((Real.arccos rho / width) * width) := hretain.symm
      _ ≤ Real.cos ((Real.arccos rho / width) * distance) :=
        (endpoint_retention_bound hbest hr).1
  · intro omega homega hfloor distance hr
    exact contrast_mono homega hbest ((retention_cap_iff hwidth hrho homega).1 hfloor) hr

example : ∃ omega, Admissible 1 omega ∧ Real.cos (omega * 1) = 0 := by
  have h := retention_constrained_optimum (by norm_num : (0 : ℝ) < 1)
    (show (0 : ℝ) ∈ Ioo (-1) 1 by constructor <;> norm_num)
  exact ⟨_, h.1, h.2.1⟩

/-- The same optimizer is unique as soon as contrast is evaluated at any
positive in-field distance, rather than only at distance zero. -/
theorem retention_optimum_unique {width rho omega distance : ℝ}
    (hwidth : 0 < width) (hrho : rho ∈ Ioo (-1) 1)
    (homega : Admissible width omega) (hfloor : rho ≤ Real.cos (omega * width))
    (hr : distance ∈ Ioc 0 width) :
    contrast omega distance = contrast (Real.arccos rho / width) distance ↔
      omega = Real.arccos rho / width := by
  have hbest := (retention_constrained_optimum hwidth hrho).1
  have hle := (retention_cap_iff hwidth hrho homega).1 hfloor
  constructor
  · intro heq
    by_contra hne
    exact (ne_of_lt (contrast_strict homega hbest (lt_of_le_of_ne hle hne) hr)) heq
  · rintro rfl
    rfl

example : contrast (Real.pi / 2) 1 = contrast (Real.arccos 0 / 1) 1 ↔
    Real.pi / 2 = Real.arccos 0 / 1 :=
  retention_optimum_unique (by norm_num) ⟨by norm_num, by norm_num⟩
    ⟨by positivity, by nlinarith [Real.pi_pos]⟩ (by norm_num) ⟨by norm_num, le_rfl⟩

/-- A real, realizable content-dependent rotary score can be smaller at a
nearer token than at the endpoint. The cosine-retention guarantee therefore
cannot be asserted for arbitrary attention coefficients. -/
theorem endpoint_need_not_minimize_score :
    Admissible 2 (Real.pi / 2) ∧
      score (-1) 0 (Real.pi / 2) 1 < score (-1) 0 (Real.pi / 2) 2 := by
  constructor
  · exact ⟨by positivity, by simp⟩
  · norm_num [score]

end Transformer.RoPE.FrequencyMatching
