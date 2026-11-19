module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Identification.Alternative.Extension

/-!
# Finite certificates for every forced unit-margin consequence

Source: `review/convex/moments.md`, CM87. Homogenize the observed inequalities
with one extra coordinate and extend the forced objective positively. The
extension's finite coordinate weights are the nonnegative certificate. The
extra coordinate proves the total-weight lower bound. This supplies both
directions without assuming a cone-closure or linear-program oracle.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Identification.Alternative

open scoped BigOperators

variable {P I : Type*} [AddCommGroup P] [Module ℝ P] [Fintype I]

/-- Homogeneous observed margins plus their common scaling coordinate,
CM87. The extra `none` coordinate retains the scale's sign. -/
def observations (A : I → P →ₗ[ℝ] ℝ) : (P × ℝ) →ₗ[ℝ] (Option I → ℝ) where
  toFun z j := match j with | none => z.2 | some i => A i z.1 - z.2
  map_add' x y := by
    funext j
    cases j <;> simp [map_add]; ring
  map_smul' r x := by
    funext j
    cases j <;> simp [map_smul, mul_sub]

/-- The homogeneous requested lower bound, CM87. -/
def objective (f : P →ₗ[ℝ] ℝ) (bound : ℝ) : (P × ℝ) →ₗ[ℝ] ℝ :=
  f.comp (LinearMap.fst ℝ P ℝ) - bound • LinearMap.snd ℝ P ℝ

/-- Every lower bound forced by a nonempty observed unit-margin system
has a finite nonnegative certificate with sufficient total mass, CM87. -/
theorem unit_margin_certificate (A : I → P →ₗ[ℝ] ℝ) (f : P →ₗ[ℝ] ℝ) (bound : ℝ)
    (p₀ : P) (h₀ : ∀ i, 1 ≤ A i p₀)
    (hforced : ∀ p, (∀ i, 1 ≤ A i p) → bound ≤ f p) :
    ∃ w : I → ℝ, (∀ i, 0 ≤ w i) ∧ bound ≤ ∑ i, w i ∧
      ∀ p, f p = ∑ i, w i * A i p := by
  classical
  have hpos : ∀ z, 0 ≤ observations A z → 0 ≤ objective f bound z := by
    intro z hz
    apply homogeneous_nonneg A f bound p₀ h₀ hforced z.1 z.2 (hz none)
    intro i
    have h := hz (some i)
    change 0 ≤ A i z.1 - z.2 at h
    linarith
  have hstrict : ∀ j, 1 ≤ observations A ((2 : ℝ) • p₀, 1) j := by
    intro j
    cases j with
    | none => exact le_rfl
    | some i =>
      change 1 ≤ A i ((2 : ℝ) • p₀) - 1
      rw [map_smul, smul_eq_mul]
      linarith [h₀ i]
  obtain ⟨g, hg, hgpos⟩ := positive_extension (observations A) (objective f bound)
    hpos ((2 : ℝ) • p₀, 1) hstrict
  let w : I → ℝ := fun i => g (Pi.single (some i) 1)
  have hmass := hg (0, 1)
  rw [coordinate_expansion] at hmass
  simp [observations, objective, Fintype.sum_option, Finset.sum_neg_distrib] at hmass
  refine ⟨w, fun i => positive_coordinate g hgpos (some i), ?_, ?_⟩
  · have hnone := positive_coordinate g hgpos none
    dsimp [w]
    linarith
  · intro p
    have h := hg (p, 0)
    rw [coordinate_expansion] at h
    simpa [observations, objective, Fintype.sum_option, w, mul_comm] using h.symm

/-- A complete nonnegative certificate transfers every observed unit
margin to the requested lower bound, CM87. -/
theorem unit_margin_sound (A : I → P →ₗ[ℝ] ℝ) (f : P →ₗ[ℝ] ℝ) (bound : ℝ)
    (w : I → ℝ) (hw : ∀ i, 0 ≤ w i) (hmass : bound ≤ ∑ i, w i)
    (heq : ∀ p, f p = ∑ i, w i * A i p) (p : P) (hp : ∀ i, 1 ≤ A i p) :
    bound ≤ f p := by
  rw [heq]
  apply hmass.trans
  apply Finset.sum_le_sum
  intro i hi
  simpa only [mul_one] using mul_le_mul_of_nonneg_left (hp i) (hw i)

/-- Exact finite unit-margin alternative, CM87. The consistency premise
is the only hypothesis beyond linearity and finitely many observations. -/
theorem unit_margin_duality (A : I → P →ₗ[ℝ] ℝ) (f : P →ₗ[ℝ] ℝ) (bound : ℝ)
    (hfeasible : ∃ p₀, ∀ i, 1 ≤ A i p₀) :
    (∀ p, (∀ i, 1 ≤ A i p) → bound ≤ f p) ↔
      ∃ w : I → ℝ, (∀ i, 0 ≤ w i) ∧ bound ≤ ∑ i, w i ∧
        ∀ p, f p = ∑ i, w i * A i p := by
  constructor
  · intro h
    obtain ⟨p₀, h₀⟩ := hfeasible
    exact unit_margin_certificate A f bound p₀ h₀ h
  · rintro ⟨w, hw, hmass, heq⟩ p hp
    exact unit_margin_sound A f bound w hw hmass heq p hp

example : ∃ w : Unit → ℝ, (∀ i, 0 ≤ w i) ∧ 1 ≤ ∑ i, w i ∧
    ∀ p : ℝ, p = ∑ i, w i * p :=
  unit_margin_certificate (fun _ => LinearMap.id) LinearMap.id 1 1
    (by simp) (by intro p hp; exact hp ())

example : (1 : ℝ) ≤ 2 :=
  unit_margin_sound (fun _ : Unit => LinearMap.id) LinearMap.id 1 (fun _ => 1)
    (by simp) (by simp) (by simp) 2 (by norm_num)

example := unit_margin_duality (fun _ : Unit => (LinearMap.id : ℝ →ₗ[ℝ] ℝ))
  LinearMap.id 1 ⟨1, by simp⟩

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear.Identification.Alternative
