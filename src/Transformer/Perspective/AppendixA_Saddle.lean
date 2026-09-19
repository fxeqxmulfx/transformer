/-
# Appendix A — Proof of Theorem (p:beta0), part 2

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

This file formalizes the second half of Appendix A of the survey:

* `Lemma lem: yury.lemma` — every non-trivial critical point of `𝖤_0` is a
                            strict saddle,
* `Lemma l:nosaddleconv`  — gradient ascent avoids strict saddles,
* the assembly of `Theorem p:beta0`.

The energy itself, its gradient flow and `eq: taylor` are in
`Perspective.AppendixA_Beta0`; `e:russiantrick` and `IsSkew` are in
`Perspective.RussianTrick`, where the identity is proved; `e:helpcl`, the
perturbation `PerturbationBy` it is read along and the second derivative
`SecondDerivE0At` it computes are in `Perspective.AppendixA_Hessian`, where
the identity is proved; that the perturbation exists at all is
`Perspective.exists_perturbationBy`, in `Perspective.AppendixA_Rotation`.
`lem: yury.lemma` is proved here from those four; the other two statements are
theorems closed by `sorry`.
-/

import Transformer.Perspective.AppendixA_Rotation

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Perspective

variable (d n : ℕ)

/-- **Lemma (lem: yury.lemma).** *Every non-trivial critical point of `𝖤_0` is
a strict saddle.*

Strictness is spelled out as: there is a rotation direction `B` and a subset
`𝒮` whose perturbation has *positive* second derivative of the energy at
`t = 0`.  Since `𝖤_0` is being maximized along `e:gradfl`, such a direction
means the critical point is not a local maximum — and hence, by
`no_saddle_convergence`, is reached from a null set of initial data.  In
particular every local maximum of `𝖤_0` is a consensus configuration, hence a
global maximum.

The proof is the paper's, in three moves.  `taylor_eq` produces a block `𝒮`
whose cross-energy `T = Σ_{i ∈ 𝒮} Σ_{j ∈ 𝒮^c} ⟨x_i, x_j⟩` is negative.
`hessian_at_critical` says the rotation of that block by `e^{tB}` has second
derivative `(2/n) Σ_{i ∈ 𝒮} Σ_{j ∈ 𝒮^c} ⟨B² x_i, x_j⟩`, and
`exists_perturbationBy` says the rotation exists.  Summing that over the
`russian_trick` family `B_1, …, B_d`, where `Σ_k B_k² = -(d-1) I`, gives
`-(d-1) T > 0`, so at least one direction `B_k` has a positive second
derivative — which is the assertion.

**What the source says and what is changed here.**  The survey states the
lemma with no restriction on `d`.  It is false at `d = 1`: the only skew
endomorphism of `ℝ` is `0`, so `B² = 0` and the second derivative of every
admissible perturbation is `0`, never positive — while non-trivial critical
points do exist there (the antipodal pair).  The hypothesis `2 ≤ d` is added;
it is the same one `russian_trick` needs, and Appendix A works in `d ≥ 2`
throughout, `p:beta0` itself assuming `d, n ≥ 2`.

Source: arXiv:2312.10794v5, Appendix A, `lem: yury.lemma`. -/
theorem yury_lemma (hd : 2 ≤ d) (X : SphereTuple d n) (hcrit : IsCriticalE0 d n X)
    (hnt : NonTrivialTuple d n X) :
    ∃ (B : ParamMatrix d) (𝒮 : Finset (Idx n)) (Y : ℝ → SphereTuple d n) (c : ℝ),
      IsSkew d B ∧ PerturbationBy d n B 𝒮 X Y ∧ SecondDerivE0At d n Y c ∧ 0 < c := by
  classical
  obtain ⟨𝒮, hS⟩ := taylor_eq d n X hcrit hnt
  obtain ⟨i₀, -, -⟩ := hnt
  have hn : (0 : ℝ) < (n : ℝ) := by exact_mod_cast i₀.pos
  obtain ⟨B, hBskew, hBsum⟩ := russian_trick (d := d) hd
  have hd' : (2 : ℝ) ≤ (d : ℝ) := by exact_mod_cast hd
  have hne : ((d : ℝ) - 1) ≠ 0 := by linarith
  -- the "Russian trick" with the scalar cleared: `Σ_k B_k² = -(d-1) I`
  have hBsum' : ∀ x : EucSpace d, ∑ k : Idx d, B k (B k x) = (-((d : ℝ) - 1)) • x := by
    intro x
    have h := congrArg (fun y : EucSpace d => ((d : ℝ) - 1) • y) (hBsum x)
    simp only [smul_smul, mul_inv_cancel₀ hne, one_smul, smul_neg] at h
    rw [h, neg_smul]
  set T : ℝ := ∑ i ∈ 𝒮, ∑ j ∈ 𝒮ᶜ,
    inner (𝕜 := ℝ) ((X i : EucSpace d)) ((X j : EucSpace d)) with hT
  set F : Idx d → ℝ := fun k => ∑ i ∈ 𝒮, ∑ j ∈ 𝒮ᶜ,
    inner (𝕜 := ℝ) (B k (B k ((X i : EucSpace d)))) ((X j : EucSpace d)) with hF
  have hstep : ∀ i j : Idx n,
      ∑ k : Idx d, inner (𝕜 := ℝ) (B k (B k ((X i : EucSpace d)))) ((X j : EucSpace d))
        = (-((d : ℝ) - 1)) * inner (𝕜 := ℝ) ((X i : EucSpace d)) ((X j : EucSpace d)) := by
    intro i j
    rw [← sum_inner, hBsum', real_inner_smul_left]
  have hsumF : ∑ k : Idx d, F k = (-((d : ℝ) - 1)) * T := by
    calc ∑ k : Idx d, F k
        = ∑ i ∈ 𝒮, ∑ j ∈ 𝒮ᶜ, ∑ k : Idx d,
            inner (𝕜 := ℝ) (B k (B k ((X i : EucSpace d)))) ((X j : EucSpace d)) := by
          rw [hF, Finset.sum_comm]
          exact Finset.sum_congr rfl fun i _ => Finset.sum_comm
      _ = ∑ i ∈ 𝒮, ∑ j ∈ 𝒮ᶜ, (-((d : ℝ) - 1)) *
            inner (𝕜 := ℝ) ((X i : EucSpace d)) ((X j : EucSpace d)) :=
          Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => hstep i j
      _ = (-((d : ℝ) - 1)) * T := by
          rw [hT, Finset.mul_sum]
          exact Finset.sum_congr rfl fun i _ => by rw [Finset.mul_sum]
  -- a negative cross-energy against a negative multiple of the identity
  have hpos : (0 : ℝ) < ∑ k : Idx d, F k := by
    rw [hsumF]
    exact mul_pos_of_neg_of_neg (by linarith) hS
  obtain ⟨k, -, hk⟩ := Finset.exists_lt_of_sum_lt
    (f := fun _ : Idx d => (0 : ℝ)) (g := F) (s := Finset.univ) (by simpa using hpos)
  obtain ⟨Y, hY⟩ := exists_perturbationBy d n (B k) (hBskew k) 𝒮 X
  refine ⟨B k, 𝒮, Y, (2 * (n : ℝ)⁻¹) * F k, hBskew k, hY,
    hessian_at_critical d n X (B k) 𝒮 (hBskew k) Y hY, ?_⟩
  have h2 : (0 : ℝ) < 2 * (n : ℝ)⁻¹ := by
    have := inv_pos.mpr hn
    linarith
  exact mul_pos h2 hk

/-- The hypotheses of `yury_lemma` are satisfiable: in the plane the antipodal
pair is a non-trivial critical point. -/
example : 2 ≤ 2 ∧ IsCriticalE0 2 2 (antipodalPair 2 (basePoint 1)) ∧
    NonTrivialTuple 2 2 (antipodalPair 2 (basePoint 1)) :=
  ⟨le_rfl, antipodalPair_critical_nonTrivial 2 (basePoint 1)⟩

/-- **Lemma (l:nosaddleconv).** *No-saddle-convergence lemma.*

On a compact Riemannian manifold the set of initial conditions whose
gradient-ascent trajectory converges to a strict saddle of a smooth `f` has
zero volume (center-stable manifold theorem).  It is stated here in the only
instance the survey uses it: `ℳ = (𝕊^{d-1})^n`, `f = 𝖤_0`, and — by
`YuryLemma` — the strict saddles are the non-trivial critical points.  Volume
is the uniform measure `UniformTuple`.

Not proved here: the center-stable manifold theorem is not available.

Source: arXiv:2312.10794v5, Appendix A, `l:nosaddleconv`. -/
theorem no_saddle_convergence :
    ∀ P : Measure (SphereTuple d n), UniformTuple d n P →
      P { X₀ : SphereTuple d n |
          ∃ X : ℝ → SphereTuple d n, X 0 = X₀ ∧ E0GradientAscent d n X ∧
            ∃ Z : SphereTuple d n, IsCriticalE0 d n Z ∧ NonTrivialTuple d n Z ∧
              ∀ i : Idx n,
                Filter.Tendsto
                  (fun t : ℝ => ((X t i : EucSpace d) - (Z i : EucSpace d)))
                  Filter.atTop (nhds 0) } = 0 := by
  sorry

/-- **Theorem (p:beta0)**, in the almost-sure form Appendix A proves: for
`d, n ≥ 2` the set of initial sequences whose `β = 0` trajectory does *not*
converge to a single point is null for the uniform law on `(𝕊^{d-1})^n`.

This is the statement `Perspective.beta0_consensus` should have; the latter
quantifies over every initial sequence, which is false at the exceptional
null set.

Not proved here.  The paper's proof assembles three ingredients, of which one
is now available: (i) Łojasiewicz — `𝖤_0` is analytic on a compact analytic
manifold, so every trajectory of `e:gradfl` converges to a critical point —
not stated here; (ii) `no_saddle_convergence` — the non-trivial critical
points are reached from a null set — a `sorry` above; (iii) `yury_lemma` —
those are exactly the strict saddles — proved.  The assembly is not recorded
as a statement of its own: with the conclusion already a sorried theorem, the
implication would be provable in one line and would assert nothing.

Source: arXiv:2312.10794v5, §4, `p:beta0`; Appendix A for the proof. -/
theorem almost_sure_consensus_beta0 (hd : 2 ≤ d) (hn : 2 ≤ n) :
    ∀ P : Measure (SphereTuple d n), UniformTuple d n P →
      P { X₀ : SphereTuple d n | ¬ ∃ x_star : SSphere d,
            ∀ X : ℝ → SphereTuple d n, X 0 = X₀ → beta0Dynamics d n X →
              ∀ i : Idx n,
                Filter.Tendsto
                  (fun t : ℝ => ((X t i : EucSpace d) - (x_star : EucSpace d)))
                  Filter.atTop (nhds 0) } = 0 := by
  sorry

/-- The hypotheses of `almost_sure_consensus_beta0` are satisfiable:
`d = n = 2`. -/
example : 2 ≤ 2 ∧ 2 ≤ 2 := ⟨le_rfl, le_rfl⟩

end Perspective
end Transformer
