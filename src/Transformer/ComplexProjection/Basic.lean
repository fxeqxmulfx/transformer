module

public import Transformer.ComplexProjection.Defs
public import Transformer.GPTMini.RMSNorm

/-!
# The exact price of complex linearity

Source: the splitting of `Defs` (DiffGPT `5a37c51`). Here `J` and `J'` are complex
structures on input and output, `J'` an isometry, and `f` any map with
`f (J z) = J' (f z)`: a complex-linear projection, or a composite of such projections
with maps that commute with the complex structure, the RMS among them
(`rmsNormEps_comm`).

On a set closed under `J`, such a map errs on a real projection `L` by at least the
conjugate-linear part `A` of `L` (`norm_antilinearPart_le`), and the complex part of `L`
errs by exactly `A` (`norm_complexPart_sub`). So a uniform error `δ` is reachable at
the same width if and only if `‖A z‖ ≤ δ` on the set (`exists_comm_approx_iff`): the
complex class is the real class without its conjugate-linear half, and it loses exactly
that half. Conjugation on the unit circle is missed by `1`, as by the zero map
(`one_le_of_approx_conj`).

The general form is `gap_of_intertwine`; with `τ = id` and `τ = -id` it gives the gaps
of `Transformer.ComplexFFN.Basic`. A target that anticommutes with the complex
structures is missed by its whole norm (`norm_le_of_anticomm`). The branch RMS keeps
this: a conjugate-linear target normalized by the RMS is missed by its full norm, about
`√d`, as by the zero map (`norm_rmsNormEps_le`).
-/

@[expose] public section

namespace Transformer.ComplexProjection

section Gap

variable {V W : Type*} [NormedAddCommGroup W]

/-- An approximant `f` with `f ∘ σ = τ ∘ f`, for a `1`-Lipschitz `τ`, misses the part of
the target that breaks this law: an error bound `δ` on a set holding `z` and `σ z`
bounds `‖τ (T z) - T (σ z)‖` by `2δ`. -/
theorem gap_of_intertwine {f T : V → W} {σ : V → V} {τ : W → W}
    (hτ : ∀ x y, ‖τ x - τ y‖ ≤ ‖x - y‖) (hf : ∀ z, f (σ z) = τ (f z)) {K : Set V}
    {δ : ℝ} (hδ : ∀ z ∈ K, ‖f z - T z‖ ≤ δ) {z : V} (hz : z ∈ K) (hσ : σ z ∈ K) :
    ‖τ (T z) - T (σ z)‖ ≤ 2 * δ := by
  have h : τ (T z) - T (σ z) = (τ (T z) - τ (f z)) + (f (σ z) - T (σ z)) := by
    rw [hf]
    abel
  rw [h, two_mul]
  exact (norm_add_le _ _).trans (add_le_add
    ((hτ _ _).trans (by rw [norm_sub_rev]; exact hδ z hz)) (hδ _ hσ))

example (c z : ℂ) : ‖Complex.I * (c * z) - c * (Complex.I * z)‖ ≤ 2 * 0 :=
  gap_of_intertwine (f := (c * ·)) (T := (c * ·)) (σ := (Complex.I * ·))
    (τ := (Complex.I * ·)) (fun x y => le_of_eq (by rw [← mul_sub, norm_mul,
      Complex.norm_I, one_mul])) (fun x => by show c * (Complex.I * x) = Complex.I * (c * x); ring)
    (K := Set.univ) (fun x _ => by simp) (Set.mem_univ z) (Set.mem_univ _)

end Gap

section Anti

variable {V W : Type*} [NormedAddCommGroup W] [NormedSpace ℝ W]

/-- A map that commutes with the complex structures misses a target that anticommutes
with them by the target's full norm: an error bound `δ` on a set holding `z` and `J z`
bounds `‖T z‖` by `δ`, which the zero map attains. -/
theorem norm_le_of_anticomm {J : V → V} {J' : W →ₗ[ℝ] W} (hn : ∀ x, ‖J' x‖ = ‖x‖)
    {f T : V → W} (hf : ∀ z, f (J z) = J' (f z)) (hT : ∀ z, T (J z) = -J' (T z))
    {K : Set V} {δ : ℝ} (hδ : ∀ z ∈ K, ‖f z - T z‖ ≤ δ) {z : V} (hz : z ∈ K)
    (hJz : J z ∈ K) : ‖T z‖ ≤ δ := by
  have h := gap_of_intertwine (τ := J') (fun x y => le_of_eq (by rw [← map_sub, hn]))
    hf hδ hz hJz
  rw [hT, sub_neg_eq_add, ← two_smul ℝ, norm_smul, hn, Real.norm_eq_abs, abs_two] at h
  linarith

end Anti

section Projection

variable {V W : Type*} [AddCommGroup V] [Module ℝ V] [NormedAddCommGroup W] [NormedSpace ℝ W]

variable {J : V →ₗ[ℝ] V} {J' : W →ₗ[ℝ] W}

/-- No map commuting with the complex structures approximates a real projection better
than its conjugate-linear part: an error bound `δ` on a set holding `z` and `J z` bounds
`‖A z‖` by `δ`. The map need not be linear. -/
theorem norm_antilinearPart_le (hJ' : ∀ x, J' (J' x) = -x) (hn : ∀ x, ‖J' x‖ = ‖x‖)
    (L : V →ₗ[ℝ] W) {f : V → W} (hf : ∀ z, f (J z) = J' (f z)) {K : Set V} {δ : ℝ}
    (hδ : ∀ z ∈ K, ‖f z - L z‖ ≤ δ) {z : V} (hz : z ∈ K) (hJz : J z ∈ K) :
    ‖antilinearPart J J' L z‖ ≤ δ := by
  have h := gap_of_intertwine (T := L) (τ := J')
    (fun x y => le_of_eq (by rw [← map_sub, hn])) hf hδ hz hJz
  have he : J' (L z) - L (J z) = (2 : ℝ) • J' (antilinearPart J J' L z) := by
    simp only [antilinearPart_apply, map_smul, map_add, hJ']
    module
  rw [he, norm_smul, hn, Real.norm_eq_abs, abs_two] at h
  linarith

/-- The complex part attains that bound: it errs on `L` by exactly the conjugate-linear
part. -/
theorem norm_complexPart_sub (L : V →ₗ[ℝ] W) (z : V) :
    ‖complexPart J J' L z - L z‖ = ‖antilinearPart J J' L z‖ := by
  have h : antilinearPart J J' L z = -(complexPart J J' L z - L z) := by
    rw [complexPart_apply, antilinearPart_apply]
    module
  rw [h, norm_neg]

/-- Complex projections at the same width: on a set closed under `J`, a uniform error `δ`
on the real projection `L` is reachable by a map commuting with the complex structures,
and then by the complex part of `L`, exactly when the conjugate-linear part of `L` stays
within `δ`. -/
theorem exists_comm_approx_iff (hJ : ∀ z, J (J z) = -z) (hJ' : ∀ x, J' (J' x) = -x)
    (hn : ∀ x, ‖J' x‖ = ‖x‖) (L : V →ₗ[ℝ] W) {K : Set V} (hK : ∀ z ∈ K, J z ∈ K) (δ : ℝ) :
    (∃ f : V → W, (∀ z, f (J z) = J' (f z)) ∧ ∀ z ∈ K, ‖f z - L z‖ ≤ δ) ↔
      ∀ z ∈ K, ‖antilinearPart J J' L z‖ ≤ δ :=
  ⟨fun ⟨_, hf, hδ⟩ z hz => norm_antilinearPart_le hJ' hn L hf hδ hz (hK z hz),
    fun h => ⟨complexPart J J' L, complexPart_comm L hJ hJ',
      fun z hz => (norm_complexPart_sub L z).trans_le (h z hz)⟩⟩

end Projection

/-- Multiplication by `i` keeps the norm. -/
theorem norm_mulLeft_I (z : ℂ) : ‖LinearMap.mulLeft ℝ Complex.I z‖ = ‖z‖ := by
  simp

/-- On the unit circle, maps commuting with `i` reach `c z + conj z` within `δ` exactly
when `δ` bounds the conjugate part. -/
example (c : ℂ) (δ : ℝ) :=
  exists_comm_approx_iff mulLeft_I_mulLeft_I mulLeft_I_mulLeft_I norm_mulLeft_I
    (LinearMap.mulLeft ℝ c + Complex.conjAe.toLinearMap) (K := Metric.sphere 0 1)
    (fun z hz => by simpa using hz) δ

/-- Conjugation on the unit circle: every map commuting with multiplication by `i`
errs by at least `1` there, as much as the zero map. -/
theorem one_le_of_approx_conj {f : ℂ → ℂ} (hf : ∀ z, f (Complex.I * z) = Complex.I * f z)
    {δ : ℝ} (hδ : ∀ z : ℂ, ‖z‖ = 1 → ‖f z - (starRingEnd ℂ) z‖ ≤ δ) : 1 ≤ δ := by
  have h := norm_le_of_anticomm (J := LinearMap.mulLeft ℝ Complex.I)
    (J' := LinearMap.mulLeft ℝ Complex.I) norm_mulLeft_I (T := starRingEnd ℂ) hf
    (fun z => by simp) (K := {z | ‖z‖ = 1}) hδ (z := 1) (by simp) (by simp)
  simpa using h

example : (1 : ℝ) ≤ 1 :=
  one_le_of_approx_conj (f := fun _ => 0) (fun z => by simp) (fun z hz => by simp [hz])

/-- The branch RMS `x √d / √(‖x‖² + dε)` commutes with every linear isometry, the
complex structure among them. -/
theorem rmsNormEps_comm {d : ℕ} (eps : ℝ) {J : EucSpace d →ₗ[ℝ] EucSpace d}
    (hn : ∀ x, ‖J x‖ = ‖x‖) (x : EucSpace d) :
    GPTMini.rmsNormEps eps (J x) = J (GPTMini.rmsNormEps eps x) := by
  rw [GPTMini.rmsNormEps, GPTMini.rmsNormEps, hn, map_smul]

example {d : ℕ} (eps : ℝ) (x : EucSpace d) :=
  rmsNormEps_comm eps (J := -(LinearMap.id : EucSpace d →ₗ[ℝ] EucSpace d)) (fun y => by simp) x

/-- Normalizing does not help: if `L` anticommutes with the complex structures, a map
commuting with them misses the normalized target `R (L z)` on a set holding `z` and
`J z` by at least `‖R (L z)‖`, what the zero map errs. -/
theorem norm_rmsNormEps_le {V : Type*} [AddCommGroup V] [Module ℝ V] {d : ℕ} (eps : ℝ)
    {J : V →ₗ[ℝ] V} {J' : EucSpace d →ₗ[ℝ] EucSpace d} (hn : ∀ x, ‖J' x‖ = ‖x‖)
    {L : V →ₗ[ℝ] EucSpace d} (hL : ∀ z, L (J z) = -J' (L z)) {f : V → EucSpace d}
    (hf : ∀ z, f (J z) = J' (f z)) {K : Set V} {δ : ℝ}
    (hδ : ∀ z ∈ K, ‖f z - GPTMini.rmsNormEps eps (L z)‖ ≤ δ) {z : V} (hz : z ∈ K)
    (hJz : J z ∈ K) : ‖GPTMini.rmsNormEps eps (L z)‖ ≤ δ := by
  refine norm_le_of_anticomm hn hf (T := fun z => GPTMini.rmsNormEps eps (L z)) (fun x => ?_)
    hδ hz hJz
  have h := rmsNormEps_comm eps (J := -J') (fun y => by rw [LinearMap.neg_apply, norm_neg, hn])
    (L x)
  simp only [LinearMap.neg_apply] at h
  rw [hL, h]

/-- Even maps miss the normalized identity by its norm on a set closed under negation. -/
example {d : ℕ} (hd : 0 < d) {eps : ℝ} (heps : 0 < eps) (z : EucSpace d) :
    ‖GPTMini.rmsNormEps eps z‖ ≤ Real.sqrt d :=
  norm_rmsNormEps_le eps (J := -(LinearMap.id : EucSpace d →ₗ[ℝ] EucSpace d))
    (J' := LinearMap.id) (fun x => by simp) (L := LinearMap.id) (fun x => by simp)
    (f := fun _ => 0) (fun x => by simp) (K := Set.univ)
    (fun x _ => by simpa using GPTMini.rmsNormEps_norm_le eps heps hd x) (Set.mem_univ z)
    (Set.mem_univ _)

end Transformer.ComplexProjection
