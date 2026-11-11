module

public import Transformer.Grokking.FlatSpectrum.FlatSupport
public import Transformer.Grokking.FlatSpectrum.Training.MaxMarginLimit
public import Transformer.Grokking.FlatSpectrum.Training.SquaredLoss

/-!
# A loss on the seen cells: a trained fit is a minimum of M from its own values

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), item XE7.
A loss reads only the seen cells `Ω` when it has equal values at matrices equal on `Ω`. The
squared loss on `Ω` does (`fitLoss_eq_of_seen`), and so does the cross-entropy whose probes
vanish off `Ω` (`ceLoss_eq_of_seen`): its logits read only `Ω` (`pairing_eq_of_seen`).

For such a loss and a weight decay `λ > 0`, the product `X = U Vᴴ` of a trained fit has the
least nuclear norm among the matrices equal to it on `Ω`
(`IsDecayedMin.isMinOn_nuclearNorm`): a matrix `Y` equal to `X` on `Ω` has the loss of `X`, and
`L X + λ ‖X‖_S1 ≤ L Y + λ ‖Y‖_S1` (item WD2). So `X` is a minimum of M from `Ω` for its own
values, whatever the loss fits. Being a minimum of M survives scaling (`not_isMinOn_smul`), so

* no trained product is a nonzero multiple of a kernel that is not a minimum of M from `Ω`
  (`IsDecayedMin.ne_smul_of_not_isMinOn`), and by item R2 of a flat kernel with a nonzero
  unseen cell (`IsDecayedMin.ne_smul_of_isFlat`);
* under the bounds of item XE3 the run on the cross-entropy converges, and its limit is not
  such a multiple (`exists_tendsto_decayRun_ceLoss_ne_smul`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix Filter Topology

variable {m n k S C : Type*} [Fintype m] [Fintype n] [Fintype k] {Ω : Set (m × n)}
  {K X Y : Matrix m n ℂ} {lam : ℝ} {U : Matrix m k ℂ} {V : Matrix n k ℂ}

/-- The squared loss on `Ω` reads only the cells of `Ω`. -/
lemma fitLoss_eq_of_seen (K : Matrix m n ℂ) (h : ∀ ij ∈ Ω, X ij.1 ij.2 = Y ij.1 ij.2) :
    fitLoss Ω K X = fitLoss Ω K Y := by
  have e : mask Ω (X - K) = mask Ω (Y - K) := by
    ext i j
    simp only [mask, of_apply, Matrix.sub_apply]
    split_ifs with hij
    · rw [h (i, j) hij]
    · rfl
  rw [fitLoss, fitLoss, e]

/-- A probe that vanishes off `Ω` reads only the cells of `Ω`. -/
lemma pairing_eq_of_seen {B : Matrix m n ℂ} (hB : ∀ i j, (i, j) ∉ Ω → B i j = 0)
    (h : ∀ ij ∈ Ω, X ij.1 ij.2 = Y ij.1 ij.2) : pairing B X = pairing B Y := by
  simp only [pairing, trace_conjTranspose_mul]
  congr 1
  refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
  by_cases hij : (i, j) ∈ Ω
  · rw [h (i, j) hij]
  · rw [hB i j hij, star_zero, zero_mul, zero_mul]

/-- The cross-entropy whose probes vanish off `Ω` reads only the cells of `Ω`. -/
lemma ceLoss_eq_of_seen [Fintype S] [Fintype C] {A : S → C → Matrix m n ℂ} {y : S → C}
    (hA : ∀ s c i j, (i, j) ∉ Ω → A s c i j = 0) (h : ∀ ij ∈ Ω, X ij.1 ij.2 = Y ij.1 ij.2) :
    ceLoss A y X = ceLoss A y Y := by
  simp only [ceLoss, fun s c => pairing_eq_of_seen (hA s c) h]

variable [DecidableEq n] {L : Matrix m n ℂ → ℝ}

/-- **A trained fit is a minimum of M from its own values** (`review/flat_spectrum.md`, item
XE7): if the loss reads only the cells of `Ω` and `λ > 0`, the product of a factorization of
least weight-decayed loss has the least nuclear norm among the matrices equal to it on `Ω`. -/
theorem IsDecayedMin.isMinOn_nuclearNorm
    (hL : ∀ X Y : Matrix m n ℂ, (∀ ij ∈ Ω, X ij.1 ij.2 = Y ij.1 ij.2) → L X = L Y)
    (hlam : 0 < lam) (h : IsDecayedMin L lam U V) :
    IsMinOn nuclearNorm {X | ∀ ij ∈ Ω, X ij.1 ij.2 = (U * Vᴴ) ij.1 ij.2} (U * Vᴴ) := by
  refine isMinOn_iff.mpr fun Y hY => ?_
  have h1 := h.add_nuclearNorm_le hlam.le Y
  rw [hL Y (U * Vᴴ) hY] at h1
  exact le_of_mul_le_mul_left (by linarith) hlam

/-- A nonzero multiple of a kernel that is not a minimum of M from `Ω` is not one either. -/
lemma not_isMinOn_smul
    (h : ¬ IsMinOn nuclearNorm {X | ∀ ij ∈ Ω, X ij.1 ij.2 = K ij.1 ij.2} K) {t : ℝ}
    (ht : t ≠ 0) :
    ¬ IsMinOn nuclearNorm {X | ∀ ij ∈ Ω, X ij.1 ij.2 = (t • K) ij.1 ij.2} (t • K) := by
  rw [isMinOn_iff] at h ⊢
  push Not at h ⊢
  obtain ⟨Y, hY, hlt⟩ := h
  refine ⟨t • Y, fun ij hij => ?_, ?_⟩
  · rw [Matrix.smul_apply, Matrix.smul_apply, hY ij hij]
  · rw [nuclearNorm_real_smul_eq_abs, nuclearNorm_real_smul_eq_abs]
    exact mul_lt_mul_of_pos_left hlt (abs_pos.mpr ht)

/-- **No trained product is a multiple of a kernel that M does not return**
(`review/flat_spectrum.md`, item XE7): if the loss reads only the cells of `Ω`, `λ > 0` and `K`
is not a minimum of M from `Ω`, a trained product is not `t K` for any `t ≠ 0`. -/
theorem IsDecayedMin.ne_smul_of_not_isMinOn
    (hL : ∀ X Y : Matrix m n ℂ, (∀ ij ∈ Ω, X ij.1 ij.2 = Y ij.1 ij.2) → L X = L Y)
    (hlam : 0 < lam) (h : IsDecayedMin L lam U V)
    (hK : ¬ IsMinOn nuclearNorm {X | ∀ ij ∈ Ω, X ij.1 ij.2 = K ij.1 ij.2} K) {t : ℝ}
    (ht : t ≠ 0) : U * Vᴴ ≠ t • K := by
  intro e
  refine not_isMinOn_smul hK ht ?_
  rw [← e]
  exact h.isMinOn_nuclearNorm hL hlam

/-- **No trained product is a multiple of a flat kernel with a nonzero unseen cell**
(`review/flat_spectrum.md`, item XE7 with R2), for a loss that reads only the cells of `Ω` and
a weight decay `λ > 0`. -/
theorem IsDecayedMin.ne_smul_of_isFlat [DecidableEq m]
    (hL : ∀ X Y : Matrix m n ℂ, (∀ ij ∈ Ω, X ij.1 ij.2 = Y ij.1 ij.2) → L X = L Y)
    (hlam : 0 < lam) (h : IsDecayedMin L lam U V) {c : ℝ} (hc : 0 < c) (hK : IsFlat K c)
    {i : m} {j : n} (hij : (i, j) ∉ Ω) (hKij : K i j ≠ 0) {t : ℝ} (ht : t ≠ 0) :
    U * Vᴴ ≠ t • K :=
  h.ne_smul_of_not_isMinOn hL hlam (hK.not_isMinOn_of_ne_zero hc hij hKij) ht

/-- **The run on the cross-entropy does not return a kernel that M does not return**
(`review/flat_spectrum.md`, items XE3 and XE7): for probes of norm at most `a` that vanish off
`Ω`, under the bounds of item XE3 the run from a start of full row rank converges, and the
product of its limit is not `t K` for any `t ≠ 0`. -/
theorem exists_tendsto_decayRun_ceLoss_ne_smul [DecidableEq m] [Fintype S] [Fintype C]
    {A : S → C → Matrix m n ℂ} {y : S → C} {η a ρ : ℝ} (hη : 0 < η) (hlam : 0 < lam)
    (ha0 : 0 ≤ a) (ha : ∀ s c, frobeniusSq (A s c) ≤ a ^ 2)
    (hρ : 4 * (Fintype.card S * Real.log (Fintype.card C)) ≤ lam * ρ)
    (hstep : 2 * η * (lam ^ 2 + (2 * a * Fintype.card S) ^ 2) ≤ lam)
    (hdesc : η * (lam + 2 * a * Fintype.card S + 2 * (Fintype.card S * a ^ 2) * ρ) ≤ 1)
    {U₀ : Matrix m k ℂ} {V₀ : Matrix n k ℂ} (hw : frobeniusSq U₀ + frobeniusSq V₀ ≤ ρ)
    (hfull : Function.Surjective (stack U₀ V₀).mulVec)
    (hA : ∀ s c i j, (i, j) ∉ Ω → A s c i j = 0)
    (hK : ¬ IsMinOn nuclearNorm {X | ∀ ij ∈ Ω, X ij.1 ij.2 = K ij.1 ij.2} K) :
    ∃ (U : Matrix m k ℂ) (V : Matrix n k ℂ),
      Tendsto (decayRun (ceGrad A y) η lam U₀ V₀) atTop (𝓝 (U, V)) ∧
        ∀ t : ℝ, t ≠ 0 → U * Vᴴ ≠ t • K := by
  obtain ⟨U, V, hlim, hmin⟩ :=
    exists_tendsto_decayRun_ceLoss_isDecayedMin hη hlam ha0 ha hρ hstep hdesc hw hfull
  exact ⟨U, V, hlim, fun t ht =>
    hmin.ne_smul_of_not_isMinOn (fun X Y h => ceLoss_eq_of_seen hA h) hlam hK ht⟩

/-! The examples: the `1 × 1` kernel `1` with its cell seen, and the `1 × 2` kernel `(1, 1)`,
flat with `c = 2`, with no cell seen and the probes `0`. -/

example := fitLoss_eq_of_seen (Ω := Set.univ) (1 : Matrix (Fin 1) (Fin 1) ℂ) (X := 1) (Y := 1)
  fun _ _ => rfl

example := pairing_eq_of_seen (Ω := Set.univ) (B := (1 : Matrix (Fin 1) (Fin 1) ℂ))
  (fun _ _ h => absurd (Set.mem_univ _) h) (X := 1) (Y := 1) fun _ _ => rfl

example := ceLoss_eq_of_seen (Ω := Set.univ) (A := signProbe) (y := fun _ => true)
  (fun _ _ _ _ h => absurd (Set.mem_univ _) h) (X := 1) (Y := 1) fun _ _ => rfl

/-- The trained product `¼` of the `1 × 1` kernel `1` at `λ = ¾`. -/
example := (isDecayedMin_one_sub (s := 1 / 2) (by norm_num) (by norm_num)).isMinOn_nuclearNorm
  (fun _ _ h => fitLoss_eq_of_seen 1 h) (by norm_num)

example := not_isMinOn_smul (isFlat_row_ones.not_isMinOn_of_ne_zero two_pos (Ω := ∅) (i := 0)
  (j := 0) (by simp) (by simp)) two_ne_zero

/-- A trained fit of the probes `0` is not a nonzero multiple of `(1, 1)`. -/
example : ∃ (U : Matrix (Fin 1) (Fin 1) ℂ) (V : Matrix (Fin 2) (Fin 1) ℂ),
    ∀ t : ℝ, t ≠ 0 → U * Vᴴ ≠ t • Matrix.of ![![(1 : ℂ), 1]] := by
  obtain ⟨U, V, h⟩ := exists_isDecayedMin_ceLoss (k := Fin 1)
    (0 : Fin 1 → Bool → Matrix (Fin 1) (Fin 2) ℂ) (fun _ => true) one_pos (Or.inl le_rfl)
  exact ⟨U, V, fun t ht => h.ne_smul_of_isFlat (Ω := ∅)
    (fun X Y hXY => ceLoss_eq_of_seen (fun _ _ _ _ _ => rfl) hXY) one_pos two_pos
    isFlat_row_ones (i := 0) (j := 0) (by simp) (by simp) ht⟩

/-- With no cell seen, the run on the probes `0` from `U₀ = [1 0]`, `V₀ = [0 1]` converges, to
no nonzero multiple of the `1 × 1` kernel `1`. -/
example := exists_tendsto_decayRun_ceLoss_ne_smul (y := fun _ : Fin 1 => true) (η := 1 / 10)
  (a := 0) (by norm_num) one_pos le_rfl (A := 0) (fun _ _ => by simp [frobeniusSq])
  signWitness_radius (by norm_num) (by norm_num) (stepWitness_ball.trans (by norm_num))
  (fun v => ⟨v, by rw [stack_fromCols_one, one_mulVec]⟩) (Ω := ∅) (fun _ _ _ _ _ => rfl)
  (isFlat_one.not_isMinOn_of_ne_zero one_pos (Ω := ∅) (i := 0) (j := 0) (by simp) (by simp))

/-- The same from the kernel not being a minimum of M. -/
example : ∃ (U : Matrix (Fin 1) (Fin 1) ℂ) (V : Matrix (Fin 2) (Fin 1) ℂ),
    U * Vᴴ ≠ (2 : ℝ) • Matrix.of ![![(1 : ℂ), 1]] := by
  obtain ⟨U, V, h⟩ := exists_isDecayedMin_ceLoss (k := Fin 1)
    (0 : Fin 1 → Bool → Matrix (Fin 1) (Fin 2) ℂ) (fun _ => true) one_pos (Or.inl le_rfl)
  exact ⟨U, V, h.ne_smul_of_not_isMinOn (Ω := ∅)
    (fun X Y hXY => ceLoss_eq_of_seen (fun _ _ _ _ _ => rfl) hXY) one_pos
    (isFlat_row_ones.not_isMinOn_of_ne_zero two_pos (Ω := ∅) (i := 0) (j := 0) (by simp)
      (by simp)) two_ne_zero⟩

end Transformer.Grokking.FlatSpectrum.Training
