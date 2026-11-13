module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DensityBounds.Perturbation

/-!
# Objective accuracy and score separation control mixed mass

Source: `review/flat_spectrum.md`, CM22. A strict scalar score gap gives a
quantitative concentration bound. Real-score ties require further analysis;
CM21's exact second-stage result does not justify approximate tie breaking.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.DensityBounds

open Finset Matrix DensityExtraction FourierExtraction
open scoped ComplexOrder

variable {ι X : Type*} [Fintype ι] [DecidableEq ι]

/-- A scalar score gap bounds the total probability outside the best index,
CM22. Written without division, the bound also displays why a zero gap is
uninformative. No weight is assumed to vanish. -/
theorem weighted_gap {w : ι → ℝ} (hw : Weights w) (v : ι → ℝ) (i : ι) (γ : ℝ)
    (hg : ∀ j, j ≠ i → γ ≤ v i - v j) :
    γ * (1 - w i) ≤ v i - ∑ j, w j * v j := by
  have hm : ∑ j ∈ univ.erase i, w j = 1 - w i := by
    have h := sum_erase_add univ w (mem_univ i)
    rw [hw.2] at h
    linarith
  have hs : ∑ j ∈ univ.erase i, w j * (v i - v j) = v i - ∑ j, w j * v j := by
    have h := sum_erase_add univ (fun j => w j * (v i - v j)) (mem_univ i)
    simp only [sub_self, mul_zero, add_zero] at h
    rw [h]
    simp_rw [mul_sub]
    rw [sum_sub_distrib, ← sum_mul, hw.2, one_mul]
  calc
    _ = ∑ j ∈ univ.erase i, w j * γ := by rw [← sum_mul, hm]; ring
    _ ≤ ∑ j ∈ univ.erase i, w j * (v i - v j) :=
      sum_le_sum fun j hj => mul_le_mul_of_nonneg_left (hg j (mem_erase.mp hj).1) (hw.1 j)
    _ = _ := hs

/-- The observable real score is within `d² ε` of the actual diagonal
probability average, CM22. This is derived from the relaxed Fourier readings. -/
theorem relaxed_score_error {q : ℕ} [NeZero q] (r : X ≃ ZMod q)
    {k : ι → ZMod q} (hk : Separated k) {M : Matrix ι ι ℂ} {ε : ℝ}
    (hM : Relaxed (basis r k) ε M) (x y : X) :
    |(reading (basis r k) M x y).re -
      ∑ i, (M i i).re * (ZMod.stdAddChar ((r x - r y) * k i)).re| ≤
      (Fintype.card ι : ℝ)^2 * ε := by
  have hε : 0 ≤ ε := (norm_nonneg _).trans (hM.2 x)
  have hb := diagonal_reading_error r k M hε
    (fun i j hij => off_diagonal_le r hk M ε hM.2 hij) x y
  have hd : (diagonal fun i => M i i) = diagonal (fun i => ((M i i).re : ℂ)) := by
    congr 1
    funext i
    apply Complex.ext
    · rfl
    · exact (Complex.nonneg_iff.mp (hM.1.1.diag_nonneg (i := i))).2.symm
  have hr := (Complex.abs_re_le_norm _).trans hb
  rw [hd, diagonal_reading] at hr
  simpa using hr

/-- A separated scalar objective and approximate optimal value force most
diagonal mass onto its best mode, CM22. Amplitude error and optimization
error are separate terms; neither exact rank nor exact extraction is claimed. -/
theorem score_concentration {q : ℕ} [NeZero q] (r : X ≃ ZMod q)
    {k : ι → ZMod q} (hk : Separated k) {M : Matrix ι ι ℂ} {ε η γ : ℝ}
    (hM : Relaxed (basis r k) ε M) (x y : X) (i : ι)
    (hg : ∀ j, j ≠ i → γ ≤ (ZMod.stdAddChar ((r x - r y) * k i)).re -
      (ZMod.stdAddChar ((r x - r y) * k j)).re)
    (hopt : (ZMod.stdAddChar ((r x - r y) * k i)).re ≤
      (reading (basis r k) M x y).re + η) :
    γ * (1 - (M i i).re) ≤ η + (Fintype.card ι : ℝ)^2 * ε := by
  have hw := weighted_gap (density_weights hM.1)
    (fun j => (ZMod.stdAddChar ((r x - r y) * k j)).re) i γ hg
  have he := (abs_le.mp (relaxed_score_error r hk hM x y)).2
  linarith

/-- Combined error bound in an estimated, rotated subspace, CM22. The unknown
unitary and Fourier coordinates occur only as structural witnesses. A supplied
approximate optimizer is compared with every relaxed competitor, and all pure
reference competitors are admitted by the explicit tolerance budget. -/
theorem rotated_concentration {q : ℕ} [NeZero q] (r : X ≃ ZMod q)
    {k : ι → ZMod q} (hk : Separated k) (B : Matrix X ι ℂ) (Q : Matrix ι ι ℂ)
    (hQ : Q.conjTranspose * Q = 1) (hQ' : Q * Q.conjTranspose = 1)
    {M : Matrix ι ι ℂ} {ε δ R η γ : ℝ} (hM : Relaxed B ε M)
    (hδ : 0 ≤ δ) (hR : 0 ≤ R)
    (he : ∀ x i, ‖B x i - (basis r k * Q) x i‖ ≤ δ)
    (hc : ∀ x i, ‖(basis r k * Q) x i‖ ≤ R)
    (hbudget : (Fintype.card ι : ℝ)^2 * δ * (2 * R + δ) ≤ ε)
    (x y : X) (i : ι)
    (hg : ∀ j, j ≠ i → γ ≤ (ZMod.stdAddChar ((r x - r y) * k i)).re -
      (ZMod.stdAddChar ((r x - r y) * k j)).re)
    (hopt : ∀ N, Relaxed B ε N → (reading B N x y).re ≤ (reading B M x y).re + η) :
    γ * (1 - ((Q * M * Q.conjTranspose) i i).re) ≤
      η + 2 * ((Fintype.card ι : ℝ)^2 * δ * (2 * R + δ)) +
      (Fintype.card ι : ℝ)^2 * (ε + (Fintype.card ι : ℝ)^2 * δ * (2 * R + δ)) := by
  have ht := relaxed_transfer B (basis r k * Q) hM hδ hR he hc
  have hh : Relaxed (basis r k)
      (ε + (Fintype.card ι : ℝ)^2 * δ * (2 * R + δ)) (Q * M * Q.conjTranspose) := by
    refine ⟨density_conjugate Q M hQ hM.1, fun a => ?_⟩
    simpa only [reading_change] using ht.2 a
  apply score_concentration r hk hh x y i hg
  have ho := objective_transfer B (basis r k * Q) hM.1 hδ hR he hc hbudget x y hopt
    (rotated_projector_feasible r k Q hQ hQ' i)
  have hu : Q * (Q.conjTranspose * projector i * Q) * Q.conjTranspose = projector i := by
    simpa using undo_conjugate Q.conjTranspose (projector i) (by simpa using hQ')
  simpa only [reading_change, hu, projector_reading, add_assoc] using ho

end Transformer.Grokking.FlatSpectrum.CoupledMoments.DensityBounds
