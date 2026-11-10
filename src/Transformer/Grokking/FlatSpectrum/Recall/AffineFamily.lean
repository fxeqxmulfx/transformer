module

public import Mathlib.Algebra.Field.ZMod
public import Transformer.Grokking.FlatSpectrum.Recall.Family

/-!
# The affine dictionaries over a finite field: `p²` contexts as hard as `p^p`

Source: `review/flat_spectrum.md`, *Families of dictionaries and the multi-query task* (added),
item FD4, for the task of Arora et al., arXiv:2312.04927v1, §3, and the scope item of AR10 on
the family of contexts.

Let the keys and the values be a field `𝔽` of `p` elements. The dictionary `q ↦ a q + b` has
two parameters (`affineDict`), so there are `p²` of them among the `p^p` dictionaries. They
are pairwise uniform of index 1: through two points with different keys passes one line
(`isPairwiseUniform_affineDict`). So a readout of cut width `d` misses recall on them by at
least `p ∑_v (φ(v) − μ)² (p − d)`, the fraction `1 − d / p` of the variance of `φ` as on all
dictionaries (`HasCutWidth.mul_le_sum_sq_affineDict`), and exact recall on them has `d ≥ p`
(`HasCutWidth.card_le_of_recall_affineDict`).

Uniformity on pairs of keys is needed: the constant dictionaries are recalled at cut width one
for every number of keys (`exists_hasCutWidth_one_recall_const`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Recall

variable {𝔽 : Type*} [Field 𝔽] [Fintype 𝔽] [DecidableEq 𝔽]

/-- The affine dictionary `q ↦ a q + b` over a finite field (`review/flat_spectrum.md`, FD4). -/
def affineDict (ab : 𝔽 × 𝔽) : 𝔽 → 𝔽 := fun q => ab.1 * q + ab.2

/-- **The affine dictionaries are pairwise uniform of index 1** (`review/flat_spectrum.md`,
FD4): a key takes a value on `p` of them, and two keys take a pair of values on one. -/
theorem isPairwiseUniform_affineDict : IsPairwiseUniform (affineDict (𝔽 := 𝔽)) 1 where
  single q v := by
    rw [one_mul, ← Finset.card_univ (α := 𝔽)]
    refine (Finset.card_bij (fun (a : 𝔽) _ => (a, v - a * q)) ?_ ?_ ?_).symm
    · intro a _
      exact Finset.mem_filter.2 ⟨Finset.mem_univ _, by simp [affineDict]⟩
    · intro a _ a' _ h
      exact (Prod.mk.inj h).1
    · rintro ⟨a, b⟩ hab
      have hv : a * q + b = v := (Finset.mem_filter.1 hab).2
      exact ⟨a, Finset.mem_univ _, by rw [← hv]; simp⟩
  pair q q' hqq' v v' := by
    have hne : q - q' ≠ 0 := sub_ne_zero.2 hqq'
    rw [Finset.card_eq_one]
    refine ⟨((v - v') / (q - q'), v - (v - v') / (q - q') * q), ?_⟩
    ext ⟨a, b⟩
    rw [Finset.mem_filter, Finset.mem_singleton, Prod.mk.injEq]
    constructor
    · rintro ⟨-, h1, h2⟩
      have h1' : a * q + b = v := h1
      have h2' : a * q' + b = v' := h2
      have ha : a = (v - v') / (q - q') := by
        rw [eq_div_iff hne, ← h1', ← h2']
        ring
      exact ⟨ha, by rw [← ha, ← h1']; ring⟩
    · rintro ⟨rfl, rfl⟩
      refine ⟨Finset.mem_univ _, ?_, ?_⟩
      · show (v - v') / (q - q') * q + (v - (v - v') / (q - q') * q) = v
        ring
      · show (v - v') / (q - q') * q' + (v - (v - v') / (q - q') * q) = v'
        field_simp
        ring

variable {d : ℕ} {F : 𝔽 → (𝔽 → 𝔽) → ℝ} {φ : 𝔽 → ℝ}

/-- **The error of a readout of cut width `d` on the affine dictionaries**
(`review/flat_spectrum.md`, FD4): at least `p ∑_v (φ(v) − μ)² (p − d)`. -/
theorem HasCutWidth.mul_le_sum_sq_affineDict (hF : HasCutWidth d F) {μ : ℝ}
    (hμ : ∑ v, (φ v - μ) = 0) :
    Fintype.card 𝔽 * (∑ v, (φ v - μ) ^ 2) * ((Fintype.card 𝔽 : ℝ) - d) ≤
      ∑ q, ∑ ab, (φ (affineDict ab q) - F q (affineDict ab)) ^ 2 := by
  have h := hF.mul_le_sum_sq_of_isPairwiseUniform isPairwiseUniform_affineDict hμ
  rwa [Fintype.card_prod, min_eq_left (Nat.le_mul_self _), Nat.cast_one, one_mul] at h

/-- **Exact recall on the `p²` affine dictionaries takes cut width `p`**
(`review/flat_spectrum.md`, FD4), for a readout `φ` that is not constant. -/
theorem HasCutWidth.card_le_of_recall_affineDict (hF : HasCutWidth d F) {v v' : 𝔽}
    (hφ : φ v ≠ φ v') (h : ∀ q ab, F q (affineDict ab) = φ (affineDict ab q)) :
    Fintype.card 𝔽 ≤ d := by
  have h1 := hF.card_le_of_recall_of_isPairwiseUniform isPairwiseUniform_affineDict one_pos hφ h
  rwa [Fintype.card_prod, min_eq_left (Nat.le_mul_self _)] at h1

/-- **The constant dictionaries are recalled at cut width one** (`review/flat_spectrum.md`,
FD4), for every number of keys: one number, the readout of the common value, crosses. -/
theorem exists_hasCutWidth_one_recall_const {κ ν : Type*} [Nonempty κ] (φ : ν → ℝ) :
    ∃ F : κ → (κ → ν) → ℝ, HasCutWidth 1 F ∧ ∀ q (c : ν), F q (fun _ => c) = φ c := by
  obtain ⟨q₀⟩ := ‹Nonempty κ›
  exact ⟨fun _ π => φ (π q₀),
    ⟨fun _ => 0, fun _ _ => 1, fun π _ => φ (π q₀), fun q π => by simp⟩, fun _ _ => rfl⟩

/-- Three keys and three values over `ZMod 3`: recall, at cut width three, on the nine affine
dictionaries. -/
example : Fintype.card (ZMod 3) ≤ 3 :=
  (hasCutWidth_recall (κ := ZMod 3) (d := 3) (by simp)
    fun v : ZMod 3 => if v = 0 then (1 : ℝ) else 0).card_le_of_recall_affineDict
      (φ := fun v : ZMod 3 => if v = 0 then (1 : ℝ) else 0) (v := 0) (v' := 1) (by simp)
      fun _ _ => rfl

example := (hasCutWidth_recall (κ := ZMod 3) (d := 3) (by simp)
  fun v : ZMod 3 => if v = 0 then (1 : ℝ) else 0).mul_le_sum_sq_affineDict
    (φ := fun v : ZMod 3 => if v = 0 then (1 : ℝ) else 0) (μ := 1 / 3) (by
      rw [Finset.sum_sub_distrib, Finset.sum_ite_eq' Finset.univ (0 : ZMod 3)]
      simp [ZMod.card])

example : ∃ F : Fin 5 → (Fin 5 → Bool) → ℝ, HasCutWidth 1 F ∧
    ∀ q (c : Bool), F q (fun _ => c) = if c then 1 else -1 :=
  exists_hasCutWidth_one_recall_const fun c : Bool => if c then (1 : ℝ) else -1

end Transformer.Grokking.FlatSpectrum.Recall
