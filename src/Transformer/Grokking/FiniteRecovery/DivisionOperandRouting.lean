module

public import Transformer.Grokking.FiniteRecovery.DivisionAttentionWeights

/-! Two native heads read the same value projection. Their invertible
two-by-two mixing matrix lets the output projection write the numerator
phase, the conjugate denominator phase and the zero flag into disjoint
residual coordinates. All coefficients are independent of the operands. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand.Division

open Transformer.PhasorV2 Finset
open scoped ComplexConjugate

noncomputable def queryKey : Stream →ₗ[ℂ] Heads 4 16 :=
  LinearMap.pi fun h => LinearMap.pi fun m =>
    if h = 1 ∧ m = 0 then Parity.coord 0 else 0

def bandIndex (m : Fin 16) : Fin 64 := ⟨m.val+1,by omega⟩

noncomputable def values : Stream →ₗ[ℂ] Heads 4 16 :=
  LinearMap.pi fun _ => LinearMap.pi fun m => Parity.coord (bandIndex m)

noncomputable def headSlot (h : Fin 4) (m : Fin 16) : Heads 4 16 →ₗ[ℂ] ℂ :=
  (LinearMap.proj (R := ℂ) (φ := fun _ : Fin 16 => ℂ) m).comp
    (LinearMap.proj (R := ℂ) (φ := fun _ : Fin 4 => Fin 16 → ℂ) h)

noncomputable def operandSlot (numerator : Bool) (m : Fin 16) : Heads 4 16 →ₗ[ℂ] ℂ :=
  let u := mix 0 1
  let p := mix 1 1
  let q := mix 1 3
  (p-q)⁻¹ • (if numerator then headSlot 1 m - (q/u) • headSlot 0 m
    else (p/u) • headSlot 0 m - headSlot 1 m)

noncomputable def attentionMap : Attention 64 4 16 where
  WQ := queryKey
  WK := queryKey
  WV := values
  WO := Parity.assemble fun i => if i = 5 then operandSlot true 0 else
    if i = 6 then operandSlot false 1 else if i = 7 then operandSlot true 2 else 0

noncomputable def normalized (χ : Character) (x : DivisionInput) (j : Fin 5) : Stream :=
  (posGain j : ℂ) • embedding χ (divisionPrompt x j)

private theorem native_head (χ : Character) (x : DivisionInput) (j : Fin 5) (h : Fin 4) :
    queryKey (normalized χ x j) h = rawHead h j := by
  funext m
  change (if h=1 ∧ m=0 then Parity.coord 0 else (0 : Stream →ₗ[ℂ] ℂ)) _ = _
  by_cases he : h=1 ∧ m=0
  · rw [ite_eq_left he,rawHead,ite_eq_left he]
    change (posGain j : ℂ) * embedding χ (divisionPrompt x j) 0 = (posGain j : ℂ)
    rw [(embedding_coords _ _).1,mul_one]
  · simp [rawHead,he]

noncomputable def mixture (χ : Character) (x : DivisionInput) (h : Fin 4) (m : Fin 16) : ℂ :=
  ∑ j, mix h j * normalized χ x j (bandIndex m)

theorem native_mixture (χ : Character) (x : DivisionInput) :
    attentionWith (causalGeometry 0) attentionMap Parity.offset 1 (normalized χ x) (4 : Fin 5) =
      attentionMap.WO (mixture χ x) := by
  let geo := causalGeometry (T := 5) (0 : Fin 16 → ℝ)
  let head (j : Fin 5) (h : Fin 4) := geo.rotate j
    (normalizeHead Parity.offset (queryKey (normalized χ x j) h))
  let score (h : Fin 4) (j : Fin 5) :=
    (1 : ℂ) * ((Real.log ((geo.time (4 : Fin 5) : ℝ)+1) / Real.sqrt 16 : ℝ) : ℂ) *
      ∑ m, conj (head 4 h m) * head j h m
  have hs (h : Fin 4) : score h = posScore h := by
    funext j
    have ht : (geo.time (4:Fin 5) : ℝ)+1 = 5 := by change ((4:ℕ):ℝ)+1 = 5; norm_num
    dsimp only [score,head]
    rw [native_head χ x 4 h,native_head χ x j h,ht,one_mul]
    rfl
  apply congrArg attentionMap.WO
  funext h m
  change (∑ j, (⟨maskedSoftmax (geo.visible 4) (fun k => (score h k).re) j,
    maskedSoftmax (geo.visible 4) (fun k => (score h k).im) j⟩ : ℂ) *
      normalized χ x j (bandIndex m)) = _
  rw [hs h]
  rfl

private theorem special_zero (χ : Character) {t : Fin 239} (ht : t.val < 22) (m : Fin 16) :
    embedding χ t (bandIndex m) = 0 := by
  obtain ⟨hp,hf⟩ := small_features χ ht
  have hi : bandIndex m ≠ 0 := by intro h; have := congrArg Fin.val h; simp [bandIndex] at this
  simp [embedding,hp,hf,EuclideanSpace.single,PiLp.single_apply,hi]

private theorem mixture_two (χ : Character) (x : DivisionInput) (h : Fin 4) (m : Fin 16) :
    mixture χ x h m = mix h 1 * normalized χ x 1 (bandIndex m) +
      mix h 3 * normalized χ x 3 (bandIndex m) := by
  have hz (j : Fin 5) (ht : (divisionPrompt x j).val < 22) :
      normalized χ x j (bandIndex m) = 0 := by
    change (posGain j : ℂ) * embedding χ (divisionPrompt x j) (bandIndex m) = _
    rw [special_zero χ ht,mul_zero]
  unfold mixture
  rw [Fin.sum_univ_succ,Fin.sum_univ_succ,Fin.sum_univ_succ,Fin.sum_univ_succ,Fin.sum_univ_one]
  let v (j : Fin 5) := normalized χ x j (bandIndex m)
  change mix h 0 * v 0 + (mix h 1 * v 1 + (mix h 2 * v 2 + (mix h 3 * v 3 + mix h 4 * v 4))) = _
  have z0 : v 0 = 0 := hz 0 (by change 0 < 22; decide)
  have z2 : v 2 = 0 := hz 2 (by change 10 < 22; decide)
  have z4 : v 4 = 0 := hz 4 (by change 1 < 22; decide)
  rw [z0,z2,z4]
  change _ = mix h 1 * v 1 + mix h 3 * v 3
  ring

private theorem solve_mixture (χ : Character) (x : DivisionInput) (m : Fin 16) :
    operandSlot true m (mixture χ x) = normalized χ x 1 (bandIndex m) ∧
    operandSlot false m (mixture χ x) = normalized χ x 3 (bandIndex m) := by
  let p := mix 1 1
  let q := mix 1 3
  let u := mix 0 1
  have hu : u ≠ 0 := by
    intro h
    have he := congrArg Complex.re h
    norm_num [u,mix_uniform] at he
  have hd : p-q ≠ 0 := sub_ne_zero.mpr mix_unequal
  have h3 : mix 0 3 = u := (mix_uniform 3).trans (mix_uniform 1).symm
  change (p-q)⁻¹ * (mixture χ x 1 m - (q/u)*mixture χ x 0 m) = _ ∧
    (p-q)⁻¹ * ((p/u)*mixture χ x 0 m - mixture χ x 1 m) = _
  rw [mixture_two,mixture_two,h3]
  change (p-q)⁻¹ * (p*_+q*_-(q/u)*(u*_+u*_)) = _ ∧
    (p-q)⁻¹ * ((p/u)*(u*_+u*_)-(p*_+q*_)) = _
  constructor <;> field_simp [hu,hd] <;> ring

/-- Exact output coordinates of the final causal row of the attention
branch, before the residual addition and RMS. -/
theorem routed (χ : Character) (x : DivisionInput) (i : Fin 64) :
    attentionWith (causalGeometry 0) attentionMap Parity.offset 1 (normalized χ x) (4 : Fin 5) i =
      if i = 5 then normalized χ x 1 1 else if i = 6 then normalized χ x 3 2
      else if i = 7 then normalized χ x 1 3 else 0 := by
  rw [native_mixture]
  change (if i=5 then operandSlot true 0 else if i=6 then operandSlot false 1 else if i=7 then operandSlot true 2 else
    (0 : Heads 4 16 →ₗ[ℂ] ℂ)) (mixture χ x) = _
  split
  · exact (solve_mixture χ x 0).1
  · split
    · exact (solve_mixture χ x 1).2
    · split
      · exact (solve_mixture χ x 2).1
      · rfl

end Transformer.Grokking.FiniteRecovery.Argand.Division
