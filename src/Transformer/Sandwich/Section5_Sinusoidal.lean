module

public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
public import Mathlib.LinearAlgebra.Matrix.NonsingularInverse
public import Mathlib.Order.Interval.Set.Card
public import Transformer.Sandwich.Section5_Causal
public import Transformer.Sandwich.Section5_ReceptiveField

/-!
# §5.3 Sinusoidal and Rotary

Chi, Fan, Rudnicky, Ramadge, arXiv:2212.10356v2, §5.3 and Eq. (4)–(5): Sinusoidal adds
`p_m` to the input embedding `e_m`, so the logit decomposes as in Eq. (4); "Rotary multiplies each
token embedding `e_m` with a position-specific rotation matrix `R_m e_m`"; and "they both fail to
focus on the most recent `L_tr` tokens because neither of their formulations guarantees a
`L_tr`-bounded receptive field."

* `decomposition`: Eq. (4), the logit of `e_m + p_m` and `e_n + p_n` as the semantic term plus
  three terms with positions. The first term's `e_n^⊤` in the TeX is a stray transpose; the
  approximation `≈ e_m^⊤ W_q^⊤ W_k e_n + p_m^⊤ p_n` that follows is a heuristic, not an identity.
* `sinusoid`, `apeFreq`, `sinusoidalAPE`: Eq. (5). Its `i ∈ [1, d/2]` gives the coordinates
  `2i, 2i + 1 ∈ [2, d + 1]`, of which `d + 1` is no coordinate and `0, 1` are never set; here
  `i ∈ [0, d/2)` as in Vaswani et al. and Appendix E, with the sines before the cosines as in
  Appendix E's `np.concatenate` (a permutation of the coordinates).
* `Contributes.comp_surjective`: a position-wise surjective transform of the inputs keeps the
  contributing tokens. Adding `p_n` is one, and so is multiplying by an invertible `R_n`.
* `ape_contributes`, `rotary_contributes`: with any APE, or any invertible `R_n`, every token
  `n ≤ m` contributes to the averaging model's prediction at `m`, so `m + 1` tokens contribute
  (`ape_ncard_contributes`): no bound in terms of `L_tr`.
-/

@[expose] public section

namespace Transformer.Sandwich

open Matrix

variable {α β γ : Type*} {d k R : ℕ}

/-- **Eq. (4)**: the logit of `e_m + p_m` and `e_n + p_n` is the semantic term
`e_m^⊤ W_q^⊤ W_k e_n` plus the three terms with positions. -/
theorem decomposition (Wq Wk : Matrix (Fin k) (Fin d) ℝ) (em pm en pn : Fin d → ℝ) :
    (Wq *ᵥ (em + pm)) ⬝ᵥ (Wk *ᵥ (en + pn)) =
      (Wq *ᵥ em) ⬝ᵥ (Wk *ᵥ en) + ((Wq *ᵥ em) ⬝ᵥ (Wk *ᵥ pn) + (Wq *ᵥ pm) ⬝ᵥ (Wk *ᵥ en) +
        (Wq *ᵥ pm) ⬝ᵥ (Wk *ᵥ pn)) := by
  simp only [mulVec_add, add_dotProduct, dotProduct_add]
  ring

/-- The sinusoid at the position `m` with the frequencies `θ`: `sin(m θ_i)`, then `cos(m θ_i)`. -/
noncomputable def sinusoid {D : ℕ} (θ : Fin D → ℝ) (m : ℝ) : Fin D ⊕ Fin D → ℝ :=
  Sum.elim (fun i => Real.sin (m * θ i)) fun i => Real.cos (m * θ i)

/-- Sinusoidal's frequencies `1/10000^{2i/d}` for `d = 2D` and `i = 0, …, D − 1` (Eq. 5). -/
noncomputable def apeFreq (D : ℕ) (i : Fin D) : ℝ := 1 / 10000 ^ ((i : ℝ) / D)

/-- **Eq. (5)**, Sinusoidal's `p_m ∈ ℝ^{2D}`: `sin(m/10000^{2i/d})` and `cos(m/10000^{2i/d})`, the
sines first. -/
noncomputable def sinusoidalAPE (D m : ℕ) : Fin (D + D) → ℝ :=
  fun j => sinusoid (apeFreq D) m (finSumFinEquiv.symm j)

/-- A position-wise surjective transform `T_j` of the inputs keeps the contributing tokens. -/
theorem Contributes.comp_surjective {F : (ℕ → α) → ℕ → β} {T : ℕ → γ → α}
    (hT : ∀ j, Function.Surjective (T j)) {m n : ℕ} (h : Contributes F m n) :
    Contributes (fun e => F fun j => T j (e j)) m n := by
  obtain ⟨x, y, hxy, hne⟩ := h
  refine ⟨fun j => Function.surjInv (hT j) (x j), fun j => Function.surjInv (hT j) (y j),
    fun j hj => by simp only [hxy j hj], ?_⟩
  simpa only [Function.surjInv_eq] using hne

/-- The hypotheses of `Contributes.comp_surjective` hold for the identity map and transforms. -/
example {m n : ℕ} (h : Contributes (fun x : ℕ → α => x) m n) :
    Contributes (fun e : ℕ → α => fun j => id (e j)) m n :=
  h.comp_surjective (T := fun _ => id) fun _ => Function.surjective_id

/-- **An APE does not bound the receptive field** (§5.3): for every APE `p`, in the averaging
model with `R ≥ 1` layers, no positional bias and the inputs `e_n + p_n`, every token `n ≤ m`
contributes to the prediction at `m`. -/
theorem ape_contributes (p : ℕ → Fin 2 → ℝ) (hR : 1 ≤ R) {m n : ℕ} (hn : n ≤ m) :
    Contributes (fun e => (avgModel R).predict (fun _ _ => 0) fun j => e j + p j) m n := by
  refine Contributes.comp_surjective (T := fun j a => a + p j)
    (fun j a => ⟨a - p j, sub_add_cancel a (p j)⟩) ?_
  refine avgModel_contributes_of_finite (w := m + 1) (fun _ _ => ⟨by simp, by simp⟩) hR
    (by omega) hn ?_
  have : m ≤ R * m := Nat.le_mul_of_pos_left m hR
  rw [Nat.add_sub_cancel]
  omega

/-- The hypotheses of `ape_contributes` hold for Sinusoidal with `d = 2` and one layer. -/
example : Contributes (fun e => (avgModel 1).predict (fun _ _ => 0)
    fun j => e j + sinusoidalAPE 1 j) 0 0 :=
  ape_contributes (sinusoidalAPE 1) le_rfl le_rfl

/-- **With an APE, all `m + 1` tokens contribute** to the averaging model's prediction at `m`. -/
theorem ape_ncard_contributes (p : ℕ → Fin 2 → ℝ) (hR : 1 ≤ R) (m : ℕ) :
    {n | Contributes (fun e => (avgModel R).predict (fun _ _ => 0) fun j => e j + p j) m n}.ncard =
      m + 1 := by
  have hset : {n | Contributes (fun e => (avgModel R).predict (fun _ _ => 0)
      fun j => e j + p j) m n} = Set.Icc 0 m := by
    ext n
    simp only [Set.mem_ofPred_eq, Set.mem_Icc, zero_le, true_and]
    exact ⟨fun h => IsCausal.contributes
      (F := fun e => (avgModel R).predict (fun _ _ => 0) fun j => e j + p j)
      (((avgModel R).isCausal_predict _).map (fun j a => a + p j) id) h,
      fun hn => ape_contributes p hR hn⟩
  rw [hset, Set.ncard_Icc, Nat.card_Icc, Nat.sub_zero]

/-- The hypothesis of `ape_ncard_contributes` holds for one layer. -/
example (p : ℕ → Fin 2 → ℝ) (m : ℕ) :
    {n | Contributes (fun e => (avgModel 1).predict (fun _ _ => 0) fun j => e j + p j) m n}.ncard =
      m + 1 :=
  ape_ncard_contributes p le_rfl m

/-- **Rotary, as §5.3 describes it, does not bound the receptive field**: for invertible `R_n`, in
the averaging model with `R ≥ 1` layers, no positional bias and the inputs `R_n e_n`, every token
`n ≤ m` contributes to the prediction at `m`. -/
theorem rotary_contributes (Rot : ℕ → Matrix (Fin 2) (Fin 2) ℝ) (hRot : ∀ j, IsUnit (Rot j))
    (hR : 1 ≤ R) {m n : ℕ} (hn : n ≤ m) :
    Contributes (fun e => (avgModel R).predict (fun _ _ => 0) fun j => Rot j *ᵥ e j) m n := by
  refine Contributes.comp_surjective (T := fun j a => Rot j *ᵥ a)
    (fun j => mulVec_surjective_iff_isUnit.2 (hRot j)) ?_
  refine avgModel_contributes_of_finite (w := m + 1) (fun _ _ => ⟨by simp, by simp⟩) hR
    (by omega) hn ?_
  have : m ≤ R * m := Nat.le_mul_of_pos_left m hR
  rw [Nat.add_sub_cancel]
  omega

/-- The hypotheses of `rotary_contributes` hold for the identity rotations. -/
example : Contributes (fun e => (avgModel 1).predict (fun _ _ => 0)
    fun j => (1 : Matrix (Fin 2) (Fin 2) ℝ) *ᵥ e j) 0 0 :=
  rotary_contributes (fun _ => 1) (fun _ => isUnit_one) le_rfl le_rfl

end Transformer.Sandwich
