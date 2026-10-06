import Transformer.GPTMini.Sparsemax.NearestPrototypeCodes
import Transformer.GPTMini.Sparsemax.CausalPrefixKeys

/-!
# Nearest memory inputs from actual causal prefix distances

Derived architecture before arXiv:1602.02068v2, Eq. (1). Compare masked
visible-prefix signatures by their Hamming distance. This is a separating
nonnegative distance computed directly on observations, without a complete
signature dictionary or output labels. Nearest codes remain exactly one-hot.

Future changes in queries or prototype records cannot affect the encoder.
Distinct registered prefixes receive identity training codes. Learned
memory attention can still vary: the encoder selects its query row, not
its eventual sparse routing weights. Generalization requires an explicit
regularity condition on the target function, stated in the next module.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open scoped BigOperators

/-- Hamming distance on visible-prefix signatures, computed without enumerating all signatures.
Source: the derived nearest input architecture for arXiv:1602.02068v2, Eq. (1). -/
def prefixSignatureDistance {T V : ℕ}
    (x y : Fin T → Option (Fin V)) : ℝ := ∑ j, if x j = y j then 0 else 1

/-- Every signature distance is nonnegative, including empty windows.
Source: the observation distance in the derived arXiv:1602.02068v2, Eq. (1) encoder. -/
theorem prefixSignatureDistance_nonneg {T V : ℕ} (x y : Fin T → Option (Fin V)) :
    0 ≤ prefixSignatureDistance x y := by
  apply Finset.sum_nonneg
  intro j hj
  split_ifs <;> norm_num

/-- Every registered observation has distance zero to itself.
Source: the data-only nearest encoder before arXiv:1602.02068v2, Eq. (1). -/
theorem prefixSignatureDistance_self {T V : ℕ} (x : Fin T → Option (Fin V)) :
    prefixSignatureDistance x x = 0 := by
  unfold prefixSignatureDistance
  simp only [ite_true, Finset.sum_const_zero]

/-- Zero distance identifies the full masked observation, preserving order and prefix length.
Source: the separating distance for the derived arXiv:1602.02068v2, Eq. (1) encoder. -/
theorem prefixSignatureDistance_eq_zero {T V : ℕ} (x y : Fin T → Option (Fin V)) :
    prefixSignatureDistance x y = 0 ↔ x = y := by
  constructor
  · intro h
    have hn (j : Fin T) : 0 ≤ (if x j = y j then (0 : ℝ) else 1) := by
      split_ifs <;> norm_num
    have he := (Finset.sum_eq_zero_iff_of_nonneg (fun j hj => hn j)).1 h
    funext j
    have hz := he j (Finset.mem_univ j)
    by_contra hj
    simp only [ite_eq_right hj] at hz
    norm_num at hz
  · intro h
    rw [h]
    exact prefixSignatureDistance_self y

/-- Interchanging two observations preserves their distance.
Source: the symmetric data comparison preceding arXiv:1602.02068v2, Eq. (1). -/
theorem prefixSignatureDistance_symm {T V : ℕ} (x y : Fin T → Option (Fin V)) :
    prefixSignatureDistance x y = prefixSignatureDistance y x := by
  apply Finset.sum_congr rfl
  intro j hj
  simp only [eq_comm]

/-- Signature distance obeys the triangle inequality on observed prefix records.
Source: the derived Hamming geometry before arXiv:1602.02068v2, Eq. (1). -/
theorem prefixSignatureDistance_triangle {T V : ℕ} (x y z : Fin T → Option (Fin V)) :
    prefixSignatureDistance x z ≤ prefixSignatureDistance x y + prefixSignatureDistance y z := by
  rw [prefixSignatureDistance, prefixSignatureDistance, prefixSignatureDistance,
    ← Finset.sum_add_distrib]
  apply Finset.sum_le_sum
  intro j hj
  by_cases hxy : x j = y j
  · simp only [hxy, ite_true, zero_add, le_refl]
  · by_cases hyz : y j = z j
    · simp only [← hyz, hxy, ite_false, ite_true, add_zero, le_refl]
    · split_ifs <;> norm_num

/-- Actual causal observations give sparse nearest-prototype probability codes.
Source: the derived input architecture before arXiv:1602.02068v2, Eq. (1). -/
def prefixNearestCodes {R T V N : ℕ} (queries : Fin R → Fin T → Fin V)
    (queryRows : Fin R → Fin T) (prototypes : Fin (N + 1) → Fin T → Fin V)
    (prototypeRows : Fin (N + 1) → Fin T) : Matrix (Fin R) (Fin (N + 1)) ℝ :=
  nearestPrototypeCodes prefixSignatureDistance
    (fun j => causalPrefixSignature (prototypes j) (prototypeRows j))
    (fun r => causalPrefixSignature (queries r) (queryRows r))

/-- Every actual or unseen prefix supplies a feasible probability code.
Source: the nearest categorical input for arXiv:1602.02068v2, Eq. (1). -/
theorem prefixNearestCodes_mem {R T V N : ℕ} (queries : Fin R → Fin T → Fin V)
    (queryRows : Fin R → Fin T) (prototypes : Fin (N + 1) → Fin T → Fin V)
    (prototypeRows : Fin (N + 1) → Fin T) :
    prefixNearestCodes queries queryRows prototypes prototypeRows ∈ contextCodeDomain R N :=
  nearestPrototypeCodes_mem _ _ _

/-- Query changes beyond the observed prefix leave the sparse code unchanged.
Source: masked observation distances before arXiv:1602.02068v2, Eq. (1). -/
theorem prefixNearestCodes_causal {R T V N : ℕ} (queries other : Fin R → Fin T → Fin V)
    (queryRows : Fin R → Fin T) (prototypes : Fin (N + 1) → Fin T → Fin V)
    (prototypeRows : Fin (N + 1) → Fin T)
    (h : ∀ r j, j ≤ queryRows r → queries r j = other r j) :
    prefixNearestCodes queries queryRows prototypes prototypeRows =
      prefixNearestCodes other queryRows prototypes prototypeRows := by
  have he : (fun r => causalPrefixSignature (queries r) (queryRows r)) =
      (fun r => causalPrefixSignature (other r) (queryRows r)) := by
    funext r
    exact (causalPrefixSignature_eq_iff _ _ _ _).2 ⟨rfl, h r⟩
  unfold prefixNearestCodes
  rw [he]

/-- A genuine future-token change inhabits sparse-code causality. -/
example : prefixNearestCodes (fun _ : Fin 1 => fun _ : Fin 2 => (0 : Fin 2)) (fun _ => 0)
    (fun r : Fin 2 => fun _ : Fin 2 => r) (fun _ => 1) =
    prefixNearestCodes (fun _ : Fin 1 => fun j : Fin 2 => if j = 0 then 0 else (1 : Fin 2))
      (fun _ => 0) (fun r : Fin 2 => fun _ : Fin 2 => r) (fun _ => 1) := by
  apply prefixNearestCodes_causal
  intro r j hj
  fin_cases j
  · rfl
  · norm_num at hj

/-- Distinct registered prefixes give identity training codes at every finite window size.
Source: proved separating Hamming distance before arXiv:1602.02068v2, Eq. (1). -/
theorem prefixNearestCodes_identity {T V N : ℕ} (prototypes : Fin (N + 1) → Fin T → Fin V)
    (rows : Fin (N + 1) → Fin T)
    (hs : Function.Injective (fun j => causalPrefixSignature (prototypes j) (rows j))) :
    prefixNearestCodes prototypes rows prototypes rows = 1 := by
  apply nearestPrototypeCodes_identity
  · exact prefixSignatureDistance_nonneg
  · exact prefixSignatureDistance_self
  · intro x y h
    exact (prefixSignatureDistance_eq_zero x y).1 h
  · exact hs

/-- Distinct actual one-token observations inhabit every registration premise. -/
example : prefixNearestCodes (fun r : Fin 2 => fun _ : Fin 1 => r) (fun _ => 0)
    (fun r : Fin 2 => fun _ : Fin 1 => r) (fun _ => 0) = 1 := by
  apply prefixNearestCodes_identity
  intro r s h
  exact ((causalPrefixSignature_eq_iff _ _ _ _).1 h).2 0 (by norm_num)

/-- Changing hidden prototype continuations leaves registration and nearest codes unchanged.
Source: the masked data encoder before arXiv:1602.02068v2, Eq. (1). -/
theorem prefixNearestCodes_prototypes_causal {R T V N : ℕ}
    (queries : Fin R → Fin T → Fin V) (queryRows : Fin R → Fin T)
    (prototypes other : Fin (N + 1) → Fin T → Fin V) (prototypeRows : Fin (N + 1) → Fin T)
    (h : ∀ r j, j ≤ prototypeRows r → prototypes r j = other r j) :
    prefixNearestCodes queries queryRows prototypes prototypeRows =
      prefixNearestCodes queries queryRows other prototypeRows := by
  have he : (fun r => causalPrefixSignature (prototypes r) (prototypeRows r)) =
      (fun r => causalPrefixSignature (other r) (prototypeRows r)) := by
    funext r
    exact (causalPrefixSignature_eq_iff _ _ _ _).2 ⟨rfl, h r⟩
  unfold prefixNearestCodes
  rw [he]

/-- Actual future prototype changes inhabit registration causality. -/
example : prefixNearestCodes (fun _ : Fin 1 => fun _ : Fin 2 => (1 : Fin 2)) (fun _ => 1)
    (fun r : Fin 2 => fun _ : Fin 2 => r) (fun _ => 0) =
    prefixNearestCodes (fun _ : Fin 1 => fun _ : Fin 2 => (1 : Fin 2)) (fun _ => 1)
      (fun r : Fin 2 => fun j : Fin 2 => if j = 0 then r else (1 : Fin 2)) (fun _ => 0) := by
  apply prefixNearestCodes_prototypes_causal
  intro r j hj
  fin_cases j
  · rfl
  · norm_num at hj

end Transformer.GPTMini.Sparsemax
