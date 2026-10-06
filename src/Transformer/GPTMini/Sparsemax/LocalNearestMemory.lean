import Transformer.GPTMini.Sparsemax.LocalJointMemory
import Transformer.GPTMini.Sparsemax.PrefixNearestCodes
import Transformer.GPTMini.Sparsemax.NearestMemoryGeneralization

/-!
# Compact sparse causal memory with conditional unseen prediction guarantees

Derived architecture for arXiv:1602.02068v2, Eq. (1), and common values at
`73f8a0b`. Fixed nearest-prefix codes feed the compact trainable path Gram.
Every actual query has at most three attention routes, with genuine learned
Q/K norms and a global decoded value table. Distinct registered causal
observations have identity data codes and fit arbitrary vector targets.

Actual outputs evaluate the nearest row of the global table, so the earlier
epsilon plus L times data-cover-radius guarantee applies to this compact
family without extra routing labels. Target regularity and coverage are
explicit assumptions; finite fitting does not establish them for text.
The prototype table, metric and possible path remain fixed data choices.
Output-only attention nonidentifiability and task-loss/FFN selection remain
separate questions. Future query tokens cannot affect the actual forward.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

/-- Actual compact attention and common values predict the nearest global output row.
Source: the structural value chart after arXiv:1602.02068v2, Eq. (1). -/
theorem localJointNearestForward_eq {Key : Type*} {R N D : ℕ} (cap floor : ℝ)
    (p : LocalMemoryParameters N) (distance : Key → Key → ℝ)
    (prototypes : Fin (N + 1) → Key) (queries : Fin R → Key)
    (Z : Matrix (Fin (N + 1)) (Fin D) ℝ) (hf : 1 / 2 < floor)
    (hp : p ∈ localMemoryParameterDomain N cap floor) :
    localJointMemoryForward (nearestPrototypeCodes distance prototypes queries) (p, Z) =
      Matrix.of (fun r => Z (nearestPrototype distance prototypes (queries r))) :=
  jointNearestForward_eq cap floor _ distance prototypes queries Z hf
    (localMemoryGram_mem cap floor p (by linarith) hp)

/-- A changed compact memory and a real unseen query inhabit the nearest-forward premises. -/
example : localJointMemoryForward (nearestPrototypeCodes (fun x y : ℝ => |x - y|)
    (fun j : Fin 2 => (2 * j.val : ℝ)) (fun _ : Fin 1 => (1 / 2 : ℝ)))
    (localMemoryExampleParameters, Matrix.of (fun j : Fin 2 => fun _ : Fin 1 => (2 * j.val : ℝ))) =
    Matrix.of (fun _ : Fin 1 => fun _ : Fin 1 =>
      (2 * (nearestPrototype (fun x y : ℝ => |x - y|)
        (fun j : Fin 2 => (2 * j.val : ℝ)) (1 / 2)).val : ℝ)) :=
  localJointNearestForward_eq _ _ _ _ _ _ _ (by norm_num) localMemoryExampleParameters_mem

/-- Every vector target table on distinct registered prefixes is fit by the compact joint chart.
Source: the target-independent nearest encoder and actual arXiv:1602.02068v2, Eq. (1) decoder.
Compact path connections do not reduce finite-prototype output attainability. -/
theorem localPrefixNearestForward_registered {T V N D : ℕ} (cap floor : ℝ)
    (p : LocalMemoryParameters N) (prototypes : Fin (N + 1) → Fin T → Fin V)
    (rows : Fin (N + 1) → Fin T) (Y : Matrix (Fin (N + 1)) (Fin D) ℝ)
    (hf : 1 / 2 < floor) (hp : p ∈ localMemoryParameterDomain N cap floor)
    (hs : Function.Injective (fun j => causalPrefixSignature (prototypes j) (rows j))) :
    localJointMemoryForward (prefixNearestCodes prototypes rows prototypes rows) (p, Y) = Y := by
  rw [localJointMemoryForward_eq cap floor _ _ hf (prefixNearestCodes_mem _ _ _ _) hp,
    prefixNearestCodes_identity prototypes rows hs, Matrix.one_mul]

/-- Actual different prefixes, changed Q/K and nonconstant vector outputs inhabit all fit premises. -/
example : localJointMemoryForward
    (prefixNearestCodes (fun j : Fin 2 => fun _ : Fin 1 => j) (fun _ => 0)
      (fun j : Fin 2 => fun _ : Fin 1 => j) (fun _ => 0))
    (localMemoryExampleParameters, Matrix.of (fun j : Fin 2 => fun _ : Fin 1 => (j.val : ℝ))) =
    Matrix.of (fun j : Fin 2 => fun _ : Fin 1 => (j.val : ℝ)) := by
  apply localPrefixNearestForward_registered 4 (3 / 4) _ _ _ _ (by norm_num)
    localMemoryExampleParameters_mem
  intro i j h
  exact ((causalPrefixSignature_eq_iff _ _ _ _).1 h).2 0 (by norm_num)

/-- Actual nearest attention on causal prefixes has at most three active routes at every query.
Source: the proved compact memory support for arXiv:1602.02068v2, Eq. (1). -/
theorem localPrefixNearestAttention_support_card {R T V N : ℕ} (cap floor : ℝ)
    (p : LocalMemoryParameters N) (queries : Fin R → Fin T → Fin V)
    (queryRows : Fin R → Fin T) (prototypes : Fin (N + 1) → Fin T → Fin V)
    (prototypeRows : Fin (N + 1) → Fin T) (hf : 0 ≤ floor)
    (hp : p ∈ localMemoryParameterDomain N cap floor) (r : Fin R) :
    (Finset.univ.filter (fun j => contextMemoryAttention (localMemoryGram p)
      (prefixNearestCodes queries queryRows prototypes prototypeRows) r j ≠ 0)).card ≤ 3 :=
  contextNearestLocalAttention_support_card cap floor p _ _ _ hf hp r

/-- Four actual prefix prototypes inhabit the genuine three-route bound. -/
example : (Finset.univ.filter (fun j => contextMemoryAttention
    (localMemoryGram (0 : LocalMemoryParameters 3))
    (prefixNearestCodes (fun _ : Fin 1 => fun _ : Fin 1 => (0 : Fin 4)) (fun _ => 0)
      (fun j : Fin 4 => fun _ : Fin 1 => j) (fun _ => 0)) 0 j ≠ 0)).card ≤ 3 :=
  localPrefixNearestAttention_support_card 4 (3 / 4) _ _ _ _ _ (by norm_num)
    (zero_mem_localMemoryParameterDomain _ _ _ (by norm_num) (by norm_num)) _

/-- Compact actual attention and common values inherit the uniform unseen error guarantee.
Source: nearest prediction after arXiv:1602.02068v2, Eq. (1), with explicit regularity and coverage.
No hypothesis supplies target attention routes. -/
theorem localNearestForward_cover_error {Key : Type*} {R N D : ℕ} (cap floor : ℝ)
    (p : LocalMemoryParameters N) (distance : Key → Key → ℝ)
    (prototypes : Fin (N + 1) → Key) (queries : Fin R → Key)
    (Z : Matrix (Fin (N + 1)) (Fin D) ℝ) (target : Key → Fin D → ℝ) (epsilon L radius : ℝ)
    (hf : 1 / 2 < floor) (hp : p ∈ localMemoryParameterDomain N cap floor)
    (hfit : ∀ j d, |Z j d - target (prototypes j) d| ≤ epsilon)
    (hreg : ∀ x y d, |target x d - target y d| ≤ L * distance x y) (hL : 0 ≤ L)
    (hcover : ∀ r, ∃ j, distance (queries r) (prototypes j) ≤ radius) (r : Fin R) (d : Fin D) :
    |localJointMemoryForward (nearestPrototypeCodes distance prototypes queries) (p, Z) r d -
      target (queries r) d| ≤ epsilon + L * radius :=
  jointNearestForward_cover_error cap floor _ distance prototypes queries Z target epsilon L radius
    hf (localMemoryGram_mem cap floor p (by linarith) hp) hfit hreg hL hcover r d

/-- Nonzero learned edges/norms, imperfect fit and an unseen query satisfy every generalization premise. -/
example : |localJointMemoryForward (nearestPrototypeCodes (fun x y : ℝ => |x - y|)
    (fun j : Fin 2 => (2 * j.val : ℝ)) (fun _ : Fin 1 => (1 / 2 : ℝ)))
    (localMemoryExampleParameters,
      Matrix.of (fun j : Fin 2 => fun _ : Fin 1 => (2 * j.val + 1 / 10 : ℝ))) 0 0 - 1 / 2| ≤ 3 / 5 := by
  calc
    _ ≤ (1 / 10 : ℝ) + 1 * (1 / 2) := by
      apply localNearestForward_cover_error 4 (3 / 4) _ _ _ _ _
        (fun x : ℝ => fun _ : Fin 1 => x) (1 / 10) 1 (1 / 2) (by norm_num)
        localMemoryExampleParameters_mem
      · intro j d
        change |(2 * (j.val : ℝ) + 1 / 10) - 2 * (j.val : ℝ)| ≤ 1 / 10
        ring_nf
        norm_num
      · intro x y d
        simp only [one_mul, le_refl]
      · norm_num
      · intro r
        exact ⟨0, by norm_num⟩
    _ = _ := by norm_num

/-- Future query changes leave the actual compact common-value forward unchanged.
Source: the masked prefix encoder before arXiv:1602.02068v2, Eq. (1). -/
theorem localPrefixNearestForward_causal {R T V N D : ℕ}
    (queries other : Fin R → Fin T → Fin V) (queryRows : Fin R → Fin T)
    (prototypes : Fin (N + 1) → Fin T → Fin V) (prototypeRows : Fin (N + 1) → Fin T)
    (x : LocalMemoryParameters N × Matrix (Fin (N + 1)) (Fin D) ℝ)
    (h : ∀ r j, j ≤ queryRows r → queries r j = other r j) :
    localJointMemoryForward (prefixNearestCodes queries queryRows prototypes prototypeRows) x =
      localJointMemoryForward (prefixNearestCodes other queryRows prototypes prototypeRows) x := by
  rw [prefixNearestCodes_causal queries other queryRows prototypes prototypeRows h]

/-- A genuine future-token change inhabits actual compact-forward causality. -/
example : localJointMemoryForward
    (prefixNearestCodes (fun _ : Fin 1 => fun _ : Fin 2 => (0 : Fin 2)) (fun _ => 0)
      (fun j : Fin 2 => fun _ : Fin 2 => j) (fun _ => 0))
    (localMemoryExampleParameters, Matrix.of (fun j : Fin 2 => fun _ : Fin 1 => (j.val : ℝ))) =
    localJointMemoryForward (prefixNearestCodes
      (fun _ : Fin 1 => fun j : Fin 2 => if j = 0 then 0 else (1 : Fin 2)) (fun _ => 0)
      (fun j : Fin 2 => fun _ : Fin 2 => j) (fun _ => 0))
      (localMemoryExampleParameters, Matrix.of (fun j : Fin 2 => fun _ : Fin 1 => (j.val : ℝ))) := by
  apply localPrefixNearestForward_causal
  intro r j hj
  fin_cases j
  · rfl
  · norm_num at hj

end Transformer.GPTMini.Sparsemax
