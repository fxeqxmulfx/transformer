module

public import Transformer.Weil.HasseDeriv
public import Transformer.Weil.Independence
public import Transformer.Weil.Auxiliary
public import Transformer.Weil.LinearSystem
public import Transformer.Weil.Stepanov
public import Transformer.Weil.Hyperelliptic

/-!
# Stepanov's method for `y² = f(x)`

Over a finite field `F` of odd order `q`, for `f ∈ F[X]` of odd degree `m`, Stepanov's method
counts the points with `f(x)^{(q−1)/2} = a` with an auxiliary polynomial
`R = ∑_{j < J} (r_j + s_j f^{(q−1)/2}) X^{q j}` that vanishes to order `ℓ` at each of them:

* `HasseDeriv`: Hasse derivatives; `X^{q j}` passes through `E^{(k)}` for `k < q`, and vanishing
  to order `ℓ` on `S` forces `ℓ |S| ≤ deg R`;
* `Independence`: `R ≠ 0` unless every `r_j`, `s_j` is `0`, for `deg r_j, deg s_j < K` with
  `m + 2K ≤ q + 1`;
* `Auxiliary`: at a point with `f(x)^{(q−1)/2} = a`, `f^k E^{(k)} R` takes the value of a
  polynomial `T_k`, linear in the `r_j`, `s_j`;
* `LinearSystem`: Siegel's lemma makes every `T_k`, `k < ℓ`, vanish when
  `ℓ (K + J + (ℓ − 1)(m − 1)) < 2 J K`;
* `Stepanov`: then `ℓ · #{x : f(x)^{(q−1)/2} = a} ≤ K − 1 + m (q − 1)/2 + q (J − 1)`;
* `Hyperelliptic`: for `q ≥ 64` and `8 m ≤ q`, `#{x : f(x)^{(q−1)/2} = a} ≤ q/2 + 3 m √q`,
  `|∑_x χ₂(f(x))| ≤ 7 m √q` and `|#{(x, y) : y² = f(x)} − q| ≤ 7 m √q`.

The proofs are written from the mathematics (S. A. Stepanov, 1969; W. M. Schmidt, *Equations
over Finite Fields*, Ch. I); no other formalization is used.
-/
