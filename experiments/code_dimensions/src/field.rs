//! Exact arithmetic in a prime field `F_ℓ` with `ℓ ≡ 1 mod n`. There the `n`-th roots of unity
//! exist, and sending `ζ_n` to one of them is a ring homomorphism from `ℤ[ζ_n]`. A minor that
//! vanishes over `ℂ` vanishes after the reduction, so the rank over `F_ℓ` is a certified lower
//! bound on the rank over `ℂ`.

/// `base^exp mod modulus`, for `modulus < 2³²`. O(log exp) time.
pub fn pow_mod(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    let mut out = 1 % modulus;
    base %= modulus;
    while exp > 0 {
        if exp & 1 == 1 {
            out = out * base % modulus;
        }
        base = base * base % modulus;
        exp >>= 1;
    }
    out
}

/// O(√n) time.
pub fn is_prime(n: u64) -> bool {
    n >= 2 && (2..).take_while(|d| d * d <= n).all(|d| n % d != 0)
}

/// The distinct prime factors of `n`. O(√n) time.
pub fn prime_factors(mut n: u64) -> Vec<u64> {
    let mut factors = Vec::new();
    let mut d = 2;
    while d * d <= n {
        if n % d == 0 {
            factors.push(d);
            while n % d == 0 {
                n /= d;
            }
        }
        d += 1;
    }
    if n > 1 {
        factors.push(n);
    }
    factors
}

/// The first `count` primes `ℓ > start` with `ℓ ≡ 1 mod n`.
pub fn primes_above(start: u64, n: u64, count: usize) -> Vec<u64> {
    (start.div_ceil(n)..)
        .map(|t| t * n + 1)
        .filter(|&l| is_prime(l))
        .take(count)
        .collect()
}

/// An element of order exactly `n` in `F_ℓ`, for `n | ℓ - 1`.
pub fn root_of_order(n: u64, l: u64) -> u64 {
    let factors = prime_factors(n);
    (2..l)
        .map(|g| pow_mod(g, (l - 1) / n, l))
        .find(|&z| factors.iter().all(|&q| pow_mod(z, n / q, l) != 1))
        .expect("ℓ ≡ 1 mod n")
}

/// The rank of a row-major `rows × cols` matrix over `F_ℓ`, `ℓ < 2³¹`, by elimination in place.
/// O(rows² cols) time, O(1) extra memory.
pub fn rank_mod(m: &mut [u64], rows: usize, cols: usize, l: u64) -> usize {
    let mut rank = 0;
    for col in 0..cols {
        if rank == rows {
            break;
        }
        let Some(pivot) = (rank..rows).find(|&r| m[r * cols + col] != 0) else {
            continue;
        };
        if pivot != rank {
            for c in col..cols {
                m.swap(pivot * cols + c, rank * cols + c);
            }
        }
        let inverse = pow_mod(m[rank * cols + col], l - 2, l);
        let (head, tail) = m.split_at_mut((rank + 1) * cols);
        let pivot_row = &head[rank * cols..];
        for row in tail.chunks_exact_mut(cols) {
            let factor = row[col] * inverse % l;
            if factor != 0 {
                for c in col..cols {
                    row[c] = (row[c] + l - factor * pivot_row[c] % l) % l;
                }
            }
        }
        rank += 1;
    }
    rank
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_of_a_small_matrix() {
        let mut m = [1, 2, 3, 2, 4, 6, 1, 0, 1];
        assert_eq!(rank_mod(&mut m, 3, 3, 7), 2);
    }

    #[test]
    fn root_has_the_order() {
        let l = primes_above(1 << 30, 97 * 96, 1)[0];
        assert_eq!(l % (97 * 96), 1);
        let z = root_of_order(96, l);
        assert_eq!(pow_mod(z, 96, l), 1);
        assert_ne!(pow_mod(z, 48, l), 1);
        assert_ne!(pow_mod(z, 32, l), 1);
    }
}
