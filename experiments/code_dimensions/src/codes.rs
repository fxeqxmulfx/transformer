//! The tasks `f(a, b)` over `ℤ/p` and their character codes: the additive `ψ_r(f) = ζ_p^{r f}`
//! and the multiplicative `χ^j(f) = ζ_{p-1}^{j log f}`, with `χ^j(0) = 0`.
use crate::field::{pow_mod, prime_factors, root_of_order};
use num_complex::Complex64 as C;

#[derive(Clone, Copy, Debug)]
pub enum Code {
    Additive(u32),
    Multiplicative(u32),
}

pub struct Task {
    pub name: &'static str,
    pub f: fn(u64, u64, u64) -> u64,
}

pub const TASKS: &[Task] = &[
    Task { name: "a³ + ab", f: |a, b, p| (a * a % p * a + a * b) % p },
    Task { name: "ab", f: |a, b, p| a * b % p },
    Task { name: "a²b", f: |a, b, p| a * a % p * b % p },
    Task { name: "a + ab", f: |a, b, p| (a + a * b) % p },
    Task { name: "a² + ab", f: |a, b, p| (a * a + a * b) % p },
    Task { name: "a² + ab + b²", f: |a, b, p| (a * a + a * b + b * b) % p },
    Task {
        name: "(a + b)²",
        f: |a, b, p| {
            let s = (a + b) % p;
            s * s % p
        },
    },
];

/// Discrete logarithms in `ℤ/p` to a primitive root. O(p) time and memory.
pub fn logs(p: u64) -> Vec<u32> {
    let q = p - 1;
    let factors = prime_factors(q);
    let g = (2..p)
        .find(|&g| factors.iter().all(|&f| pow_mod(g, q / f, p) != 1))
        .expect("a prime has a primitive root");
    let mut table = vec![0; p as usize];
    let mut x = 1;
    for k in 0..q {
        table[x as usize] = k as u32;
        x = x * g % p;
    }
    table
}

/// The root of unity `code(value)` as an exponent and its order; `None` for `χ(0) = 0`.
fn exponent(code: Code, value: u64, p: u64, log: &[u32]) -> Option<(u64, u64)> {
    match code {
        Code::Additive(r) => Some((u64::from(r) * value % p, p)),
        Code::Multiplicative(_) if value == 0 => None,
        Code::Multiplicative(j) => {
            Some((u64::from(j) * u64::from(log[value as usize]) % (p - 1), p - 1))
        }
    }
}

/// `F_ℓ` with the powers of a root of order `p` and of one of order `p - 1`.
pub struct Field {
    pub l: u64,
    additive: Vec<u64>,
    multiplicative: Vec<u64>,
}

impl Field {
    pub fn new(l: u64, p: u64) -> Self {
        let powers = |n: u64| {
            let z = root_of_order(n, l);
            let mut out = Vec::with_capacity(n as usize);
            let mut x = 1;
            for _ in 0..n {
                out.push(x);
                x = x * z % l;
            }
            out
        };
        Self { l, additive: powers(p), multiplicative: powers(p - 1) }
    }

    fn value(&self, code: Code, value: u64, p: u64, log: &[u32]) -> u64 {
        exponent(code, value, p, log).map_or(0, |(e, n)| {
            if n == p { self.additive[e as usize] } else { self.multiplicative[e as usize] }
        })
    }
}

/// The row-major `p × p` table of a code over `F_ℓ`. O(p²).
pub fn table_mod(task: &Task, code: Code, p: u64, log: &[u32], field: &Field) -> Vec<u64> {
    let mut out = Vec::with_capacity((p * p) as usize);
    for a in 0..p {
        for b in 0..p {
            out.push(field.value(code, (task.f)(a, b, p), p, log));
        }
    }
    out
}

/// The column-major `p × p` table of a code over `ℂ`. O(p²).
pub fn table_complex(task: &Task, code: Code, p: u64, log: &[u32]) -> Vec<C> {
    let mut out = Vec::with_capacity((p * p) as usize);
    for b in 0..p {
        for a in 0..p {
            out.push(exponent(code, (task.f)(a, b, p), p, log).map_or(C::new(0.0, 0.0), |(e, n)| {
                C::from_polar(1.0, std::f64::consts::TAU * e as f64 / n as f64)
            }));
        }
    }
    out
}
