//! Finite f64 values are converted from their exact binary representation.
//! Compute every saved coefficient in two different complete contractions;
//! no ideal theta/decoder formula replaces the saved tensor. num-rational0.4.2
//! from_float uses integer_decode; to_f64 uses integer round-to-even division.
use crate::*;
use num_rational::BigRational as Q;
use num_traits::{One, Signed, ToPrimitive, Zero};

pub(crate) fn rational(x: f64) -> Q {
    Q::from_float(x).expect("finite inputs checked before exact arithmetic")
}
fn check(q: &Q, cap: u64) -> Result<()> {
    if q.numer().bits() > cap || q.denom().bits() > cap {
        return Err("exact product integer precision cap exceeded".into());
    }
    Ok(())
}

pub(crate) fn products(e: &Evaluator<'_>, rows: &[Features]) -> Result<(Vec<Q>, Vec<Q>, Q)> {
    let v = e.forecast.classes;
    let mut real = vec![Q::zero(); v];
    let mut complex = vec![Q::zero(); v];
    let mut normalizer = Q::zero();
    let weights: Vec<_> = e.coefficients.iter().copied().map(rational).collect();
    for (p, h) in rows.iter().enumerate() {
        for x in h {
            normalizer += rational(*x).abs();
        }
        normalizer += Q::one();
        for k in 0..16 {
            let phi: Q = row(h, k / 8)
                .into_iter()
                .enumerate()
                .map(|(s, x)| rational(x) * rational(source(k % 8, s)))
                .sum();
            for (c, z) in real.iter_mut().enumerate() {
                *z += &phi * &weights[(p * 16 + k) * v + c];
            }
        }
    }
    // Class, program, row, frequency order differs from real feature-first
    // contraction. Re(U*H) includes the independently constructed imaginary term.
    for (c, z) in complex.iter_mut().enumerate() {
        for (p, h) in rows.iter().enumerate() {
            for b in 0..2 {
                for (s, x) in row(h, b).into_iter().enumerate() {
                    let x = rational(x);
                    for f in 0..4 {
                        let phase = (s * f) % 4;
                        let re = rational([0.5, 0., -0.5, 0.][phase]);
                        let im = rational([0., 0.5, 0., -0.5][phase]);
                        let k = (p * 16 + b * 8 + 2 * f) * v + c;
                        *z += &x * (re * &weights[k] - im * &weights[k + v]);
                    }
                }
            }
        }
    }
    normalizer = normalizer.max(Q::one());
    for q in real
        .iter()
        .chain(&complex)
        .chain(std::iter::once(&normalizer))
    {
        check(q, e.forecast.exact_integer_bit_cap)?;
    }
    if real != complex {
        return Err("exact independent complete real/complex numerators differ".into());
    }
    Ok((real, complex, normalizer))
}

pub(crate) fn canonical(numerators: &[Q], normalizer: &Q, cap: u64) -> Result<Scores> {
    let maximum = numerators.iter().max().ok_or("empty exact class vector")?;
    let maximizers: Vec<_> = numerators
        .iter()
        .enumerate()
        .filter_map(|(c, x)| (x == maximum).then_some(c))
        .collect();
    let best = maximizers[0];
    let mut logits = Vec::with_capacity(numerators.len());
    let mut width = 0.0f64;
    for x in numerators {
        let shifted = (x - maximum) / normalizer;
        check(&shifted, cap)?;
        let rounded = shifted
            .to_f64()
            .ok_or("exact normalized score cannot convert")?;
        if !rounded.is_finite() || (!shifted.is_zero() && rounded == 0.) {
            return Err("exact strict gap lost in binary64 conversion".into());
        }
        // Verify the claimed rounding enclosure by exact rational comparisons;
        // the conversion implementation is not trusted as an argmax oracle.
        let lo = rounded.next_down();
        let hi = rounded.next_up();
        if !lo.is_finite() || !hi.is_finite() || rational(lo) > shifted || rational(hi) < shifted {
            return Err("rounded exact score escaped its outward enclosure".into());
        }
        width = width.max((hi - lo).next_up());
        logits.push(rounded);
    }
    if joint::argmax(&logits) != best {
        return Err("rounded exact raw winner differs".into());
    }
    Ok(Scores {
        logits,
        predicted: best,
        maximizers,
        method: Method::ExactRationals,
        independent_residual: 0.,
        logit_enclosure_width: width,
        target_realizability_certified: false,
    })
}

pub(crate) fn evaluate(e: &Evaluator<'_>, rows: &[Features]) -> Result<Scores> {
    let (real, complex, n) = products(e, rows)?;
    let primary = canonical(&real, &n, e.forecast.exact_integer_bit_cap)?;
    let independent = canonical(&complex, &n, e.forecast.exact_integer_bit_cap)?;
    if primary != independent {
        return Err("independent exact normalized vectors differ".into());
    }
    Ok(primary)
}
