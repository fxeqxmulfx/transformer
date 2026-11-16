//! All nonzero source entries are signed halves. A common denominator2^1075
//! represents every such product of a finite binary64 coefficient, including
//! subnormals. Bounds/ordering concern the saved tensor, not an ideal decoder.
//! Ported from reduced_latent at 6797e8ab. Shared columns are still visited
//! separately; no arithmetic is replaced by multiplying their multiplicity.
use super::*;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, ToPrimitive, Zero};

fn half_numerator(x: f64, sign: i8) -> BigInt {
    if sign == 0 || x == 0. {
        return BigInt::zero();
    }
    let bits = x.to_bits();
    let exponent = (bits >> 52) & 0x7ff;
    let fraction = bits & ((1u64 << 52) - 1);
    let mut n = if exponent == 0 {
        BigInt::from(fraction)
    } else {
        BigInt::from(fraction | (1u64 << 52)) << (exponent as usize - 1)
    };
    if x.is_sign_negative() != (sign < 0) {
        n = -n;
    }
    n
}
pub(super) fn predict(head: &Head, states: &[usize], work: &mut Work) -> Result<Prediction> {
    // Includes two contractions, 128-limb ceilings per addition/shift,
    // classwise rational conversion, and its independent enclosure checks.
    work.charge(
        8192 * (head.query(states)? as u128
            + head.shape.programs as u128 * head.shape.classes as u128)
            + 1_000_000 * head.shape.classes as u128,
    )?;
    let mut real = vec![BigInt::zero(); head.shape.classes];
    let mut independent = vec![BigInt::zero(); head.shape.classes];
    for (p, &s) in states.iter().enumerate() {
        for f in s / 4 * 4..s / 4 * 4 + 4 {
            let (u, v) = source(s, f);
            let b = &head.blocks[(p * head.shape.states + s) / 4];
            for (c, value) in head.columns(b) {
                real[c] += half_numerator(value[f % 4], (2. * u) as i8);
                real[c] += half_numerator(value[4 + f % 4], (-2. * v) as i8);
            }
        }
    }
    for (c, z) in independent.iter_mut().enumerate() {
        for (p, &s) in states.iter().enumerate().rev() {
            for f in (s / 4 * 4..s / 4 * 4 + 4).rev() {
                let (u, v) = reference_source(s, f);
                let (re, im) = head.coefficient(p, f, c)?;
                *z += half_numerator(re, u) - half_numerator(im, v);
            }
        }
    }
    if real != independent || real.iter().any(|n| n.bits() > 8192) {
        return Err("independent exact block products disagree or exceed 8192 bits".into());
    }
    let maximum = real.iter().max().ok_or("empty exact logits")?;
    let maximizers: Vec<_> = real
        .iter()
        .enumerate()
        .filter_map(|(c, z)| (z == maximum).then_some(c))
        .collect();
    let denominator = BigInt::one() << 1075usize;
    let mut logits = Vec::with_capacity(head.shape.classes);
    let mut width = 0.0f64;
    for n in &real {
        let q = BigRational::new(n - maximum, denominator.clone());
        let z = q.to_f64().ok_or("exact dyadic score conversion failed")?;
        if !z.is_finite() || (!q.is_zero() && z == 0.) {
            return Err("exact strict gap lost during binary64 conversion".into());
        }
        let (lo, hi) = (z.next_down(), z.next_up());
        if !lo.is_finite()
            || !hi.is_finite()
            || BigRational::from_float(lo).unwrap() > q
            || BigRational::from_float(hi).unwrap() < q
        {
            return Err("exact dyadic conversion escaped its enclosure".into());
        }
        width = width.max((hi - lo).next_up());
        logits.push(z);
    }
    let winner = maximizers[0];
    if predict::argmax(&logits) != winner {
        return Err("exact raw winner was lost".into());
    }
    Ok(Prediction {
        logits,
        winner,
        maximizers,
        certificate: Certificate::ExactDyadics,
        maximum_independent_residual: 0.,
        maximum_enclosure_width: width,
        target_realizability_certified: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dyadic_conversion_matches_independent_rational_products() {
        let denominator = BigInt::one() << 1075usize;
        let mut values = vec![
            0.,
            -0.,
            1.,
            -1.,
            1e12,
            -1e-12,
            f64::MIN_POSITIVE,
            f64::from_bits(1),
            -f64::from_bits(1),
            f64::from_bits((1u64 << 52) - 1),
        ];
        let mut state = 11u64;
        for _ in 0..100 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let exponent = (state % 1065) << 52;
            values.push(f64::from_bits((state & ((1u64 << 52) - 1)) | exponent));
        }
        for x in values {
            for sign in [-1, 0, 1] {
                let direct = BigRational::from_float(x).unwrap()
                    * BigRational::new(BigInt::from(sign), BigInt::from(2));
                assert_eq!(
                    BigRational::new(half_numerator(x, sign), denominator.clone()),
                    direct
                );
            }
        }
    }
}
