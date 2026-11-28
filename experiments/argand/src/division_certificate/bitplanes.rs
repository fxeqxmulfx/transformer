//! Lossless fixed-point bit planes for a row of at most 256 finite FP32 values.
//! Plane zero holds the most significant bit of the biased signed numerator.
use crate::Result;
use std::fmt::{self, Write};

type Mask = [u64; 4];

fn set(mask: &mut Mask, j: usize) {
    mask[j / 64] |= 1_u64 << (j % 64);
}
fn get(mask: &Mask, j: usize) -> bool {
    mask[j / 64] >> (j % 64) & 1 != 0
}

pub(super) struct Planes {
    shift: u32,
    negative_zero: Mask,
    planes: Vec<Mask>,
}

// A finite FP32 number in units of 2^-149, as odd significand times 2^shift.
fn magnitude(bits: u32) -> (u32, u32) {
    let exp = (bits >> 23) & 255;
    let sig = (bits & 0x7fffff) | if exp == 0 { 0 } else { 1 << 23 };
    if sig == 0 {
        return (0, 0);
    }
    let zeros = sig.trailing_zeros();
    (sig >> zeros, exp.saturating_sub(1) + zeros)
}

impl Planes {
    pub fn encode(z: &[f32]) -> Result<Self> {
        if z.len() > 256 || z.iter().any(|x| !x.is_finite()) {
            return Err("expected up to 256 finite scores".into());
        }
        let terms: Vec<_> = z.iter().map(|x| magnitude(x.to_bits())).collect();
        let shift = terms
            .iter()
            .filter(|(s, _)| *s != 0)
            .map(|&(_, e)| e)
            .min()
            .unwrap_or(0);
        let width = terms
            .iter()
            .filter(|(s, _)| *s != 0)
            .map(|&(s, e)| 32 - s.leading_zeros() + e - shift)
            .max()
            .unwrap_or(1);
        let mut result = Self {
            shift,
            negative_zero: [0; 4],
            planes: vec![[0; 4]; width as usize + 1],
        };
        for (j, (&x, &(sig, e))) in z.iter().zip(&terms).enumerate() {
            let negative = x.is_sign_negative() && sig != 0;
            if !negative {
                set(&mut result.planes[0], j);
            }
            if x.to_bits() == 0x80000000 {
                set(&mut result.negative_zero, j);
            }
            let low = e.saturating_sub(shift);
            for k in 0..width {
                let bit = sig.checked_shr(k.wrapping_sub(low)).unwrap_or(0) & 1 != 0;
                // Negation keeps the lowest set bit and complements higher bits.
                if if negative && k > low { !bit } else { bit } {
                    set(&mut result.planes[(width - k) as usize], j);
                }
            }
        }
        for (j, x) in z.iter().enumerate() {
            if result.bits(j)? != x.to_bits() {
                return Err("bit-plane encoding changed an FP32 logit".into());
            }
        }
        Ok(result)
    }

    // Reconstruct the raw IEEE fields independently, including subnormal values
    // and signed zero. Every exported coordinate must round-trip bit for bit.
    fn bits(&self, j: usize) -> Result<u32> {
        let negative = !get(&self.planes[0], j);
        let mut magnitude = [false; 277];
        let mut first = None;
        let mut highest = None;
        for (k, p) in self.planes[1..].iter().rev().enumerate() {
            let bit = get(p, j);
            let raw = if negative && first.is_some() {
                !bit
            } else {
                bit
            };
            if bit && first.is_none() {
                first = Some(k);
            }
            if raw {
                let i = k + self.shift as usize;
                *magnitude.get_mut(i).ok_or("FP32 numerator overflow")? = true;
                highest = Some(i);
            }
        }
        let Some(hi) = highest else {
            return Ok(u32::from(get(&self.negative_zero, j)) << 31);
        };
        let offset = hi.saturating_sub(23);
        if magnitude[..offset].iter().any(|&x| x) {
            return Err("fraction not representable in FP32".into());
        }
        let mut fraction = 0_u32;
        for k in 0..23 {
            fraction |= u32::from(magnitude.get(offset + k).copied().unwrap_or(false)) << k;
        }
        let exp = hi.saturating_sub(22) as u32;
        Ok((u32::from(negative) << 31) | (exp << 23) | fraction)
    }

    pub fn write_row(&self, out: &mut String, quotient: usize) -> fmt::Result {
        fn write_mask(out: &mut String, m: &Mask) -> fmt::Result {
            let Some(last) = m.iter().rposition(|&x| x != 0) else {
                out.push('0');
                return Ok(());
            };
            write!(out, "0x{:x}", m[last])?;
            for v in m[..last].iter().rev() {
                write!(out, "{v:016x}")?;
            }
            Ok(())
        }
        write!(out, "  ⟨{quotient}, {}, ", self.shift)?;
        write_mask(out, &self.negative_zero)?;
        out.push_str(", [");
        for (i, m) in self.planes.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            write_mask(out, m)?;
        }
        out.push_str("]⟩");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_zero_subnormals_extreme_exponents_and_all_vocabulary_lanes_round_trip() -> Result<()>
    {
        let edges = [
            0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000, 0x80800000, 0x3f800000,
            0xbf800000, 0x7f7fffff, 0xff7fffff,
        ];
        let z: Vec<_> = (0..239)
            .map(|j| f32::from_bits(edges[j % edges.len()]))
            .collect();
        let p = Planes::encode(&z)?;
        for (j, x) in z.iter().enumerate() {
            assert_eq!(p.bits(j)?, x.to_bits());
        }
        assert!(Planes::encode(&[f32::NAN]).is_err());
        assert!(Planes::encode(&[0.0; 257]).is_err());
        let only_zero = Planes::encode(&[-0.0, 0.0])?;
        assert_eq!(only_zero.bits(0)?, 0x80000000);
        Ok(())
    }
}
