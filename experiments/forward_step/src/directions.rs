//! Fixed A-only directions, normalized in independent real coordinates.
//! See PROTOCOL.md; scaling realized AdamW motion is a parameter probe,
//! not a claim of equivalence to rerunning AdamW at another learning rate.
use crate::{
    coordinates as coord,
    domain::Result,
    domain::model::{self, Arm},
    inputs::require,
};
use serde_json::{Value, json};
pub const SCALES: [f64; 5] = [1., 0.25, 0.0625, 0.015625, 0.00390625];
pub struct Direction {
    pub name: &'static str,
    pub independent: Vec<f64>,
}
pub struct Directions {
    pub choices: [Direction; 2],
    pub gradient: Vec<f64>,
    pub checks: Value,
}
pub fn build(arm: Arm, old: &[f32], updated: &[f32], gradient: &[f32]) -> Result<Directions> {
    let a = coord::groups()[1].1.clone();
    let expanded: Vec<_> = updated
        .iter()
        .zip(old)
        .map(|(&x, &y)| x as f64 - y as f64)
        .collect();
    let mut d = coord::owners(arm, &expanded);
    let mut g = coord::pullback(arm, &gradient.iter().map(|&x| x as f64).collect::<Vec<_>>());
    for k in 0..coord::COUNT {
        if !a.contains(&k) {
            d[k] = 0.;
            g[k] = 0.;
        }
    }
    let radius = coord::norm(&d);
    let gradient_norm = coord::norm(&g);
    require(
        radius.is_finite() && gradient_norm.is_finite(),
        "nonfinite direction norm",
    )?;
    let raw: Vec<_> = g
        .iter()
        .map(|&v| {
            if gradient_norm > 0. {
                -radius * v / gradient_norm
            } else {
                0.
            }
        })
        .collect();
    let norm_error = (coord::norm(&raw) - radius).abs();
    let cosine = if radius * gradient_norm > 0. {
        Some(coord::dot(&g, &raw) / (gradient_norm * coord::norm(&raw)))
    } else {
        None
    };
    let passed = (gradient_norm == 0. || norm_error <= 1e-12 * (1. + radius))
        && cosine.is_none_or(|c| (c + 1.).abs() <= 1e-12);
    let checks = json!({"passed":passed,"radius":radius,"local_gradient_norm":gradient_norm,
        "norm_match_error":norm_error,"negative_gradient_cosine":cosine,
        "zero_adamw_direction":radius==0.,"zero_gradient":gradient_norm==0.});
    Ok(Directions {
        choices: [
            Direction {
                name: "adamw",
                independent: d,
            },
            Direction {
                name: "gradient",
                independent: raw,
            },
        ],
        gradient: g,
        checks,
    })
}
pub fn displace(arm: Arm, old: &[f32], direction: &[f64], scale: f64) -> Vec<f32> {
    let mut owners = coord::owners(arm, &old.iter().map(|&x| x as f64).collect::<Vec<_>>());
    for i in coord::groups()[1].1.clone() {
        owners[i] = ((owners[i] + scale * direction[i]) as f32) as f64;
    }
    let expanded = coord::expand(arm, &owners);
    let range = coord::expanded_groups(arm)[1].1.clone();
    let mut new = old.to_vec();
    for i in range {
        new[i] = expanded[i] as f32;
    }
    new
}
pub fn valid(arm: Arm, old: &[f32], new: &[f32]) -> bool {
    let a = coord::expanded_groups(arm)[1].1.clone();
    model::valid(arm, new)
        && old
            .iter()
            .zip(new)
            .enumerate()
            .all(|(i, (&x, &y))| a.contains(&i) || x.to_bits() == y.to_bits())
}
pub fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn independent_rounding_keeps_other_blocks_and_complex_aliases() {
        for arm in Arm::ALL {
            let p = model::initialize(arm, 7);
            let a = coord::groups()[1].1.start;
            let mut d = vec![0.; coord::COUNT];
            d[a] = 0.125;
            d[a + 1] = -0.25;
            let q = displace(arm, &p, &d, 0.25);
            assert!(valid(arm, &p, &q));
            let old = coord::owners(arm, &p.iter().map(|&v| v as f64).collect::<Vec<_>>());
            let new = coord::owners(arm, &q.iter().map(|&v| v as f64).collect::<Vec<_>>());
            assert_eq!(new[a], ((old[a] + 0.03125) as f32) as f64);
            assert_eq!(new[a + 1], ((old[a + 1] - 0.0625) as f32) as f64);
            assert!(same_bits(&p, &displace(arm, &p, &d, 0.)));
        }
    }
    #[test]
    fn negative_gradient_has_matching_length_and_handles_zero_without_nan() {
        let arm = Arm::Swiglu;
        let p = vec![0.; coord::COUNT];
        let mut q = p.clone();
        let mut g = p.clone();
        let a = coord::groups()[1].1.start;
        q[a] = 0.25;
        g[a] = 3.;
        g[a + 1] = 4.;
        let directions = build(arm, &p, &q, &g).unwrap();
        assert_eq!(directions.checks["passed"], true);
        assert!((directions.choices[1].independent[a] + 0.15).abs() < 1e-15);
        assert!((directions.choices[1].independent[a + 1] + 0.20).abs() < 1e-15);
        assert!(
            build(arm, &p, &q, &p).unwrap().choices[1]
                .independent
                .iter()
                .all(|&x| x == 0.)
        );
    }
}
