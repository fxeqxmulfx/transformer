//! Scalar dual solves for full shared-mass decoder rows. The uniform-row
//! fast path gives the same projection as the generic simplex solve; it does
//! not fix unknown rows or remove their coordinates from the model domain.
use super::*;

pub(super) struct Row {
    sorted: Vec<f64>,
    prefix: Vec<f64>,
    pub uniform: bool,
}
impl Row {
    fn new(values: &[f64]) -> Self {
        let uniform = values.iter().all(|v| v.to_bits() == values[0].to_bits());
        let mut sorted = values.to_vec();
        sorted.sort_unstable_by(|a, b| b.total_cmp(a));
        let mut prefix = Vec::with_capacity(values.len() + 1);
        prefix.push(0.);
        for &v in &sorted {
            prefix.push(prefix.last().unwrap() + v);
        }
        Self {
            sorted,
            prefix,
            uniform,
        }
    }
    pub fn threshold(&self, mass: f64, work: &mut Work) -> Result<f64> {
        work.charge(16 + 8 * self.sorted.len().ilog2() as u128)?;
        if mass == 0. {
            return Ok(self.sorted[0]);
        }
        let (mut lo, mut hi) = (1, self.sorted.len());
        while lo < hi {
            let mid = (lo + hi).div_ceil(2);
            if self.sorted[mid - 1] > (self.prefix[mid] - mass) / mid as f64 {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        Ok((self.prefix[lo] - mass) / lo as f64)
    }
}

pub(super) struct Plan {
    linear: f64,
    known_weight: f64,
    pub unknown: Vec<Row>,
    pub uniform: bool,
    uniform_linear: f64,
    uniform_weight: f64,
}
impl Plan {
    pub fn prepare(domain: &Domain, target: &Parameters, work: &mut Work) -> Result<Vec<Self>> {
        let sh = domain.shape;
        work.charge(sh.parameters() as u128 * (16 + 8 * sh.classes.ilog2() as u128))?;
        let mut plans = Vec::with_capacity(sh.programs);
        for p in 0..sh.programs {
            let mut linear = target.values[p];
            let mut known_weight = 1.;
            let mut unknown = Vec::with_capacity(sh.states);
            for s in 0..sh.states {
                if let Some(y) = domain.labels[p * sh.states + s] {
                    linear += target.values[sh.at(p, s, y)];
                    known_weight += 1.;
                } else {
                    unknown.push(Row::new(
                        &target.values[sh.at(p, s, 0)..sh.at(p, s, 0) + sh.classes],
                    ));
                }
            }
            let uniform = unknown.iter().all(|r| r.uniform);
            let uniform_linear = linear
                + unknown
                    .iter()
                    .map(|r| r.prefix[sh.classes] / sh.classes as f64)
                    .sum::<f64>();
            let uniform_weight = known_weight + unknown.len() as f64 / sh.classes as f64;
            plans.push(Self {
                linear,
                known_weight,
                unknown,
                uniform,
                uniform_linear,
                uniform_weight,
            });
        }
        Ok(plans)
    }
    fn derivative(&self, mass: f64, work: &mut Work) -> Result<f64> {
        work.charge(4 + self.unknown.len() as u128)?;
        let mut d = self.known_weight * mass - self.linear;
        for row in &self.unknown {
            d -= row.threshold(mass, work)?;
        }
        Ok(d)
    }
    pub fn mass(&self, lambda: f64, work: &mut Work) -> Result<f64> {
        work.charge(8)?;
        if self.uniform {
            return Ok(((self.uniform_linear - lambda) / self.uniform_weight).max(0.));
        }
        if self.derivative(0., work)? + lambda >= 0. {
            return Ok(0.);
        }
        let (mut lo, mut hi) = (0., 1.);
        while self.derivative(hi, work)? + lambda < 0. {
            hi *= 2.;
            if hi > 1e18 {
                return Err("joint row mass could not be bracketed".into());
            }
        }
        for _ in 0..80 {
            let mid = (lo + hi) / 2.;
            if self.derivative(mid, work)? + lambda > 0. {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        Ok((lo + hi) / 2.)
    }
}
