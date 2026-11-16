//! Same CM66 scalar solves as reduced_latent at b3aa191c. For an entirely
//! uniform program only the final repeated-add prefix sum is needed; mixed
//! programs retain every sorted/prefix entry, including their uniform rows.
use super::*;
use std::mem::size_of;

pub(super) struct Row {
    sorted: Vec<f64>,
    prefix: Vec<f64>,
    sum: f64,
    pub uniform: bool,
    pub offset: usize,
}
impl Row {
    fn new(values: &Rows, index: usize, compact: bool, limit: u64) -> Result<Self> {
        let uniform = values.repeated(index).is_some();
        let (mut sorted, mut prefix) = (Vec::new(), Vec::new());
        let mut sum = 0.;
        if compact {
            if !uniform {
                return Err("nonuniform row cannot omit threshold prefixes".into());
            }
            for &v in values.row(index).ok_or("invalid projection row")? {
                sum += v;
            }
        } else {
            sorted = memory::vector(values.classes(), limit)?;
            sorted.extend(values.row(index).ok_or("invalid projection row")?.copied());
            sorted.sort_unstable_by(|a, b| b.total_cmp(a));
            prefix = memory::vector(values.classes() + 1, limit)?;
            prefix.push(0.);
            for &v in &sorted {
                prefix.push(prefix.last().unwrap() + v);
            }
            sum = prefix[values.classes()];
        }
        Ok(Self {
            sorted,
            prefix,
            sum,
            uniform,
            offset: index * values.classes(),
        })
    }
    pub fn threshold(&self, mass: f64, work: &mut Work) -> Result<f64> {
        if self.sorted.is_empty() {
            return Err("uniform program does not need a generic threshold".into());
        }
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
fn uniform_program(domain: &Domain, target: &Target, p: usize) -> bool {
    (0..domain.shape.states).all(|s| {
        let at = p * domain.shape.states + s;
        domain.labels[at].is_some()
            || target
                .free
                .repeated((domain.offsets[at] - domain.shape.programs) / domain.shape.classes)
                .is_some()
    })
}
impl Plan {
    pub fn allocation_bytes(plans: &Vec<Self>) -> u128 {
        memory::vector_bytes::<Self>(plans.capacity())
            + plans
                .iter()
                .map(|p| {
                    p.unknown.capacity() as u128 * size_of::<Row>() as u128
                        + p.unknown
                            .iter()
                            .map(|r| (r.sorted.capacity() + r.prefix.capacity()) as u128 * 8)
                            .sum::<u128>()
                })
                .sum::<u128>()
    }
    pub fn prepare(
        domain: &Domain,
        target: &Target,
        limit: u64,
        work: &mut Work,
    ) -> Result<Vec<Self>> {
        let sh = domain.shape;
        work.charge(
            32 * sh.rows() as u128
                + domain.parameters() as u128 * (16 + 8 * sh.classes.ilog2() as u128),
        )?;
        let mut forecast = memory::vector_bytes::<Self>(sh.programs)
            + domain.unknown.len() as u128 * size_of::<Row>() as u128;
        for p in 0..sh.programs {
            if !uniform_program(domain, target, p) {
                forecast +=
                    (sh.states - domain.known[p]) as u128 * (2 * sh.classes + 1) as u128 * 8;
            }
        }
        memory::remaining(limit, forecast)?;
        let mut plans = memory::vector(sh.programs, limit)?;
        for p in 0..sh.programs {
            let linear = target.linear[p];
            let known_weight = (1 + domain.known[p]) as f64;
            let uniform = uniform_program(domain, target, p);
            let mut unknown = memory::vector(sh.states - domain.known[p], limit)?;
            for s in 0..sh.states {
                let at = p * sh.states + s;
                if domain.labels[at].is_none() {
                    let index = (domain.offsets[at] - sh.programs) / sh.classes;
                    unknown.push(Row::new(&target.free, index, uniform, limit)?);
                }
            }
            let uniform_linear = linear
                + unknown
                    .iter()
                    .map(|r| r.sum / sh.classes as f64)
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
        if Self::allocation_bytes(&plans) != forecast {
            return Err("actual projection capacities differ from complete forecast".into());
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
