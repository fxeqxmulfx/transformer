//! Full CM66 proposal, with every original unknown coordinate retained.
//! Arithmetic ported from reduced_latent at b3aa191c, following CM70 storage.
use super::*;
use std::mem::size_of;

#[derive(Debug)]
pub struct Target {
    pub(super) linear: Vec<f64>,
    pub(super) free: Rows,
}
impl Target {
    pub fn from_legacy(
        domain: &Domain,
        old: &legacy::projection::Target,
        limit: u64,
        work: &mut Work,
    ) -> Result<Self> {
        domain::validate(domain, work)?;
        old.validate(domain)?;
        let mut linear = memory::vector(domain.shape.programs, limit)?;
        linear.extend_from_slice(&old.linear);
        let room = memory::remaining(limit, memory::vector_bytes::<f64>(linear.capacity()))?;
        let free = Rows::from_dense(domain.shape.classes, &old.free, room)?;
        Ok(Self { linear, free })
    }
    pub fn validate(&self, domain: &Domain) -> Result<()> {
        if self.linear.len() != domain.shape.programs
            || self.free.rows() != domain.unknown.len()
            || self.free.classes() != domain.shape.classes
            || self
                .linear
                .iter()
                .chain(self.free.iter())
                .any(|x| !x.is_finite() || x.abs() > 1e15)
        {
            return Err("invalid reduced weighted proposal".into());
        }
        Ok(())
    }
    pub fn linear(&self) -> &[f64] {
        &self.linear
    }
    pub fn unknown_rows(&self) -> &Rows {
        &self.free
    }
    pub fn allocation_bytes(&self) -> u64 {
        (size_of::<Self>() - size_of::<Rows>()) as u64
            + self.linear.capacity() as u64 * 8
            + self.free.allocation_bytes()
    }
    pub fn step(
        domain: &Domain,
        q: &Parameters,
        gradient: &Parameters,
        rate: f64,
        limit: u64,
        work: &mut Work,
    ) -> Result<Self> {
        q.validate(domain)?;
        gradient.validate(domain)?;
        if !rate.is_finite() || !(1e-6..=1e6).contains(&rate) {
            return Err("invalid gradient rate".into());
        }
        work.charge(12 * domain.parameters() as u128)?;
        let mut linear = memory::vector(domain.shape.programs, limit)?;
        for j in 0..domain.shape.programs {
            linear.push((1 + domain.known[j]) as f64 * q.masses()[j] - rate * gradient.masses()[j]);
        }
        let room = memory::remaining(limit, memory::vector_bytes::<f64>(linear.capacity()))?;
        let free = q
            .unknown_rows()
            .zip_map(gradient.unknown_rows(), room, |q, g| q - rate * g)?;
        let out = Self { linear, free };
        out.validate(domain)?;
        Ok(out)
    }
    pub fn objective(&self, domain: &Domain, q: &Parameters) -> Result<f64> {
        self.validate(domain)?;
        q.validate(domain)?;
        let masses: f64 = (0..domain.shape.programs)
            .map(|j| {
                let k = (1 + domain.known[j]) as f64;
                k * (q.masses()[j] - self.linear[j] / k).powi(2)
            })
            .sum();
        Ok(masses
            + q.unknown_rows()
                .iter()
                .zip(self.free.iter())
                .map(|(q, v)| (q - v).powi(2))
                .sum::<f64>())
    }
}
