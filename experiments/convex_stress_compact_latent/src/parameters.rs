//! CM70 parameter representation after cc6931a4. Anchor expansion, checks and
//! weighted distance follow reduced_latent at 6797e8ab in original order.
use crate::{
    Result, Work,
    rows::{Builder, Rows},
};
use legacy::domain::Domain;
use std::mem::size_of;

#[derive(Debug)]
pub struct Parameters {
    masses: Vec<f64>,
    free: Rows,
}

fn allocate_masses(programs: usize, limit: u64) -> Result<(Vec<f64>, u64)> {
    if limit == 0 || limit > 96_000_000_000 {
        return Err("invalid compact parameter storage limit".into());
    }
    let overhead = (size_of::<Parameters>() - size_of::<Rows>()) as u128
        + programs as u128 * size_of::<f64>() as u128;
    if overhead >= limit as u128 {
        return Err("complete mass vector exceeds admitted storage".into());
    }
    let mut masses = Vec::new();
    masses.try_reserve_exact(programs)?;
    if masses.capacity() != programs {
        return Err("mass allocator capacity differs from admitted size".into());
    }
    Ok((masses, limit - overhead as u64))
}

impl Parameters {
    /// Reference import; the full input tensor is separately owned and must
    /// be charged by the caller. Production initialization never expands it.
    pub fn from_legacy(
        domain: &Domain,
        source: &legacy::Parameters,
        limit: u64,
        work: &mut Work,
    ) -> Result<Self> {
        domain.validate(work)?;
        source.validate(domain)?;
        let (mut masses, remaining) = allocate_masses(domain.shape.programs, limit)?;
        masses.extend_from_slice(&source.values[..domain.shape.programs]);
        let free = Rows::from_dense(
            domain.shape.classes,
            &source.values[domain.shape.programs..],
            remaining,
        )?;
        Ok(Self { masses, free })
    }
    pub fn initial(domain: &Domain, total: f64, limit: u64, work: &mut Work) -> Result<Self> {
        domain.validate(work)?;
        if !total.is_finite() || total < domain.minimum_mass || total > domain.maximum_mass {
            return Err("initial confidence outside bounded domain".into());
        }
        work.charge(8 * domain.parameters() as u128)?;
        let (mut masses, remaining) = allocate_masses(domain.shape.programs, limit)?;
        let mass = total / domain.shape.programs as f64;
        masses.resize(domain.shape.programs, mass);
        let mut free = Builder::new(domain.shape.classes, domain.unknown.len(), remaining)?;
        for _ in &domain.unknown {
            free.push_repeated(mass / domain.shape.classes as f64)?;
        }
        let out = Self {
            masses,
            free: free.finish()?,
        };
        out.check(domain, 1e-10, work)?;
        Ok(out)
    }
    pub fn validate(&self, domain: &Domain) -> Result<()> {
        if self.masses.len() != domain.shape.programs
            || self.free.rows() != domain.unknown.len()
            || self.free.classes() != domain.shape.classes
            || self.values().any(|x| !x.is_finite() || x.abs() > 1e12)
        {
            return Err("invalid compact parameter tensor".into());
        }
        Ok(())
    }
    pub fn masses(&self) -> &[f64] {
        &self.masses
    }
    pub fn unknown_rows(&self) -> &Rows {
        &self.free
    }
    pub fn values(&self) -> impl Iterator<Item = &f64> {
        self.masses.iter().chain(self.free.iter())
    }
    pub fn allocation_bytes(&self) -> u64 {
        (size_of::<Self>() - size_of::<Rows>() + self.masses.capacity() * size_of::<f64>()) as u64
            + self.free.allocation_bytes()
    }
    // Internal callers validate the complete dictionary and tensor first.
    pub(crate) fn gamma(&self, domain: &Domain, p: usize, s: usize, c: usize) -> f64 {
        let row = p * domain.shape.states + s;
        match domain.labels[row] {
            Some(y) => {
                if c == y {
                    self.masses[p]
                } else {
                    0.
                }
            }
            None => *self
                .free
                .get(
                    (domain.offsets[row] - domain.shape.programs) / domain.shape.classes,
                    c,
                )
                .unwrap(),
        }
    }
    pub fn check(&self, domain: &Domain, tolerance: f64, work: &mut Work) -> Result<f64> {
        self.validate(domain)?;
        if !tolerance.is_finite() || !(0. ..=1e-7).contains(&tolerance) {
            return Err("invalid reduced feasibility tolerance".into());
        }
        work.charge(8 * domain.parameters() as u128)?;
        let total: f64 = self.masses.iter().sum();
        let mut error = (domain.minimum_mass - total)
            .max(total - domain.maximum_mass)
            .max(0.);
        for &x in self.values() {
            error = error.max(-x);
        }
        for (i, &(p, _)) in domain.unknown.iter().enumerate() {
            error = error.max((self.free.row(i).unwrap().sum::<f64>() - self.masses[p]).abs());
        }
        if error > tolerance {
            return Err(format!("reduced feasibility residual {error}").into());
        }
        Ok(error)
    }
    /// Full CM65 distance: all anchored mass copies and every unknown class.
    pub fn distance(&self, other: &Self, domain: &Domain, work: &mut Work) -> Result<f64> {
        self.validate(domain)?;
        other.validate(domain)?;
        work.charge(8 * domain.parameters() as u128)?;
        let masses: f64 = (0..domain.shape.programs)
            .map(|p| (1 + domain.known[p]) as f64 * (self.masses[p] - other.masses[p]).powi(2))
            .sum();
        Ok(masses
            + self
                .free
                .iter()
                .zip(other.free.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>())
    }
}
