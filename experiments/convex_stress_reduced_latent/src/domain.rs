//! Explicit reduced CM65 domain. Its dictionaries are rechecked against
//! every original record before training. No unseen completion is imposed.
use crate::*;

#[derive(Clone, Debug, PartialEq)]
pub struct Domain {
    pub shape: Shape,
    pub labels: Vec<Option<usize>>,
    pub offsets: Vec<usize>,
    pub unknown: Vec<(usize, usize)>,
    pub known: Vec<usize>,
    pub minimum_mass: f64,
    pub maximum_mass: f64,
    pub head_columns_bound: usize,
}
impl Domain {
    pub fn record(shape: Shape, row: &Record) -> Result<()> {
        if row.states.len() != shape.programs
            || row.label >= shape.classes
            || row.states.iter().any(|&s| s >= shape.states)
        {
            return Err("invalid original reduced decoder record".into());
        }
        Ok(())
    }
    pub fn new(
        shape: Shape,
        records: &[Record],
        minimum: f64,
        maximum: f64,
        work: &mut Work,
    ) -> Result<Self> {
        shape.admit(records.len(), 0, 0)?;
        if records.is_empty() {
            return Err("empty original training records".into());
        }
        work.charge(
            16 * shape.rows() as u128 + 8 * records.len() as u128 * shape.programs as u128,
        )?;
        let mut labels = vec![None; shape.rows()];
        for row in records {
            Self::record(shape, row)?;
            for (p, &s) in row.states.iter().enumerate() {
                let slot = &mut labels[p * shape.states + s];
                if slot.is_some_and(|y| y != row.label) {
                    return Err(
                        "conflicting candidate requires exact zero-block elimination".into(),
                    );
                }
                *slot = Some(row.label);
            }
        }
        let out = Self::from_labels(shape, labels, minimum, maximum, work)?;
        out.admit(records.len())?;
        Ok(out)
    }
    pub fn from_labels(
        shape: Shape,
        labels: Vec<Option<usize>>,
        minimum: f64,
        maximum: f64,
        work: &mut Work,
    ) -> Result<Self> {
        shape.admit(0, 0, 0)?;
        if labels.len() != shape.rows()
            || labels.iter().flatten().any(|&y| y >= shape.classes)
            || !minimum.is_finite()
            || !maximum.is_finite()
            || minimum <= 0.
            || minimum > maximum
            || maximum > 64.
        {
            return Err("invalid reduced domain dictionary or mass bounds".into());
        }
        work.charge(32 * shape.rows() as u128)?;
        let count = labels.iter().filter(|x| x.is_none()).count();
        let columns: usize = labels
            .as_chunks::<4>()
            .0
            .iter()
            .map(|block| {
                if block.iter().any(Option::is_none) {
                    shape.classes
                } else {
                    4.min(shape.classes)
                }
            })
            .sum();
        shape.admit(0, count, columns)?;
        let mut offsets = vec![usize::MAX; shape.rows()];
        let mut unknown = Vec::with_capacity(count);
        let mut known = vec![0; shape.programs];
        for (i, label) in labels.iter().enumerate() {
            let (p, s) = (i / shape.states, i % shape.states);
            if label.is_some() {
                known[p] += 1;
            } else {
                offsets[i] = shape.programs + unknown.len() * shape.classes;
                unknown.push((p, s));
            }
        }
        Ok(Self {
            shape,
            labels,
            offsets,
            unknown,
            known,
            minimum_mass: minimum,
            maximum_mass: maximum,
            head_columns_bound: columns,
        })
    }
    pub fn parameters(&self) -> usize {
        self.shape.programs + self.unknown.len() * self.shape.classes
    }
    pub fn admit(&self, records: usize) -> Result<u64> {
        self.shape
            .admit(records, self.unknown.len(), self.head_columns_bound)
    }
    pub fn validate(&self, work: &mut Work) -> Result<()> {
        self.admit(0)?;
        let rebuilt = Self::from_labels(
            self.shape,
            self.labels.clone(),
            self.minimum_mass,
            self.maximum_mass,
            work,
        )?;
        if *self != rebuilt {
            return Err("reduced domain indices differ from dictionary".into());
        }
        Ok(())
    }
    pub fn initial(&self, total: f64, work: &mut Work) -> Result<Parameters> {
        self.validate(work)?;
        if !total.is_finite() || total < self.minimum_mass || total > self.maximum_mass {
            return Err("initial confidence outside bounded domain".into());
        }
        work.charge(8 * self.parameters() as u128)?;
        let mass = total / self.shape.programs as f64;
        let mut values = vec![mass / self.shape.classes as f64; self.parameters()];
        values[..self.shape.programs].fill(mass);
        let q = Parameters { values };
        self.check(&q, 1e-10, work)?;
        Ok(q)
    }
    pub fn check(&self, q: &Parameters, tolerance: f64, work: &mut Work) -> Result<f64> {
        q.validate(self)?;
        if !tolerance.is_finite() || !(0. ..=1e-7).contains(&tolerance) {
            return Err("invalid reduced feasibility tolerance".into());
        }
        work.charge(8 * self.parameters() as u128)?;
        let total: f64 = q.values[..self.shape.programs].iter().sum();
        let mut error = (self.minimum_mass - total)
            .max(total - self.maximum_mass)
            .max(0.);
        for &x in &q.values {
            error = error.max(-x);
        }
        for &(p, s) in &self.unknown {
            let at = self.offsets[p * self.shape.states + s];
            error = error.max(
                (q.values[at..at + self.shape.classes].iter().sum::<f64>() - q.values[p]).abs(),
            );
        }
        if error > tolerance {
            return Err(format!("reduced feasibility residual {error}").into());
        }
        Ok(error)
    }
    /// CM65 full coordinate distance, including every eliminated mass copy.
    pub fn distance(&self, a: &Parameters, b: &Parameters, work: &mut Work) -> Result<f64> {
        a.validate(self)?;
        b.validate(self)?;
        work.charge(8 * self.parameters() as u128)?;
        let masses: f64 = (0..self.shape.programs)
            .map(|p| (1 + self.known[p]) as f64 * (a.values[p] - b.values[p]).powi(2))
            .sum();
        Ok(masses
            + a.values[self.shape.programs..]
                .iter()
                .zip(&b.values[self.shape.programs..])
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>())
    }
}
