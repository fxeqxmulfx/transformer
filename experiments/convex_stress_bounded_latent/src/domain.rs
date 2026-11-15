//! Explicit CM64 joint domain. Labels constrain only observed states. Every
//! unknown row remains a full learned vector with the program's shared mass.
use crate::*;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Domain {
    pub shape: Shape,
    pub labels: Vec<Option<usize>>,
    pub minimum_mass: f64,
    pub maximum_mass: f64,
}
impl Domain {
    pub fn new(shape: Shape, records: &[Record], minimum: f64, maximum: f64) -> Result<Self> {
        shape.admit(records.len())?;
        if records.is_empty() {
            return Err("empty original joint training pool".into());
        }
        let mut labels = vec![None; shape.programs * shape.states];
        for row in records {
            Self::record(shape, row)?;
            for (p, &s) in row.states.iter().enumerate() {
                let slot = &mut labels[p * shape.states + s];
                if slot.is_some_and(|y| y != row.label) {
                    return Err(
                        "inconsistent program must be eliminated as an exact zero block".into(),
                    );
                }
                *slot = Some(row.label);
            }
        }
        let out = Self {
            shape,
            labels,
            minimum_mass: minimum,
            maximum_mass: maximum,
        };
        out.validate()?;
        Ok(out)
    }
    pub fn record(shape: Shape, row: &Record) -> Result<()> {
        if row.states.len() != shape.programs
            || row.states.iter().any(|&s| s >= shape.states)
            || row.label >= shape.classes
        {
            return Err("invalid original joint record".into());
        }
        Ok(())
    }
    pub fn validate(&self) -> Result<()> {
        self.shape.admit(0)?;
        if self.labels.len() != self.shape.programs * self.shape.states
            || self
                .labels
                .iter()
                .flatten()
                .any(|&c| c >= self.shape.classes)
            || !self.minimum_mass.is_finite()
            || !self.maximum_mass.is_finite()
            || self.minimum_mass <= 0.
            || self.minimum_mass > self.maximum_mass
            || self.maximum_mass > 64.
        {
            return Err("invalid bounded shared-mass domain".into());
        }
        Ok(())
    }
    pub fn initial(&self, total: f64) -> Result<Parameters> {
        self.validate()?;
        if !total.is_finite() || total < self.minimum_mass || total > self.maximum_mass {
            return Err("initial confidence outside declared interval".into());
        }
        let sh = self.shape;
        let mass = total / sh.programs as f64;
        let mut values = vec![0.; sh.parameters()];
        for p in 0..sh.programs {
            values[p] = mass;
            for s in 0..sh.states {
                for c in 0..sh.classes {
                    values[sh.at(p, s, c)] = match self.labels[p * sh.states + s] {
                        Some(y) => {
                            if c == y {
                                mass
                            } else {
                                0.
                            }
                        }
                        // Uniform unknown rows are feasible initialization,
                        // not an asserted raw target or a fixed decoder rule.
                        None => mass / sh.classes as f64,
                    };
                }
            }
        }
        let q = Parameters { values };
        self.check(&q, 1e-10)?;
        Ok(q)
    }
    pub fn violation(&self, q: &Parameters) -> Result<f64> {
        self.validate()?;
        q.validate(self.shape)?;
        let sh = self.shape;
        let total = q.values[..sh.programs].iter().sum::<f64>();
        let mut error = (self.minimum_mass - total)
            .max(total - self.maximum_mass)
            .max(0.);
        for &v in &q.values {
            error = error.max(-v);
        }
        for p in 0..sh.programs {
            let m = q.values[p];
            for s in 0..sh.states {
                let row = &q.values[sh.at(p, s, 0)..sh.at(p, s, 0) + sh.classes];
                error = error.max((row.iter().sum::<f64>() - m).abs());
                if let Some(y) = self.labels[p * sh.states + s] {
                    for (c, &v) in row.iter().enumerate() {
                        error = error.max((v - if c == y { m } else { 0. }).abs());
                    }
                }
            }
        }
        Ok(error)
    }
    pub fn check(&self, q: &Parameters, tolerance: f64) -> Result<f64> {
        if !tolerance.is_finite() || !(0. ..=1e-7).contains(&tolerance) {
            return Err("invalid joint feasibility tolerance".into());
        }
        let v = self.violation(q)?;
        if v > tolerance {
            return Err(format!("joint feasibility residual{v} exceeds{tolerance}").into());
        }
        Ok(v)
    }
}
