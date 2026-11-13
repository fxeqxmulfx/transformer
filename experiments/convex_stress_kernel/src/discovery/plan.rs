//! CM34 categorical compaction and original matrix construction.
//! Unlike the one-direction producer, every supplied equation is retained.
use super::*;
use std::collections::BTreeSet;
fn name(names: &mut Vec<usize>, value: usize) {
    if !names.contains(&value) {
        names.push(value);
    }
}
impl Request {
    pub(crate) fn names(&self) -> Result<Names> {
        let i = &self.input;
        if !(2..=257).contains(&i.n)
            || !(2..=1024).contains(&i.vocabulary)
            || i.observations.is_empty()
            || i.observations.len() > i.n * i.n
            || !finite_system::prime(self.prime)
            || !(1..=8190).contains(&i.policy.equations)
            || i.policy.elimination.nullity > 8
            || i.policy.elimination.scalar_updates > 1_000_000_000
            || i.policy.elimination.row_operations > 2_000_000
        {
            return Err("invalid bounded complete-kernel request".into());
        }
        let mut names = Names {
            rows: vec![],
            columns: vec![],
            classes: vec![],
        };
        let mut cells = BTreeSet::new();
        for &[cell, label] in &i.observations {
            if cell >= i.n * i.n || label >= i.vocabulary || !cells.insert(cell) {
                return Err("invalid or repeated original cell/label".into());
            }
            name(&mut names.rows, cell % i.n);
            name(&mut names.columns, cell / i.n);
            name(&mut names.classes, label);
        }
        Ok(names)
    }
    pub(crate) fn refusal(&self, names: &Names) -> Option<&'static str> {
        if self.input.observations.len() > self.input.policy.equations {
            Some("all_observed_equation_cap")
        } else if names.rows.len() + names.columns.len() + names.classes.len() > 1024 {
            Some("coefficient_variable_cap")
        } else {
            None
        }
    }
    pub(crate) fn problem(&self, names: &Names) -> Result<finite_system::Problem> {
        if self.refusal(names).is_some() {
            return Err("refused matrix cannot be allocated".into());
        }
        let (nr, nc) = (names.rows.len(), names.columns.len());
        let variables = nr + nc + names.classes.len();
        let mut rows = Vec::with_capacity(self.input.observations.len() + 2);
        for &[cell, label] in &self.input.observations {
            let mut row = vec![0; variables];
            row[names
                .rows
                .iter()
                .position(|&r| r == cell % self.input.n)
                .unwrap()] = 1;
            row[nr
                + names
                    .columns
                    .iter()
                    .position(|&c| c == cell / self.input.n)
                    .unwrap()] = 1;
            row[nr + nc + names.classes.iter().position(|&c| c == label).unwrap()] = self.prime - 1;
            rows.push(row);
        }
        for coordinate in [0, nr + nc] {
            let mut row = vec![0; variables];
            row[coordinate] = 1;
            rows.push(row);
        }
        Ok(finite_system::Problem {
            system: finite_system::System {
                prime: self.prime,
                variables,
                right: vec![0; rows.len()],
                rows,
            },
            policy: self.input.policy.elimination.clone(),
        })
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Code {
    pub prime: u16,
    pub dimension: usize,
    pub names: Names,
    pub rows: Vec<Vec<u16>>,
    pub columns: Vec<Vec<u16>>,
    pub classes: Vec<Vec<u16>>,
}
impl Code {
    pub(crate) fn from_directions(names: Names, prime: u16, basis: &[Vec<u16>]) -> Result<Self> {
        let (nr, nc) = (names.rows.len(), names.columns.len());
        let variables = nr + nc + names.classes.len();
        if !finite_system::prime(prime)
            || basis.len() > 8
            || basis
                .iter()
                .any(|v| v.len() != variables || v.iter().any(|&x| x >= prime))
        {
            return Err("invalid vector-code basis dimensions".into());
        }
        let coordinates: Vec<Vec<_>> = (0..variables)
            .map(|v| basis.iter().map(|d| d[v]).collect())
            .collect();
        Ok(Self {
            prime,
            dimension: basis.len(),
            names,
            rows: coordinates[..nr].into(),
            columns: coordinates[nr..nr + nc].into(),
            classes: coordinates[nr + nc..].into(),
        })
    }
    pub fn query(&self, row: usize, column: usize) -> Option<Vec<u16>> {
        let i = self.names.rows.iter().position(|&r| r == row)?;
        let j = self.names.columns.iter().position(|&c| c == column)?;
        Some(
            self.rows[i]
                .iter()
                .zip(&self.columns[j])
                .map(|(&a, &b)| ((a as u32 + b as u32) % self.prime as u32) as u16)
                .collect(),
        )
    }
    pub fn label(&self, label: usize) -> Option<&[u16]> {
        self.names
            .classes
            .iter()
            .position(|&c| c == label)
            .map(|c| self.classes[c].as_slice())
    }
    pub fn validate(&self, n: usize, vocabulary: usize) -> Result<()> {
        if !finite_system::prime(self.prime)
            || self.dimension > 8
            || self.rows.len() != self.names.rows.len()
            || self.columns.len() != self.names.columns.len()
            || self.classes.len() != self.names.classes.len()
        {
            return Err("invalid full-vector code shape".into());
        }
        for (names, values, bound) in [
            (&self.names.rows, &self.rows, n),
            (&self.names.columns, &self.columns, n),
            (&self.names.classes, &self.classes, vocabulary),
        ] {
            if names.is_empty()
                || names.len() > bound
                || names.iter().any(|&x| x >= bound)
                || names.iter().collect::<BTreeSet<_>>().len() != names.len()
                || values
                    .iter()
                    .any(|v| v.len() != self.dimension || v.iter().any(|&x| x >= self.prime))
            {
                return Err("invalid categorical names or vector coordinates".into());
            }
        }
        Ok(())
    }
}
