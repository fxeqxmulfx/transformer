use super::*;

impl Request {
    pub fn validate(&self) -> Result<(Vec<usize>, u64)> {
        let p = &self.policy;
        if !(2..=257).contains(&self.n)
            || !(2..=1024).contains(&self.vocabulary)
            || !(2..=257).contains(&self.modulus)
            || self.observations.is_empty()
            || self.observations.len() > self.n * self.n
            || self.predictions.len() != self.n * self.n
            || self
                .predictions
                .iter()
                .any(|&label| label >= self.vocabulary)
            || !(1..=8192).contains(&p.equations)
            || p.queries > 257 * 257
            || p.reduction_products > 2_000_000_000
            || p.certificate_products > 2_000_000_000
            || p.certificate_operations > 2_000_000
        {
            return Err("decoded certificate input or work limits exceeded".into());
        }
        let mut seen = vec![false; self.n * self.n];
        let mut classes = vec![];
        for &[cell, label] in &self.observations {
            if cell >= seen.len() || label >= self.vocabulary || seen[cell] {
                return Err("decoded certificate has invalid or repeated observed cells".into());
            }
            seen[cell] = true;
            if !classes.contains(&label) {
                classes.push(label);
            }
        }
        if classes.len() < 2 || classes.len() > self.modulus as usize {
            return Err("declared injective observed-class ring family exceeds its domain".into());
        }
        let variables = 2 * self.n + classes.len();
        let bound = self.problem(&classes).validate()?
            + 512 * 1024 * 1024
            + 128 * p.queries.min(self.n * self.n) as u64 * (variables + 64) as u64
            + 256 * p.certificate_operations as u64;
        if bound > 16 * 1024 * 1024 * 1024 {
            return Err("decoded prefix/query/serialization forecast exceeds16GiB".into());
        }
        Ok((classes, bound))
    }
    pub(super) fn row(&self, classes: usize, cell: usize, class: usize) -> Vec<u16> {
        let mut row = vec![0; 2 * self.n + classes];
        row[cell % self.n] = 1;
        row[self.n + cell / self.n] = 1;
        row[2 * self.n + class] = self.modulus - 1;
        row
    }
    pub(super) fn problem(&self, classes: &[usize]) -> row_program::Problem {
        row_program::Problem {
            modulus: self.modulus,
            variables: 2 * self.n + classes.len(),
            rows: self
                .observations
                .iter()
                .take(self.policy.equations)
                .map(|&[cell, label]| {
                    let class = classes
                        .iter()
                        .position(|&c| c == label)
                        .expect("actual observed class");
                    self.row(classes.len(), cell, class)
                })
                .collect(),
            policy: self.prefix_policy.clone(),
        }
    }
}

/// The suffix only changes its receiver. Other rows stay equal to the shared
/// prefix matrix, exactly as in CM28 applyRow and CM29 decoded_of_program.
pub(super) fn suffix(terms: &[Term]) -> (usize, Vec<Op>) {
    let first = &terms[0];
    let mut operations = vec![];
    if first.factor != 1 {
        operations.push(Op::Scale {
            i: first.row,
            factor: first.factor,
        });
    }
    for term in &terms[1..] {
        operations.push(Op::Add {
            i: first.row,
            j: term.row,
            factor: term.factor,
        });
    }
    (first.row, operations)
}
