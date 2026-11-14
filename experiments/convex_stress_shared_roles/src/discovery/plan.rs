//! CM40 actual matrix: every grouped observed row, two legitimate gauges,
//! then all aligned row-minus-column differences against the first alignment.
//! The anchor's zero row is retained. Limits precede dense matrix allocation.
use super::*;
use std::collections::BTreeSet;
mod code;
pub(super) struct Plan {
    pub names: Names,
    pub raw_names: Vec<usize>,
    pub alignments: Vec<[usize; 2]>,
    pub shape: [usize; 2],
}
fn append(names: &mut Vec<usize>, value: usize) {
    if !names.contains(&value) {
        names.push(value);
    }
}
impl Plan {
    pub fn new(r: &Request) -> Result<Self> {
        let i = &r.input;
        limits::dimensions(i.n, i.vocabulary)?;
        if !finite_system::prime(r.prime)
            || !(2..=1024).contains(&r.answers)
            || r.readout.len() != i.vocabulary
            || r.readout.iter().any(|&d| d >= r.answers)
            || r.readout.iter().copied().collect::<BTreeSet<_>>().len() != r.answers
            || i.observations.is_empty()
            || i.observations.len() > i.n * i.n
            || !(1..=8190).contains(&i.policy.equations)
            || i.policy.elimination.nullity > 8
            || i.policy.elimination.scalar_updates > 1_000_000_000
            || i.policy.elimination.row_operations > 2_000_000
        {
            return Err("invalid bounded shared-role request or declared readout".into());
        }
        let mut names = Names {
            rows: vec![],
            columns: vec![],
            classes: vec![],
        };
        let (mut raw_names, mut cells) = (vec![], BTreeSet::new());
        for &[cell, label] in &i.observations {
            if cell >= i.n * i.n || label >= i.vocabulary || !cells.insert(cell) {
                return Err("invalid or repeated algebraic observation cell".into());
            }
            append(&mut names.rows, cell % i.n);
            append(&mut names.columns, cell / i.n);
            append(&mut names.classes, r.readout[label]);
            append(&mut raw_names, label);
        }
        let alignments: Vec<_> = names
            .rows
            .iter()
            .enumerate()
            .filter_map(|(i, token)| {
                names
                    .columns
                    .iter()
                    .position(|c| c == token)
                    .map(|j| [i, j])
            })
            .collect();
        let shape = [
            i.observations.len() + 2 + alignments.len(),
            names.rows.len() + names.columns.len() + names.classes.len(),
        ];
        Ok(Self {
            names,
            raw_names,
            alignments,
            shape,
        })
    }
    pub fn buffer_bytes(&self, r: &Request) -> u64 {
        // Four complete solver/checker buffers, plus independent headroom.
        4 * (512 * 1024 * 1024
            + 8 * self.shape[0] as u64 * (self.shape[1] + 1) as u64
            + 64 * r.input.policy.elimination.row_operations as u64)
            + 512 * 1024 * 1024
    }
    pub fn refusal(&self, r: &Request) -> Option<&'static str> {
        if r.answers != usize::from(r.prime) {
            Some("answer_code_cardinality_mismatch")
        } else if self.alignments.is_empty() {
            Some("no_shared_categorical_identity")
        } else if self.shape[0] > r.input.policy.equations + 2 {
            Some("all_role_equation_cap")
        } else if self.shape[1] > 1024 {
            Some("coefficient_variable_cap")
        } else if self.buffer_bytes(r) > 8 * 1024 * 1024 * 1024 {
            Some("discovery_memory_cap")
        } else {
            None
        }
    }
    pub fn matrix(&self, r: &Request) -> Result<finite_system::Problem> {
        if self.refusal(r).is_some() {
            return Err("refused role matrix cannot allocate".into());
        }
        let (nr, nc) = (self.names.rows.len(), self.names.columns.len());
        let mut rows = Vec::with_capacity(self.shape[0]);
        for &[cell, label] in &r.input.observations {
            let mut row = vec![0; self.shape[1]];
            row[self
                .names
                .rows
                .iter()
                .position(|&x| x == cell % r.input.n)
                .unwrap()] = 1;
            row[nr
                + self
                    .names
                    .columns
                    .iter()
                    .position(|&x| x == cell / r.input.n)
                    .unwrap()] = 1;
            row[nr
                + nc
                + self
                    .names
                    .classes
                    .iter()
                    .position(|&x| x == r.readout[label])
                    .unwrap()] = r.prime - 1;
            rows.push(row);
        }
        for coordinate in [0, nr + nc] {
            let mut row = vec![0; self.shape[1]];
            row[coordinate] = 1;
            rows.push(row);
        }
        let [i0, j0] = self.alignments[0];
        for &[i, j] in &self.alignments {
            let mut row = vec![0; self.shape[1]];
            for (k, sign) in [(i, 1_i32), (nr + j, -1), (i0, -1), (nr + j0, 1)] {
                row[k] = (i32::from(row[k]) + sign).rem_euclid(i32::from(r.prime)) as u16;
            }
            rows.push(row);
        }
        let problem = finite_system::Problem {
            system: finite_system::System {
                prime: r.prime,
                variables: self.shape[1],
                right: vec![0; rows.len()],
                rows,
            },
            policy: r.input.policy.elimination.clone(),
        };
        if self.buffer_bytes(r) != 4 * problem.validate()? + 512 * 1024 * 1024 {
            return Err("role pre-allocation forecast differs from actual matrix".into());
        }
        Ok(problem)
    }
}
/// The actual augmented matrix is public for independent certificate audits.
pub fn problem(request: &Request) -> Result<finite_system::Problem> {
    Plan::new(request)?.matrix(request)
}
