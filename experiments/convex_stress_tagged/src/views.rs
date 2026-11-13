//! CM25: propose categorical constant strips from observed agreement only.
//! Missing axes are explicitly flagged for a proposed constant, never claimed
//! identified. Unequal active axis counts are padded with unobserved categories.
use crate::{Model, Result, Sample};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct View {
    pub n: usize,
    pub constant: Option<usize>,
    pub rows: Vec<Option<usize>>,
    pub columns: Vec<Option<usize>>,
    pub row_support: Vec<usize>,
    pub column_support: Vec<usize>,
    pub missing_rows: Vec<usize>,
    pub missing_columns: Vec<usize>,
    pub observations: Vec<[usize; 2]>,
    pub stripped_observations: usize,
    pub row_padding: usize,
    pub column_padding: usize,
    pub reason: String,
}
impl View {
    pub fn samples(&self) -> Vec<Sample> {
        self.observations
            .iter()
            .map(|&[cell, label]| Sample { cell, label })
            .collect()
    }
    pub fn lift(&self, base: &Model) -> Result<Model> {
        if base.n != self.n {
            return Err("view/base dimension mismatch".into());
        }
        if let Some(c) = self.constant {
            base.lift(c, &self.rows, &self.columns)
        } else if self.rows == (0..base.n).map(Some).collect::<Vec<_>>()
            && self.columns == self.rows
        {
            base.validate()?;
            Ok(base.clone())
        } else {
            Err("nonidentity untagged view".into())
        }
    }
}
fn mapping(flags: &[bool]) -> (Vec<Option<usize>>, usize) {
    let mut count = 0;
    let map = flags
        .iter()
        .map(|&flag| {
            if flag {
                None
            } else {
                let i = count;
                count += 1;
                Some(i)
            }
        })
        .collect();
    (map, count)
}
pub fn propose(
    n: usize,
    vocab: usize,
    samples: &[Sample],
    minimum_support: usize,
) -> Result<Vec<View>> {
    crate::validate_observations(n, vocab, samples)?;
    if !(1..=257).contains(&minimum_support) {
        return Err("invalid strip support threshold".into());
    }
    let (mut rows, mut columns) = (vec![vec![]; n], vec![vec![]; n]);
    let mut classes = vec![];
    for o in samples {
        rows[o.cell % n].push(o.label);
        columns[o.cell / n].push(o.label);
        if !classes.contains(&o.label) {
            classes.push(o.label);
        }
    }
    let r_support: Vec<_> = rows.iter().map(Vec::len).collect();
    let c_support: Vec<_> = columns.iter().map(Vec::len).collect();
    let missing_rows: Vec<_> = (0..n).filter(|&i| rows[i].is_empty()).collect();
    let missing_columns: Vec<_> = (0..n).filter(|&i| columns[i].is_empty()).collect();
    let mut out = vec![View {
        n,
        constant: None,
        rows: (0..n).map(Some).collect(),
        columns: (0..n).map(Some).collect(),
        row_support: r_support.clone(),
        column_support: c_support.clone(),
        missing_rows: missing_rows.clone(),
        missing_columns: missing_columns.clone(),
        observations: samples.iter().map(|o| [o.cell, o.label]).collect(),
        stripped_observations: 0,
        row_padding: 0,
        column_padding: 0,
        reason: "base_view".into(),
    }];
    for constant in classes {
        let supported =
            |s: &Vec<usize>| s.len() >= minimum_support && s.iter().all(|&c| c == constant);
        if !rows.iter().chain(&columns).any(supported) {
            continue;
        }
        let r_flags: Vec<_> = rows.iter().map(|s| s.is_empty() || supported(s)).collect();
        let c_flags: Vec<_> = columns
            .iter()
            .map(|s| s.is_empty() || supported(s))
            .collect();
        let (r, nr) = mapping(&r_flags);
        let (c, nc) = mapping(&c_flags);
        let interior_n = nr.max(nc);
        let mut observations = vec![];
        let mut stripped = 0;
        for o in samples {
            if let (Some(i), Some(j)) = (r[o.cell % n], c[o.cell / n]) {
                observations.push([i + interior_n * j, o.label]);
            } else {
                if o.label != constant {
                    return Err("observed strip proposal contradicts its constant".into());
                }
                stripped += 1;
            }
        }
        let reason = if interior_n < 2 || observations.is_empty() {
            "empty_or_small_interior"
        } else {
            "observed_constant_strips"
        };
        out.push(View {
            n: interior_n,
            constant: Some(constant),
            rows: r,
            columns: c,
            row_support: r_support.clone(),
            column_support: c_support.clone(),
            missing_rows: missing_rows.clone(),
            missing_columns: missing_columns.clone(),
            observations,
            stripped_observations: stripped,
            row_padding: interior_n - nr,
            column_padding: interior_n - nc,
            reason: reason.into(),
        });
    }
    Ok(out)
}

/// Rebuild the original observation partition, including flags and support.
/// It must be checked before treating interior descent as full-data descent.
pub fn replay(
    n: usize,
    vocab: usize,
    samples: &[Sample],
    minimum_support: usize,
    views: &[View],
) -> Result<()> {
    if propose(n, vocab, samples, minimum_support)? != views {
        return Err("categorical strip partition replay mismatch".into());
    }
    Ok(())
}
